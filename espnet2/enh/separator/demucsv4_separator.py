# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# First author is Simon Rouard.
"""
This code contains the spectrogram and Hybrid version of Demucs
from https://github.com/facebookresearch/demucs and modified 
for our use.
"""
import functools
import math
import random
import typing as tp
from collections import OrderedDict
from copy import deepcopy
from fractions import Fraction
from typing import Dict, List, Optional, Tuple

import julius
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from openunmix.filtering import wiener

from espnet2.enh.separator.abs_separator import AbsSeparator


def capture_init(init):
    @functools.wraps(init)
    def __init__(self, *args, **kwargs):
        self._init_args_kwargs = (args, kwargs)
        init(self, *args, **kwargs)

    return __init__


def unfold(a, kernel_size, stride):
    """Given input of size [*OT, T], output Tensor of size [*OT, F, K]
    with K the kernel size, by extracting frames with the given stride.

    This will pad the input so that `F = ceil(T / K)`.

    see https://github.com/pytorch/pytorch/issues/60466
    """
    *shape, length = a.shape
    n_frames = math.ceil(length / stride)
    tgt_length = (n_frames - 1) * stride + kernel_size
    a = F.pad(a, (0, tgt_length - length))
    strides = list(a.stride())
    assert strides[-1] == 1, "data should be contiguous"
    strides = strides[:-1] + [stride, 1]
    return a.as_strided([*shape, n_frames, kernel_size], strides)


def center_trim(tensor: torch.Tensor, reference: tp.Union[torch.Tensor, int]):
    """
    Center trim `tensor` with respect to `reference`, along the last dimension.
    `reference` can also be a number, representing the length to trim to.
    If the size difference != 0 mod 2, the extra sample is removed on the right side.
    """
    ref_size: int
    if isinstance(reference, torch.Tensor):
        ref_size = reference.size(-1)
    else:
        ref_size = reference
    delta = tensor.size(-1) - ref_size
    if delta < 0:
        raise ValueError("tensor must be larger than reference. " f"Delta is {delta}.")
    if delta:
        tensor = tensor[..., delta // 2 : -(delta - delta // 2)]
    return tensor


def spectro(x, n_fft=512, hop_length=None, pad=0):
    *other, length = x.shape
    x = x.reshape(-1, length)
    is_mps = x.device.type == "mps"
    if is_mps:
        x = x.cpu()
    z = torch.stft(
        x,
        n_fft * (1 + pad),
        hop_length or n_fft // 4,
        window=torch.hann_window(n_fft).to(x),
        win_length=n_fft,
        normalized=True,
        center=True,
        return_complex=True,
        pad_mode="reflect",
    )
    _, freqs, frame = z.shape
    return z.view(*other, freqs, frame)


def ispectro(z, hop_length=None, length=None, pad=0):
    *other, freqs, frames = z.shape
    n_fft = 2 * freqs - 2
    z = z.view(-1, freqs, frames)
    win_length = n_fft // (1 + pad)
    is_mps = z.device.type == "mps"
    if is_mps:
        z = z.cpu()
    x = torch.istft(
        z,
        n_fft,
        hop_length,
        window=torch.hann_window(win_length).to(z.real),
        win_length=win_length,
        normalized=True,
        length=length,
        center=True,
    )
    _, length = x.shape
    return x.view(*other, length)


class BLSTM(nn.Module):
    """
    BiLSTM with same hidden units as input dim.
    If `max_steps` is not None, input will be splitting in overlapping
    chunks and the LSTM applied separately on each chunk.
    """

    def __init__(self, dim, layers=1, max_steps=None, skip=False):
        super().__init__()
        assert max_steps is None or max_steps % 4 == 0
        self.max_steps = max_steps
        self.lstm = nn.LSTM(
            bidirectional=True, num_layers=layers, hidden_size=dim, input_size=dim
        )
        self.linear = nn.Linear(2 * dim, dim)
        self.skip = skip

    def forward(self, x):
        B, C, T = x.shape
        y = x
        framed = False
        if self.max_steps is not None and T > self.max_steps:
            width = self.max_steps
            stride = width // 2
            frames = unfold(x, width, stride)
            nframes = frames.shape[2]
            framed = True
            x = frames.permute(0, 2, 1, 3).reshape(-1, C, width)

        x = x.permute(2, 0, 1)

        x = self.lstm(x)[0]
        x = self.linear(x)
        x = x.permute(1, 2, 0)
        if framed:
            out = []
            frames = x.reshape(B, -1, C, width)
            limit = stride // 2
            for k in range(nframes):
                if k == 0:
                    out.append(frames[:, k, :, :-limit])
                elif k == nframes - 1:
                    out.append(frames[:, k, :, limit:])
                else:
                    out.append(frames[:, k, :, limit:-limit])
            out = torch.cat(out, -1)
            out = out[..., :T]
            x = out
        if self.skip:
            x = x + y
        return x


def rescale_conv(conv, reference):
    """Rescale initial weight scale. It is unclear why it helps but it certainly does."""
    std = conv.weight.std().detach()
    scale = (std / reference) ** 0.5
    conv.weight.data /= scale
    if conv.bias is not None:
        conv.bias.data /= scale


def rescale_module(module, reference):
    for sub in module.modules():
        if isinstance(
            sub, (nn.Conv1d, nn.ConvTranspose1d, nn.Conv2d, nn.ConvTranspose2d)
        ):
            rescale_conv(sub, reference)


class DConv(nn.Module):
    """
    New residual branches in each encoder layer.
    This alternates dilated convolutions, potentially with LSTMs and attention.
    Also before entering each residual branch, dimension is projected on a smaller subspace,
    e.g. of dim `channels // compress`.
    """

    def __init__(
        self,
        channels: int,
        compress: float = 4,
        depth: int = 2,
        init: float = 1e-4,
        norm=True,
        attn=False,
        heads=4,
        ndecay=4,
        lstm=False,
        gelu=True,
        kernel=3,
        dilate=True,
    ):
        """
        Args:
            channels: input/output channels for residual branch.
            compress: amount of channel compression inside the branch.
            depth: number of layers in the residual branch. Each layer has its own
                projection, and potentially LSTM and attention.
            init: initial scale for LayerNorm.
            norm: use GroupNorm.
            attn: use LocalAttention.
            heads: number of heads for the LocalAttention.
            ndecay: number of decay controls in the LocalAttention.
            lstm: use LSTM.
            gelu: Use GELU activation.
            kernel: kernel size for the (dilated) convolutions.
            dilate: if true, use dilation, increasing with the depth.
        """

        super().__init__()
        assert kernel % 2 == 1
        self.channels = channels
        self.compress = compress
        self.depth = abs(depth)
        dilate = depth > 0

        norm_fn: tp.Callable[[int], nn.Module]
        norm_fn = lambda d: nn.Identity()  # noqa
        if norm:
            norm_fn = lambda d: nn.GroupNorm(1, d)  # noqa

        hidden = int(channels / compress)

        act: tp.Type[nn.Module]
        if gelu:
            act = nn.GELU
        else:
            act = nn.ReLU

        self.layers = nn.ModuleList([])
        for d in range(self.depth):
            dilation = 2**d if dilate else 1
            padding = dilation * (kernel // 2)
            mods = [
                nn.Conv1d(channels, hidden, kernel, dilation=dilation, padding=padding),
                norm_fn(hidden),
                act(),
                nn.Conv1d(hidden, 2 * channels, 1),
                norm_fn(2 * channels),
                nn.GLU(1),
                LayerScale(channels, init),
            ]
            if attn:
                mods.insert(3, LocalState(hidden, heads=heads, ndecay=ndecay))
            if lstm:
                mods.insert(3, BLSTM(hidden, layers=2, max_steps=200, skip=True))
            layer = nn.Sequential(*mods)
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return x


class LocalState(nn.Module):
    """Local state allows to have attention based only on data (no positional embedding),
    but while setting a constraint on the time window (e.g. decaying penalty term).

    Also a failed experiments with trying to provide some frequency based attention.
    """

    def __init__(self, channels: int, heads: int = 4, nfreqs: int = 0, ndecay: int = 4):
        super().__init__()
        assert channels % heads == 0, (channels, heads)
        self.heads = heads
        self.nfreqs = nfreqs
        self.ndecay = ndecay
        self.content = nn.Conv1d(channels, channels, 1)
        self.query = nn.Conv1d(channels, channels, 1)
        self.key = nn.Conv1d(channels, channels, 1)
        if nfreqs:
            self.query_freqs = nn.Conv1d(channels, heads * nfreqs, 1)
        if ndecay:
            self.query_decay = nn.Conv1d(channels, heads * ndecay, 1)
            # Initialize decay close to zero (there is a sigmoid), for maximum initial window.
            self.query_decay.weight.data *= 0.01
            assert self.query_decay.bias is not None  # stupid type checker
            self.query_decay.bias.data[:] = -2
        self.proj = nn.Conv1d(channels + heads * nfreqs, channels, 1)

    def forward(self, x):
        B, C, T = x.shape
        heads = self.heads
        indexes = torch.arange(T, device=x.device, dtype=x.dtype)
        # left index are keys, right index are queries
        delta = indexes[:, None] - indexes[None, :]

        queries = self.query(x).view(B, heads, -1, T)
        keys = self.key(x).view(B, heads, -1, T)
        # t are keys, s are queries
        dots = torch.einsum("bhct,bhcs->bhts", keys, queries)
        dots /= keys.shape[2] ** 0.5
        if self.nfreqs:
            periods = torch.arange(1, self.nfreqs + 1, device=x.device, dtype=x.dtype)
            freq_kernel = torch.cos(2 * math.pi * delta / periods.view(-1, 1, 1))
            freq_q = self.query_freqs(x).view(B, heads, -1, T) / self.nfreqs**0.5
            dots += torch.einsum("fts,bhfs->bhts", freq_kernel, freq_q)
        if self.ndecay:
            decays = torch.arange(1, self.ndecay + 1, device=x.device, dtype=x.dtype)
            decay_q = self.query_decay(x).view(B, heads, -1, T)
            decay_q = torch.sigmoid(decay_q) / 2
            decay_kernel = -decays.view(-1, 1, 1) * delta.abs() / self.ndecay**0.5
            dots += torch.einsum("fts,bhfs->bhts", decay_kernel, decay_q)

        # Kill self reference.
        dots.masked_fill_(torch.eye(T, device=dots.device, dtype=torch.bool), -100)
        weights = torch.softmax(dots, dim=2)

        content = self.content(x).view(B, heads, -1, T)
        result = torch.einsum("bhts,bhct->bhcs", weights, content)
        if self.nfreqs:
            time_sig = torch.einsum("bhts,fts->bhfs", weights, freq_kernel)
            result = torch.cat([result, time_sig], 2)
        result = result.reshape(B, -1, T)
        return x + self.proj(result)


class Demucs(nn.Module):
    @capture_init
    def __init__(
        self,
        sources,
        # Channels
        audio_channels=2,
        channels=64,
        growth=2.0,
        # Main structure
        depth=6,
        rewrite=True,
        lstm_layers=0,
        # Convolutions
        kernel_size=8,
        stride=4,
        context=1,
        # Activations
        gelu=True,
        glu=True,
        # Normalization
        norm_starts=4,
        norm_groups=4,
        # DConv residual branch
        dconv_mode=1,
        dconv_depth=2,
        dconv_comp=4,
        dconv_attn=4,
        dconv_lstm=4,
        dconv_init=1e-4,
        # Pre/post processing
        normalize=True,
        resample=True,
        # Weight init
        rescale=0.1,
        # Metadata
        samplerate=44100,
        segment=4 * 10,
    ):
        """
        Args:
            sources (list[str]): list of source names
            audio_channels (int): stereo or mono
            channels (int): first convolution channels
            depth (int): number of encoder/decoder layers
            growth (float): multiply (resp divide) number of channels by that
                for each layer of the encoder (resp decoder)
            depth (int): number of layers in the encoder and in the decoder.
            rewrite (bool): add 1x1 convolution to each layer.
            lstm_layers (int): number of lstm layers, 0 = no lstm. Deactivated
                by default, as this is now replaced by the smaller and faster small LSTMs
                in the DConv branches.
            kernel_size (int): kernel size for convolutions
            stride (int): stride for convolutions
            context (int): kernel size of the convolution in the
                decoder before the transposed convolution. If > 1,
                will provide some context from neighboring time steps.
            gelu: use GELU activation function.
            glu (bool): use glu instead of ReLU for the 1x1 rewrite conv.
            norm_starts: layer at which group norm starts being used.
                decoder layers are numbered in reverse order.
            norm_groups: number of groups for group norm.
            dconv_mode: if 1: dconv in encoder only, 2: decoder only, 3: both.
            dconv_depth: depth of residual DConv branch.
            dconv_comp: compression of DConv branch.
            dconv_attn: adds attention layers in DConv branch starting at this layer.
            dconv_lstm: adds a LSTM layer in DConv branch starting at this layer.
            dconv_init: initial scale for the DConv branch LayerScale.
            normalize (bool): normalizes the input audio on the fly, and scales back
                the output by the same amount.
            resample (bool): upsample x2 the input and downsample /2 the output.
            rescale (float): rescale initial weights of convolutions
                to get their standard deviation closer to `rescale`.
            samplerate (int): stored as meta information for easing
                future evaluations of the model.
            segment (float): duration of the chunks of audio to ideally evaluate the model on.
                This is used by `demucs.apply.apply_model`.
        """

        super().__init__()
        self.audio_channels = audio_channels
        self.sources = sources
        self.kernel_size = kernel_size
        self.context = context
        self.stride = stride
        self.depth = depth
        self.resample = resample
        self.channels = channels
        self.normalize = normalize
        self.samplerate = samplerate
        self.segment = segment
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.skip_scales = nn.ModuleList()

        if glu:
            activation = nn.GLU(dim=1)
            ch_scale = 2
        else:
            activation = nn.ReLU()
            ch_scale = 1
        if gelu:
            act2 = nn.GELU
        else:
            act2 = nn.ReLU

        in_channels = audio_channels
        padding = 0
        for index in range(depth):
            norm_fn = lambda d: nn.Identity()  # noqa
            if index >= norm_starts:
                norm_fn = lambda d: nn.GroupNorm(norm_groups, d)  # noqa

            encode = []
            encode += [
                nn.Conv1d(in_channels, channels, kernel_size, stride),
                norm_fn(channels),
                act2(),
            ]
            attn = index >= dconv_attn
            lstm = index >= dconv_lstm
            if dconv_mode & 1:
                encode += [
                    DConv(
                        channels,
                        depth=dconv_depth,
                        init=dconv_init,
                        compress=dconv_comp,
                        attn=attn,
                        lstm=lstm,
                    )
                ]
            if rewrite:
                encode += [
                    nn.Conv1d(channels, ch_scale * channels, 1),
                    norm_fn(ch_scale * channels),
                    activation,
                ]
            self.encoder.append(nn.Sequential(*encode))

            decode = []
            if index > 0:
                out_channels = in_channels
            else:
                out_channels = len(self.sources) * audio_channels
            if rewrite:
                decode += [
                    nn.Conv1d(
                        channels, ch_scale * channels, 2 * context + 1, padding=context
                    ),
                    norm_fn(ch_scale * channels),
                    activation,
                ]
            if dconv_mode & 2:
                decode += [
                    DConv(
                        channels,
                        depth=dconv_depth,
                        init=dconv_init,
                        compress=dconv_comp,
                        attn=attn,
                        lstm=lstm,
                    )
                ]
            decode += [
                nn.ConvTranspose1d(
                    channels, out_channels, kernel_size, stride, padding=padding
                )
            ]
            if index > 0:
                decode += [norm_fn(out_channels), act2()]
            self.decoder.insert(0, nn.Sequential(*decode))
            in_channels = channels
            channels = int(growth * channels)

        channels = in_channels
        if lstm_layers:
            self.lstm = BLSTM(channels, lstm_layers)
        else:
            self.lstm = None

        if rescale:
            rescale_module(self, reference=rescale)

    def valid_length(self, length):
        """
        Return the nearest valid length to use with the model so that
        there is no time steps left over in a convolution, e.g. for all
        layers, size of the input - kernel_size % stride = 0.

        Note that input are automatically padded if necessary to ensure that the output
        has the same length as the input.
        """
        if self.resample:
            length *= 2

        for _ in range(self.depth):
            length = math.ceil((length - self.kernel_size) / self.stride) + 1
            length = max(1, length)

        for idx in range(self.depth):
            length = (length - 1) * self.stride + self.kernel_size

        if self.resample:
            length = math.ceil(length / 2)
        return int(length)

    def forward(self, mix):
        x = mix
        length = x.shape[-1]

        if self.normalize:
            mono = mix.mean(dim=1, keepdim=True)
            mean = mono.mean(dim=-1, keepdim=True)
            std = mono.std(dim=-1, keepdim=True)
            x = (x - mean) / (1e-5 + std)
        else:
            mean = 0
            std = 1

        delta = self.valid_length(length) - length
        x = F.pad(x, (delta // 2, delta - delta // 2))

        if self.resample:
            x = julius.resample_frac(x, 1, 2)

        saved = []
        for encode in self.encoder:
            x = encode(x)
            saved.append(x)

        if self.lstm:
            x = self.lstm(x)

        for decode in self.decoder:
            skip = saved.pop(-1)
            skip = center_trim(skip, x)
            x = decode(x + skip)

        if self.resample:
            x = julius.resample_frac(x, 2, 1)
        x = x * std + mean
        x = center_trim(x, length)
        x = x.view(x.size(0), len(self.sources), self.audio_channels, x.size(-1))
        return x

    def load_state_dict(self, state, strict=True):
        # fix a mismatch with previous generation Demucs models.
        for idx in range(self.depth):
            for a in ["encoder", "decoder"]:
                for b in ["bias", "weight"]:
                    new = f"{a}.{idx}.3.{b}"
                    old = f"{a}.{idx}.2.{b}"
                    if old in state and new not in state:
                        state[new] = state.pop(old)
        super().load_state_dict(state, strict=strict)


def create_sin_embedding(
    length: int, dim: int, shift: int = 0, device="cpu", max_period=10000
):
    # We aim for TBC format
    assert dim % 2 == 0
    pos = shift + torch.arange(length, device=device).view(-1, 1, 1)
    half_dim = dim // 2
    adim = torch.arange(dim // 2, device=device).view(1, 1, -1)
    phase = pos / (max_period ** (adim / (half_dim - 1)))
    return torch.cat(
        [
            torch.cos(phase),
            torch.sin(phase),
        ],
        dim=-1,
    )


def create_2d_sin_embedding(d_model, height, width, device="cpu", max_period=10000):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError(
            "Cannot use sin/cos positional encoding with "
            "odd dimension (got dim={:d})".format(d_model)
        )
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(
        torch.arange(0.0, d_model, 2) * -(math.log(max_period) / d_model)
    )
    pos_w = torch.arange(0.0, width).unsqueeze(1)
    pos_h = torch.arange(0.0, height).unsqueeze(1)
    pe[0:d_model:2, :, :] = (
        torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    )
    pe[1:d_model:2, :, :] = (
        torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    )
    pe[d_model::2, :, :] = (
        torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    )
    pe[d_model + 1 :: 2, :, :] = (
        torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    )

    return pe[None, :].to(device)


def create_sin_embedding_cape(
    length: int,
    dim: int,
    batch_size: int,
    mean_normalize: bool,
    augment: bool,  # True during training
    max_global_shift: float = 0.0,  # delta max
    max_local_shift: float = 0.0,  # epsilon max
    max_scale: float = 1.0,
    device: str = "cpu",
    max_period: float = 10000.0,
):
    # We aim for TBC format
    assert dim % 2 == 0
    pos = 1.0 * torch.arange(length).view(-1, 1, 1)  # (length, 1, 1)
    pos = pos.repeat(1, batch_size, 1)  # (length, batch_size, 1)
    if mean_normalize:
        pos -= torch.nanmean(pos, dim=0, keepdim=True)

    if augment:
        delta = np.random.uniform(
            -max_global_shift, +max_global_shift, size=[1, batch_size, 1]
        )
        delta_local = np.random.uniform(
            -max_local_shift, +max_local_shift, size=[length, batch_size, 1]
        )
        log_lambdas = np.random.uniform(
            -np.log(max_scale), +np.log(max_scale), size=[1, batch_size, 1]
        )
        pos = (pos + delta + delta_local) * np.exp(log_lambdas)

    pos = pos.to(device)

    half_dim = dim // 2
    adim = torch.arange(dim // 2, device=device).view(1, 1, -1)
    phase = pos / (max_period ** (adim / (half_dim - 1)))
    return torch.cat(
        [
            torch.cos(phase),
            torch.sin(phase),
        ],
        dim=-1,
    ).float()


def get_causal_mask(length):
    pos = torch.arange(length)
    return pos > pos[:, None]


def get_elementary_mask(
    T1,
    T2,
    mask_type,
    sparse_attn_window,
    global_window,
    mask_random_seed,
    sparsity,
    device,
):
    """
    When the input of the Decoder has length T1 and the output T2
    The mask matrix has shape (T2, T1)
    """
    assert mask_type in ["diag", "jmask", "random", "global"]

    if mask_type == "global":
        mask = torch.zeros(T2, T1, dtype=torch.bool)
        mask[:, :global_window] = True
        line_window = int(global_window * T2 / T1)
        mask[:line_window, :] = True

    if mask_type == "diag":
        mask = torch.zeros(T2, T1, dtype=torch.bool)
        rows = torch.arange(T2)[:, None]
        cols = (
            (T1 / T2 * rows + torch.arange(-sparse_attn_window, sparse_attn_window + 1))
            .long()
            .clamp(0, T1 - 1)
        )
        mask.scatter_(1, cols, torch.ones(1, dtype=torch.bool).expand_as(cols))

    elif mask_type == "jmask":
        mask = torch.zeros(T2 + 2, T1 + 2, dtype=torch.bool)
        rows = torch.arange(T2 + 2)[:, None]
        t = torch.arange(0, int((2 * T1) ** 0.5 + 1))
        t = (t * (t + 1) / 2).int()
        t = torch.cat([-t.flip(0)[:-1], t])
        cols = (T1 / T2 * rows + t).long().clamp(0, T1 + 1)
        mask.scatter_(1, cols, torch.ones(1, dtype=torch.bool).expand_as(cols))
        mask = mask[1:-1, 1:-1]

    elif mask_type == "random":
        gene = torch.Generator(device=device)
        gene.manual_seed(mask_random_seed)
        mask = (
            torch.rand(T1 * T2, generator=gene, device=device).reshape(T2, T1)
            > sparsity
        )

    mask = mask.to(device)
    return mask


def get_mask(
    T1,
    T2,
    mask_type,
    sparse_attn_window,
    global_window,
    mask_random_seed,
    sparsity,
    device,
):
    """
    Return a SparseCSRTensor mask that is a combination of elementary masks
    mask_type can be a combination of multiple masks: for instance "diag_jmask_random"
    """
    from xformers.sparse import SparseCSRTensor

    # create a list
    mask_types = mask_type.split("_")

    all_masks = [
        get_elementary_mask(
            T1,
            T2,
            mask,
            sparse_attn_window,
            global_window,
            mask_random_seed,
            sparsity,
            device,
        )
        for mask in mask_types
    ]

    final_mask = torch.stack(all_masks).sum(axis=0) > 0

    return SparseCSRTensor.from_dense(final_mask[None])


class ScaledEmbedding2(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        scale: float = 1.0,
        boost: float = 3.0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data *= scale / boost
        self.boost = boost

    @property
    def weight(self):
        return self.embedding.weight * self.boost

    def forward(self, x):
        return self.embedding(x) * self.boost


class LayerScale(nn.Module):
    """Layer scale from [Touvron et al 2021] (https://arxiv.org/pdf/2103.17239.pdf).
    This rescales diagonaly residual outputs close to 0 initially, then learnt.
    """

    def __init__(self, channels: int, init: float = 0, channel_last=False):
        """
        channel_last = False corresponds to (B, C, T) tensors
        channel_last = True corresponds to (T, B, C) tensors
        """
        super().__init__()
        self.channel_last = channel_last
        self.scale = nn.Parameter(torch.zeros(channels, requires_grad=True))
        self.scale.data[:] = init

    def forward(self, x):
        if self.channel_last:
            return self.scale * x
        else:
            return self.scale[:, None] * x


class MyGroupNorm(nn.GroupNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        """
        x: (B, T, C)
        if num_groups=1: Normalisation on all T and C together for each B
        """
        x = x.transpose(1, 2)
        return super().forward(x).transpose(1, 2)


class MyTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation=F.relu,
        group_norm=0,
        norm_first=False,
        norm_out=False,
        layer_norm_eps=1e-5,
        layer_scale=False,
        init_values=1e-4,
        device=None,
        dtype=None,
        sparse=False,
        mask_type="diag",
        mask_random_seed=42,
        sparse_attn_window=500,
        global_window=50,
        auto_sparsity=False,
        sparsity=0.95,
        batch_first=False,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=batch_first,
            norm_first=norm_first,
            device=device,
            dtype=dtype,
        )
        self.sparse = sparse
        self.auto_sparsity = auto_sparsity
        if sparse:
            if not auto_sparsity:
                self.mask_type = mask_type
                self.sparse_attn_window = sparse_attn_window
                self.global_window = global_window
            self.sparsity = sparsity
        if group_norm:
            self.norm1 = MyGroupNorm(
                int(group_norm), d_model, eps=layer_norm_eps, **factory_kwargs
            )
            self.norm2 = MyGroupNorm(
                int(group_norm), d_model, eps=layer_norm_eps, **factory_kwargs
            )

        self.norm_out = None
        if self.norm_first & norm_out:
            self.norm_out = MyGroupNorm(num_groups=int(norm_out), num_channels=d_model)
        self.gamma_1 = (
            LayerScale(d_model, init_values, True) if layer_scale else nn.Identity()
        )
        self.gamma_2 = (
            LayerScale(d_model, init_values, True) if layer_scale else nn.Identity()
        )

        if sparse:
            self.self_attn = MultiheadAttention(
                d_model,
                nhead,
                dropout=dropout,
                batch_first=batch_first,
                auto_sparsity=sparsity if auto_sparsity else 0,
            )
            self.__setattr__("src_mask", torch.zeros(1, 1))
            self.mask_random_seed = mask_random_seed

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        if batch_first = False, src shape is (T, B, C)
        the case where batch_first=True is not covered
        """
        device = src.device
        x = src
        T, B, C = x.shape
        if self.sparse and not self.auto_sparsity:
            assert src_mask is None
            src_mask = self.src_mask
            if src_mask.shape[-1] != T:
                src_mask = get_mask(
                    T,
                    T,
                    self.mask_type,
                    self.sparse_attn_window,
                    self.global_window,
                    self.mask_random_seed,
                    self.sparsity,
                    device,
                )
                self.__setattr__("src_mask", src_mask)

        if self.norm_first:
            x = x + self.gamma_1(
                self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            )
            x = x + self.gamma_2(self._ff_block(self.norm2(x)))

            if self.norm_out:
                x = self.norm_out(x)
        else:
            x = self.norm1(
                x + self.gamma_1(self._sa_block(x, src_mask, src_key_padding_mask))
            )
            x = self.norm2(x + self.gamma_2(self._ff_block(x)))

        return x


class CrossTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation=F.relu,
        layer_norm_eps: float = 1e-5,
        layer_scale: bool = False,
        init_values: float = 1e-4,
        norm_first: bool = False,
        group_norm: bool = False,
        norm_out: bool = False,
        sparse=False,
        mask_type="diag",
        mask_random_seed=42,
        sparse_attn_window=500,
        global_window=50,
        sparsity=0.95,
        auto_sparsity=None,
        device=None,
        dtype=None,
        batch_first=False,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.sparse = sparse
        self.auto_sparsity = auto_sparsity
        if sparse:
            if not auto_sparsity:
                self.mask_type = mask_type
                self.sparse_attn_window = sparse_attn_window
                self.global_window = global_window
            self.sparsity = sparsity

        self.cross_attn: nn.Module
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1: nn.Module
        self.norm2: nn.Module
        self.norm3: nn.Module
        if group_norm:
            self.norm1 = MyGroupNorm(
                int(group_norm), d_model, eps=layer_norm_eps, **factory_kwargs
            )
            self.norm2 = MyGroupNorm(
                int(group_norm), d_model, eps=layer_norm_eps, **factory_kwargs
            )
            self.norm3 = MyGroupNorm(
                int(group_norm), d_model, eps=layer_norm_eps, **factory_kwargs
            )
        else:
            self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)

        self.norm_out = None
        if self.norm_first & norm_out:
            self.norm_out = MyGroupNorm(num_groups=int(norm_out), num_channels=d_model)

        self.gamma_1 = (
            LayerScale(d_model, init_values, True) if layer_scale else nn.Identity()
        )
        self.gamma_2 = (
            LayerScale(d_model, init_values, True) if layer_scale else nn.Identity()
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = self._get_activation_fn(activation)
        else:
            self.activation = activation

        if sparse:
            self.cross_attn = MultiheadAttention(
                d_model,
                nhead,
                dropout=dropout,
                batch_first=batch_first,
                auto_sparsity=sparsity if auto_sparsity else 0,
            )
            if not auto_sparsity:
                self.__setattr__("mask", torch.zeros(1, 1))
                self.mask_random_seed = mask_random_seed

    def forward(self, q, k, mask=None):
        """
        Args:
            q: tensor of shape (T, B, C)
            k: tensor of shape (S, B, C)
            mask: tensor of shape (T, S)

        """
        device = q.device
        T, B, C = q.shape
        S, B, C = k.shape
        if self.sparse and not self.auto_sparsity:
            assert mask is None
            mask = self.mask
            if mask.shape[-1] != S or mask.shape[-2] != T:
                mask = get_mask(
                    S,
                    T,
                    self.mask_type,
                    self.sparse_attn_window,
                    self.global_window,
                    self.mask_random_seed,
                    self.sparsity,
                    device,
                )
                self.__setattr__("mask", mask)

        if self.norm_first:
            x = q + self.gamma_1(self._ca_block(self.norm1(q), self.norm2(k), mask))
            x = x + self.gamma_2(self._ff_block(self.norm3(x)))
            if self.norm_out:
                x = self.norm_out(x)
        else:
            x = self.norm1(q + self.gamma_1(self._ca_block(q, k, mask)))
            x = self.norm2(x + self.gamma_2(self._ff_block(x)))

        return x

    # self-attention block
    def _ca_block(self, q, k, attn_mask=None):
        x = self.cross_attn(q, k, k, attn_mask=attn_mask, need_weights=False)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

    def _get_activation_fn(self, activation):
        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu

        raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


# ----------------- MULTI-BLOCKS MODELS: -----------------------


class CrossTransformerEncoder(nn.Module):
    def __init__(
        self,
        dim: int,
        emb: str = "sin",
        hidden_scale: float = 4.0,
        num_heads: int = 8,
        num_layers: int = 6,
        cross_first: bool = False,
        dropout: float = 0.0,
        max_positions: int = 1000,
        norm_in: bool = True,
        norm_in_group: bool = False,
        group_norm: int = False,
        norm_first: bool = False,
        norm_out: bool = False,
        max_period: float = 10000.0,
        weight_decay: float = 0.0,
        lr: tp.Optional[float] = None,
        layer_scale: bool = False,
        gelu: bool = True,
        sin_random_shift: int = 0,
        weight_pos_embed: float = 1.0,
        cape_mean_normalize: bool = True,
        cape_augment: bool = True,
        cape_glob_loc_scale: list = [5000.0, 1.0, 1.4],
        sparse_self_attn: bool = False,
        sparse_cross_attn: bool = False,
        mask_type: str = "diag",
        mask_random_seed: int = 42,
        sparse_attn_window: int = 500,
        global_window: int = 50,
        auto_sparsity: bool = False,
        sparsity: float = 0.95,
    ):
        super().__init__()
        """
        """
        assert dim % num_heads == 0

        hidden_dim = int(dim * hidden_scale)

        self.num_layers = num_layers
        # classic parity = 1 means that if idx%2 == 1 there is a
        # classical encoder else there is a cross encoder
        self.classic_parity = 1 if cross_first else 0
        self.emb = emb
        self.max_period = max_period
        self.weight_decay = weight_decay
        self.weight_pos_embed = weight_pos_embed
        self.sin_random_shift = sin_random_shift
        if emb == "cape":
            self.cape_mean_normalize = cape_mean_normalize
            self.cape_augment = cape_augment
            self.cape_glob_loc_scale = cape_glob_loc_scale
        if emb == "scaled":
            self.position_embeddings = ScaledEmbedding2(max_positions, dim, scale=0.2)

        self.lr = lr

        activation: tp.Any = F.gelu if gelu else F.relu

        self.norm_in: nn.Module
        self.norm_in_t: nn.Module
        if norm_in:
            self.norm_in = nn.LayerNorm(dim)
            self.norm_in_t = nn.LayerNorm(dim)
        elif norm_in_group:
            self.norm_in = MyGroupNorm(int(norm_in_group), dim)
            self.norm_in_t = MyGroupNorm(int(norm_in_group), dim)
        else:
            self.norm_in = nn.Identity()
            self.norm_in_t = nn.Identity()

        # spectrogram layers
        self.layers = nn.ModuleList()
        # temporal layers
        self.layers_t = nn.ModuleList()

        kwargs_common = {
            "d_model": dim,
            "nhead": num_heads,
            "dim_feedforward": hidden_dim,
            "dropout": dropout,
            "activation": activation,
            "group_norm": group_norm,
            "norm_first": norm_first,
            "norm_out": norm_out,
            "layer_scale": layer_scale,
            "mask_type": mask_type,
            "mask_random_seed": mask_random_seed,
            "sparse_attn_window": sparse_attn_window,
            "global_window": global_window,
            "sparsity": sparsity,
            "auto_sparsity": auto_sparsity,
            "batch_first": True,
        }

        kwargs_classic_encoder = dict(kwargs_common)
        kwargs_classic_encoder.update(
            {
                "sparse": sparse_self_attn,
            }
        )
        kwargs_cross_encoder = dict(kwargs_common)
        kwargs_cross_encoder.update(
            {
                "sparse": sparse_cross_attn,
            }
        )

        for idx in range(num_layers):
            if idx % 2 == self.classic_parity:
                self.layers.append(MyTransformerEncoderLayer(**kwargs_classic_encoder))
                self.layers_t.append(
                    MyTransformerEncoderLayer(**kwargs_classic_encoder)
                )

            else:
                self.layers.append(CrossTransformerEncoderLayer(**kwargs_cross_encoder))

                self.layers_t.append(
                    CrossTransformerEncoderLayer(**kwargs_cross_encoder)
                )

    def forward(self, x, xt):
        B, C, Fr, T1 = x.shape
        pos_emb_2d = create_2d_sin_embedding(
            C, Fr, T1, x.device, self.max_period
        )  # (1, C, Fr, T1)
        pos_emb_2d = rearrange(pos_emb_2d, "b c fr t1 -> b (t1 fr) c")
        x = rearrange(x, "b c fr t1 -> b (t1 fr) c")
        x = self.norm_in(x)
        x = x + self.weight_pos_embed * pos_emb_2d

        B, C, T2 = xt.shape
        xt = rearrange(xt, "b c t2 -> b t2 c")  # now T2, B, C
        pos_emb = self._get_pos_embedding(T2, B, C, x.device)
        pos_emb = rearrange(pos_emb, "t2 b c -> b t2 c")
        xt = self.norm_in_t(xt)
        xt = xt + self.weight_pos_embed * pos_emb

        for idx in range(self.num_layers):
            if idx % 2 == self.classic_parity:
                x = self.layers[idx](x)
                xt = self.layers_t[idx](xt)
            else:
                old_x = x
                x = self.layers[idx](x, xt)
                xt = self.layers_t[idx](xt, old_x)

        x = rearrange(x, "b (t1 fr) c -> b c fr t1", t1=T1)
        xt = rearrange(xt, "b t2 c -> b c t2")
        return x, xt

    def _get_pos_embedding(self, T, B, C, device):
        if self.emb == "sin":
            shift = random.randrange(self.sin_random_shift + 1)
            pos_emb = create_sin_embedding(
                T, C, shift=shift, device=device, max_period=self.max_period
            )
        elif self.emb == "cape":
            if self.training:
                pos_emb = create_sin_embedding_cape(
                    T,
                    C,
                    B,
                    device=device,
                    max_period=self.max_period,
                    mean_normalize=self.cape_mean_normalize,
                    augment=self.cape_augment,
                    max_global_shift=self.cape_glob_loc_scale[0],
                    max_local_shift=self.cape_glob_loc_scale[1],
                    max_scale=self.cape_glob_loc_scale[2],
                )
            else:
                pos_emb = create_sin_embedding_cape(
                    T,
                    C,
                    B,
                    device=device,
                    max_period=self.max_period,
                    mean_normalize=self.cape_mean_normalize,
                    augment=False,
                )

        elif self.emb == "scaled":
            pos = torch.arange(T, device=device)
            pos_emb = self.position_embeddings(pos)[:, None]

        return pos_emb

    def make_optim_group(self):
        group = {"params": list(self.parameters()), "weight_decay": self.weight_decay}
        if self.lr is not None:
            group["lr"] = self.lr
        return group


# Attention Modules


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
        batch_first=False,
        auto_sparsity=None,
    ):
        super().__init__()
        assert auto_sparsity is not None, "sanity check"
        self.num_heads = num_heads
        self.q = torch.nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k = torch.nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v = torch.nn.Linear(embed_dim, embed_dim, bias=bias)
        self.attn_drop = torch.nn.Dropout(dropout)
        self.proj = torch.nn.Linear(embed_dim, embed_dim, bias)
        self.proj_drop = torch.nn.Dropout(dropout)
        self.batch_first = batch_first
        self.auto_sparsity = auto_sparsity

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=None,
        average_attn_weights=True,
    ):
        if not self.batch_first:  # N, B, C
            query = query.permute(1, 0, 2)  # B, N_q, C
            key = key.permute(1, 0, 2)  # B, N_k, C
            value = value.permute(1, 0, 2)  # B, N_k, C
        B, N_q, C = query.shape
        B, N_k, C = key.shape

        q = (
            self.q(query)
            .reshape(B, N_q, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        q = q.flatten(0, 1)
        k = (
            self.k(key)
            .reshape(B, N_k, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        k = k.flatten(0, 1)
        v = (
            self.v(value)
            .reshape(B, N_k, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        v = v.flatten(0, 1)

        if self.auto_sparsity:
            assert attn_mask is None
            x = dynamic_sparse_attention(q, k, v, sparsity=self.auto_sparsity)
        else:
            x = scaled_dot_product_attention(q, k, v, attn_mask, dropout=self.attn_drop)
        x = x.reshape(B, self.num_heads, N_q, C // self.num_heads)

        x = x.transpose(1, 2).reshape(B, N_q, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if not self.batch_first:
            x = x.permute(1, 0, 2)
        return x, None


def scaled_query_key_softmax(q, k, att_mask):
    from xformers.ops import masked_matmul

    q = q / (k.size(-1)) ** 0.5
    att = masked_matmul(q, k.transpose(-2, -1), att_mask)
    att = torch.nn.functional.softmax(att, -1)
    return att


def scaled_dot_product_attention(q, k, v, att_mask, dropout):
    att = scaled_query_key_softmax(q, k, att_mask=att_mask)
    att = dropout(att)
    y = att @ v
    return y


def _compute_buckets(x, R):
    qq = torch.einsum("btf,bfhi->bhti", x, R)
    qq = torch.cat([qq, -qq], dim=-1)
    buckets = qq.argmax(dim=-1)

    return buckets.permute(0, 2, 1).byte().contiguous()


def dynamic_sparse_attention(
    query, key, value, sparsity, infer_sparsity=True, attn_bias=None
):
    # assert False, "The code for the custom sparse kernel is not ready for release yet."
    from xformers.ops import find_locations, sparse_memory_efficient_attention

    n_hashes = 32
    proj_size = 4
    query, key, value = [x.contiguous() for x in [query, key, value]]
    with torch.no_grad():
        R = torch.randn(
            1, query.shape[-1], n_hashes, proj_size // 2, device=query.device
        )
        bucket_query = _compute_buckets(query, R)
        bucket_key = _compute_buckets(key, R)
        row_offsets, column_indices = find_locations(
            bucket_query, bucket_key, sparsity, infer_sparsity
        )
    return sparse_memory_efficient_attention(
        query, key, value, row_offsets, column_indices, attn_bias
    )


def pad1d(
    x: torch.Tensor,
    paddings: tp.Tuple[int, int],
    mode: str = "constant",
    value: float = 0.0,
):
    """Tiny wrapper around F.pad, just to allow for reflect padding on small input.
    If this is the case, we insert extra 0 padding to the right before the reflection happen.
    """
    x0 = x
    length = x.shape[-1]
    padding_left, padding_right = paddings
    if mode == "reflect":
        max_pad = max(padding_left, padding_right)
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            extra_pad_right = min(padding_right, extra_pad)
            extra_pad_left = extra_pad - extra_pad_right
            paddings = (padding_left - extra_pad_left, padding_right - extra_pad_right)
            x = F.pad(x, (extra_pad_left, extra_pad_right))
    out = F.pad(x, paddings, mode, value)
    assert out.shape[-1] == length + padding_left + padding_right
    assert (out[..., padding_left : padding_left + length] == x0).all()
    return out


class ScaledEmbedding(nn.Module):
    """
    Boost learning rate for embeddings (with `scale`).
    Also, can make embeddings continuous with `smooth`.
    """

    def __init__(
        self, num_embeddings: int, embedding_dim: int, scale: float = 10.0, smooth=False
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        if smooth:
            weight = torch.cumsum(self.embedding.weight.data, dim=0)
            # when summing gaussian, overscale raises as sqrt(n), so we nornalize by that.
            weight = (
                weight / torch.arange(1, num_embeddings + 1).to(weight).sqrt()[:, None]
            )
            self.embedding.weight.data[:] = weight
        self.embedding.weight.data /= scale
        self.scale = scale

    @property
    def weight(self):
        return self.embedding.weight * self.scale

    def forward(self, x):
        out = self.embedding(x) * self.scale
        return out


class HEncLayer(nn.Module):
    def __init__(
        self,
        chin,
        chout,
        kernel_size=8,
        stride=4,
        norm_groups=1,
        empty=False,
        freq=True,
        dconv=True,
        norm=True,
        context=0,
        dconv_kw={},
        pad=True,
        rewrite=True,
    ):
        """Encoder layer. This used both by the time and the frequency branch.

        Args:
            chin: number of input channels.
            chout: number of output channels.
            norm_groups: number of groups for group norm.
            empty: used to make a layer with just the first conv. this is used
                before merging the time and freq. branches.
            freq: this is acting on frequencies.
            dconv: insert DConv residual branches.
            norm: use GroupNorm.
            context: context size for the 1x1 conv.
            dconv_kw: list of kwargs for the DConv class.
            pad: pad the input. Padding is done so that the output size is
                always the input size / stride.
            rewrite: add 1x1 conv at the end of the layer.
        """
        super().__init__()
        norm_fn = lambda d: nn.Identity()  # noqa
        if norm:
            norm_fn = lambda d: nn.GroupNorm(norm_groups, d)  # noqa
        if pad:
            pad = kernel_size // 4
        else:
            pad = 0
        klass = nn.Conv1d
        self.freq = freq
        self.kernel_size = kernel_size
        self.stride = stride
        self.empty = empty
        self.norm = norm
        self.pad = pad
        if freq:
            kernel_size = [kernel_size, 1]
            stride = [stride, 1]
            pad = [pad, 0]
            klass = nn.Conv2d
        self.conv = klass(chin, chout, kernel_size, stride, pad)
        if self.empty:
            return
        self.norm1 = norm_fn(chout)
        self.rewrite = None
        if rewrite:
            self.rewrite = klass(chout, 2 * chout, 1 + 2 * context, 1, context)
            self.norm2 = norm_fn(2 * chout)

        self.dconv = None
        if dconv:
            self.dconv = DConv(chout, **dconv_kw)

    def forward(self, x, inject=None):
        """
        `inject` is used to inject the result from the time branch into the frequency branch,
        when both have the same stride.
        """
        if not self.freq and x.dim() == 4:
            B, C, Fr, T = x.shape
            x = x.view(B, -1, T)

        if not self.freq:
            le = x.shape[-1]
            if not le % self.stride == 0:
                x = F.pad(x, (0, self.stride - (le % self.stride)))
        y = self.conv(x)
        if self.empty:
            return y
        if inject is not None:
            assert inject.shape[-1] == y.shape[-1], (inject.shape, y.shape)
            if inject.dim() == 3 and y.dim() == 4:
                inject = inject[:, :, None]
            y = y + inject
        y = F.gelu(self.norm1(y))
        if self.dconv:
            if self.freq:
                B, C, Fr, T = y.shape
                y = y.permute(0, 2, 1, 3).reshape(-1, C, T)
            y = self.dconv(y)
            if self.freq:
                y = y.view(B, Fr, C, T).permute(0, 2, 1, 3)
        if self.rewrite:
            z = self.norm2(self.rewrite(y))
            z = F.glu(z, dim=1)
        else:
            z = y
        return z


class MultiWrap(nn.Module):
    """
    Takes one layer and replicate it N times. each replica will act
    on a frequency band. All is done so that if the N replica have the same weights,
    then this is exactly equivalent to applying the original module on all frequencies.

    This is a bit over-engineered to avoid edge artifacts when splitting
    the frequency bands, but it is possible the naive implementation would work as well...
    """

    def __init__(self, layer, split_ratios):
        """
        Args:
            layer: module to clone, must be either HEncLayer or HDecLayer.
            split_ratios: list of float indicating which ratio to keep for each band.
        """
        super().__init__()
        self.split_ratios = split_ratios
        self.layers = nn.ModuleList()
        self.conv = isinstance(layer, HEncLayer)
        assert not layer.norm
        assert layer.freq
        assert layer.pad
        if not self.conv:
            assert not layer.context_freq
        for k in range(len(split_ratios) + 1):
            lay = deepcopy(layer)
            if self.conv:
                lay.conv.padding = (0, 0)
            else:
                lay.pad = False
            for m in lay.modules():
                if hasattr(m, "reset_parameters"):
                    m.reset_parameters()
            self.layers.append(lay)

    def forward(self, x, skip=None, length=None):
        B, C, Fr, T = x.shape

        ratios = list(self.split_ratios) + [1]
        start = 0
        outs = []
        for ratio, layer in zip(ratios, self.layers):
            if self.conv:
                pad = layer.kernel_size // 4
                if ratio == 1:
                    limit = Fr
                    frames = -1
                else:
                    limit = int(round(Fr * ratio))
                    le = limit - start
                    if start == 0:
                        le += pad
                    frames = round((le - layer.kernel_size) / layer.stride + 1)
                    limit = start + (frames - 1) * layer.stride + layer.kernel_size
                    if start == 0:
                        limit -= pad
                assert limit - start > 0, (limit, start)
                assert limit <= Fr, (limit, Fr)
                y = x[:, :, start:limit, :]
                if start == 0:
                    y = F.pad(y, (0, 0, pad, 0))
                if ratio == 1:
                    y = F.pad(y, (0, 0, 0, pad))
                outs.append(layer(y))
                start = limit - layer.kernel_size + layer.stride
            else:
                if ratio == 1:
                    limit = Fr
                else:
                    limit = int(round(Fr * ratio))
                last = layer.last
                layer.last = True

                y = x[:, :, start:limit]
                s = skip[:, :, start:limit]
                out, _ = layer(y, s, None)
                if outs:
                    outs[-1][:, :, -layer.stride :] += out[
                        :, :, : layer.stride
                    ] - layer.conv_tr.bias.view(1, -1, 1, 1)
                    out = out[:, :, layer.stride :]
                if ratio == 1:
                    out = out[:, :, : -layer.stride // 2, :]
                if start == 0:
                    out = out[:, :, layer.stride // 2 :, :]
                outs.append(out)
                layer.last = last
                start = limit
        out = torch.cat(outs, dim=2)
        if not self.conv and not last:
            out = F.gelu(out)
        if self.conv:
            return out
        else:
            return out, None


class HDecLayer(nn.Module):
    def __init__(
        self,
        chin,
        chout,
        last=False,
        kernel_size=8,
        stride=4,
        norm_groups=1,
        empty=False,
        freq=True,
        dconv=True,
        norm=True,
        context=1,
        dconv_kw={},
        pad=True,
        context_freq=True,
        rewrite=True,
    ):
        """
        Same as HEncLayer but for decoder. See `HEncLayer` for documentation.
        """
        super().__init__()
        norm_fn = lambda d: nn.Identity()  # noqa
        if norm:
            norm_fn = lambda d: nn.GroupNorm(norm_groups, d)  # noqa
        if pad:
            pad = kernel_size // 4
        else:
            pad = 0
        self.pad = pad
        self.last = last
        self.freq = freq
        self.chin = chin
        self.empty = empty
        self.stride = stride
        self.kernel_size = kernel_size
        self.norm = norm
        self.context_freq = context_freq
        klass = nn.Conv1d
        klass_tr = nn.ConvTranspose1d
        if freq:
            kernel_size = [kernel_size, 1]
            stride = [stride, 1]
            klass = nn.Conv2d
            klass_tr = nn.ConvTranspose2d
        self.conv_tr = klass_tr(chin, chout, kernel_size, stride)
        self.norm2 = norm_fn(chout)
        if self.empty:
            return
        self.rewrite = None
        if rewrite:
            if context_freq:
                self.rewrite = klass(chin, 2 * chin, 1 + 2 * context, 1, context)
            else:
                self.rewrite = klass(
                    chin, 2 * chin, [1, 1 + 2 * context], 1, [0, context]
                )
            self.norm1 = norm_fn(2 * chin)

        self.dconv = None
        if dconv:
            self.dconv = DConv(chin, **dconv_kw)

    def forward(self, x, skip, length):
        if self.freq and x.dim() == 3:
            B, C, T = x.shape
            x = x.view(B, self.chin, -1, T)

        if not self.empty:
            x = x + skip

            if self.rewrite:
                y = F.glu(self.norm1(self.rewrite(x)), dim=1)
            else:
                y = x
            if self.dconv:
                if self.freq:
                    B, C, Fr, T = y.shape
                    y = y.permute(0, 2, 1, 3).reshape(-1, C, T)
                y = self.dconv(y)
                if self.freq:
                    y = y.view(B, Fr, C, T).permute(0, 2, 1, 3)
        else:
            y = x
            assert skip is None
        z = self.norm2(self.conv_tr(y))
        if self.freq:
            if self.pad:
                z = z[..., self.pad : -self.pad, :]
        else:
            z = z[..., self.pad : self.pad + length]
            assert z.shape[-1] == length, (z.shape[-1], length)
        if not self.last:
            z = F.gelu(z)
        return z, y


class HTDemucs(AbsSeparator):
    """
    Spectrogram and hybrid Demucs model.
    The spectrogram model has the same structure as Demucs, except the first few layers are over the
    frequency axis, until there is only 1 frequency, and then it moves to time convolutions.
    Frequency layers can still access information across time steps thanks to the DConv residual.

    Hybrid model have a parallel time branch. At some layer, the time branch has the same stride
    as the frequency branch and then the two are combined. The opposite happens in the decoder.

    Models can either use naive iSTFT from masking, Wiener filtering ([Ulhih et al. 2017]),
    or complex as channels (CaC) [Choi et al. 2020]. Wiener filtering is based on
    Open Unmix implementation [Stoter et al. 2019].

    The loss is always on the temporal domain, by backpropagating through the above
    output methods and iSTFT. This allows to define hybrid models nicely. However, this breaks
    a bit Wiener filtering, as doing more iteration at test time will change the spectrogram
    contribution, without changing the one from the waveform, which will lead to worse performance.
    I tried using the residual option in OpenUnmix Wiener implementation, but it didn't improve.
    CaC on the other hand provides similar performance for hybrid, and works naturally with
    hybrid models.

    This model also uses frequency embeddings are used to improve efficiency on convolutions
    over the freq. axis, following [Isik et al. 2020] (https://arxiv.org/pdf/2008.04470.pdf).

    Unlike classic Demucs, there is no resampling here, and normalization is always applied.
    """

    @capture_init
    def __init__(
        self,
        input_dim,
        sources=["speech"],
        # Channels
        audio_channels=1,
        channels=48,
        channels_time=None,
        growth=2,
        # STFT
        nfft=4096,
        wiener_iters=0,
        end_iters=0,
        wiener_residual=False,
        cac=True,
        # Main structure
        depth=4,
        rewrite=True,
        # Frequency branch
        multi_freqs=None,
        multi_freqs_depth=3,
        freq_emb=0.2,
        emb_scale=10,
        emb_smooth=True,
        # Convolutions
        kernel_size=8,
        time_stride=2,
        stride=4,
        context=1,
        context_enc=0,
        # Normalization
        norm_starts=4,
        norm_groups=4,
        # DConv residual branch
        dconv_mode=1,
        dconv_depth=2,
        dconv_comp=8,
        dconv_init=1e-3,
        # Before the Transformer
        bottom_channels=0,
        # Transformer
        t_layers=5,
        t_emb="sin",
        t_hidden_scale=4.0,
        t_heads=8,
        t_dropout=0.0,
        t_max_positions=10000,
        t_norm_in=True,
        t_norm_in_group=False,
        t_group_norm=False,
        t_norm_first=True,
        t_norm_out=True,
        t_max_period=10000.0,
        t_weight_decay=0.0,
        t_lr=None,
        t_layer_scale=True,
        t_gelu=True,
        t_weight_pos_embed=1.0,
        t_sin_random_shift=0,
        t_cape_mean_normalize=True,
        t_cape_augment=True,
        t_cape_glob_loc_scale=[5000.0, 1.0, 1.4],
        t_sparse_self_attn=False,
        t_sparse_cross_attn=False,
        t_mask_type="diag",
        t_mask_random_seed=42,
        t_sparse_attn_window=500,
        t_global_window=100,
        t_sparsity=0.95,
        t_auto_sparsity=False,
        # ------ Particuliar parameters
        t_cross_first=False,
        # Weight init
        rescale=0.1,
        # Metadata
        samplerate=44100,
        segment=10,
        use_train_segment=True,
    ):
        """
        Args:
            sources (list[str]): list of source names.
            audio_channels (int): input/output audio channels.
            channels (int): initial number of hidden channels.
            channels_time: if not None, use a different `channels` value for the time branch.
            growth: increase the number of hidden channels by this factor at each layer.
            nfft: number of fft bins. Note that changing this require careful computation of
                various shape parameters and will not work out of the box for hybrid models.
            wiener_iters: when using Wiener filtering, number of iterations at test time.
            end_iters: same but at train time. For a hybrid model, must be equal to `wiener_iters`.
            wiener_residual: add residual source before wiener filtering.
            cac: uses complex as channels, i.e. complex numbers are 2 channels each
                in input and output. no further processing is done before ISTFT.
            depth (int): number of layers in the encoder and in the decoder.
            rewrite (bool): add 1x1 convolution to each layer.
            multi_freqs: list of frequency ratios for splitting frequency bands with `MultiWrap`.
            multi_freqs_depth: how many layers to wrap with `MultiWrap`. Only the outermost
                layers will be wrapped.
            freq_emb: add frequency embedding after the first frequency layer if > 0,
                the actual value controls the weight of the embedding.
            emb_scale: equivalent to scaling the embedding learning rate
            emb_smooth: initialize the embedding with a smooth one (with respect to frequencies).
            kernel_size: kernel_size for encoder and decoder layers.
            stride: stride for encoder and decoder layers.
            time_stride: stride for the final time layer, after the merge.
            context: context for 1x1 conv in the decoder.
            context_enc: context for 1x1 conv in the encoder.
            norm_starts: layer at which group norm starts being used.
                decoder layers are numbered in reverse order.
            norm_groups: number of groups for group norm.
            dconv_mode: if 1: dconv in encoder only, 2: decoder only, 3: both.
            dconv_depth: depth of residual DConv branch.
            dconv_comp: compression of DConv branch.
            dconv_attn: adds attention layers in DConv branch starting at this layer.
            dconv_lstm: adds a LSTM layer in DConv branch starting at this layer.
            dconv_init: initial scale for the DConv branch LayerScale.
            bottom_channels: if >0 it adds a linear layer (1x1 Conv) before and after the
                transformer in order to change the number of channels
            t_layers: number of layers in each branch (waveform and spec) of the transformer
            t_emb: "sin", "cape" or "scaled"
            t_hidden_scale: the hidden scale of the Feedforward parts of the transformer
                for instance if C = 384 (the number of channels in the transformer) and
                t_hidden_scale = 4.0 then the intermediate layer of the FFN has dimension
                384 * 4 = 1536
            t_heads: number of heads for the transformer
            t_dropout: dropout in the transformer
            t_max_positions: max_positions for the "scaled" positional embedding, only
                useful if t_emb="scaled"
            t_norm_in: (bool) norm before addinf positional embedding and getting into the
                transformer layers
            t_norm_in_group: (bool) if True while t_norm_in=True, the norm is on all the
                timesteps (GroupNorm with group=1)
            t_group_norm: (bool) if True, the norms of the Encoder Layers are on all the
                timesteps (GroupNorm with group=1)
            t_norm_first: (bool) if True the norm is before the attention and before the FFN
            t_norm_out: (bool) if True, there is a GroupNorm (group=1) at the end of each layer
            t_max_period: (float) denominator in the sinusoidal embedding expression
            t_weight_decay: (float) weight decay for the transformer
            t_lr: (float) specific learning rate for the transformer
            t_layer_scale: (bool) Layer Scale for the transformer
            t_gelu: (bool) activations of the transformer are GeLU if True, ReLU else
            t_weight_pos_embed: (float) weighting of the positional embedding
            t_cape_mean_normalize: (bool) if t_emb="cape", normalisation of positional embeddings
                see: https://arxiv.org/abs/2106.03143
            t_cape_augment: (bool) if t_emb="cape", must be True during training and False
                during the inference, see: https://arxiv.org/abs/2106.03143
            t_cape_glob_loc_scale: (list of 3 floats) if t_emb="cape", CAPE parameters
                see: https://arxiv.org/abs/2106.03143
            t_sparse_self_attn: (bool) if True, the self attentions are sparse
            t_sparse_cross_attn: (bool) if True, the cross-attentions are sparse (don't use it
                unless you designed really specific masks)
            t_mask_type: (str) can be "diag", "jmask", "random", "global" or any combination
                with '_' between: i.e. "diag_jmask_random" (note that this is permutation
                invariant i.e. "diag_jmask_random" is equivalent to "jmask_random_diag")
            t_mask_random_seed: (int) if "random" is in t_mask_type, controls the seed
                that generated the random part of the mask
            t_sparse_attn_window: (int) if "diag" is in t_mask_type, for a query (i), and
                a key (j), the mask is True id |i-j|<=t_sparse_attn_window
            t_global_window: (int) if "global" is in t_mask_type, mask[:t_global_window, :]
                and mask[:, :t_global_window] will be True
            t_sparsity: (float) if "random" is in t_mask_type, t_sparsity is the sparsity
                level of the random part of the mask.
            t_cross_first: (bool) if True cross attention is the first layer of the
                transformer (False seems to be better)
            rescale: weight rescaling trick
            use_train_segment: (bool) if True, the actual size that is used during the
                training is used during inference.
        """
        super().__init__()
        self.cac = cac
        self.wiener_residual = wiener_residual
        self.audio_channels = audio_channels
        self.sources = sources
        self.kernel_size = kernel_size
        self.context = context
        self.stride = stride
        self.depth = depth
        self.bottom_channels = bottom_channels
        self.channels = channels
        self.samplerate = samplerate
        self.segment = segment
        self.use_train_segment = use_train_segment
        self.nfft = nfft
        self.hop_length = nfft // 4
        self.wiener_iters = wiener_iters
        self.end_iters = end_iters
        self.freq_emb = None
        assert wiener_iters == end_iters

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        self.tencoder = nn.ModuleList()
        self.tdecoder = nn.ModuleList()

        chin = audio_channels
        chin_z = chin  # number of channels for the freq branch
        if self.cac:
            chin_z *= 2
        chout = channels_time or channels
        chout_z = channels
        freqs = nfft // 2

        for index in range(depth):
            norm = index >= norm_starts
            freq = freqs > 1
            stri = stride
            ker = kernel_size
            if not freq:
                assert freqs == 1
                ker = time_stride * 2
                stri = time_stride

            pad = True
            last_freq = False
            if freq and freqs <= kernel_size:
                ker = freqs
                pad = False
                last_freq = True

            kw = {
                "kernel_size": ker,
                "stride": stri,
                "freq": freq,
                "pad": pad,
                "norm": norm,
                "rewrite": rewrite,
                "norm_groups": norm_groups,
                "dconv_kw": {
                    "depth": dconv_depth,
                    "compress": dconv_comp,
                    "init": dconv_init,
                    "gelu": True,
                },
            }
            kwt = dict(kw)
            kwt["freq"] = 0
            kwt["kernel_size"] = kernel_size
            kwt["stride"] = stride
            kwt["pad"] = True
            kw_dec = dict(kw)
            multi = False
            if multi_freqs and index < multi_freqs_depth:
                multi = True
                kw_dec["context_freq"] = False

            if last_freq:
                chout_z = max(chout, chout_z)
                chout = chout_z

            enc = HEncLayer(
                chin_z, chout_z, dconv=dconv_mode & 1, context=context_enc, **kw
            )
            if freq:
                tenc = HEncLayer(
                    chin,
                    chout,
                    dconv=dconv_mode & 1,
                    context=context_enc,
                    empty=last_freq,
                    **kwt,
                )
                self.tencoder.append(tenc)

            if multi:
                enc = MultiWrap(enc, multi_freqs)
            self.encoder.append(enc)
            if index == 0:
                chin = self.audio_channels * len(self.sources)
                chin_z = chin
                if self.cac:
                    chin_z *= 2
            dec = HDecLayer(
                chout_z,
                chin_z,
                dconv=dconv_mode & 2,
                last=index == 0,
                context=context,
                **kw_dec,
            )
            if multi:
                dec = MultiWrap(dec, multi_freqs)
            if freq:
                tdec = HDecLayer(
                    chout,
                    chin,
                    dconv=dconv_mode & 2,
                    empty=last_freq,
                    last=index == 0,
                    context=context,
                    **kwt,
                )
                self.tdecoder.insert(0, tdec)
            self.decoder.insert(0, dec)

            chin = chout
            chin_z = chout_z
            chout = int(growth * chout)
            chout_z = int(growth * chout_z)
            if freq:
                if freqs <= kernel_size:
                    freqs = 1
                else:
                    freqs //= stride
            if index == 0 and freq_emb:
                self.freq_emb = ScaledEmbedding(
                    freqs, chin_z, smooth=emb_smooth, scale=emb_scale
                )
                self.freq_emb_scale = freq_emb

        if rescale:
            rescale_module(self, reference=rescale)

        transformer_channels = channels * growth ** (depth - 1)
        if bottom_channels:
            self.channel_upsampler = nn.Conv1d(transformer_channels, bottom_channels, 1)
            self.channel_downsampler = nn.Conv1d(
                bottom_channels, transformer_channels, 1
            )
            self.channel_upsampler_t = nn.Conv1d(
                transformer_channels, bottom_channels, 1
            )
            self.channel_downsampler_t = nn.Conv1d(
                bottom_channels, transformer_channels, 1
            )

            transformer_channels = bottom_channels

        if t_layers > 0:
            self.crosstransformer = CrossTransformerEncoder(
                dim=transformer_channels,
                emb=t_emb,
                hidden_scale=t_hidden_scale,
                num_heads=t_heads,
                num_layers=t_layers,
                cross_first=t_cross_first,
                dropout=t_dropout,
                max_positions=t_max_positions,
                norm_in=t_norm_in,
                norm_in_group=t_norm_in_group,
                group_norm=t_group_norm,
                norm_first=t_norm_first,
                norm_out=t_norm_out,
                max_period=t_max_period,
                weight_decay=t_weight_decay,
                lr=t_lr,
                layer_scale=t_layer_scale,
                gelu=t_gelu,
                sin_random_shift=t_sin_random_shift,
                weight_pos_embed=t_weight_pos_embed,
                cape_mean_normalize=t_cape_mean_normalize,
                cape_augment=t_cape_augment,
                cape_glob_loc_scale=t_cape_glob_loc_scale,
                sparse_self_attn=t_sparse_self_attn,
                sparse_cross_attn=t_sparse_cross_attn,
                mask_type=t_mask_type,
                mask_random_seed=t_mask_random_seed,
                sparse_attn_window=t_sparse_attn_window,
                global_window=t_global_window,
                sparsity=t_sparsity,
                auto_sparsity=t_auto_sparsity,
            )
        else:
            self.crosstransformer = None

    def _spec(self, x):
        hl = self.hop_length
        nfft = self.nfft
        x0 = x  # noqa

        # We re-pad the signal in order to keep the property
        # that the size of the output is exactly the size of the input
        # divided by the stride (here hop_length), when divisible.
        # This is achieved by padding by 1/4th of the kernel size (here nfft).
        # which is not supported by torch.stft.
        # Having all convolution operations follow this convention allow to easily
        # align the time and frequency branches later on.
        assert hl == nfft // 4
        le = int(math.ceil(x.shape[-1] / hl))
        pad = hl // 2 * 3
        x = pad1d(x, (pad, pad + le * hl - x.shape[-1]), mode="reflect")

        z = spectro(x, nfft, hl)[..., :-1, :]
        assert z.shape[-1] == le + 4, (z.shape, x.shape, le)
        z = z[..., 2 : 2 + le]
        return z

    def _ispec(self, z, length=None, scale=0):
        hl = self.hop_length // (4**scale)
        z = F.pad(z, (0, 0, 0, 1))
        z = F.pad(z, (2, 2))
        pad = hl // 2 * 3
        le = hl * int(math.ceil(length / hl)) + 2 * pad
        x = ispectro(z, hl, length=le)
        x = x[..., pad : pad + length]
        return x

    def _magnitude(self, z):
        # return the magnitude of the spectrogram, except when cac is True,
        # in which case we just move the complex dimension to the channel one.
        if self.cac:
            B, C, Fr, T = z.shape
            m = torch.view_as_real(z).permute(0, 1, 4, 2, 3)
            m = m.reshape(B, C * 2, Fr, T)
        else:
            m = z.abs()
        return m

    def _mask(self, z, m):
        # Apply masking given the mixture spectrogram `z` and the estimated mask `m`.
        # If `cac` is True, `m` is actually a full spectrogram and `z` is ignored.
        niters = self.wiener_iters
        if self.cac:
            B, S, C, Fr, T = m.shape
            out = m.view(B, S, -1, 2, Fr, T).permute(0, 1, 2, 4, 5, 3)
            out = torch.view_as_complex(out.contiguous())
            return out
        if self.training:
            niters = self.end_iters
        if niters < 0:
            z = z[:, None]
            return z / (1e-8 + z.abs()) * m
        else:
            return self._wiener(m, z, niters)

    def _wiener(self, mag_out, mix_stft, niters):
        # apply wiener filtering from OpenUnmix.
        init = mix_stft.dtype
        wiener_win_len = 300
        residual = self.wiener_residual

        B, S, C, Fq, T = mag_out.shape
        mag_out = mag_out.permute(0, 4, 3, 2, 1)
        mix_stft = torch.view_as_real(mix_stft.permute(0, 3, 2, 1))

        outs = []
        for sample in range(B):
            pos = 0
            out = []
            for pos in range(0, T, wiener_win_len):
                frame = slice(pos, pos + wiener_win_len)
                z_out = wiener(
                    mag_out[sample, frame],
                    mix_stft[sample, frame],
                    niters,
                    residual=residual,
                )
                out.append(z_out.transpose(-1, -2))
            outs.append(torch.cat(out, dim=0))
        out = torch.view_as_complex(torch.stack(outs, 0))
        out = out.permute(0, 4, 3, 2, 1).contiguous()
        if residual:
            out = out[:, :-1]
        assert list(out.shape) == [B, S, C, Fq, T]
        return out.to(init)

    def valid_length(self, length: int):
        """
        Return a length that is appropriate for evaluation.
        In our case, always return the training length, unless
        it is smaller than the given length, in which case this
        raises an error.
        """
        if not self.use_train_segment:
            return length
        training_length = int(self.segment * self.samplerate)
        if training_length < length:
            raise ValueError(
                f"Given length {length} is longer than "
                f"training length {training_length}"
            )
        return training_length

    def forward(
        self,
        mix: torch.Tensor,
        ilens: torch.Tensor,
        additional: Optional[Dict] = None,
    ) -> Tuple[List[torch.Tensor], torch.Tensor, OrderedDict]:
        if mix.ndim == 2:
            mix = mix.unsqueeze(1)
        length = mix.shape[-1]
        length_pre_pad = None
        if self.use_train_segment:
            if self.training:
                self.segment = Fraction(mix.shape[-1], self.samplerate)
            else:
                training_length = int(self.segment * self.samplerate)
                if mix.shape[-1] < training_length:
                    length_pre_pad = mix.shape[-1]
                    mix = F.pad(mix, (0, training_length - length_pre_pad))
        z = self._spec(mix)
        mag = self._magnitude(z).to(mix.device)
        x = mag

        B, C, Fq, T = x.shape

        # unlike previous Demucs, we always normalize because it is easier.
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / (1e-5 + std)
        # x will be the freq. branch input.

        # Prepare the time branch input.
        xt = mix
        meant = xt.mean(dim=(1, 2), keepdim=True)
        stdt = xt.std(dim=(1, 2), keepdim=True)
        xt = (xt - meant) / (1e-5 + stdt)

        # okay, this is a giant mess I know...
        saved = []  # skip connections, freq.
        saved_t = []  # skip connections, time.
        lengths = []  # saved lengths to properly remove padding, freq branch.
        lengths_t = []  # saved lengths for time branch.
        for idx, encode in enumerate(self.encoder):
            lengths.append(x.shape[-1])
            inject = None
            if idx < len(self.tencoder):
                # we have not yet merged branches.
                lengths_t.append(xt.shape[-1])
                tenc = self.tencoder[idx]
                xt = tenc(xt)
                if not tenc.empty:
                    # save for skip connection
                    saved_t.append(xt)
                else:
                    # tenc contains just the first conv., so that now time and freq.
                    # branches have the same shape and can be merged.
                    inject = xt
            x = encode(x, inject)
            if idx == 0 and self.freq_emb is not None:
                # add frequency embedding to allow for non equivariant convolutions
                # over the frequency axis.
                frs = torch.arange(x.shape[-2], device=x.device)
                emb = self.freq_emb(frs).t()[None, :, :, None].expand_as(x)
                x = x + self.freq_emb_scale * emb

            saved.append(x)
            # print(
            #     idx,
            #     x.shape,
            #     xt.shape,
            #     x.requires_grad,
            #     xt.requires_grad,
            #     self.training,
            #     flush=True,
            # )
        if self.crosstransformer:
            if self.bottom_channels:
                b, c, f, t = x.shape
                x = rearrange(x, "b c f t-> b c (f t)")
                x = self.channel_upsampler(x)
                x = rearrange(x, "b c (f t)-> b c f t", f=f)
                xt = self.channel_upsampler_t(xt)

            x, xt = self.crosstransformer(x, xt)

            if self.bottom_channels:
                x = rearrange(x, "b c f t-> b c (f t)")
                x = self.channel_downsampler(x)
                x = rearrange(x, "b c (f t)-> b c f t", f=f)
                xt = self.channel_downsampler_t(xt)

        for idx, decode in enumerate(self.decoder):
            skip = saved.pop(-1)
            x, pre = decode(x, skip, lengths.pop(-1))
            # `pre` contains the output just before final transposed convolution,
            # which is used when the freq. and time branch separate.

            offset = self.depth - len(self.tdecoder)
            if idx >= offset:
                tdec = self.tdecoder[idx - offset]
                length_t = lengths_t.pop(-1)
                if tdec.empty:
                    assert pre.shape[2] == 1, pre.shape
                    pre = pre[:, :, 0]
                    xt, _ = tdec(pre, None, length_t)
                else:
                    skip = saved_t.pop(-1)
                    xt, _ = tdec(xt, skip, length_t)

        # Let's make sure we used all stored skip connections.
        assert len(saved) == 0
        assert len(lengths_t) == 0
        assert len(saved_t) == 0

        S = len(self.sources)
        x = x.view(B, S, -1, Fq, T)
        x = x * std[:, None] + mean[:, None]

        # to cpu as mps doesnt support complex numbers
        # demucs issue #435 ##432
        # NOTE: in this case z already is on cpu
        # TODO: remove this when mps supports complex numbers
        x_is_mps = x.device.type == "mps"
        if x_is_mps:
            x = x.cpu()

        zout = self._mask(z, x)
        if self.use_train_segment:
            if self.training:
                x = self._ispec(zout, length)
            else:
                x = self._ispec(zout, training_length)
        else:
            x = self._ispec(zout, length)

        # back to mps device
        if x_is_mps:
            x = x.to("mps")

        if self.use_train_segment:
            if self.training:
                xt = xt.view(B, S, -1, length)
            else:
                xt = xt.view(B, S, -1, training_length)
        else:
            xt = xt.view(B, S, -1, length)
        xt = xt * stdt[:, None] + meant[:, None]
        x = xt + x
        if length_pre_pad:
            x = x[..., :length_pre_pad]

        # x: [b, n, ch, t]
        x = x[:, :, 0]
        x = [x[:, src] for src in range(self.num_spk)]
        return x, ilens, OrderedDict()

    @property
    def num_spk(self):
        return len(self.sources)
