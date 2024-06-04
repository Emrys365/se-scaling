# Description
This repository contains configurations of the following SE models with different model sizes:
  * Conv-TasNet
  * DEMUCS-v4
  * BSRNN
  * TF-GridNet

The configruations are based on [ESPnet](https://github.com/espnet/espnet).

<!-- <style type="text/css">
.tg  {border-collapse:collapse;border-color:#ccc;border-spacing:0;}
.tg td{background-color:#fff;border-color:#ccc;border-style:solid;border-width:1px;color:#333;
  font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{background-color:#f0f0f0;border-color:#ccc;border-style:solid;border-width:1px;color:#333;
  font-family:Arial, sans-serif;font-size:14px;font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-z4i2{border-color:#ffffff;text-align:left;vertical-align:middle}
.tg .tg-m7vq{background-color:#ffffff;border-color:#ffffff;font-weight:bold;text-align:left;vertical-align:middle}
.tg .tg-qj9o{background-color:#ffffff;border-color:#ffffff;font-weight:bold;text-align:center;vertical-align:middle}
.tg .tg-hxqw{background-color:#f0f0f0;border-color:#ffffff;font-style:italic;text-align:left;vertical-align:middle}
.tg .tg-7bul{background-color:#f0f0f0;border-color:#ffffff;font-style:italic;text-align:right;vertical-align:middle}
.tg .tg-v0mg{border-color:#ffffff;text-align:center;vertical-align:middle}
</style> -->
<table class="tg">
<thead>
  <tr>
    <th class="tg-m7vq" rowspan="2">Model</th>
    <th class="tg-qj9o" rowspan="2">Causal</th>
    <th class="tg-qj9o" rowspan="2">#Params (M)</th>
    <th class="tg-qj9o" colspan="2">#MACs (G/s)</th>
    <th class="tg-m7vq" rowspan="2">Config file</th>
  </tr>
  <tr>
    <th class="tg-qj9o">16 kHz</th>
    <th class="tg-qj9o">48 kHz</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-hxqw" colspan="2">BSRNN</td>
    <td class="tg-7bul" colspan="3">(sampling-frequency-independent)</td>
    <td class="tg-hxqw" colspan="1"></td>
  </tr>
  <tr>
    <td class="tg-z4i2" rowspan="2">xtiny</td>
    <td class="tg-v0mg">✔︎</td>
    <td class="tg-v0mg">0.5</td>
    <td class="tg-v0mg">0.1</td>
    <td class="tg-v0mg">0.4</td>
    <td class="tg-z4i2"><a href="https://github.com/anonymous-link/se-scaling/blob/main/conf/bsrnn_xtiny.yaml">conf/bsrnn_xtiny.yaml</a></td>
  </tr>
  <tr>
    <td class="tg-v0mg">✘</td>
    <td class="tg-v0mg">0.5</td>
    <td class="tg-v0mg">0.2</td>
    <td class="tg-v0mg">0.6</td>
    <td class="tg-z4i2"><a href="https://github.com/anonymous-link/se-scaling/blob/main/conf/bsrnn_xtiny_noncausal.yaml">conf/bsrnn_xtiny_noncausal.yaml</a></td>
  </tr>
  <tr>
    <td class="tg-z4i2" rowspan="2">tiny</td>
    <td class="tg-v0mg">✔︎</td>
    <td class="tg-v0mg">1.3</td>
    <td class="tg-v0mg">0.6</td>
    <td class="tg-v0mg">1.7</td>
    <td class="tg-z4i2"><a href="https://github.com/anonymous-link/se-scaling/blob/main/conf/bsrnn_tiny.yaml">conf/bsrnn_tiny.yaml</a></td>
  </tr>
  <tr>
    <td class="tg-v0mg">✘</td>
    <td class="tg-v0mg">1.5</td>
    <td class="tg-v0mg">0.7</td>
    <td class="tg-v0mg">2.2</td>
    <td class="tg-z4i2"><a href="https://github.com/anonymous-link/se-scaling/blob/main/conf/bsrnn_tiny_noncausal.yaml">conf/bsrnn_tiny_noncausal.yaml</a></td>
  </tr>
  <tr>
    <td class="tg-z4i2" rowspan="2">small</td>
    <td class="tg-v0mg">✔︎</td>
    <td class="tg-v0mg">4.1</td>
    <td class="tg-v0mg">2.1</td>
    <td class="tg-v0mg">6.4</td>
    <td class="tg-z4i2"><a href="https://github.com/anonymous-link/se-scaling/blob/main/conf/bsrnn_small.yaml">conf/bsrnn_small.yaml</a></td>
  </tr>
  <tr>
    <td class="tg-v0mg">✘</td>
    <td class="tg-v0mg">4.8</td>
    <td class="tg-v0mg">2.8</td>
    <td class="tg-v0mg">8.5</td>
    <td class="tg-z4i2"><a href="https://github.com/anonymous-link/se-scaling/blob/main/conf/bsrnn_small_noncausal.yaml">conf/bsrnn_small_noncausal.yaml</a></td>
  </tr>
  <tr>
    <td class="tg-z4i2" rowspan="2">medium</td>
    <td class="tg-v0mg">✔︎</td>
    <td class="tg-v0mg">14.3</td>
    <td class="tg-v0mg">8.4</td>
    <td class="tg-v0mg">25.2</td>
    <td class="tg-z4i2"><a href="https://github.com/anonymous-link/se-scaling/blob/main/conf/bsrnn_medium.yaml">conf/bsrnn_medium.yaml</a></td>
  </tr>
  <tr>
    <td class="tg-v0mg">✘</td>
    <td class="tg-v0mg">16.9</td>
    <td class="tg-v0mg">11.2</td>
    <td class="tg-v0mg">33.4</td>
    <td class="tg-z4i2"><a href="https://github.com/anonymous-link/se-scaling/blob/main/conf/bsrnn_medium_noncausal.yaml">conf/bsrnn_medium_noncausal.yaml</a></td>
  </tr>
  <tr>
    <td class="tg-z4i2" rowspan="2">large</td>
    <td class="tg-v0mg">✔︎</td>
    <td class="tg-v0mg">52.9</td>
    <td class="tg-v0mg">33.4</td>
    <td class="tg-v0mg">99.9</td>
    <td class="tg-z4i2"><a href="https://github.com/anonymous-link/se-scaling/blob/main/conf/bsrnn_large.yaml">conf/bsrnn_large.yaml</a></td>
  </tr>
  <tr>
    <td class="tg-v0mg">✘</td>
    <td class="tg-v0mg">63.1</td>
    <td class="tg-v0mg">44.3</td>
    <td class="tg-v0mg">132.5</td>
    <td class="tg-z4i2"><a href="https://github.com/anonymous-link/se-scaling/blob/main/conf/bsrnn_large_noncausal.yaml">conf/bsrnn_large_noncausal.yaml</a></td>
  </tr>
  <tr>
    <td class="tg-z4i2" rowspan="2">xlarge</td>
    <td class="tg-v0mg">✔︎</td>
    <td class="tg-v0mg">83.6</td>
    <td class="tg-v0mg">66.1</td>
    <td class="tg-v0mg">197.7</td>
    <td class="tg-z4i2"><a href="https://github.com/anonymous-link/se-scaling/blob/main/conf/bsrnn_large_double.yaml">conf/bsrnn_large_double.yaml</a></td>
  </tr>
  <tr>
    <td class="tg-v0mg">✘</td>
    <td class="tg-v0mg">104.1</td>
    <td class="tg-v0mg">87.9</td>
    <td class="tg-v0mg">262.3</td>
    <td class="tg-z4i2"><a href="https://github.com/anonymous-link/se-scaling/blob/main/conf/bsrnn_large_double_noncausal.yaml">conf/bsrnn_large_double_noncausal.yaml</a></td>
  </tr>
</tbody>
</table>

--------

<!-- <style type="text/css">
.tg  {border-collapse:collapse;border-color:#ccc;border-spacing:0;}
.tg td{background-color:#fff;border-color:#ccc;border-style:solid;border-width:0px;color:#333;
  font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{background-color:#f0f0f0;border-color:#ccc;border-style:solid;border-width:0px;color:#333;
  font-family:Arial, sans-serif;font-size:14px;font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-lboi{border-color:inherit;text-align:left;vertical-align:middle}
.tg .tg-9wq8{border-color:inherit;text-align:center;vertical-align:middle}
.tg .tg-ee30{background-color:#ffffff;border-color:inherit;font-weight:bold;text-align:center;vertical-align:middle}
.tg .tg-8f26{background-color:#ffffff;border-color:inherit;font-weight:bold;text-align:left;vertical-align:middle}
.tg .tg-uzvj{border-color:inherit;font-weight:bold;text-align:center;vertical-align:middle}
.tg .tg-vqkg{background-color:#f0f0f0;border-color:inherit;font-style:italic;text-align:left;vertical-align:middle}
.tg .tg-35at{background-color:#f0f0f0;border-color:inherit;font-style:italic;text-align:right;vertical-align:middle}
</style> -->
<table class="tg">
<thead>
  <tr>
    <th class="tg-8f26" rowspan="2">Model</th>
    <th class="tg-ee30" rowspan="2">Causal</th>
    <th class="tg-ee30" rowspan="2">#Params (M)</th>
    <th class="tg-ee30" colspan="2">#MACs (G/s)</th>
    <th class="tg-8f26" rowspan="2">Config file</th>
  </tr>
  <tr>
    <th class="tg-ee30">16 kHz</th>
    <th class="tg-ee30">48 kHz</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-vqkg" colspan="2">Conv-TasNet</td>
    <td class="tg-35at" colspan="3">(input is always resampled to 48 kHz)</td>
    <td class="tg-vqkg" colspan="2"></td>
  </tr>
  <tr>
    <td class="tg-lboi">small</td>
    <td class="tg-9wq8">✘</td>
    <td class="tg-9wq8">1.1</td>
    <td class="tg-9wq8">-</td>
    <td class="tg-9wq8">8.9</td>
    <td class="tg-lboi"><a href="https://github.com/anonymous-link/se-scaling/blob/main/conf/conv_tasnet_small.yaml">conf/conv_tasnet_small.yaml</a></td>
  </tr>
  <tr>
    <td class="tg-lboi">medium</td>
    <td class="tg-9wq8">✘</td>
    <td class="tg-9wq8">14.3</td>
    <td class="tg-9wq8">-</td>
    <td class="tg-9wq8">18.7</td>
    <td class="tg-lboi"><a href="https://github.com/anonymous-link/se-scaling/blob/main/conf/conv_tasnet_medium.yaml">conf/conv_tasnet_medium.yaml</a></td>
  </tr>
  <tr>
    <td class="tg-lboi">large</td>
    <td class="tg-9wq8">✘</td>
    <td class="tg-9wq8">52.6</td>
    <td class="tg-9wq8">-</td>
    <td class="tg-9wq8">47.2</td>
    <td class="tg-lboi"><a href="https://github.com/anonymous-link/se-scaling/blob/main/conf/conv_tasnet_large.yaml">conf/conv_tasnet_large.yaml</a></td>
  </tr>
  <tr>
    <td class="tg-lboi">xlarge</td>
    <td class="tg-9wq8">✘</td>
    <td class="tg-9wq8">103.9</td>
    <td class="tg-9wq8">-</td>
    <td class="tg-9wq8">85.4</td>
    <td class="tg-lboi"><a href="https://github.com/anonymous-link/se-scaling/blob/main/conf/conv_tasnet_xlarge.yaml">conf/conv_tasnet_xlarge.yaml</a></td>
  </tr>
  <tr>
    <td class="tg-vqkg" colspan="2">DEMUCS-v4</td>
    <td class="tg-35at" colspan="3">(input is always resampld to 48 kHz)</td>
    <td class="tg-vqkg" colspan="2"></td>
  </tr>
  <tr>
    <td class="tg-lboi">tiny</td>
    <td class="tg-9wq8">✘</td>
    <td class="tg-9wq8">1.0</td>
    <td class="tg-9wq8">-</td>
    <td class="tg-9wq8">1.0</td>
    <td class="tg-lboi"><a href="https://github.com/anonymous-link/se-scaling/blob/main/conf/demucsv4_tiny.yaml">conf/demucsv4_tiny.yaml</a></td>
  </tr>
  <tr>
    <td class="tg-lboi">small</td>
    <td class="tg-9wq8">✘</td>
    <td class="tg-9wq8">4.1</td>
    <td class="tg-9wq8">-</td>
    <td class="tg-9wq8">3.5</td>
    <td class="tg-lboi"><a href="https://github.com/anonymous-link/se-scaling/blob/main/conf/demucsv4_small.yaml">conf/demucsv4_small.yaml</a></td>
  </tr>
  <tr>
    <td class="tg-lboi">medium</td>
    <td class="tg-9wq8">✘</td>
    <td class="tg-9wq8">16.2</td>
    <td class="tg-9wq8">-</td>
    <td class="tg-9wq8">13.0</td>
    <td class="tg-lboi"><a href="https://github.com/anonymous-link/se-scaling/blob/main/conf/demucsv4_medium.yaml">conf/demucsv4_medium.yaml</a></td>
  </tr>
  <tr>
    <td class="tg-lboi">large</td>
    <td class="tg-9wq8">✘</td>
    <td class="tg-9wq8">26.9</td>
    <td class="tg-9wq8">-</td>
    <td class="tg-9wq8">17.2</td>
    <td class="tg-lboi"><a href="https://github.com/anonymous-link/se-scaling/blob/main/conf/demucsv4_large.yaml">conf/demucsv4_large.yaml</a></td>
  </tr>
  <tr>
    <td class="tg-lboi">xlarge</td>
    <td class="tg-9wq8">✘</td>
    <td class="tg-9wq8">79.3</td>
    <td class="tg-9wq8">-</td>
    <td class="tg-9wq8">40.7</td>
    <td class="tg-lboi"><a href="https://github.com/anonymous-link/se-scaling/blob/main/conf/demucsv4_xlarge.yaml">conf/demucsv4_xlarge.yaml</a></td>
  </tr>
  <tr>
    <td class="tg-vqkg" colspan="2">TF-GridNet</td>
    <td class="tg-35at" colspan="3">(sampling-frequency-independent)</td>
    <td class="tg-vqkg" colspan="2"></td>
  </tr>
  <tr>
    <td class="tg-lboi">xxtiny</td>
    <td class="tg-9wq8">✘</td>
    <td class="tg-9wq8">0.1</td>
    <td class="tg-9wq8">1.9</td>
    <td class="tg-9wq8">5.6</td>
    <td class="tg-lboi"><a href="https://github.com/anonymous-link/se-scaling/blob/main/conf/tfgridnet_xxtiny.yaml">conf/tfgridnet_xxtiny.yaml</a></td>
  </tr>
  <tr>
    <td class="tg-lboi">xtiny</td>
    <td class="tg-9wq8">✘</td>
    <td class="tg-9wq8">0.5</td>
    <td class="tg-9wq8">7.4</td>
    <td class="tg-9wq8">21.7</td>
    <td class="tg-lboi"><a href="https://github.com/anonymous-link/se-scaling/blob/main/conf/tfgridnet_xtiny.yaml">conf/tfgridnet_xtiny.yaml</a></td>
  </tr>
  <tr>
    <td class="tg-lboi">tiny</td>
    <td class="tg-9wq8">✘</td>
    <td class="tg-9wq8">1.5</td>
    <td class="tg-9wq8">24.1</td>
    <td class="tg-9wq8">70.5</td>
    <td class="tg-lboi"><a href="https://github.com/anonymous-link/se-scaling/blob/main/conf/tfgridnet_tiny.yaml">conf/tfgridnet_tiny.yaml</a></td>
  </tr>
  <tr>
    <td class="tg-lboi">small</td>
    <td class="tg-9wq8">✘</td>
    <td class="tg-9wq8">5.7</td>
    <td class="tg-9wq8">89.5</td>
    <td class="tg-9wq8">261.8</td>
    <td class="tg-lboi"><a href="https://github.com/anonymous-link/se-scaling/blob/main/conf/tfgridnet_small.yaml">conf/tfgridnet_small.yaml</a></td>
  </tr>
</tbody>
</table>
