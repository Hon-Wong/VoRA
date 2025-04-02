# Vision as LoRA

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv%20paper-2503.20680-b31b1b.svg)](https://arxiv.org/pdf/2503.20680)

</div>

<table border="0" cellspacing="0" cellpadding="0" width="100%">
  <tr>
    <td width="45%" valign="top" style="padding-right: 20px;">
      <img src="assets/framework.gif" alt="Framework" width="100%">
    </td>
    <td width="55%" valign="top">
      <h3>Abstract</h3>
      <p>We introduce Vision as LoRA (VoRA), a novel paradigm for transforming an LLM into an MLLM. Unlike prevalent MLLM architectures that rely on external vision modules for vision encoding, VoRA internalizes visual capabilities by integrating vision-specific LoRA layers directly into the LLM. This design allows the added parameters to be seamlessly merged into the LLM during inference, eliminating structural complexity and minimizing computational overhead. Moreover, inheriting the LLM's ability of handling flexible context, VoRA can process inputs at arbitrary resolutions.</p>
      <p>To further strengthen VoRA’s visual capabilities, we introduce a block-wise distillation method that transfers visual priors from a pre-trained ViT into the LoRA layers, effectively accelerating training by injecting visual knowledge. Additionally, we apply bi-directional attention masks to better capture the context information of an image. We successfully demonstrate that with additional pre-training data, VoRA can perform comparably with conventional encode-based MLLMs.</p>
    </td>
  </tr>
</table>