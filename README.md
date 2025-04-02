# VoRA: Integrating Visual Capabilities Directly into LLMs

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv%20paper-2503.20680-b31b1b.svg)](https://arxiv.org/pdf/2503.20680)


</div>
<p align="center" style="font-size: larger;">
  <a href="https://arxiv.org/pdf/2503.20680">Vision as LoRA</a>
</p>

<table style="border-collapse: collapse; border: none; width: 100%;">
  <tr style="border: none;">
    <td style="width: 45%; border: none !important; padding-right: 20px; vertical-align: top;">
      <img src="assets/framework.gif" alt="Framework" style="width: 100%;">
    </td>
    <td style="width: 55%; border: none !important; vertical-align: top;">
      <h3 style="margin-top: 0;">Abstract</h3>
      <p>We introduce Vision as LoRA (VoRA), a novel paradigm for transforming an LLM into an MLLM. Unlike prevalent MLLM architectures that rely on external vision modules for vision encoding, VoRA internalizes visual capabilities by integrating vision-specific LoRA layers directly into the LLM. This design allows the added parameters to be seamlessly merged into the LLM during inference, eliminating structural complexity and minimizing computational overhead. Moreover, inheriting the LLM's ability of handling flexible context, VoRA can process inputs at arbitrary resolutions.</p>
      <p>To further strengthen VoRA’s visual capabilities, we introduce a block-wise distillation method that transfers visual priors from a pre-trained ViT into the LoRA layers, effectively accelerating training by injecting visual knowledge. Additionally, we apply bi-directional attention masks to better capture the context information of an image. We successfully demonstrate that with additional pre-training data, VoRA can perform comparably with conventional encode-based MLLMs.</p>
    </td>
  </tr>
</table>