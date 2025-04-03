# VoRA: Integrating Visual Capabilities into LLMs

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv%20paper-2503.20680-b31b1b.svg)](https://arxiv.org/pdf/2503.20680)&nbsp;
[![huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20Collection-Vision%20as%20LoRA-yellow)](https://huggingface.co/collections/Hon-Wong/vora-67ee34c9d32e9ac2358106ae)

<p style="font-size: larger; margin-top: -5px;">
  <a href="https://arxiv.org/pdf/2503.20680">Vision as LoRA</a>
</p>

<div align="center" style="width: 100%; margin: 0 auto;">
  <img src="assets/framework.gif" alt="Framework" width="50%">
</div>

</div>

## News

* **2025-04-04:** [VoRA Weights](https://huggingface.co/collections/Hon-Wong/vora-67ee34c9d32e9ac2358106ae) are released. 

<h3 align="center">Abstract</h3>

<p>We introduce Vision as LoRA (VoRA), a novel paradigm for transforming an LLM into an MLLM. Unlike prevalent MLLM architectures that rely on external vision modules for vision encoding, VoRA internalizes visual capabilities by integrating vision-specific LoRA layers directly into the LLM. This design allows the added parameters to be seamlessly merged into the LLM during inference, eliminating structural complexity and minimizing computational overhead. Moreover, inheriting the LLM's ability of handling flexible context, VoRA can process inputs at arbitrary resolutions.</p>

<p>To further strengthen VoRA’s visual capabilities, we introduce a block-wise distillation method that transfers visual priors from a pre-trained ViT into the LoRA layers, effectively accelerating training by injecting visual knowledge. Additionally, we apply bi-directional attention masks to better capture the context information of an image. We successfully demonstrate that with additional pre-training data, VoRA can perform comparably with conventional encode-based MLLMs.</p>
