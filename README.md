<p align="center">
  <h1 align="center">Vision as LoRA</h1>
</p>

<table>
  <tr>
    <td width="45%">
      <img src="assets/framework.gif" alt="VoRA Framework" width="100%">
      <br>
      <em>Illustration of Vora</em>
    </td>
    <td width="55%">
      <h3>Abstract</h3>
      <p>We introduce Vision as LoRA (VoRA), a novel paradigm for transforming an LLM into an MLLM. Unlike prevalent MLLM architectures that rely on external vision modules for vision encoding, VoRA internalizes visual capabilities by integrating vision-specific LoRA layers directly into the LLM. This design allows the added parameters to be seamlessly merged into the LLM during inference, eliminating structural complexity and minimizing computational overhead. Moreover, inheriting the LLM's ability of handling flexible context, VoRA can process inputs at arbitrary resolutions.</p>
      <p>To further strengthen VoRA’s visual capabilities, we introduce a block-wise distillation method that transfers visual priors from a pre-trained ViT into the LoRA layers, effectively accelerating training by injecting visual knowledge. Additionally, we apply bi-directional attention masks to better capture the context information of an image. We successfully demonstrate that with additional pre-training data, VoRA can perform comparably with conventional encode-based MLLMs.</p>
    </td>
  </tr>
</table>
