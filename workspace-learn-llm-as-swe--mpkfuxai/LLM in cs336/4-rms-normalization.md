# Root Mean Square (RMS) Layer Normalization

In LLMs, **normalization** is a crucial technique used to stabilize the training process, prevent exploding or vanishing gradients, and accelerate convergence. It works by rescaling the activations (the outputs of a neural network layer) to have a consistent distribution across training steps.

Without normalization, the scale of activations can grow exponentially as the network gets deeper, leading to training instability and slow convergence.

---

## RMSNorm

Here we use RMSNorm, given a vector $a \in \mathbb{R}^{d_{\text{model}}}$ of activations, RMSNorm rescales each individual activation $a_i$ as follows:

$$
\text{RMSNorm}(a_i) = \frac{a_i}{\text{RMS}(a)} g_i
$$

where the root mean square $\text{RMS}(a)$ is defined as:

$$
\text{RMS}(a) = \sqrt{\frac{1}{d_{\text{model}}} \sum_{i=1}^{d_{\text{model}}} a_i^2 + \epsilon}
$$

Here:

* $g_i$ is a learnable "gain" parameter (with $d_{\text{model}}$ parameters in total).

* $\epsilon$ is a small constant (typically $10^{-5}$) added for numerical stability to avoid division by zero.

## Dimensional Transformation in the LLM Pipeline

During a forward pass in the Transformer, RMSNorm is applied to the activation tensor (hidden states) at various points (e.g., before the Self-Attention and MLP blocks).

RMSNorm is a **shape-preserving** operation. It normalizes values along the feature dimension but does not alter the overall structure of the tensor.

* **Input Tensor Shape**: `(batch_size, sequence_length, d_model)` (often abbreviated as `(B, T, C)` where $C$ represents `d_model`).

* **RMSNorm Transformation**:

  * The Root Mean Square is calculated independently for each individual token representation (across the last dimension of size `d_model`).

  * Each activation vector is divided by its RMS and then scaled element-wise by the learnable parameter.

  * **Learnable Parameter**: 

    * Gain parameter $g$ with shape `(d_model,)` which is broadcasted across the batch and sequence dimensions.

  * **Hyper parameter**: 

    * Small constant $\epsilon$ that is often fixed at $10^{-5}$.

* **Output Tensor Shape**: `(batch_size, sequence_length, d_model)`. The sequence length and batch size remain completely unchanged, and the feature dimension remains of size `d_model`.

```
Input: Hidden states of shape (B, T, C)
[
  [ [h_11], [h_12], [h_13] ],  # Sequence 1 (each [h] is a C-dim vector)
  [ [h_21], [h_22], [h_23] ]   # Sequence 2
]

==================== [ RMSNORM (Across C-dim) ] ====================

Output: Normalized states of shape (B, T, C)
[
  [ [h'_11], [h'_12], [h'_13] ], # Each [h'] is normalized & scaled
  [ [h'_21], [h'_22], [h'_23] ]
]
```

This shape preservation ensures that the tensor can continue to flow seamlessly through subsequent layers of the network without requiring any dimensional projection.

## FAQs: Understanding the Learnable Parameter $g$

### Why do we need the learnable parameter $g$?

While normalization stabilizes training by scaling activations, restricting the activations to a strict root mean square of 1 limits the model's expressivity.

* **Restoring Representational Capacity:** The learnable gain parameter $g$ allows the network to adaptively rescale the normalized activations. If the model determines that a specific feature dimension requires a larger variance or a different scale to represent information effectively, it can learn to adjust $g_i$ accordingly.

* **Feature-Specific Scaling:** Different coordinates in the $d_{\text{model}}$ space represent different semantic or syntactic features. The parameter $g$ gives the model the flexibility to scale each of these features independently.