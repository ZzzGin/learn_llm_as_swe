# Residual Path

In a standard Transformer block, the output of the **Causal Multi-Head Self-Attention (with RoPE)** is not passed directly to the next stage of the model. Instead, it is added back to the block's original input $X$ via a **Residual Connection** (also known as a skip connection).

Mathematically, this addition is represented as:

$$
Y = X + \text{MultiHeadSelfAttention}(\text{Norm}(X))
$$

Where:

* $X$ is the input tensor to the sub-layer.

* $\text{MultiHeadSelfAttention}(\text{Norm}(X))$ is the output of the causal multi-head attention module (having already been normalized, typically using RMSNorm, in modern Pre-LN architectures).

* $Y$ is the output of this attention block, which serves as the input to the subsequent feed-forward network (FFN) sub-layer.

---

## Why Use Residual Connections?

Residual connections are one of the most critical architectural innovations enabling the training of modern, deep neural networks. Without them, scaling models to dozens or hundreds of layers would be virtually impossible.

### 1. Mitigating Vanishing and Exploding Gradients (The Gradient Highway)

In very deep networks, backpropagating gradients through many layers of matrix multiplications can cause gradients to exponentially shrink (vanish) or grow (explode).

By adding the input $X$ directly to the output of the sub-layer, we create an uninterrupted "gradient highway". Mathematically, when we compute the derivative of the output with respect to the input:

$$
\frac{\partial Y}{\partial X} = I + \frac{\partial \text{MultiHeadSelfAttention}(\text{Norm}(X))}{\partial X}
$$

The identity matrix $I$ represents the direct residual path. Even if the gradients through the attention mechanism $\frac{\partial \text{MultiHeadSelfAttention}(\text{Norm}(X))}{\partial X}$ approach zero (vanish) due to saturation or small weight initializations, the gradient can still flow back to earlier layers completely unimpeded through the $+ I$ term.

### 2. Purity of the Residual Stream (Shared Workspace)

The residual connection can be thought of as a **residual stream** that runs through the entire network. Rather than forcing each layer to completely reconstruct and redefine the representations of the tokens from scratch, the residual stream acts as a **shared workspace**:

* Each attention head or feed-forward block simply "reads" from this stream.

* It computes a localized update (or delta) representing new context or relationships.

* It "writes" (adds) this delta back into the stream.

This allows the model to incrementally refine token representations layer-by-layer while naturally retaining the basic, unprocessed semantic information from earlier layers.

### 3. Pre-Norm vs. Post-Norm and Path Uninterruptedness

In the original Transformer architecture (Post-LN), layer normalization was applied *after* the residual addition: $\text{Norm}(X + \text{SubLayer}(X))$. This meant that the residual path itself was continuously normalized at every step, which scales down the gradients and complicates optimization.

In modern architectures (such as the one implemented in CS336), we use **Pre-Normalization (Pre-LN)**:

$$
Y = X + \text{SubLayer}(\text{RMSNorm}(X))
$$

By normalizing the input *before* it enters the attention sub-layer, the main residual path $X \to Y \to \dots$ remains completely uninterrupted and free of normalization. This ensures that the variance of the gradients remains stable across arbitrary depths, allowing for stable training from step zero without requiring highly sensitive learning rate warm-up schedules.

## Dimensional Transformation Overlook of Residual Addition

For element-wise addition ($X + \text{Attention}(X)$) to be mathematically valid, the two tensors must have the exact same shape. This means that all sub-layer transformations must strictly preserve the model's hidden dimension $d_{\text{model}}$.

* **Input Tensor $X$ Shape**: `(batch_size, sequence_length, d_model)` (abbreviated as `(B, T, C)`).

* **Attention Output Shape**: `(batch_size, sequence_length, d_model)`.

  * *Note*: As explained in the previous step, the individual attention head outputs of shape `(B, N, T, d_k)` were concatenated and linearly projected back to `(B, T, d_model)` using $W_O$ specifically to restore this shape.

* **Element-wise Addition**:

  $$
Y = X + \text{MultiHeadSelfAttention}(X)
$$

  Since both operands have identical dimensions, they are added element-by-element across the batch, sequence, and channel dimensions.

* **Output Tensor $Y$ Shape**: `(batch_size, sequence_length, d_model)`.

This uniform shape is preserved across all blocks, allowing multiple Transformer layers to be cleanly stacked on top of one another.