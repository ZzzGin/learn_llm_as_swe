# Position-wise Feed-Forward

Below is the PyTorch implementation of the SwiGLU-based Position-Wise Feed-Forward Network and an explanation of how each code component maps to the underlying mathematical equations.

The SwiGLU Position-Wise Feed-Forward network is mathematically formulated as:

$$
\text{FFN}_{\text{SwiGLU}}(x) = \left( \text{SiLU}(x W_1^T) \odot (x W_3^T) \right) W_2^T
$$

Here is how the PyTorch implementation mirrors this equation step-by-step:

### 1. Activation Function

The mathematical definition of the SiLU/Swish activation function is:

$$
\text{SiLU}(x) = x \cdot \sigma(x)
$$

In code, this is implemented manually in the `silu` function using PyTorch's element-wise sigmoid function `torch.sigmoid(x)` and the multiplication operator `*`:

```
def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)
```

### 2. Linear Projections

In PyTorch, a `Linear` layer initialized without a bias term performs the matrix multiplication $x W^T$.

* `self.w1` corresponds to the weight matrix $W_1$, projecting the input from $d_{\text{model}}$ to $d_{\text{ff}}$:

  $$
x W_1^T \longrightarrow \text{Shape: } (B, T, d_{\text{ff}})
$$

  In code: `self.w1(x)`

* `self.w3` corresponds to the weight matrix $W_3$, which acts as the gate projection:

  $$
x W_3^T \longrightarrow \text{Shape: } (B, T, d_{\text{ff}})
$$

  In code: `self.w3(x)`

### 3. Gating Mechanism

The non-linear gating is implemented by applying the `silu` activation function to the output of `self.w1` and performing an element-wise multiplication ($\odot$) with the output of `self.w3`:

$$
\text{SiLU}(x W_1^T) \odot (x W_3^T)
$$

In PyTorch, this element-wise multiplication is handled via the standard `*` operator on the two intermediate tensors of shape `(batch_size, sequence_length, d_ff)`:

```
silu(self.w1(x)) * self.w3(x)
```

### 4. Down-Projection

Finally, the gated representation is projected back down to the model's residual stream dimension ($d_{\text{model}}$) using the weight matrix $W_2$ (implemented as `self.w2`):

$$
\left( \text{SiLU}(x W_1^T) \odot (x W_3^T) \right) W_2^T
$$

In the code, this wraps the entire gated expression:

```
self.w2(silu(self.w1(x)) * self.w3(x))
```

This returns a tensor with the final shape of `(batch_size, sequence_length, d_model)`.