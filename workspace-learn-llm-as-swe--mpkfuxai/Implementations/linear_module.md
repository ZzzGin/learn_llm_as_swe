# Linear Module

The **Linear Module** (also known as a Dense or Fully Connected layer) is a fundamental building block in neural networks and transformer architectures. It performs an affine transformation on the incoming data, projecting features from one dimensional space to another.

In modern LLMs, linear modules are used extensively for projections, such as in the Multi-Head Attention layer (Q, K, V, and Output projections), the Position-Wise Feed-Forward Network (SwiGLU gates), and the Final Output Embedding / LM Head.

## PyTorch Implementation

```
import math
import torch
from torch import nn
from einops import einsum

class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = math.sqrt(2 / (self.in_features + self.out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, 
            a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.weight, x, 
            "out_dim in_dim, ... in_dim -> ... out_dim")
```

### `nn.Parameter`

In PyTorch, `nn.Parameter` is a subclass of `torch.Tensor` that serves a specific purpose when defined inside an `nn.Module`.

While standard tensors can be used for intermediate computations, weights and biases that need to be learned during training must be wrapped in `nn.Parameter`. When a tensor is assigned as an `nn.Parameter` attribute of an `nn.Module`, PyTorch automatically performs several key actions:

1. **Automatic Registration:** The parameter is registered in the module's parameter list. This allows helper methods like `module.parameters()` or `module.named_parameters()` to locate and return it. Optimizers (such as Adam or SGD) rely on these methods to discover and update the model's learnable weights.

2. **Gradient Tracking:** By default, `nn.Parameter` initializes with `requires_grad=True`. This instructs PyTorch's autograd engine to track operations on this tensor and compute gradients for it during the backward pass.

3. **Serialization Support:** The parameter is automatically included in the module's state dictionary (`module.state_dict()`). This ensures the weights are saved and restored correctly when using `torch.save` and `torch.load`.

In the `Linear` implementation above, wrapping the initialized weight tensor in `nn.Parameter` ensures that the projection weights are updated during backpropagation and saved properly when exporting the model.

### `torch.empty()`

`torch.empty()` is a PyTorch tensor creation function that allocates and returns a tensor of the specified shape without initializing its memory.

#### Performance and Behavior

Unlike functions like `torch.zeros()` or `torch.ones()`, which must write specific values (`0` or `1`) to every memory address in the newly allocated block, `torch.empty()` only reserves the memory block. The resulting tensor is filled with whatever raw, "garbage" data currently resides in those allocated physical memory addresses.

Because it skips the write phase, `torch.empty()` is highly efficient and avoids the computational overhead of setting initial values.

#### Application in Parameter Initialization

In neural networks, parameter weights are rarely left at zero or one, as doing so can prevent effective learning (e.g., causing symmetry-breaking issues or vanishing/exploding gradients during training). Instead, weights are initialized using specific mathematical distributions (such as Kaiming, Xavier, or truncated normal distributions).

In the `Linear` module implementation, `torch.empty()` is used to allocate the weight tensor of size `(out_features, in_features)` using the provided `device` and `dtype` factory arguments. This uninitialized memory block is then immediately populated with structured values in `reset_parameters()` via `nn.init.trunc_normal_()`. Utilizing `torch.empty()` here avoids a redundant write pass, making model instantiation faster and more memory-efficient—which is particularly critical when initializing large language models with billions of parameters.

### `nn.init.trunc_normal_()`

`nn.init.trunc_normal_()` is an in-place PyTorch initialization function that populates a tensor with values drawn from a truncated normal distribution.

#### Mechanics and In-Place Operations

In PyTorch, functions ending with a trailing underscore (such as `trunc_normal_`) operate **in-place**. Instead of allocating new memory to create a modified copy of the tensor, the function directly overwrites the existing memory block of the input tensor. When initializing parameters, in-place operations prevent unnecessary memory allocations, which is vital for keeping resource utilization low.

A **truncated normal distribution** is derived from a standard normal distribution $\mathcal{N}(\mu, \sigma^2)$, but with a strict bounding constraint. Any random value generated outside the specified interval $[a, b]$ is discarded and redrawn until it falls within the bounds.

#### Key Parameters

* **`tensor`**: The input tensor to be filled (e.g., `self.weight` in the `Linear` module).

* **`mean` (default: `0.0`)**: The mean ($\mu$) of the underlying normal distribution.

* **`std` (default: `1.0`)**: The standard deviation ($\sigma$) of the underlying normal distribution.

* **`a` (default: `-2.0`)**: The minimum value cutoff.

* **`b` (default: `2.0`)**: The maximum value cutoff.

In the custom `Linear` implementation above, the boundaries are dynamically set to three standard deviations ($a = -3\sigma$, $b = 3\sigma$). Since approximately 99.73% of values in a normal distribution naturally fall within three standard deviations of the mean, this choice discards only the most extreme statistical outliers while preserving the overall bell curve shape of the distribution.

#### Why Use Truncated Normal Initialization?

In deep neural networks and large transformer architectures, weight initialization plays a critical role in stabilization:

1. **Preventing Extreme Outliers:** Standard normal initialization has infinite support, meaning there is a non-zero probability of drawing exceptionally large or small weights. In deep architectures, even a few extreme weight values can cause activations to saturate, trigger exploding gradients, or lead to training instability in the initial epochs.

2. **Symmetry Breaking:** Initializing weights to small, random non-zero values ensures that different neurons in the same layer learn different features during backpropagation.

3. **Stability in Transformers:** Many prominent deep learning models, including Vision Transformers (ViTs) and various LLM architectures, favor truncated normal distributions over standard normal or uniform initializations. Restricting the initial weights to a tight, bounded range (such as $[-3\sigma, 3\sigma]$) ensures reliable signal propagation across dozens of layers during the first few training steps.

### `einsum` Projection

In the `forward` pass, the linear projection is computed using `einsum` (Einstein summation notation) from the `einops` library:

```
return einsum(self.weight, x, "out_dim in_dim, ... in_dim -> ... out_dim")
```

Einstein summation provides a compact and highly expressive syntax for representing tensor operations (such as matrix multiplications, transpositions, and contractions) by defining how the dimensions of input tensors map to the dimensions of the output tensor.

#### Deconstructing the Formula: `"out_dim in_dim, ... in_dim -> ... out_dim"`

The equation string specifies how the indices of the input tensors are manipulated:

* **`self.weight` (`out_dim in_dim`)**: Maps to the first input tensor (`self.weight`), which has the shape `(out_features, in_features)`. Here, `out_dim` matches `out_features`, and `in_dim` matches `in_features`.

* **`x` (`... in_dim`)**: Maps to the second input tensor (`x`). The ellipses (`...`) represent any number of arbitrary leading dimensions, such as batch size or sequence length (e.g., `(batch_size, sequence_length, in_features)`). The final dimension, `in_dim`, matches the incoming feature dimension.

* **`-> ... out_dim`**: Dictates the shape of the output tensor. The leading dimensions represented by `...` are preserved. Because the `in_dim` label appears on both inputs but is omitted from the output side of the arrow, it is **summed over** (contracted). The final dimension of the output tensor becomes `out_dim`.

Mathematically, if $W$ is the weight matrix and $X$ is the input tensor, this expression performs the following contraction:

$$
Y_{..., j} = \sum_{i} W_{j, i} \cdot X_{..., i}
$$

where $i$ indexes the `in_dim` and $j$ indexes the `out_dim`.

#### Advantages of Using `einsum` in Transformer Architectures

1. **Implicit Transposition:** In standard PyTorch, linear layers perform the operation $x W^T + b$ where the weight matrix $W$ is stored with shape `(out_features, in_features)`. To do this with typical matrix multiplication, you would need to write `x @ self.weight.t()`. The `einsum` formulation performs this contraction directly without needing an explicit transpose operation.

2. **Handling Dynamic Batch and Sequence Dimensions:** Input tensors in Large Language Models (LLMs) frequently alternate between 2D (e.g., during inference with a single token) and 3D shapes (e.g., `(batch_size, sequence_length, hidden_dim)` during training). By using ellipses (`...`), `einsum` seamlessly adapts to any number of leading batch or sequence dimensions without requiring complex reshapes, squeezing, or unsqueezing.

3. **Readability and Self-Documentation:** Using descriptive label names like `in_dim` and `out_dim` acts as inline documentation. This makes the code self-documenting, reducing the cognitive load required to understand how dimensions are transformed.