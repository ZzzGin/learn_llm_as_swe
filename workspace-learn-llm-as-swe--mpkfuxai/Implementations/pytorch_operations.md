# PyTorch Operations

## `nn.Parameter` and `nn.Module`

[/Implementations/linear_module.md#understanding-nnparameter](/Implementations/linear_module.md#understanding-nnparameter)

## `torch.empty()`

[/Implementations/linear_module.md#torchempty](/Implementations/linear_module.md#torchempty)

## `nn.init.trunc_normal_()`

[/Implementations/linear_module.md#nninittruncnormal](/Implementations/linear_module.md#nninittruncnormal)

## `eionops.einsum`

[/Implementations/linear_module.md#einsum-projection](/Implementations/linear_module.md#einsum-projection)

## `tensor[key]` look up

[/Implementations/embedding.md#embedding-weights-lookup](/Implementations/embedding.md#embedding-weights-lookup)

[/Implementations/rope.md#4-fetching-positional-buffers-and-aligning-shapes](/Implementations/rope.md#4-fetching-positional-buffers-and-aligning-shapes)

## `tensor.to(torch.dtype)` type conversion

[/Implementations/rms_norm.md#1-numerical-stability-upcasting](/Implementations/rms_norm.md#1-numerical-stability-upcasting)

## `einops.recude`

[/Implementations/rms_norm.md#2-calculating-the-mean-square](/Implementations/rms_norm.md#2-calculating-the-mean-square)

## `torch.sigmoid()`

[/Implementations/positionwise_feedforward.md#1-activation-function](/Implementations/positionwise_feedforward.md#1-activation-function)

## `linear_module(tensor)`

$xW^T$ <=> `x @ W.t()` <=> `LieanerModuleW(x)`

<=> `einsum(W, x, "out_dim in_dim, ... in_dim -> ... out_dim")`

[/Implementations/positionwise_feedforward.md#2-linear-projections](/Implementations/positionwise_feedforward.md#2-linear-projections)

## `torch.arange(x)`

Craetes a tensor of `[0, 1, ..., x-1]`.

[/Implementations/rope.md#1-initialization-and-frequency-precomputation](/Implementations/rope.md#1-initialization-and-frequency-precomputation)

## `torch.outer(a, b)`

Computes the outer product of 1-D vectors `a` and `b`. If `a` is of shape `(M,)` and `b` is of shape `(N,)`, the resulting tensor has shape `(M, N)` where each element at index $(i, j)$ is $a[i] \cdot b[j]$.

[/Implementations/rope.md#2-computing-and-caching-rotation-angles](/Implementations/rope.md#2-computing-and-caching-rotation-angles)

## `nn.Module` `self.register_buffer`

[/Implementations/rope.md#2-computing-and-caching-rotation-angles](/Implementations/rope.md#2-computing-and-caching-rotation-angles)

## `torch.max`

[/Implementations/softmax.md#1-numerical-stability-shifting-by-max](/Implementations/softmax.md#1-numerical-stability-shifting-by-max)

## `torch.ones`

[/Implementations/attention.md#3-scaled-dot-product-attention](/Implementations/attention.md#3-scaled-dot-product-attention)

## `torch.tril`

[/Implementations/attention.md#3-scaled-dot-product-attention](/Implementations/attention.md#3-scaled-dot-product-attention)