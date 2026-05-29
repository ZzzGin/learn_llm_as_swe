# Rotary Positional Embedding (RoPE)

Below, we map the mathematical formulation of Rotary Positional Embedding (RoPE) directly to its PyTorch implementation and explain step-by-step how each operation is performed.

## Mathematical Formulas

Unlike absolute positional encodings that are added to input embeddings, Rotary Positional Embedding (RoPE) encodes relative position by rotating pairs of dimensions in the query and key vectors in the complex plane.

1. **Rotary Base and Frequencies**: For a head dimension $d_k$, we define a set of frequencies $\theta_i$ for each $2\text{D}$ subspace:

   $$
\theta_i = \theta^{-2i / d_k}, \quad i \in \left\{0, 1, \dots, \frac{d_k}{2} - 1\right\}
$$

   Here, $\theta$ is a constant base (typically $10000$).

2. **Position-dependent Rotation Angles**: For a token at position $m$, the angle of rotation for the $i$-th dimension pair is:

   $$
R_i(m) = m \theta_i
$$

3. **Rotation Matrix**: For each coordinate pair $(x^{(2i)}, x^{(2i+1)})$, the position-dependent rotation is defined as:

   $$
\begin{pmatrix} 
\tilde{x}^{(2i)} \\ 
\tilde{x}^{(2i+1)} 
\end{pmatrix} 
= 
\begin{pmatrix} 
\cos(m \theta_i) & -\sin(m \theta_i) \\ 
\sin(m \theta_i) & \cos(m \theta_i) 
\end{pmatrix} 
\begin{pmatrix} 
x^{(2i)} \\ 
x^{(2i+1)} 
\end{pmatrix}
$$

   Expanding the matrix multiplication gives:

   $$
\tilde{x}^{(2i)} = x^{(2i)} \cos(m \theta_i) - x^{(2i+1)} \sin(m \theta_i)
$$

   $$
\tilde{x}^{(2i+1)} = x^{(2i)} \sin(m \theta_i) + x^{(2i+1)} \cos(m \theta_i)
$$

## Code Implementation

```
import torch
from torch import nn


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError("RoPE requires an even embedding dimension.")

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        half_dim = d_k // 2
        dimension_indices = torch.arange(half_dim, device=device)
        frequencies = 1.0 / (theta ** (2 * dimension_indices / d_k))

        positions = torch.arange(max_seq_len, device=device)
        angles = torch.outer(positions, frequencies)
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor,
    ) -> torch.Tensor:
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        cos = self.cos[token_positions]
        sin = self.sin[token_positions]
        while cos.ndim < x_even.ndim:
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)

        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos

        output = torch.empty_like(x)
        output[..., 0::2] = rotated_even
        output[..., 1::2] = rotated_odd
        return output
```

---

## Step-by-Step Execution Analysis

### 1. Initialization and Frequency Precomputation

```
half_dim = d_k // 2
dimension_indices = torch.arange(half_dim, device=device)
frequencies = 1.0 / (theta ** (2 * dimension_indices / d_k))
```

* **The Math**: Corresponds to $\theta_i = \theta^{-2i / d_k}$.

* **How it works**:

  * RoPE splits the $d_k$ hidden size into `half_dim` ($d_k / 2$) coordinate pairs.

  * `dimension_indices` generates $[0, 1, \dots, \frac{d_k}{2} - 1]$.

  * The `frequencies` tensor precomputes the scaling factors for each $2\text{D}$ subspace. Since `theta` is typically a large positive number (e.g., $10000$), these frequencies decay as the dimension index increases, allowing different dimensions to model patterns of varying wavelengths.

### 2. Computing and Caching Rotation Angles

```
positions = torch.arange(max_seq_len, device=device)
angles = torch.outer(positions, frequencies)
self.register_buffer("cos", torch.cos(angles), persistent=False)
self.register_buffer("sin", torch.sin(angles), persistent=False)
```

* **The Math**: Corresponds to computing $m \theta_i$ for every position $m$ and frequency index $i$.

* **How it works**:

  * `positions` is a 1D tensor representing token indices $[0, 1, \dots, \text{max\_seq\_len} - 1]$.

  * `torch.outer(positions, frequencies)` computes the outer product of the position tensor and the frequency tensor. This yields a 2D matrix of shape `(max_seq_len, half_dim)` where each entry at index $(m, i)$ is $m \theta_i$.

  * We compute the cosine and sine of these angles and cache them using `self.register_buffer`. Buffers are saved alongside the model state but are not treated as trainable parameters. Setting `persistent=False` prevents them from being written to the model's `state_dict`, saving storage since they can be easily reconstructed on initialization.

### 3. Splitting Embeddings into Coordinate Pairs

```
x_even = x[..., 0::2]
x_odd = x[..., 1::2]
```

* **The Math**: Splitting the input vector $x$ into its even components $x^{(2i)}$ and odd components $x^{(2i+1)}$.

* **How it works**:

  * The input tensor `x` (representing query or key states) typically has a shape of `(batch_size, num_heads, sequence_length, d_k)`.

  * Slicing `0::2` and `1::2` along the last dimension partitions the $d_k$ embedding space. `x_even` collects all elements with even indices, and `x_odd` collects those with odd indices.

  * Both resulting tensors have the shape `(batch_size, num_heads, sequence_length, half_dim)`.

* **Example**: Suppose we have a single head vector for one token of dimension $d_k = 4$ (meaning `half_dim = 2`):

  $$
x = [a, b, c, d]
$$

  Slicing splits this vector into even and odd components:

  * `x_even` $= [a, c]$ (indices $0, 2$)

  * `x_odd` $= [b, d]$ (indices $1, 3$)

### 4. Fetching Positional Buffers and Aligning Shapes

```
cos = self.cos[token_positions]
sin = self.sin[token_positions]
while cos.ndim < x_even.ndim:
    cos = cos.unsqueeze(-3)
    sin = sin.unsqueeze(-3)
```

* **How it works**:

  * `token_positions` is a 1D or 2D tensor indicating the actual indices of the tokens in the sequence (e.g., of shape `(batch_size, sequence_length)`).

  * Indexing `self.cos` and `self.sin` with `token_positions` retrieves the cached trigonometric values corresponding to those positions. The indexed tensors have the shape `(batch_size, sequence_length, half_dim)`.

  * **Dimension Alignment**: The tensor `x_even` has dimensions corresponding to `(batch_size, num_heads, sequence_length, half_dim)`. The indexed `cos`/`sin` tensors are missing the `num_heads` dimension.

  * The `while` loop unsqueezes dimensions at index `-3` (which represents the head dimension) until `cos` has the same number of dimensions as `x_even`. This transforms the shape of `cos`/`sin` from `(batch_size, sequence_length, half_dim)` to `(batch_size, 1, sequence_length, half_dim)`. PyTorch can now automatically broadcast this across the `num_heads` dimension of `x_even` during operations.

* **Example**: Suppose `batch_size = 2`, `num_heads = 8`, `sequence_length = 3`, and `half_dim = 2` (corresponding to a head dimension $d_k = 4$ and base $\theta = 10000$).

  * We precompute the cosine buffer `self.cos` of shape `(max_seq_len = 5, half_dim = 2)` with decaying frequencies:

    $$
\text{self.cos} = \begin{pmatrix} 
1.0000 & 1.0000 \\ 
0.5403 & 1.0000 \\ 
-0.4161 & 0.9998 \\ 
-0.9900 & 0.9995 \\ 
-0.6536 & 0.9992 
\end{pmatrix}
$$

  * Let `token_positions` be a tensor of shape `(2, 3)` indicating the logical sequence positions of the tokens for each batch item (e.g., during sequence packing or batched generation with padding):

    $$
\text{token\_positions} = \begin{pmatrix} 
0 & 1 & 2 \\ 
2 & 3 & 4 
\end{pmatrix}
$$

  * Indexing the buffer retrieves a `cos` tensor of shape `(2, 3, 2)`:

    $$
\text{cos} = \left[ 
\begin{pmatrix} 
1.0000 & 1.0000 \\ 
0.5403 & 1.0000 \\ 
-0.4161 & 0.9998 
\end{pmatrix}, 
\begin{pmatrix} 
-0.4161 & 0.9998 \\ 
-0.9900 & 0.9995 \\ 
-0.6536 & 0.9992 
\end{pmatrix} 
\right]
$$

  * `x_even` has the shape `(2, 8, 3, 2)`.

  * The loop detects that `cos.ndim` (3) is less than `x_even.ndim` (4).

  * It unsqueezes at index `-3` (which is index `1` in the resulting 4D tensor, inserting a singleton dimension), transforming the shape of `cos` to `(2, 1, 3, 2)`.

  * When multiplying `x_even * cos`, PyTorch automatically broadcasts the dimension of size `1` in `cos` to match the `8` heads in `x_even`.

### Why Use an Explicit Position Tensor instead of `torch.arange`?

A common question is: *Why can't we simply determine the sequence length dynamically within the forward pass (e.g., using `torch.arange(sequence_length)`) instead of requiring an external `token_positions` tensor?*

There are two primary reasons for this design:

#### A. Autoregressive Generation with KV Caching

During inference, Large Language Models generate text token-by-token using **Key-Value (KV) Caching** to avoid redundant computations:

1. **Prefill Phase**: The model processes the entire prompt of length $L$ in a single forward pass. Here, the positions range from $0$ to $L - 1$.

2. **Generation Phase**: For each subsequent step, the model only receives the single most recently generated token as input (sequence length is $1$), while retrieval of past context is handled by fetching cached $K$ and $V$ representations.

If the forward pass dynamically computed positions using `torch.arange(sequence_length)`:

* During the generation phase, the input sequence length is $1$.

* `torch.arange(1)` would always yield position $0$.

* The newly generated token would incorrectly receive the positional embedding for position $0$, destroying its relative distance relationships with the cached keys (which are at positions $0, 1, \dots, L-1$).

By requiring `token_positions` explicitly (e.g., passing `token_positions = torch.tensor([[L]])`), we ensure that the new token receives the correct positional rotation corresponding to its absolute sequence index $L$, preserving the spatial relationships with previously cached tokens.

#### B. Sequence Packing and Batched Padding

In modern training and inference pipelines, efficiency techniques often decouple a token's tensor index from its logical sequence position:

* **Sample Packing (Flash Attention)**: Multiple independent, shorter sequences are often packed into a single training sequence to minimize padding overhead. For example, two sequences of length $3$ and $4$ might be concatenated into a single tensor of length $7$.

* **Batched Generation**: When generating for batches with varying sequence lengths, padding tokens are often used.

By passing `token_positions` explicitly, we can map each token to its true, logical position within its respective sequence (e.g., `[0, 1, 2, 0, 1, 2, 3]` for packed sequences) rather than blindly relying on its physical index in the tensor.

### 5. Applying Rotation and Reassembling the Tensor

```
rotated_even = x_even * cos - x_odd * sin
rotated_odd = x_even * sin + x_odd * cos

output = torch.empty_like(x)
output[..., 0::2] = rotated_even
output[..., 1::2] = rotated_odd
```

* **The Math**:

  $$
\tilde{x}^{(2i)} = x^{(2i)} \cos(m \theta_i) - x^{(2i+1)} \sin(m \theta_i)
$$

  $$
\tilde{x}^{(2i+1)} = x^{(2i)} \sin(m \theta_i) + x^{(2i+1)} \cos(m \theta_i)
$$

* **How it works**:

  * PyTorch performs element-wise additions, subtractions, and multiplications, broadcasting `cos` and `sin` across the `num_heads` dimension of `x_even` and `x_odd`.

  * `output = torch.empty_like(x)` allocates an uninitialized tensor of the original shape `(batch_size, num_heads, sequence_length, d_k)` directly in memory, bypassing unnecessary initialization overhead.

  * The rotated even and odd vectors are mapped back to their respective positions in the final output tensor using step slices (`0::2` and `1::2`), restoring the original structure.

* **Example**: Continuing with our single token of dimension $d_k = 4$:

  * Let the inputs be `x_even` $= [a, c]$ and `x_odd` $= [b, d]$.

  * Let the corresponding trigonometric scalars be `cos` $= [C_1, C_2]$ and `sin` $= [S_1, S_2]$.

  * We calculate the intermediate rotated arrays:

    $$
\text{rotated\_even} = [a C_1 - b S_1, \; c C_2 - d S_2]
$$

    $$
\text{rotated\_odd} = [a S_1 + b C_1, \; c S_2 + d C_2]
$$

  * `output` starts as an empty placeholder of size $4$: `[_, _, _, _]`.

  * Mapping `rotated_even` to even indices (`0::2`) and `rotated_odd` to odd indices (`1::2`) yields:

    $$
\text{output} = [a C_1 - b S_1, \; a S_1 + b C_1, \; c C_2 - d S_2, \; c S_2 + d C_2]
$$

  * This matches the analytical 2D rotation matrix applied individually to the pairs $(a, b)$ and $(c, d)$!