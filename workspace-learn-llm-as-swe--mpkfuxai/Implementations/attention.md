# Causal Multi-head Self-attention With RoPE

### 1. Multi-Head Split and Linear Projections

In the initialization block of `MultiheadSelfAttention`, we define four linear transformations representing the projection weights ($W_Q, W_K, W_V, W_O$):

```
self.w_q = Linear(d_model, d_model, device=device, dtype=dtype)
self.w_k = Linear(d_model, d_model, device=device, dtype=dtype)
self.w_v = Linear(d_model, d_model, device=device, dtype=dtype)
self.w_o = Linear(d_model, d_model, device=device, dtype=dtype)
```

These project the input sequence representation $X$ into the global query, key, and value spaces.

To transition from global spaces to independent head subspaces, the code utilizes `rearrange` from the `einops` library in `project_qkv`:

```
q = rearrange(
    self.w_q(x),
    "... seq (head d_head) -> ... head seq d_head",
    head=self.num_heads,
)
```

This splits the last dimension `d_model` (which equals `head * d_head`) into individual heads and moves the `head` dimension forward. This matches the mathematical transformation:

$$
\text{Shape: } (B, T, d_{\text{model}}) \longrightarrow (B, N, T, d_k)
$$

where $B$ represents the batch size (handled dynamically as `...`), $T$ is the sequence length (`seq`), $N$ is the number of heads (`head`), and $d_k$ is the head dimension (`d_head`).

---

### 2. Rotary Positional Embedding (RoPE)

Mathematically, RoPE rotates the query and key vectors in 2D complex planes to naturally encode positional details:

$$
Q_{\text{rope}} = \mathbf{R}_Q Q, \quad K_{\text{rope}} = \mathbf{R}_K K
$$

where $\mathbf{R}$ represents the position-dependent rotation matrix applied to each pair of dimensions.

In `MultiheadSelfAttentionWithRoPE`, we apply this transformation only to $Q$ and $K$ (leaving $V$ position-agnostic):

```
q = self.rope(q, token_positions)
k = self.rope(k, token_positions)
```

This injects relative sequence positions directly into the representations, ensuring that when query-key dot products are computed, they naturally reflect distance-based relationships.

---

### 3. Scaled Dot-Product Attention

The core attention formula is:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V
$$

This mathematical sequence is written directly in `scaled_dot_product_attention`:

#### A. Matrix Multiplication ($Q K^T$)

Instead of manual transposes and matrix multiplication, the code utilizes `einsum` to perform the dot product over the head dimension $d_k$:

```
attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key")
```

This multiplies each query token's representation with each key token's representation, outputting a score matrix of shape `(..., seq_len, seq_len)`.

#### B. Scaling Factor ($\frac{1}{\sqrt{d_k}}$)

To prevent the variance of the dot products from growing too large with higher dimensions—which leads to vanishing gradients in the softmax step—we scale the scores:

```
attention_scores = attention_scores / math.sqrt(d_k)
```

#### C. Causal Masking ($M$)

To avoid future information leakage during autoregressive text generation, a causal lower-triangular mask is generated in `attend`:

```
causal_mask = torch.tril(
    torch.ones(seq_len, seq_len, dtype=torch.bool, device=q.device)
)
```

This mask is a boolean matrix where indices $i \ge j$ are `True` (allowed) and $i < j$ are `False` (masked).

In `scaled_dot_product_attention`, we apply this mask by filling all `False` indices with negative infinity ($-\infty$):

```
if mask is not None:
    attention_scores = attention_scores.masked_fill(~mask, float("-inf"))
```

*Note:* The bitwise negation operator `~` converts the boolean mask so that all the future sequence indices (formerly `False`) become `True`, identifying exactly what to fill with $-\infty$.

#### D. Softmax

The scaled and masked scores are passed through the softmax activation along the last dimension:

```
attention_weights = soft_max(attention_scores, dim=-1)
```

Because $e^{-\infty} = 0$, masked "future" tokens receive an attention weight of exactly $0$, preventing information leakage.

#### E. Weighted Value Context Retrieval

The normalized weights are multiplied by the value matrix $V$ using `einsum`:

```
return einsum(attention_weights, V, "... query key, ... key d_v -> ... query d_v")
```

This produces the final context-rich token representations.

---

### 4. Head Concatenation and Output Projection

Once self-attention is computed for each head, we must reconstruct the original model dimensions.

In `attend`, `rearrange` handles the concatenation back into a single vector of shape `(..., seq_len, d_model)`:

```
return rearrange(
    attention_output,
    "... head seq d_head -> ... seq (head d_head)",
)
```

This is equivalent to the mathematical concatenation step:

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_N)
$$

Finally, in the `forward` pass, we project the concatenated hidden states back using the output projection matrix $W_O$:

```
return self.w_o(attention_output)
```

This projection acts as a mixing layer, allowing information captured by different attention heads to interact and blend cohesively before being passed to the next block of the architecture.