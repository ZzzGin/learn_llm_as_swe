# KV Cache

During autoregressive text generation, a Large Language Model (LLM) predicts tokens one by one. In a naive implementation, at each generation step, the entire sequence (including all previously generated tokens) is fed into the model. This results in highly redundant computations, which is why we use **Key-Value (KV) Caching**.

---

## The Inefficiency of Naive Generation

Let's visualize why naive generation is inefficient. Suppose we have a prompt with two tokens, $[x_1, x_2]$, and we want to generate the next two tokens ($x_3$ and then $x_4$).

### Step 1: Predict $x_3$ from $[x_1, x_2]$

We feed the sequence of length 2 into the model. For a single attention head, we project the input representations to compute $Q, K, V \in \mathbb{R}^{2 \times d_k}$:

$$
Q^{(1..2)} = \begin{bmatrix} \mathbf{q}_1 \\ \mathbf{q}_2 \end{bmatrix}, \quad
K^{(1..2)} = \begin{bmatrix} \mathbf{k}_1 \\ \mathbf{k}_2 \end{bmatrix}, \quad
V^{(1..2)} = \begin{bmatrix} \mathbf{v}_1 \\ \mathbf{v}_2 \end{bmatrix}
$$

We compute attention and sample the next token, $x_3$.

### Step 2: Predict $x_4$ from $[x_1, x_2, x_3]$

In a naive implementation, we must feed the entire sequence $[x_1, x_2, x_3]$ back into the model. We perform linear projections on all three tokens:

$$
Q^{(1..3)} = \begin{bmatrix} \mathbf{q}_1 \\ \mathbf{q}_2 \\ \mathbf{q}_3 \end{bmatrix}, \quad
K^{(1..3)} = \begin{bmatrix} \mathbf{k}_1 \\ \mathbf{k}_2 \\ \mathbf{k}_3 \end{bmatrix}, \quad
V^{(1..3)} = \begin{bmatrix} \mathbf{v}_1 \\ \mathbf{v}_2 \\ \mathbf{v}_3 \end{bmatrix}
$$

Notice that the vectors $\mathbf{k}_1, \mathbf{k}_2$ and $\mathbf{v}_1, \mathbf{v}_2$ are **identical** to the ones computed in Step 1. Re-projecting $x_1$ and $x_2$ at every subsequent step is a waste of computational resources. As the sequence length $T$ grows, the cost of recomputing these keys and values scales quadratically $O(T^2)$.

---

## How KV Cache Works

The KV Cache optimizes this process by storing the computed key and value vectors of previous tokens in memory. When generating the next token, we only pass the **most recently generated token** through the model.

### 1. Initial Phase (Prefill)

When we process the initial prompt $[x_1, x_2]$, we compute and save the initial keys and values in the cache:

$$
K_{\text{cache}} = \begin{bmatrix} \mathbf{k}_1 \\ \mathbf{k}_2 \end{bmatrix}, \quad
V_{\text{cache}} = \begin{bmatrix} \mathbf{v}_1 \\ \mathbf{v}_2 \end{bmatrix}
$$

We output $x_3$.

### 2. Autoregressive Phase (Decoding)

To generate $x_4$, instead of processing $[x_1, x_2, x_3]$, we feed **only** the single new token $x_3$ into the model.

1. **Projection**: We perform linear projection only for $x_3$ to obtain its single-row query, key, and value vectors:

   $$
\mathbf{q}_3 \in \mathbb{R}^{1 \times d_k}, \quad \mathbf{k}_3 \in \mathbb{R}^{1 \times d_k}, \quad \mathbf{v}_3 \in \mathbb{R}^{1 \times d_k}
$$

2. **Cache Update**: We append the newly computed $\mathbf{k}_3$ and $\mathbf{v}_3$ to our existing caches:

   $$
K_{\text{cache}} \leftarrow \begin{bmatrix} K_{\text{cache}} \\ \mathbf{k}_3 \end{bmatrix} = \begin{bmatrix} \mathbf{k}_1 \\ \mathbf{k}_2 \\ \mathbf{k}_3 \end{bmatrix} \in \mathbb{R}^{3 \times d_k}
$$

   $$
V_{\text{cache}} \leftarrow \begin{bmatrix} V_{\text{cache}} \\ \mathbf{v}_3 \end{bmatrix} = \begin{bmatrix} \mathbf{v}_1 \\ \mathbf{v}_2 \\ \mathbf{v}_3 \end{bmatrix} \in \mathbb{R}^{3 \times d_k}
$$

3. **Attention Computation**: Since we only care about the outputs of the current step (the final position), we calculate the attention scores of our single active query $\mathbf{q}_3$ against all keys in $K_{\text{cache}}$:

   $$
A = \text{softmax}\left(\frac{\mathbf{q}_3 K_{\text{cache}}^T}{\sqrt{d_k}}\right) \in \mathbb{R}^{1 \times 3}
$$

   Explicitly, the multiplication $\mathbf{q}_3 K_{\text{cache}}^T$ yields a row vector of raw attention scores:

   $$
\mathbf{q}_3 K_{\text{cache}}^T = \begin{bmatrix} \mathbf{q}_3 \mathbf{k}_1^T & \mathbf{q}_3 \mathbf{k}_2^T & \mathbf{q}_3 \mathbf{k}_3^T \end{bmatrix}
$$

4. **Context Output**: We multiply the attention weights by $V_{\text{cache}}$ to obtain the final attention output for the token $x_3$:

   $$
\text{Attention Output} = A V_{\text{cache}} = a_1 \mathbf{v}_1 + a_2 \mathbf{v}_2 + a_3 \mathbf{v}_3 \in \mathbb{R}^{1 \times d_k}
$$

   where $a_1, a_2, a_3$ are the elements of the softmax row vector $A$.

This output is identical to the one produced by the naive method, but we have avoided projecting $x_1$ and $x_2$ again, reducing the linear projection complexity of each decoding step from $O(T)$ to $O(1)$ per layer.

---

## KV Caching across Multiple Layers

In a deep Transformer architecture, each layer operates on its own distinct representation space. Therefore, we must maintain a separate KV Cache for **every single attention layer** in the model.

### Why Layer-by-Layer Caching is Necessary

Consider a Transformer with $L$ layers. The forward pass at any given layer $l$ depends sequentially on the output (activations) of the previous layer $l-1$:

$$
\mathbf{h}_t^{(l-1)} \xrightarrow{\text{Layer } l \text{ Attention}} \mathbf{h}_t^{(l)}
$$

Because the hidden representations $\mathbf{h}_t^{(l)}$ change at each layer as they progress through the network, the keys and values computed at layer $l$ are completely different from those computed at layer $l-1$:

* **Layer 1** computes keys and values $K^{(1)}, V^{(1)}$ using the initial token embeddings.

* **Layer 2** computes keys and values $K^{(2)}, V^{(2)}$ using the output representations of Layer 1.

* **Layer $L$** computes keys and values $K^{(L)}, V^{(L)}$ using the output representations of Layer $L-1$.

To generate a token autoregressively, the model must perform the full sequential forward pass through all $L$ layers. At each layer $l$, the active query $\mathbf{q}^{(l)}$ must attend to the historical keys $K_{\text{cache}}^{(l)}$ and retrieve context from $V_{\text{cache}}^{(l)}$ specific to that layer.

Consequently, the complete KV Cache is not a single pair of matrices, but a collection of pairs indexed by the layer number:

$$
\text{KV Cache} = \left\{ \left(K_{\text{cache}}^{(1)}, V_{\text{cache}}^{(1)}\right), \left(K_{\text{cache}}^{(2)}, V_{\text{cache}}^{(2)}\right), \dots, \left(K_{\text{cache}}^{(L)}, V_{\text{cache}}^{(L)}\right) \right\}
$$

---

## Memory Footprint of the KV Cache

While the KV Cache drastically speeds up decoding by eliminating redundant FLOPs, it introduces a severe memory bottleneck. The size of the KV cache grows linearly with the batch size, sequence length, and model depth.

For a model using 16-bit precision (`fp16` or `bf16`), each parameter occupies 2 bytes. The memory footprint of the KV cache can be calculated as:

$$
\text{KV Cache Size (Bytes)} = 2 \times B \times T \times L \times H_{\text{KV}} \times d_k \times 2
$$

Where:

* **$2$**: Factor for storing both Keys ($K$) and Values ($V$).

* **$B$**: Batch size.

* **$T$**: Sequence length (prompt tokens + generated tokens).

* **$L$**: Number of layers.

* **$H_{\text{KV}}$**: Number of key-value heads per layer (equals the number of query heads in standard Multi-Head Attention, but is smaller in Grouped-Query Attention).

* **$d_k$**: Head dimension (size of each key/value vector).

* **$2$**: Bytes per element (for 16-bit float representations).

### Practical Example: Llama-3-8B

To put this into perspective, let's calculate the KV Cache size for a single instance of a Llama-3-8B model with the following configuration:

* $L = 32$ layers

* $H_{\text{KV}} = 8$ key-value heads (utilizing Grouped-Query Attention)

* $d_k = 128$ head dimension

* Precision: 16-bit float ($2$ bytes)

For a batch size ($B$) of $1$ and a sequence length ($T$) of $8,192$ tokens:

$$
\text{KV Cache Size} = 2 \times 1 \times 8192 \times 32 \times 8 \times 128 \times 2 \text{ bytes}
$$

$$
\text{KV Cache Size} = 1,073,741,824 \text{ bytes} = 1 \text{ GiB}
$$

If the batch size increases to $32$ with the same context window, the KV cache alone requires **32 GiB of GPU VRAM**, which exceeds the capacity of standard consumer GPUs. This massive memory footprint is why optimizations like Grouped-Query Attention (GQA), Multi-Query Attention (MQA), and PagedAttention are crucial for serving large models efficiently.