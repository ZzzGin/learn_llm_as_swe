# 5 - Causal Multi-Head Self-Attention With RoPE

From the previous step, we obtain a matrix of shape `(batch_size, sequence_length, d_model)` representing the hidden states for each input token. However, the nature of language is that preceding tokens heavily influence the meaning of subsequent ones. For example, in the sentence *"The chef cooked the soup,"* the meaning of the action *"cooked"* is directly shaped by the subject *"chef"*. To generate text autoregressively, we must model this directional flow so that each token only gathers context from itself and past tokens.

Additionally, the relative distance between tokens affects how they relate to one another. For example, in the sentence *"The dog, which was running through the park, barked,"* the subject *"dog"* remains strongly tied to the verb *"barked"* despite the physical distance between them. This layer uses rotation to naturally encode these relative positions, allowing the model to capture these dependencies across varying spans.

These are done by 4 parts:

* **Multi-head mechanism**: Splits the attention process into multiple independent subspaces to capture diverse semantic interactions in parallel.

* **Self-attention**: Calculates dynamic relationships between all tokens in a sequence.

* **Causal mask**: Restricts attention so that each token can only attend to itself and preceding tokens, preventing future information leakage during autoregressive generation.

* **RoPE (Rotary Position Embedding)**: Injects positional context by rotating the query and key vectors in the complex plane, allowing the model to naturally capture relative distances and scale to longer sequence lengths.

---

## Multi-head Mechanism: Split Hidden Dimensions by Heads

The core motivation behind the **Multi-Head Mechanism** is to enable the model to simultaneously attend to different types of relationships within the same sequence layer.

While a single attention head computes a single attention pattern (determining which tokens a given token should focus on), natural language is rich with concurrent, multi-faceted relationships:

* **Syntactic dependencies** (e.g., linking verbs to their subjects)

* **Coreference resolution** (e.g., linking pronouns to their referring nouns)

* **Local collocations** (e.g., recognizing common multi-word phrases)

* **Long-range context** (e.g., tracking a subject across multiple clauses)

* **Structural cues** (e.g., parsing the role of punctuation)

A single attention pattern has limited capacity and struggles to capture all of these diverse features at once. To resolve this, multi-head attention splits the global representation space into multiple independent, lower-dimensional subspaces.

### An Intuitive Example

Consider the following sentence:

> *"The animal didn't cross the street because it was tired."*

In a multi-head architecture, different heads can focus on different linguistic aspects of the same token simultaneously:

* **Head 1 (Coreference)**: Might learn to associate the pronoun *"it"* heavily with *"animal"*.

* **Head 2 (Causal Structure)**: Might focus on linking the word *"because"* to the main clause and subordinate clause to capture the logical flow.

* **Head 3 (Subject-Verb Relation)**: Might map the adjective *"tired"* back to its subject *"animal"* (or *"it"*).

* **Head 4 (Local Attention)**: Might naturally pay attention to immediate neighbors (e.g., associating *"the"* with *"street"*).

While real-world models do not always divide their labor this cleanly, the multi-head mechanism provides the mathematical capacity for this parallel representation.

### Dimensional Splitting: Defining $d_k$

In a multi-head attention layer, the model's hidden dimension $d_{\text{model}}$ is divided across $N$ parallel attention heads (where $N = \text{num\_heads}$):

$$
d_{\text{model}} = N \times d_k
$$

Here, **$d_k$** (often referred to as $d_{\text{head}}$) represents the **dimension of the key, query, and value vectors for each individual head**.

### Dimensional Transformation Overlook of Multi-head Split

During a forward pass in the Transformer, the linear projections for the queries, keys, and values produce tensors of shape `(batch_size, sequence_length, d_model)`. To perform multi-head attention, we must partition this global feature space into $N$ parallel, lower-dimensional subspaces.

* **Input Tensor Shape**: `(batch_size, sequence_length, d_model)` (often abbreviated as `(B, T, C)` where $C$ represents `d_model`, and $C = N \times d_k$).

* **The Split and Reshape Transformation**:

  * **Reshaping**: We split the final dimension $C$ into two distinct dimensions: the number of heads $N$ and the head dimension $d_k$.

    $$
\text{Shape: } (B, T, C) \longrightarrow (B, T, N, d_k)
$$

    * Hyper parameter: $N$ for `num_heads`.

  * **Transposing**: To perform matrix multiplication on the sequence length $T$ for each head independently, we swap the sequence dimension ($T$) and the head dimension ($N$). This groups the token vectors by their respective attention heads.

    $$
\text{Shape: } (B, T, N, d_k) \longrightarrow (B, N, T, d_k)
$$

* **Output Tensor Shape**: `(batch_size, num_heads, sequence_length, d_head)` (abbreviated as `(B, N, T, d_k)`).

---

## Self-attention

At its core, **Self-Attention** allows each token in an input sequence to dynamically assign "importance" or "attention weights" to every other token. This mechanism enables the model to capture semantic and syntactic dependencies between tokens regardless of their distance in the sequence.

This is the math equation of Self-Attention. We will explain the calculation step-by-step.

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

### Queries ($Q$)

The **Query** matrix $Q$ represents the active search vectors. For each token in the sequence, its query vector asks: *"What information am I looking for in other tokens?"*

#### Dimensional Transformation Overlook of Linear Projection of Queries

* **Input Tensor Shape**: `(B, T, d_model)`, representing the hidden states from the previous layer.

* **Linear Transformation**:

  * Learnable Parameter: $W_Q$ of shape `(d_model, d_model)`.

  * The input is projected using $W_Q$ to produce a query tensor of shape `(B, T, d_model)`.

* **Split and Transpose**:

  * Following the multi-head split and transpose procedure, the query tensor is reshaped to `(B, T, N, d_k)` and transposed to `(B, N, T, d_k)`.

* **Output Tensor Shape**: `(B, N, T, d_k)`. For a single attention head, each of the $T$ tokens has a query vector of size $d_k$. Across all batch elements and heads, $Q$ has the shape `(B, N, T, d_k)`.

### Keys ($K$)

The **Key** matrix $K$ represents the indexing or profile vectors. For each token in the sequence, its key vector declares: *"Here is what kind of information I contain and can offer to others."*

Same as $Q$, to calculate $K$, we need a learnable parameter $W_K$ and its output tensor shape is `(B, N, T, d_k)`.

### Query-Key Multiplication ($Q K^T$)

To find the relationship between any pair of tokens, we compute the similarity between the queries and keys. This is done by taking the dot product between every query vector and every key vector.

Mathematically, we perform a batch matrix multiplication of the query matrix $Q$ and the transposed key matrix $K^T$:

$$
\text{Raw Scores} = Q K^T
$$

* **Transposition**: Transposing the last two dimensions of $K$ yields $K^T$ with shape `(B, N, d_k, T)`.

* **Matrix Multiplication**: Multiplying $Q$ of shape `(B, N, T, d_k)` by $K^T$ of shape `(B, N, d_k, T)` results in an attention score matrix of shape `(B, N, T, T)`.

In this `(T, T)` matrix of raw scores, the element at index $(i, j)$ represents the dot product $q_i \cdot k_j$. A higher score indicates a stronger alignment between the query of token $i$ and the key of token $j$, meaning token $i$ should pay more attention to token $j$.

### Scaling Factor ($\frac{1}{\sqrt{d_k}}$)

Once the raw attention scores are computed, they are scaled down by dividing by the square root of the head dimension, $\sqrt{d_k}$.

$$
\text{Scaled Scores} = \frac{Q K^T}{\sqrt{d_k}}
$$

#### Why Scale?

* **Mathematical Growth of Dot Products**: Assuming the components of a query vector $q$ and a key vector $k$ are independent random variables with a mean of $0$ and a variance of $1$, their dot product:

  $$
q \cdot k = \sum_{i=1}^{d_k} q_i k_i
$$

  will have a mean of $0$ and a variance of $d_k$. As the head dimension $d_k$ grows larger, the variance of the dot products increases, causing individual attention scores to grow significantly in magnitude.

* **Softmax Saturation and Vanishing Gradients**: When the inputs to the softmax function are very large in magnitude, the resulting probability distribution becomes highly peaky (concentrating almost all probability mass on a single token). In these extreme regions, the gradient of the softmax function becomes vanishingly small. During backpropagation, this "saturation" prevents gradients from flowing back through the model, severely stalling the training process.

By dividing the dot products by $\sqrt{d_k}$, we scale the variance of the raw scores back to $1$. This keeps the values input to the softmax function within a moderate range, maintaining active gradients and ensuring stable training.

### Softmax

The next step in the attention mechanism is to convert the scaled attention scores into a probability distribution using the **Softmax** function. For an unnormalized vector of scores $v \in \mathbb{R}^T$ (representing the scaled alignment scores of a single query token against all key tokens), the softmax operation is defined as:

$$
\text{softmax}(v)_i = \frac{\exp(v_i)}{\sum_{j=1}^{T} \exp(v_j)}
$$

By applying softmax, the raw scores are mapped to a range between $0$ and $1$, and they sum to $1$. These normalized values represent the **attention weights**—the relative importance that each token assigns to other tokens in the sequence.

#### Dimensional Transformation Overlook of Softmax Operation

* **Input Tensor Shape**: `(B, N, T, T)`, representing the scaled attention scores.

* **Softmax Transformation**:

  * Softmax is applied along the **last dimension** of size $T$ (representing the key tokens being attended to).

  * For each individual attention head and query token, the maximum value along that last dimension is computed and subtracted from all elements before taking the exponential.

* **Output Tensor Shape**: `(B, N, T, T)`. The shape remains completely unchanged, but the values along the last dimension now form a valid probability distribution.

### Values ($V$)

The **Value** matrix $V$ represents the actual content or representation vectors. For each token in the sequence, its value vector declares: *"Here is the semantic content I contain, which should be retrieved if someone pays attention to me."*

Like $Q$ and $K$, we project the input hidden states using a learnable weight matrix:

#### Dimensional Transformation Overlook of Leanier Projection of Values

* **Input Tensor Shape**: `(B, T, d_model)`.

* **Linear Transformation**:

  * Learnable Parameter: $W_V$ of shape `(d_model, d_model)`.

  * The input is projected to produce a value tensor of shape `(B, T, d_model)`.

* **Split and Transpose**:

  * Reshaped to `(B, T, N, d_k)` and transposed to `(B, N, T, d_k)`.

* **Output Tensor Shape**: `(B, N, T, d_k)`.

### Attention Output Calculation

The final step of the self-attention mechanism is to compute a weighted sum of the value vectors, where the weights are determined by the softmax attention weights.

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

#### Dimensional Transformation Overlook of Self Attention

* **Matrix Multiplication**:

  * We multiply the attention weights matrix of shape `(B, N, T, T)` by the value matrix $V$ of shape `(B, N, T, d_k)`.

  * For each head, this performs a standard matrix multiplication of a `(T, T)` matrix by a `(T, d_k)` matrix.

* **Output Tensor Shape**: `(B, N, T, d_k)`.

  * The resulting tensor represents the updated representations for each token, having gathered context from all other tokens in the sequence based on the calculated attention weights.

---

## Multi-head Mechanism: Concat Heads Back to Hidden Dimensions

After computing self-attention independently for each of the $N$ heads, we obtain $N$ separate output tensors. To pass these representations to subsequent layers in the Transformer block (such as the feed-forward MLP network), we must consolidate them back into a single unified hidden state.

This is achieved by **concatenating** the individual head outputs and applying a final **linear projection**.

Mathematically, the multi-head output is reconstructed as:

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \text{head}_2, \dots, \text{head}_N)
$$

The final output of the multi-head self-attention layer is then obtained by projecting this concatenated tensor using a learnable weight matrix $W_O$:

$$
\text{MultiHeadSelfAttention}(X) = \text{MultiHead}(Q, K, V) W_O
$$

Where:

* $\text{head}_i = \text{Attention}(Q_i, K_i, V_i)$ represents the attention output of the $i$-th head.

* $W_O \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$ (or $W_O \in \mathbb{R}^{N d_v \times d_{\text{model}}}$) is the output projection matrix.

### Dimensional Transformation Overlook of Multi-head Merge

During this step, the individual head representations of shape `(batch_size, num_heads, sequence_length, d_head)` are merged and projected back to the original model dimension `d_model`.

* **Input Tensor Shape**: `(B, N, T, d_k)` representing the attention outputs from all $N$ heads.

* **The Transpose and Reshape Transformation**:

  * **Transposing**: We swap the sequence length dimension ($T$) and the head dimension ($N$) to restore the sequence-first ordering.

    $$
\text{Shape: } (B, N, T, d_k) \longrightarrow (B, T, N, d_k)
$$

  * **Concatenation (Reshaping)**: Rather than explicitly concatenating $N$ separate tensors, we perform an efficient reshape. We flatten the last two dimensions ($N$ and $d_k$) into a single dimension of size $d_{\text{model}}$ (since $d_{\text{model}} = N \times d_k$). This effectively "glues" the head representations side-by-side for each token.

    $$
\text{Shape: } (B, T, N, d_k) \longrightarrow (B, T, N \times d_k) = (B, T, d_{\text{model}})
$$

* **Linear Projection ($W_O$)**:

  * **Learnable Parameter**: $W_O$ of shape `(d_model, d_model)`.

  * The concatenated tensor of shape `(B, T, d_model)` is multiplied by $W_O$.

* **Output Tensor Shape**: `(B, T, d_model)`.

### Why is the Output Projection ($W_O$) Necessary?

While concatenation brings the outputs of all attention heads back into a single tensor, it does not allow them to interact. At this stage, the features in the tensor are still strictly segregated by the heads that computed them.

The output projection matrix $W_O$ acts as a **mixing layer**:

1. **Inter-Head Communication**: It provides a learnable linear transformation that mixes information across the different subspaces of all $N$ heads.

2. **Feature Integration**: It allows the model to map the combined multi-head context back into the global representation space, ensuring that subsequent layers (like the MLP block) can process the integrated information cohesively.

---

## Causal Masking

In standard self-attention explained above, every token in a sequence is allowed to attend to every other token, regardless of its position. While this bidirectional context is incredibly powerful for tasks where the entire sequence is known beforehand (such as text classification or language translation), it poses a fundamental problem for **autoregressive text generation**.

### The Problem: Future Information Leakage

To generate text, a language model must learn to predict the next token in a sequence given the history of preceding tokens.

Consider the sequence:

> *"The chef cooked the"*

During training, we want the model to learn to predict the next token, *"soup"*, based only on the prefix *"The chef cooked the"*.

However, if we use standard self-attention on the full training sequence *"The chef cooked the soup"*, the representation of the token *"the"* at position 4 is allowed to look forward and attend to the token *"soup"* at position 5. This causes two major issues:

1. **Cheating during training**: The model can bypass learning actual linguistic patterns by simply looking ahead to copy the target token. This trivializes the pre-training task, preventing the model from learning useful representations.

2. **Inference mismatch**: During text generation (inference), future tokens do not exist yet. If the model relies on future-looking connections learned during training, it will fail to generate coherent text because those future keys and values are unavailable.

### The Naive Solution vs. Causal Masking

To prevent this lookahead, we could naively run the attention mechanism multiple times—once for every unique prefix:

* **Step 1**: Pass `["The"]` $\to$ Predict `"chef"`

* **Step 2**: Pass `["The", "chef"]` $\to$ Predict `"cooked"`

* **Step 3**: Pass `["The", "chef", "cooked"]` $\to$ Predict `"the"`

However, running self-attention $T$ times for a sequence of length $T$ is computationally expensive and highly inefficient.

Instead, we use **Causal Masking**. This technique allows us to process the entire sequence in a single parallelized forward pass while mathematically enforcing that **any token $i$ can only attend to positions $j \le i$**.

### How the Causal Mask Works

We enforce causality by modifying the raw attention scores **before** they are passed to the softmax function. We construct a 2D matrix of shape `(T, T)` where the lower-triangular part allows connections and the upper-triangular part blocks them.

For a sequence of length $T = 4$, the causal mask looks like this:

| Query \ Key | ("The") | ("chef") | ("cooked") | ("the") |
| --- | --- | --- | --- | --- |
| ("The") | Allow | Block | Block | Block |
| ("chef") | Allow | Allow | Block | Block |
| ("cooked") | Allow | Allow | Allow | Block |
| ("the") | Allow | Allow | Allow | Allow |

To apply this mask to the attention mechanism, we perform a operation called **masked fill**:

1. **Compute raw scores**: Calculate the scaled dot product scores $\frac{QK^T}{\sqrt{d_k}}$.

2. **Apply negative infinity**: For all positions where the mask blocks the connection (the upper-triangular region), we replace the raw score with negative infinity ($-\infty$).

3. **Softmax**: Since $e^{-\infty} = 0$, applying the softmax function reduces the attention weights for all future positions to exactly $0$.

$$
\text{Attention weights for } t_i = \text{softmax}\left( [s_{i,1}, s_{i,2}, \dots, s_{i,i}, -\infty, \dots, -\infty] \right)
$$

This ensures that future tokens have absolutely no influence on the representation of the current token, enabling efficient parallel training while strictly preserving autoregressive causality.

---

## Rotary Position Embedding (RoPE)

In standard self-attention, the model treats the input sequence as a **bag of words** because the mathematical operations are permutation-invariant.

### The Problem: Attention is Invariant to Order

To see why, recall the attention score computation:

$$
\text{Score}(q_i, k_j) = q_i \cdot k_j
$$

If we shuffle the order of the tokens in a sequence, their resulting query and key vectors will be identical to their unshuffled counterparts, just swapped in position. The dot product values themselves do not change. Because standard matrix multiplication does not naturally encode spatial order, a sentence like *"Dog bites man"* produces the exact same token representations as *"Man bites dog"*.

#### Limitations of Traditional Positional Encodings

Historically, models solved this using **absolute positional encodings** (either learned or sinusoidal):

* These were **added** directly to the input token embeddings before the first layer:

  $$
\vec{x}_i = \vec{e}_i + \vec{p}_i
$$

  where $\vec{e}_i$ is the semantic token embedding and $\vec{p}_i$ is the absolute position embedding.

* **Why this falls short**:

  1. **Pollution of semantic space**: Adding positional vectors directly to semantic vectors forces the model to represent two distinct types of information in the same vector space, which can degrade semantic representations.

  2. **Poor extrapolation**: Absolute positional encodings struggle to generalize to sequence lengths longer than what they were trained on. If a model is trained on sequences of up to 2048 tokens, it has no learned absolute position vector for token 2049, causing immediate performance degradation.

  3. **Weak relative awareness**: In natural language, the *relative* distance between two tokens is often much more important than their *absolute* positions in a document. Standard additive encodings do not mathematically guarantee that the dot product of two vectors will naturally reflect their relative distance.

### The Solution: RoPE (Rotary Position Embedding)

**Rotary Position Embedding (RoPE)** solves these issues by abandoning additive position encodings. Instead of adding positional offsets, RoPE **rotates** the query and key vectors in the complex plane by an angle that corresponds directly to their absolute position.

By rotating vectors, RoPE allows the self-attention mechanism to naturally capture **relative positions** between tokens through the geometric properties of rotation.

#### The 2D Rotation Intuition

To understand how rotation encodes position, imagine a 2D query vector $q = (q_1, q_2)^T$ at position $m$.

To inject its position $m$, we rotate $q$ in the 2D plane by an angle proportional to $m$. Let $\theta$ be a fixed base angle. The rotation matrix $R_{\theta}^m$ is defined as:

$$
R_{\theta}^m = \begin{pmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{pmatrix}
$$

Multiplying our vector $q$ by this matrix yields the position-aware query $q_m$:

$$
q_m = R_{\theta}^m q = \begin{pmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{pmatrix} \begin{pmatrix} q_1 \\ q_2 \end{pmatrix} = \begin{pmatrix} q_1 \cos(m\theta) - q_2 \sin(m\theta) \\ q_1 \sin(m\theta) + q_2 \cos(m\theta) \end{pmatrix}
$$

Similarly, for a key vector $k = (k_1, k_2)^T$ at position $n$, we rotate it by $n\theta$:

$$
k_n = R_{\theta}^n k
$$

### How RoPE Encodes Relative Position Mathematically

The core magic of RoPE is that when we calculate the similarity (dot product) between a query at position $m$ and a key at position $n$, **the absolute positions disappear, leaving only their relative distance $(m - n)$**.

Using the properties of 2D rotation matrices, the dot product of the rotated vectors is:

$$
q_m \cdot k_n = \langle R_{\theta}^m q, R_{\theta}^n k \rangle = q^T (R_{\theta}^m)^T R_{\theta}^n k
$$

Since a rotation matrix is orthogonal, its transpose is its inverse, representing a rotation in the opposite direction: $(R_{\theta}^m)^T = R_{\theta}^{-m}$. Sequential rotations in a 2D plane add up linearly:

$$
(R_{\theta}^m)^T R_{\theta}^n = R_{\theta}^{-m} R_{\theta}^n = R_{\theta}^{n - m}
$$

Thus, the dot product simplifies to:

$$
q_m \cdot k_n = q^T R_{\theta}^{n - m} k
$$

This means the attention score between token $m$ and token $n$ depends **only on the original query/key values and their relative distance $d = n - m$**.

### Generalizing to $d$-Dimensions

To scale RoPE to our multi-dimensional hidden states, we partition the $d$-dimensional representation space (where $d = d_k$, the head dimension) into $d/2$ independent 2D subspaces.

For a given token position $i$ and its $d$-dimensional query vector $q^{(i)} = W_q x^{(i)} \in \mathbb{R}^d$, we apply a pairwise rotation matrix $R_i \in \mathbb{R}^{d \times d}$:

$$
q'^{(i)} = R_i q^{(i)} = R_i W_q x^{(i)}
$$

Here, $R_i$ rotates pairs of embedding elements $q^{(i)}_{2k-1:2k}$ as 2D vectors by the angle $\theta_{i,k}$ for $k \in \{1, \dots, d/2\}$:

$$
\theta_{i,k} = \frac{i}{\Theta^{(2k-2)/d}}
$$

where $\Theta$ is a hyper parameter (typically $10000$).

Thus, we can represent $R_i$ as a block-diagonal matrix of size $d \times d$, with blocks $R_i^k$ for $k \in \{1, \dots, d/2\}$:

$$
R_i^k = \begin{pmatrix} \cos(\theta_{i,k}) & -\sin(\theta_{i,k}) \\ \sin(\theta_{i,k}) & \cos(\theta_{i,k}) \end{pmatrix}
$$

Giving us the full rotation matrix:

$$
R_i = \begin{pmatrix}
R_i^1 & 0 & 0 & \dots & 0 \\
0 & R_i^2 & 0 & \dots & 0 \\
0 & 0 & R_i^3 & \dots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
0 & 0 & 0 & \dots & R_i^{d/2}
\end{pmatrix}
$$

where each 0 represents a $2 \times 2$ zero matrix.

By applying this transformation, each pair of dimensions $(2k-1, 2k)$ is rotated independently:

$$
\begin{pmatrix}
q'^{(i)}_{2k-1} \\
q'^{(i)}_{2k}
\end{pmatrix} =
\begin{pmatrix}
\cos(\theta_{i,k}) & -\sin(\theta_{i,k}) \\
\sin(\theta_{i,k}) & \cos(\theta_{i,k})
\end{pmatrix}
\begin{pmatrix}
q^{(i)}_{2k-1} \\
q^{(i)}_{2k}
\end{pmatrix}
$$

The same rotation process is then performed for the key vector $k^{(j)}$ at position $j$, rotating by the corresponding $R_j$:

$$
k'^{(j)} = R_j k^{(j)} = R_j W_k x^{(j)}
$$

### A Concrete Example: "Dog bites man" vs. "Man bites dog"

To understand how RoPE preserves word order and distinguishes meaning, consider two sentences with identical words but opposite meanings:

1. **Sentence A**: *"Dog bites man"*

   * Position 0: `Dog` (Subject)

   * Position 1: `bites` (Verb)

   * Position 2: `man` (Object)

2. **Sentence B**: *"Man bites dog"*

   * Position 0: `Man` (Subject)

   * Position 1: `bites` (Verb)

   * Position 2: `dog` (Object)

Without position embeddings, the attention score between the query for `bites` and the key for `Dog` would be exactly the same in both sentences:

$$
\text{Score}(\text{bites}, \text{Dog}) = q_{\text{bites}} \cdot k_{\text{Dog}}
$$

Because dot products are permutation-invariant, the model cannot distinguish who is performing the action and who is receiving it.

#### Applying RoPE

With RoPE, we rotate the query and key vectors in a 2D subspace by an angle proportional to their sequence positions.

In **Sentence A** ("Dog bites man"):

* `Dog` is at position $0$, so its key is rotated by $0$:

  $$
k_{\text{Dog}}^{(0)} = R^0 k_{\text{Dog}}
$$

* `bites` is at position $1$, so its query is rotated by $1\theta$:

  $$
q_{\text{bites}}^{(1)} = R^1 q_{\text{bites}}
$$

* The attention score for `bites` attending to `Dog` (relative distance of $-1$) becomes:

  $$
\text{Score}_{\text{A}}(\text{bites} \to \text{Dog}) = \langle R^1 q_{\text{bites}}, R^0 k_{\text{Dog}} \rangle = q_{\text{bites}}^T R^{-1} k_{\text{Dog}}
$$

In **Sentence B** ("Man bites dog"):

* `Man` is at position $0$, so its key is rotated by $0$:

  $$
k_{\text{Man}}^{(0)} = R^0 k_{\text{Man}}
$$

* `bites` is at position $1$, so its query is rotated by $1\theta$:

  $$
q_{\text{bites}}^{(1)} = R^1 q_{\text{bites}}
$$

* The attention score for `bites` attending to `Man` (relative distance of $-1$) becomes:

  $$
\text{Score}_{\text{B}}(\text{bites} \to \text{Man}) = \langle R^1 q_{\text{bites}}, R^0 k_{\text{Man}} \rangle = q_{\text{bites}}^T R^{-1} k_{\text{Man}}
$$

By using RoPE, the relative distance of $-1$ between the verb and its preceding subject is consistently represented by the rotation $R^{-1}$, regardless of whether the subject is `Dog` or `Man`.

Similarly, the object at position $2$ attending back to the verb at position $1$ also shares a relative distance of $-1$:

* In Sentence A, `man` at position $2$ attends to `bites` at position $1$:

  $$
\text{Score}_{\text{A}}(\text{man} \to \text{bites}) = \langle R^2 q_{\text{man}}, R^1 k_{\text{bites}} \rangle = q_{\text{man}}^T R^{-1} k_{\text{bites}}
$$

* In Sentence B, `dog` at position $2$ attends to `bites` at position $1$:

  $$
\text{Score}_{\text{B}}(\text{dog} \to \text{bites}) = \langle R^2 q_{\text{dog}}, R^1 k_{\text{bites}} \rangle = q_{\text{dog}}^T R^{-1} k_{\text{bites}}
$$

This consistent mathematical encoding of relative positions allows the model to learn that a noun preceding the verb by one position (relative distance $-1$) acts as the subject, while a noun succeeding the verb by one position acts as the object, resolving the ambiguity of word order.

### Dimensional Transformation Overlook of RoPE

During the forward pass, Rotary Position Embedding is applied after the Query and Key projection steps but before calculating attention scores.

* **Input Tensor Shape**: `(batch_size, num_heads, sequence_length, d_head)` (often abbreviated as `(B, N, T, d_k)`) for both $Q$ and $K$.

* **Transformation**: For each token position index $t \in [1, T]$, the $d_k$-dimensional query and key vectors are rotated in pairs of dimensions.

* **Output Tensor Shape**: `(batch_size, num_heads, sequence_length, d_head)`. The shape remains completely unchanged, but the query and key representations now embed relative distance relationships.

---

## Sum It Up

To synthesize all the components we have explored, a single forward pass of the **Causal Multi-Head Self-Attention with RoPE** layer executes the following step-by-step pipeline:

1. **Projection**: The input token hidden states $X \in \mathbb{R}^{B \times T \times d_{\text{model}}}$ are linearly projected using three learnable weight matrices ($W_Q, W_K, W_V$) to generate the global Query ($Q$), Key ($K$), and Value ($V$) representations.

2. **Multi-Head Split & Transpose**: The global $d_{\text{model}}$ dimension of $Q, K,$ and $V$ is reshaped and split across $N$ attention heads of dimension $d_k$, and then transposed to isolate the sequence dimension $T$ for independent parallel computation.

3. **Applying RoPE**: The query and key tensors are rotated in 2D pairs across their $d_k$ dimensions according to their absolute sequence positions, encoding relative distance information directly into their geometry.

4. **Scaled Dot-Product Matrix Multiplication**: The rotated queries and keys are multiplied ($Q K^T$) and scaled by $1 / \sqrt{d_k}$ to calculate pairwise interaction scores.

5. **Causal Masking**: An upper-triangular causal mask is applied to the raw scores, setting future token positions to $-\infty$ so that softmax maps their attention weights to exactly $0$.

6. **Attention Softmax & Value Aggregation**: Softmax is applied to normalize the attention weights. These weights are then used to compute a weighted sum of the value vectors ($V$), dynamically pooling context across preceding tokens.

7. **Concat & Output Projection**: The resulting attention heads are transposed and merged (concatenated) back into the original $d_{\text{model}}$ representation space. A final output projection ($W_O$) mixes the head features, yielding the updated hidden states.

### Dimensional Transformation Overlook of Causal Multi-head Self-attention with RoPE

* **Input Tensor Shape**: `(batch_size, sequence_length, d_model)` (abbreviated as `(B, T, C)` where $C = d_{\text{model}}$).

* **Transformations In-Between**:

  1. **Linear Projection for $Q$, $K$, and $V$**:

     * **Calculation**: The input hidden states are multiplied by three independent weight matrices to project the features into Query ($Q$), Key ($K$), and Value ($V$) representations.

     * **Learnable Parameters**: $W_Q, W_K, W_V \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$.

     * **Intermediate Shape**: Three tensors of shape `(B, T, C)`.

  2. **Multi-Head Split & Transpose**:

     * **Calculation**: The global feature dimension $C$ is split into $N$ heads of dimension $d_k$ (where $C = N \times d_k$). The sequence dimension $T$ and the head dimension $N$ are swapped so attention calculations can be performed independently per head.

     * **Hyperparameters**: `num_heads` ($N$) and `head_dim` ($d_k$).

     * **Intermediate Shape**: $Q, K, V$ tensors of shape `(B, N, T, d_k)`.

  3. **Rotary Position Embedding (RoPE)**:

     * **Calculation**: Applies pairwise rotations to the query ($Q$) and key ($K$) tensors across their $d_k$ dimensions according to their absolute sequence indices. Value ($V$) remains unmodified.

     * **Hyperparameters**: Base angle $\Theta$ (typically $10000$).

     * **Intermediate Shape**: Rotated $Q$ and $K$ of shape `(B, N, T, d_k)`.

  4. **Scaled Dot-Product Attention**:

     * **Calculation**:

       * Multiplies queries and transposed keys ($Q K^T$) to compute pairwise attention scores.

       * Scales the scores by $1 / \sqrt{d_k}$.

       * Sets future positions in the upper triangular portion of the score matrix to $-\infty$ (causal masking).

       * Applies a softmax over the last dimension to get attention weights.

       * Multiplies the resulting weights by the value tensor ($V$).

     * **Intermediate Shape**: Attention scores/weights of shape `(B, N, T, T)`, resulting in an attention output of shape `(B, N, T, d_k)`.

  5. **Transpose & Concatenate (Merge Heads)**:

     * **Calculation**: Swaps the sequence dimension ($T$) and head dimension ($N$) back to their original order, then flattens the last two dimensions to reconstruct the unified hidden size $C$.

     * **Intermediate Shape**: `(B, T, C)`.

  6. **Output Projection**:

     * **Calculation**: Linearly projects the concatenated multi-head representation to mix and integrate the features computed by different attention heads.

     * **Learnable Parameters**: $W_O \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$.

     * **Intermediate Shape**: `(B, T, C)`.

* **Output Tensor Shape**: `(batch_size, sequence_length, d_model)` (abbreviated as `(B, T, C)`), preserving the exact dimensions of the input tensor.