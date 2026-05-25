# Output Embedding

After processing the sequence through all the Transformer blocks, the final layer outputs a 3D tensor of shape `(batch_size, sequence_length, d_model)` representing the contextualized hidden states. While these dense vectors contain rich semantic and syntactic information, they are continuous vectors and cannot directly represent discrete words.

To map these continuous representations back to our discrete vocabulary, we use an **Output Embedding** layer (often referred to as the **Language Model Head** or **Output Projection**). This layer projects the hidden states back into the vocabulary space, generating a set of unnormalized scores (logits) for each vocabulary item at every token position.

---

## Why We Need an Output Embedding Layer

The final objective of an autoregressive language model is to predict a probability distribution over the vocabulary for the next token. The output embedding layer acts as the bridge that translates the model's inner representations back into a format we can decode.

### From Continuous to Discrete Space

At each step, the Transformer's hidden state summarizes all the preceding context into a single $d_{\text{model}}$-dimensional vector. To select a concrete token, we need a mechanism to compute how well this context vector matches every word in our dictionary:

1. **Mapping to Vocabulary Dimension**: By projecting the $d_{\text{model}}$ dimensions to $V$ dimensions (where $V$ is `vocab_size`), we generate a score (logit) for every possible word.

2. **Generating Probabilities**: These raw logits are transformed into a normalized probability distribution using the Softmax function:

   $$
P(x_{t+1} \mid x_{1:t}) = \text{softmax}(\text{logits}_t)
$$

   From this distribution, we can either sample the next token during text generation (inference) or compute the cross-entropy loss against the target token (training).

### Weight Tying (Shared Input-Output Embeddings)

Because the vocabulary size $V$ is extremely large (e.g., $32,000$ to $50,257$), the output projection weight matrix contains a massive number of parameters ($V \times d_{\text{model}}$). To optimize training and memory usage, we often use **Weight Tying** to share the weights between the **Token Embedding** layer and the **Output Embedding** layer:

* **Parameter Reduction**: Reusing the same matrix reduces the overall parameter count of the model significantly, leading to faster training and lower GPU memory utilization.

* **Semantic Coherence**: The token embedding layer learns a continuous vector space where semantically similar words are grouped together. By using the same weights to project the hidden states back, the model can leverage these structured geometric relationships. A hidden state containing "canine" context will naturally align closest to the representations of related words like *"dog"* or *"puppy"*.

## The Projection Mechanics

Conceptually, the output embedding layer performs a linear transformation on the hidden states. In frameworks like PyTorch, the weight matrix $W_{\text{out}}$ (often named `lm_head.weight`) is represented with transposed dimensions for standard matrix operations:

$$
\text{Shape of } W_{\text{out}} = (\text{Vocabulary Size}, d_{\text{model}})
$$

Given the final layer hidden states $X$ of shape `(batch_size, sequence_length, d_model)`, we compute the unnormalized logits by multiplying the hidden states by the transpose of the output weight matrix:

$$
\text{Logits} = X W_{\text{out}}^T
$$

Each element in the resulting tensor represents the dot-product similarity between a token's hidden state vector and a specific vocabulary token's embedding vector.

## Dimensional Transformation Overlook of Output Embedding Layer

During the forward pass, the dense hidden state representation is projected up to the vocabulary dimension.

* **Input Tensor Shape**: `(batch_size, sequence_length, d_model)` (abbreviated as `(B, T, C)`). Each token in the sequence is represented by a dense continuous vector of size `d_model`.

* **Output Embedding Layer Transformation**: The layer projects each continuous token vector to a vector of size `vocab_size`.

  * **Learnable Parameter**:

    * Output embedding weights - matrix of shape: `(vocab_size, d_model)`.

* **Output Tensor Shape**: `(batch_size, sequence_length, vocab_size)` (abbreviated as `(B, T, V)`).