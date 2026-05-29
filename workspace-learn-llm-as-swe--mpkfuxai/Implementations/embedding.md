# Embedding Module

## Embedding Weights Lookup

This module's forward function look like this:

```
def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
    return self.weight[token_ids]
```

The `forward` method relies on **tensor indexing** (lookups) to retrieve embedding vectors. Instead of performing a computationally expensive matrix multiplication with high-dimensional one-hot vectors, PyTorch uses the index values in `token_ids` to directly slice and return rows from the learnable `self.weight` matrix.

### Dimensional Transformation

The dimensional transformation during this tensor indexing operation can be broken down as follows:

* **Input Tensor (`token_ids`)**: A 2D integer tensor of shape `(batch_size, sequence_length)`. Each element is a discrete token ID (an index).

* **Weight Tensor (`self.weight`)**: A 2D continuous parameter matrix of shape `(vocab_size, embedding_dim)`.

* **Output Tensor**: A 3D continuous tensor of shape `(batch_size, sequence_length, embedding_dim)`.

During the forward pass, the indexing operation replaces each integer in the 2D input tensor with its corresponding 1D embedding vector of size `embedding_dim` from the weight matrix:

$$
\text{Shape: } (B, T) \quad \xrightarrow{\text{Indexing / Lookup}} \quad (B, T, C)
$$

Where:

* $B$ (**Batch Size**): The number of sequences processed in parallel.

* $T$ (**Sequence Length**): The number of tokens in each sequence.

* $C$ (**Embedding Dimension** / $d_{model}$): The size of the dense vector representation for each token.