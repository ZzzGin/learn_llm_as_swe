# 3 - Token Embedding

To train a transformer model, we cannot feed raw token IDs (discrete integers) directly into the neural network layers. Token IDs are arbitrary categorical indices; a token ID of `258` is not "twice as large" or "mathematically distant" from `129` in any meaningful way.

To solve this, we use a **Token Embedding** layer. This layer acts as a lookup table that maps each discrete token ID to a continuous, high-dimensional vector. These vectors represent the semantic and syntactic meaning of the tokens in a geometric space.

---

## Why We Need an Embedding Layer

The primary reason why we transform discrete token IDs into continuous embeddings is to move the tokens to semantic spaces

In their raw form, token IDs are purely categorical. The model has no inherent way of knowing that the token for `"cat"` and the token for `"dog"` are conceptually similar, while the token for `"democracy"` is entirely different.

By mapping each token to a vector (typically of size $d_{model} = 768$, $1024$, or larger), we allow the model to learn coordinates for each token. Over training, the model adjusts these coordinates so that semantically similar words are positioned closer together in this high-dimensional space.

* **Closeness in space**: The vectors for `"puppy"` and `"dog"` will point in similar directions.

* **Relationships as directions**: The vector offset between `"man"` and `"woman"` will be similar to the offset between `"king"` and `"queen"`.

## The Lookup Mechanics

Conceptually, the embedding layer is simply a large matrix of weights with dimensions:

$$
\text{Shape} = (\text{Vocabulary Size}, \text{Embedding Dimension})
$$

> Vocabulary Size => `vocab_size`
>
> Embedding Dimension => `d_model`

Each row in this matrix corresponds to a single token in our vocabulary. When an input token ID passes through the embedding layer, it acts as an **index** to extract (or "look up") its corresponding row from the matrix.

### A Visual Toy Example

Assume we have a tiny vocabulary of 4 tokens and an embedding dimension of 3. Our embedding matrix might look like this:

```
Token ID      Byte/Subword       Embedding Vector (Dimension = 3)
   0      ->    b'the'      ->   [  0.25, -1.43,  0.89 ]
   1      ->    b'cat'      ->   [  1.12,  0.05, -0.67 ]
   2      ->    b'sat'      ->   [ -0.84,  1.21,  0.10 ]
   3      ->    b'mat'      ->   [  1.05,  0.12, -0.55 ]
```

If the model receives an input sequence of token IDs `[1, 2, 3]` (representing `"cat sat mat"`), the embedding layer retrieves the corresponding rows sequentially:

```
Input Token IDs:
[ 1, 2, 3 ]

Output Embedding Matrix:
[
  [  1.12,  0.05, -0.67 ],  # Vector for 'cat' (ID 1)
  [ -0.84,  1.21,  0.10 ],  # Vector for 'sat' (ID 2)
  [  1.05,  0.12, -0.55 ]   # Vector for 'mat' (ID 3)
]
```

## Dimensional Transformation in the LLM Pipeline

During a forward pass in training or inference, the input batch of token IDs changes shape as it moves through the embedding layer.

* **Input Tensor Shape**: `(batch_size, sequence_length)` Each value in this 2D tensor is an integer representing a token ID.

* **Embedding Layer Transformation**: The embedding layer replaces every single integer with a 1D vector of size `d_model`.

  * Learnable paramater:

    * Embedding mappings - metrics of shape: `(vocab_size, d_model)`.

* **Output Tensor Shape**: `(batch_size, sequence_length, d_model)` This 3D tensor is now a continuous numerical representation ready to be processed by the Transformer's attention blocks.

```
Input: Token IDs of shape (B, T)
[
  [ 258, 111, 261 ],
  [  39, 109,  32 ]
]

==================== [ EMBEDDING LOOKUP ] ====================

Output: Dense vectors of shape (B, T, C)
[
  [ [v_258], [v_111], [v_261] ],  # Sequence 1 (each [v] is a C-dim vector)
  [ [v_39] , [v_109], [v_32]  ]   # Sequence 2
]
```

This transformation transitions our input data from a discrete symbolic space to a continuous vector space where deep learning algorithms excel.