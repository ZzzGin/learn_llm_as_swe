# 2 - Batched Process

To train a transformer model efficiently, we cannot feed it raw, variable-length text sequences one by one. Instead, we must group the tokenized dataset into uniform, parallel batches. This process prepares the token IDs for parallel computation on hardware like GPUs.

---

## The Language Modeling Dataset

In autoregressive language modeling, the task is to predict the next token in a sequence given all the preceding tokens. To set up this supervised learning task, we slice our continuous tokenized text corpus into chunks of size `sequence_length+1`, where `sequence_length` is the maximum sequence length (often called **block size** or **context length**).

For each chunk, we create:

* **Input $X$**: The first `sequence_length` tokens.

* **Target $Y$**: The tokens shifted by one position to the right.

### Example of Offset Targets

Suppose we have a single chunk of tokens of length `sequence_length+1 = 5`: `[258, 111, 261, 108, 100]`

We split this into:

* **Input $X$**: `[258, 111, 261, 108]`

* **Target $Y$**: `[111, 261, 108, 100]`

This structure allows the model to learn multiple predictions simultaneously from a single sequence:

* Given `[258]`, predict `111`.

* Given `[258, 111]`, predict `261`.

* Given `[258, 111, 261]`, predict `108`.

* Given `[258, 111, 261, 108]`, predict `100`.

## Batching Mechanics as Metrics Dimensions

To utilize modern hardware accelerators, we process multiple sequences in parallel. We define two key dimensions:

* **Batch Size (`batch_size`)**: The number of independent sequences processed simultaneously in a single forward/backward pass.

* **Sequence Length (`sequence_length`)**: The number of tokens in each sequence (the temporal dimension).

When batched, our inputs and targets are represented as 2D tensors of shape `(batch_size, sequence_length)`.

### Visual Representation of a Batch

For a batch size `batch_size=4` and sequence length `sequence_length=8`, the input tensor $X$ is a $4 \times 8$ grid of token IDs:

```
Batch Index (B)
     ↓
     0:  [ 258,  111,  261,  108,  100,   44,   32,   73 ]  (Sequence 1)
     1:  [  39,  109,   32,   49,   48,   32,  121,  101 ]  (Sequence 2)
     2:  [  97,  114,  115,   32,  111,  108,  100,   33 ]  (Sequence 3)
     3:  [ 201,  118,  109,   32,  259,   97,  108,  108 ]  (Sequence 4)
         |<-------------------  T = 8  ------------------>|
```

The corresponding target tensor $Y$ has the exact same shape, containing the next token for every position in $X$.

### 