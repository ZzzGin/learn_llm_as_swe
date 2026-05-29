# Cross-entropy Loss

## Mathematical Formulas

1. **Cross-Entropy (Negative Log-Likelihood) Loss**:

   Given a training set $D$ consisting of sequences of length $m + 1$, the standard cross-entropy loss function is defined as:

   $$
\ell(\theta; D) = \frac{1}{|D| \cdot m} \sum_{x \in D} \sum_{i=1}^{m} -\log p_\theta(x_{i+1} \mid x_{1:i})
$$

   Here, a single forward pass through the Transformer yields $p_\theta(x_{i+1} \mid x_{1:i})$ for all positions $i = 1, \dots, m$ simultaneously.

   Where:

   * **$\theta$** represents the trainable parameters (weights and biases) of the Transformer model.

   * **$D$** is the training dataset (or the current mini-batch of sequences).

   * **$|D|$** represents the cardinality of the dataset $D$, which is the total number of sequences it contains.

   * **$p_\theta$** denotes the probability distribution predicted by the model parameterized by $\theta$.

2. **Probability Distribution (Softmax over Logits)**:

   The Transformer computes logits $o_i \in \mathbb{R}^{\text{vocab\_size}}$ for each position $i$. The probability of predicting the correct next token $x_{i+1}$ is:

   $$
p(x_{i+1} \mid x_{1:i}) = \text{Softmax}(o_i)[x_{i+1}] = \frac{\exp(o_i[x_{i+1}])}{\sum_{a=1}^{\text{vocab\_size}} \exp(o_i[a])}
$$

   Where $o_i[k]$ refers to the value at index $k$ of the vector $o_i$. This loss formulation corresponds to the cross-entropy between the Dirac delta distribution over $x_{i+1}$ (representing the ground truth target) and the predicted $\text{Softmax}(o_i)$ distribution.

3. **Numerically Stable Formulation**:

   To avoid overflow and underflow issues, we subtract the maximum logit $C_i = \max_a o_i[a]$ from the logits of each sequence element:

   $$
-\log p(x_{i+1} \mid x_{1:i}) = \log\left(\sum_{a=1}^{\text{vocab\_size}} \exp(o_i[a] - C_i)\right) - (o_i[x_{i+1}] - C_i)
$$

4. **Perplexity (PPL)**:

   Perplexity is a metric used to evaluate language models, representing the exponentiated cross-entropy loss:

   $$
\text{PPL}(D) = \exp(\ell(\theta; D)) = \exp\left( \frac{1}{|D| \cdot m} \sum_{x \in D} \sum_{i=1}^{m} -\log p_\theta(x_{i+1} \mid x_{1:i}) \right)
$$

   It can also be expressed as the geometric mean of the inverse probabilities of the target tokens:

   $$
\text{PPL}(D) = \left( \prod_{x \in D} \prod_{i=1}^{m} \frac{1}{p_\theta(x_{i+1} \mid x_{1:i})} \right)^{\frac{1}{|D| \cdot m}}
$$

   A lower perplexity indicates that the model is more confident and accurate in its predictions. Intuitively, a perplexity of $K$ suggests the model is as "perplexed" as if it were choosing uniformly at random from $K$ options at each step.

---

## Code Implementation

```
import torch
from einops import reduce


def cross_entropy(o: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # shift o by -max for each batch
    shifted = o - reduce(o, "batch vocab -> batch 1", "max")
    # calculate the exp for the last dim, then sum the exps for each batch then log
    logsumexp = torch.log(reduce(torch.exp(shifted), "batch vocab -> batch", "sum"))
    # target is a vector on dimension: tensor([2, 1])
    # target[:, None] makes it: tensor([[2], [1]])
    target_logits = torch.gather(shifted, dim=-1, index=target[:, None])[:, 0]
    return reduce(logsumexp - target_logits, "batch ->", "mean")


def perplexity(loss: torch.Tensor) -> torch.Tensor:
    return torch.exp(loss)
```

---

## Step-by-Step Execution Analysis

### 1. Numerical Stability (Shifting by Max)

```
shifted = o - reduce(o, "batch vocab -> batch 1", "max")
```

* **Why it's done**: Exponentials can easily overflow when inputs are large. By subtracting the maximum logit $C = \max_j o_j$ for each sequence/batch element, the largest value becomes $0$. This ensures $e^0 = 1$ is the largest exponent computed, preventing overflow during the subsequent exponentiation step.

* **How it works**: `reduce(..., "batch vocab -> batch 1", "max")` calculates the maximum value across the vocabulary dimension for each batch element. Keeping the dimension of size `1` (via `batch 1`) allows PyTorch to automatically broadcast the subtraction across the entire vocabulary for each batch.

### 2. Log-Sum-Exp Calculation

```
logsumexp = torch.log(reduce(torch.exp(shifted), "batch vocab -> batch", "sum"))
```

* **The Math**: Corresponds to calculating the logarithm of the denominator sum: $\log\left(\sum_{a} \exp(o_i[a] - C_i)\right)$.

* **How it works**:

  * `torch.exp(shifted)` computes the exponential of each shifted logit.

  * `reduce(..., "batch vocab -> batch", "sum")` sums these exponentials along the vocabulary dimension.

  * `torch.log(...)` computes the natural logarithm of the resulting sum.

### 3. Gathering Target Logits

```
target_logits = torch.gather(shifted, dim=-1, index=target[:, None])[:, 0]
```

* **The Math**: Corresponds to extracting the shifted logit value corresponding to the correct target index $x_{i+1}$: $o_i[x_{i+1}] - C_i$.

* **How it works**:

  * `target[:, None]` reshapes the target tensor of shape `(batch,)` to `(batch, 1)` to match the dimensionality required by `torch.gather`.

  * `torch.gather(shifted, dim=-1, index=...)` extracts the shifted logit values at the indices specified by `target` along the vocabulary dimension.

  * `[:, 0]` indexes back into the singleton dimension to return a 1D tensor of shape `(batch,)`.

### 4. Loss Reduction (Mean)

```
return reduce(logsumexp - target_logits, "batch ->", "mean")
```

* **The Math**: Subtracts the target logit from the log-sum-exp value per batch element, and computes the average across the entire batch:

  $$
\text{Loss} = \frac{1}{B} \sum_{b=1}^{B} \left( \log\sum_{a} e^{o_b[a] - C_b} - (o_b[\text{target}_b] - C_b) \right)
$$

* **How it works**: Performs element-wise subtraction and uses `reduce` to compute the average over all batch elements.

### 5. Perplexity

```
return torch.exp(loss)
```

* **The Math**: Perplexity is defined as $\exp(\text{Loss})$. It represents the exponentiated cross-entropy loss and can be interpreted as the average number of word choices the model is choosing from at each position.

---

## Example Execution

Let's trace a concrete example with a batch of size 2 and a vocabulary of size 3:

### 1. Setup Input

```
o = torch.tensor([[1.0, 2.0, 5.0], [2.0, 4.0, 3.0]])
target = torch.tensor([1, 2])
```

* **Logits `o`**:

  $$
o = \begin{bmatrix} 1.0 & 2.0 & 5.0 \\ 2.0 & 4.0 & 3.0 \end{bmatrix}
$$

* **Targets `target`**:

  * Batch 0: Correct token index is `1` (logit value `2.0`).

  * Batch 1: Correct token index is `2` (logit value `3.0`).

### 2. Shifting by Max

* **The Max per Batch**: $C = \begin{bmatrix} 5.0 \\ 4.0 \end{bmatrix}$

* **The Shift**:

  $$
o - C = \begin{bmatrix} 1.0-5.0 & 2.0-5.0 & 5.0-5.0 \\ 2.0-4.0 & 4.0-4.0 & 3.0-4.0 \end{bmatrix} = \begin{bmatrix} -4.0 & -3.0 & 0.0 \\ -2.0 & 0.0 & -1.0 \end{bmatrix}
$$

* **Code Output**: `shifted` is `tensor([[-4.0, -3.0,  0.0], [-2.0,  0.0, -1.0]])`

### 3. Log-Sum-Exp Calculation

* **The Exponentials**:

  $$
e^{\text{shifted}} \approx \begin{bmatrix} e^{-4.0} & e^{-3.0} & e^{0.0} \\ e^{-2.0} & e^{0.0} & e^{-1.0} \end{bmatrix} \approx \begin{bmatrix} 0.0183 & 0.0498 & 1.0000 \\ 0.1353 & 1.0000 & 0.3679 \end{bmatrix}
$$

* **The Sums**:

  $$
\sum e^{\text{shifted}} = \begin{bmatrix} 0.0183 + 0.0498 + 1.0000 \\ 0.1353 + 1.0000 + 0.3679 \end{bmatrix} = \begin{bmatrix} 1.0681 \\ 1.5032 \end{bmatrix}
$$

* **The Logarithm**:

  $$
\log(\text{Sums}) = \begin{bmatrix} \log(1.0681) \\ \log(1.5032) \end{bmatrix} \approx \begin{bmatrix} 0.0659 \\ 0.4076 \end{bmatrix}
$$

* **Code Output**: `logsumexp` is `tensor([0.0659, 0.4076])`

### 4. Gathering Target Logits

* **Unsqueezed Target**: `target[:, None]` is `tensor([[1], [2]])`

* **Gathering**:

  * Batch 0: Index `1` from `[-4.0, -3.0, 0.0]` $\rightarrow -3.0$.

  * Batch 1: Index `2` from `[-2.0, 0.0, -1.0]` $\rightarrow -1.0$.

* **Code Output**: `target_logits` is `tensor([-3.0, -1.0])`

### 5. Subtracting and Normalizing

* **Loss per Batch Element**:

  $$
\text{logsumexp} - \text{target\_logits} = \begin{bmatrix} 0.0659 - (-3.0) \\ 0.4076 - (-1.0) \end{bmatrix} = \begin{bmatrix} 3.0659 \\ 1.4076 \end{bmatrix}
$$

* **The Mean Loss**:

  $$
\text{Loss} = \frac{3.0659 + 1.4076}{2} \approx 2.2368
$$

* **Code Output**: `tensor(2.2368)`

### 6. Perplexity

* **The Perplexity**:

  $$
e^{2.2368} \approx 9.3633
$$

* **Code Output**: `tensor(9.3633)`

---

## Q&A

### Q: Why does the mathematical formula include the sequence length ($m$), while the code implementation only operates on `batch` and `vocab` dimensions?

**A:** In practice, the code implementation is mathematically equivalent due to **dimension flattening (reshaping)**. Here is why the sequence dimension is omitted in the code:

#### 1. Dimension Flattening

In a Transformer pipeline, the model output (logits) typically has the shape `(batch_size, sequence_length, vocab_size)`, and the targets have the shape `(batch_size, sequence_length)`.

Before passing these tensors to the loss function, they are flattened by collapsing the batch and sequence dimensions together:

* **Logits**: `(B, M, V) -> (B * M, V)`

* **Targets**: `(B, M) -> (B * M,)`

In our code implementation, the `batch` dimension in `reduce(o, "batch vocab -> batch 1", "max")` actually represents this flattened dimension of size $B \cdot m$ (the total number of tokens in the mini-batch). The average (`mean`) computed at the end of the code is over this entire $B \cdot m$ dimension, which perfectly matches the double summation and normalization $\frac{1}{|D| \cdot m} \sum_{x \in D} \sum_{i=1}^{m}$ in the mathematical formula.

#### 2. Handling Padding and Masking

Real-world training sequences often have variable lengths and must be padded with a special token (e.g., `<pad>`). By flattening the sequence dimension, we can easily mask out these padding tokens before calculating the loss:

```
# Flattening the dimensions
logits = logits.view(-1, vocab_size)  # Shape: (B * M, V)
targets = targets.view(-1)            # Shape: (B * M,)

# Filtering out padding tokens (using -100 as the standard ignore index)
active_mask = targets != -100
logits = logits[active_mask]
targets = targets[active_mask]

# Now, 'batch' simply represents all valid, non-padded tokens in the batch
loss = cross_entropy(logits, targets)
```

Designing the loss function to operate on a 2D `(batch, vocab)` tensor keeps the implementation clean, highly reusable, and fully compatible with dynamic sequence masking.

### Q: In a helper function like `loss_on_batch`, is the purpose of `logits.reshape(-1, logits.shape[-1])` to flatten the sequence dimension?

```
def loss_on_batch(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    logits = model(x)
    return cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
```

**A:** Yes, exactly. This reshape operation collapses (flattens) the sequence dimension into the batch dimension so that the 3D tensor matches the 2D input expected by the `cross_entropy` function.

Here is a detailed breakdown of how this transformation works:

#### 1. Input Dimensions

* **Model Logits**: Typically has the shape `(B, M, V)`, where $B$ is the batch size, $M$ is the sequence length, and $V$ is the vocabulary size (`logits.shape[-1]`).

* **Targets ($y$)**: Has the shape `(B, M)`.

#### 2. Dimension Collapse

By using `-1` in `reshape`, PyTorch automatically infers the size of that dimension based on the remaining elements:

* **Logits**: `logits.reshape(-1, V)` reshapes the tensor from `(B, M, V)` to `(B * M, V)`.

* **Targets**: `y.reshape(-1)` flattens the 2D targets tensor from `(B, M)` to a 1D tensor of shape `(B * M,)`.

#### 3. Execution inside `cross_entropy`

The `cross_entropy` function receives the flattened inputs:

* It treats each of the $B \cdot M$ tokens as an independent "batch" element.

* The final `mean` reduction is computed over $B \cdot M$ elements, which mathematically yields:

  $$
\frac{1}{B \cdot M} \sum_{b=1}^{B} \sum_{i=1}^{M} \mathcal{L}_{b, i}
$$

  This aligns perfectly with the standard cross-entropy loss formula defined earlier.