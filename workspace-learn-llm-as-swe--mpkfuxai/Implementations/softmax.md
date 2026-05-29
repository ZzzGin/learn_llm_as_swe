# Softmax

## Mathematical Formulas

1. **Standard Softmax**:

   $$
\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}}
$$

2. **Numerically Stable Softmax**:

   $$
\text{Softmax}(x_i) = \frac{e^{x_i - C}}{\sum_{j} e^{x_j - C}} \quad \text{where} \quad C = \max_j(x_j)
$$

---

## Code Implementation

```
import torch


def soft_max(x: torch.Tensor, dim: int) -> torch.Tensor:
    shifted = x - torch.max(x, dim=dim, keepdim=True).values
    exp_shifted = torch.exp(shifted)
    return exp_shifted / torch.sum(exp_shifted, dim=dim, keepdim=True)
```

---

## Step-by-Step Execution Analysis

### 1. Numerical Stability (Shifting by Max)

```
shifted = x - torch.max(x, dim=dim, keepdim=True).values
```

* **Why it's done**: Exponential functions grow extremely fast. For 32-bit floating-point numbers, values above approximately $88.7$ cause overflow, resulting in `inf`. Subtracting the maximum value $C = \max_j(x_j)$ shifts the range of inputs so that the maximum value becomes $0$. Since $e^0 = 1$ and $e^y \in (0, 1]$ for all $y \le 0$, the computed exponentials are guaranteed never to overflow. Mathematically, this shift does not change the resulting probabilities because the scaling factor cancels out:

  $$
\frac{e^{x_i - C}}{\sum_{j} e^{x_j - C}} = \frac{e^{x_i} \cdot e^{-C}}{\sum_{j} e^{x_j} \cdot e^{-C}} = \frac{e^{x_i} \cdot e^{-C}}{e^{-C} \sum_{j} e^{x_j}} = \frac{e^{x_i}}{\sum_{j} e^{x_j}}
$$

* **How it works**:

  * `torch.max(x, dim=dim, keepdim=True).values` finds the maximum value along the specified dimension `dim`.

  * `keepdim=True` preserves the singleton dimension so that the output shape remains broadcastable (e.g., transforming a shape of `(batch, dim)` to `(batch, 1)`). This allows PyTorch to easily subtract the maximum value from each element along that dimension.

### 2. Exponentiation

```
exp_shifted = torch.exp(shifted)
```

* **The Math**: Corresponds to calculating the numerator $e^{x_i - C}$ for each element.

* **How it works**: Computes the natural exponential element-wise for all shifted inputs.

### 3. Summing and Normalization

```
return exp_shifted / torch.sum(exp_shifted, dim=dim, keepdim=True)
```

* **The Math**: Corresponds to dividing the shifted exponentials by their sum: $\sum_{j} e^{x_j - C}$.

* **How it works**:

  * `torch.sum(exp_shifted, dim=dim, keepdim=True)` calculates the sum of the exponentials along the specified dimension. Keeping the dimension ensures the sum tensor has a size of `1` along `dim`, allowing it to be broadcasted during the division.

  * The division normalizes the exponentials, ensuring the output elements sum to $1$ and represent a valid probability distribution.

---

## Example Execution

Let's trace a concrete example with a 1D tensor:

### 1. Setup Input

```
x = torch.tensor([1.0, 2.0, 5.0])
dim = 0
```

### 2. Shifting by Max

* **The Max**: $C = 5.0$

* **The Shift**:

  $$
x - C = [1.0 - 5.0, 2.0 - 5.0, 5.0 - 5.0] = [-4.0, -3.0, 0.0]
$$

* **Code Output**: `shifted` is `tensor([-4.0, -3.0,  0.0])`

### 3. Exponentiation

* **The Exponentials**:

  $$
e^{-4.0} \approx 0.0183, \quad e^{-3.0} \approx 0.0498, \quad e^{0.0} = 1.0
$$

* **Code Output**: `exp_shifted` is `tensor([0.0183, 0.0498, 1.0000])`

### 4. Normalization

* **The Sum**:

  $$
\sum e^{x_j - C} = 0.0183 + 0.0498 + 1.0 = 1.0681
$$

* **The Division**:

  $$
\left[ \frac{0.0183}{1.0681}, \frac{0.0498}{1.0681}, \frac{1.0}{1.0681} \right] \approx [0.0171, 0.0466, 0.9362]
$$

* **Code Output**: `tensor([0.0171, 0.0466, 0.9362])`