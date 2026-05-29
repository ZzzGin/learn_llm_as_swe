# Gradient clipping

### Overview of Gradient Clipping

During training, we can sometimes hit training examples that yield large gradients, which can destabilize training. To mitigate this, one technique often employed in practice is gradient clipping. The idea is to enforce a limit on the norm of the gradient after each backward pass before taking an optimizer step.

Instead of clipping gradients element-wise (which alters the gradient direction), global gradient clipping scales the entire gradient vector proportionally. This preserves the direction of the gradient step while capping its magnitude, preventing the optimizer from making excessively large steps in parameter space that could cause divergence.

---

### Mathematical Formulation

Let $g$ represent the combined gradient vector of all model parameters:

$$
g = [g^{(1)}, g^{(2)}, \dots, g^{(P)}]
$$

Where $g^{(i)}$ is the gradient of the $i$-th parameter group, and $P$ is the total number of parameters.

To clarify terminology: in modern deep learning frameworks like PyTorch, a model's parameters are stored as a collection of $P$ individual tensors (such as weight matrices and bias vectors) rather than a single flat array.

In this mathematical formulation, $g^{(i)}$ represents the gradient tensor corresponding to the $i$-th parameter tensor, and $P$ is the total number of these parameter tensors. Note that this is distinct from PyTorch's concept of an optimizer "parameter group" (which represents a subset of parameter tensors grouped together to share hyperparameters like learning rate).

#### 1. Compute the Global $\ell_2$-Norm:

$$
\|g\|_2 = \sqrt{\sum_{i=1}^{P} \|g^{(i)}\|_2^2}
$$

The $\ell_2$-norm (often referred to as the Euclidean norm or Euclidean length) measures the overall magnitude of a vector. For a single gradient tensor $g^{(i)}$ flattened into a 1D vector of $N_i$ elements $[g^{(i)}_1, g^{(i)}_2, \dots, g^{(i)}_{N_i}]$, its individual $\ell_2$-norm is computed as the square root of the sum of its squared elements:

$$
\|g^{(i)}\|_2 = \sqrt{\sum_{j=1}^{N_i} \left(g^{(i)}_j\right)^2}
$$

The global $\ell_2$-norm $\|g\|_2$ represents the $\ell_2$-norm of all model gradients concatenated into a single massive vector. Mathematically, taking the square root of the sum of the squared individual norms yields the exact same result as flattening every gradient tensor in the entire model and calculating a single overall Euclidean norm.

#### 2. Apply Clipping Threshold:

Let $M > 0$ be the maximum norm threshold. We define a clipping coefficient $\text{clip\_coef}$ using a small scalar $\epsilon$ (typically $10^{-6}$) for numerical stability:

$$
\text{clip\_coef} = \frac{M}{\|g\|_2 + \epsilon}
$$

* If $\text{clip\_coef} \ge 1$ (meaning $\|g\|_2 \le M$), we leave the gradients unchanged.

* If $\text{clip\_coef} < 1$, we scale the gradients down:

$$
g \leftarrow g \cdot \text{clip\_coef}
$$

The resulting norm of the updated gradient vector will be just under $M$.

---

### PyTorch Implementation

Below is the PyTorch implementation of `gradient_clipping` which aligns precisely with this mathematical definition.

```
from collections.abc import Iterable

import torch


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    parameters_with_grad = [parameter for parameter in parameters if parameter.grad is not None]
    if len(parameters_with_grad) == 0:
        return

    total_norm = torch.linalg.vector_norm(
        torch.stack([torch.linalg.vector_norm(parameter.grad.detach(), ord=2) for parameter in parameters_with_grad]),
        ord=2,
    )

    clip_coef = max_l2_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for parameter in parameters_with_grad:
            parameter.grad.detach().mul_(clip_coef)
```

---

### Concrete Numerical Example

To see how this plays out, let's trace through a complete clipping step with two parameter gradients.

#### Setup Hyperparameters:

* **Max $\ell_2$-norm ($M$)**: $10.0$

* **Epsilon ($\epsilon$)**: $10^{-6}$

#### Initial Gradients:

* **Parameter 1 Gradient ($g^{(1)}$)**: $[3.0, 4.0]^T$

* **Parameter 2 Gradient ($g^{(2)}$)**: $[0.0, 12.0]^T$

---

#### Step 1: Compute Individual $\ell_2$-Norms

$$
\|g^{(1)}\|_2 = \sqrt{3.0^2 + 4.0^2} = \sqrt{9.0 + 16.0} = 5.0
$$

$$
\|g^{(2)}\|_2 = \sqrt{0.0^2 + 12.0^2} = \sqrt{144.0} = 12.0
$$

#### Step 2: Compute Total Global Norm

$$
\|g\|_2 = \sqrt{\sum_{i=1}^{2} \|g^{(i)}\|_2^2} = \sqrt{5.0^2 + 12.0^2} = \sqrt{25.0 + 144.0} = \sqrt{169.0} = 13.0
$$

Since the total norm ($13.0$) is greater than our maximum norm threshold $M = 10.0$, clipping is triggered.

#### Step 3: Compute Clipping Coefficient

$$
\text{clip\_coef} = \frac{M}{\|g\|_2 + \epsilon} = \frac{10.0}{13.0 + 10^{-6}} \approx 0.7692307
$$

#### Step 4: Scale the Gradients

Since $\text{clip\_coef} \approx 0.7692307 < 1$, we scale both gradients:

$$
g^{(1)}_{\text{clipped}} = g^{(1)} \cdot \text{clip\_coef} \approx [3.0, 4.0]^T \cdot 0.7692307 \approx [2.307692, 3.076923]^T
$$

$$
g^{(2)}_{\text{clipped}} = g^{(2)} \cdot \text{clip\_coef} \approx [0.0, 12.0]^T \cdot 0.7692307 \approx [0.0, 9.230769]^T
$$

The resulting global norm of the scaled gradients is now:

$$
\|g_{\text{clipped}}\|_2 \approx \sqrt{(2.307692^2 + 3.076923^2) + (0.0^2 + 9.230769^2)} \approx \sqrt{15.0 + 85.0} = 10.0
$$