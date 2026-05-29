# AdamW Optimizer

### Overview of AdamW Optimizer

Modern language models are typically trained with more sophisticated optimizers, instead of standard Stochastic Gradient Descent (SGD). Most modern optimizers are derivatives of the Adam optimizer, with AdamW being the standard choice in recent state-of-the-art architectures (such as GPT-3 and LLaMA).

AdamW improves generalization by decoupling the weight decay from the gradient update. In standard Adam with L2 regularization, the regularization term is added directly to the gradient, which gets mixed with the moving averages of the first and second moments. AdamW, on the other hand, applies the weight decay update directly to the parameters before updating them with the moment estimates, leading to better regularization and more stable convergence.

---

### Mathematical Formulation

Below is the formalized step-by-step algorithm for AdamW.

#### Initialization:

* Let $\theta_0$ be the initial parameters.

* Initialize first moment vector $m_0 \leftarrow 0$ (same shape as $\theta$).

* Initialize second moment vector $v_0 \leftarrow 0$ (same shape as $\theta$).

* Let $\alpha$ be the learning rate, $\lambda$ be the weight decay coefficient, $(\beta_1, \beta_2)$ be the exponential decay rates for the moment estimates, and $\epsilon$ be a small scalar for numerical stability.

#### At each time step $t = 1, \dots, T$:

1. **Compute Gradient**:

   $$
g_t \leftarrow \nabla_\theta \ell(\theta_{t-1}; B_t)
$$

2. **Compute Adjusted Step Size (Bias Correction)**:

   $$
\alpha_t \leftarrow \alpha \frac{\sqrt{1 - \beta_2^t}}{1 - \beta_1^t}
$$

3. **Decoupled Weight Decay**:

   $$
\theta_{t-1/2} \leftarrow \theta_{t-1} - \alpha \lambda \theta_{t-1}
$$

4. **Update First Moment Estimate (with momentum)**:

   $$
m_t \leftarrow \beta_1 m_{t-1} + (1 - \beta_1) g_t
$$

5. **Update Second Moment Estimate (RMSprop-style)**:

   $$
v_t \leftarrow \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
$$

6. **Apply Parameter Update**:

   $$
\theta_t \leftarrow \theta_{t-1/2} - \alpha_t \frac{m_t}{\sqrt{v_t} + \epsilon}
$$

---

### PyTorch Implementation

Here is the PyTorch implementation of the `AdamWOptimizer` which aligns precisely with this mathematical definition.

```
import torch
from collections.abc import Callable


class AdamWOptimizer(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        # check inputs are valid.

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                if grad.is_sparse:
                    raise RuntimeError(
                          "AdamW does not support sparse gradients")

                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param)
                    state["exp_avg_sq"] = torch.zeros_like(param)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                step = state["step"]

                with torch.no_grad():
                    # 1. Apply decoupled weight decay: 
                    # θ_half ← θ * (1 - lr * weight_decay)
                    param.mul_(1 - lr * weight_decay)

                    # 2. Update first moment: 
                    # m_t ← β1 * m_{t-1} + (1 - β1) * g_t
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    
                    # 3. Update second moment: 
                    # v_t ← β2 * v_{t-1} + (1 - β2) * g_t^2
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                    # 4. Compute bias corrections
                    bias_correction1 = 1 - beta1**step
                    bias_correction2 = 1 - beta2**step
                    
                    # 5. Compute step size: 
                    # α_t = α * sqrt(1 - β2^t) / (1 - β1^t)
                    step_size = lr * bias_correction2**0.5 / bias_correction1
                    
                    # 6. Apply moment-adjusted update: 
                    # θ_t ← θ_half - α_t * m_t / (sqrt(v_t) + ε)
                    param.addcdiv_(exp_avg, exp_avg_sq.sqrt().add_(eps), 
                            value=-step_size)

        return loss
```

---

### Concrete Numerical Example

To see how these formulas play out, let's trace through **two complete update steps** for a single parameter.

#### Setup Hyperparameters:

* **Learning Rate ($\alpha$)**: $0.1$

* **Weight Decay ($\lambda$)**: $0.01$

* **Betas ($\beta_1, \beta_2$)**: $(0.9, 0.99)$

* **Epsilon ($\epsilon$)**: $10^{-8}$

#### Initial State:

* **Parameter ($\theta_0$)**: $1.0$

* **First Moment ($m_0$)**: $0.0$

* **Second Moment ($v_0$)**: $0.0$

---

#### Step 1 ($t = 1$)

Suppose we sample a batch and compute the gradient: **$g_1 = 0.5$**.

1. **Apply Weight Decay**:

   $$
\theta_{0.5} = \theta_0 \cdot (1 - \alpha \lambda) = 1.0 \cdot (1 - 0.1 \times 0.01) = 1.0 \cdot 0.999 = 0.999
$$

2. **Update First Moment**:

   $$
m_1 = \beta_1 m_0 + (1 - \beta_1) g_1 = 0.9 \cdot 0.0 + 0.1 \cdot 0.5 = 0.05
$$

3. **Update Second Moment**:

   $$
v_1 = \beta_2 v_0 + (1 - \beta_2) g_1^2 = 0.99 \cdot 0.0 + 0.01 \cdot (0.5)^2 = 0.01 \cdot 0.25 = 0.0025
$$

4. **Calculate Bias Corrections**:

   $$
\text{bias\_correction}_1 = 1 - \beta_1^1 = 1 - 0.9 = 0.1
$$

   $$
\text{bias\_correction}_2 = 1 - \beta_2^1 = 1 - 0.99 = 0.01
$$

5. **Compute Adjusted Step Size**:

   $$
\alpha_1 = \alpha \frac{\sqrt{\text{bias\_correction}_2}}{\text{bias\_correction}_1} = 0.1 \cdot \frac{\sqrt{0.01}}{0.1} = 0.1 \cdot \frac{0.1}{0.1} = 0.1
$$

6. **Apply Parameter Update**:

   $$
\theta_1 = \theta_{0.5} - \alpha_1 \frac{m_1}{\sqrt{v_1} + \epsilon} = 0.999 - 0.1 \cdot \frac{0.05}{\sqrt{0.0025} + 10^{-8}}
$$

   Since $\sqrt{0.0025} = 0.05$ and $\epsilon$ is negligibly small:

   $$
\theta_1 \approx 0.999 - 0.1 \cdot \frac{0.05}{0.05} = 0.999 - 0.1 = 0.899
$$

---

#### Step 2 ($t = 2$)

Suppose we sample a new batch and compute the gradient: **$g_2 = 0.2$**.

1. **Apply Weight Decay**:

   $$
\theta_{1.5} = \theta_1 \cdot (1 - \alpha \lambda) = 0.899 \cdot 0.999 = 0.898101
$$

2. **Update First Moment**:

   $$
m_2 = \beta_1 m_1 + (1 - \beta_1) g_2 = 0.9 \cdot 0.05 + 0.1 \cdot 0.2 = 0.045 + 0.02 = 0.065
$$

3. **Update Second Moment**:

   $$
v_2 = \beta_2 v_1 + (1 - \beta_2) g_2^2 = 0.99 \cdot 0.0025 + 0.01 \cdot (0.2)^2 = 0.002475 + 0.01 \cdot 0.04 = 0.002875
$$

4. **Calculate Bias Corrections**:

   $$
\text{bias\_correction}_1 = 1 - \beta_1^2 = 1 - 0.81 = 0.19
$$

   $$
\text{bias\_correction}_2 = 1 - \beta_2^2 = 1 - 0.9801 = 0.0199
$$

5. **Compute Adjusted Step Size**:

   $$
\alpha_2 = \alpha \frac{\sqrt{\text{bias\_correction}_2}}{\text{bias\_correction}_1} = 0.1 \cdot \frac{\sqrt{0.0199}}{0.19} \approx 0.1 \cdot \frac{0.141067}{0.19} \approx 0.074246
$$

6. **Apply Parameter Update**:

   $$
\sqrt{v_2} \approx \sqrt{0.002875} \approx 0.053619
$$

   $$
\frac{m_2}{\sqrt{v_2} + \epsilon} \approx \frac{0.065}{0.053619} \approx 1.212257
$$

   $$
\theta_2 = \theta_{1.5} - \alpha_2 \frac{m_2}{\sqrt{v_2} + \epsilon} \approx 0.898101 - 0.074246 \cdot 1.212257 \approx 0.898101 - 0.090005 = 0.808096
$$