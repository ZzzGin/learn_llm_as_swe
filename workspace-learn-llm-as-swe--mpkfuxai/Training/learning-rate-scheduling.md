# Learning rate scheduling

During the training of deep learning models—especially large architectures like Transformers—the optimal learning rate is rarely constant. In the initial phases of training, the model's weights are randomly initialized, requiring a carefully managed learning rate to prevent divergence. As training progresses and the model begins to converge, a smaller learning rate is needed to make fine-tuned adjustments and settle into a stable, low-loss local minimum.

A **learning rate scheduler** is a function that dynamically adjusts the learning rate $\alpha_t$ at each training step $t$ based on predefined parameters.

One of the most widely adopted schedules in modern LLM training (such as LLaMA \[H. Touvron et al., 2023\]) is **Cosine Annealing with Warm-up**. This schedule adjusts the learning rate through three distinct phases: warm-up, cosine decay, and a constant post-annealing floor.

---

### Parameters of the Schedule

To define this schedule, five parameters are required:

* $t$: The current training iteration or step.

* $\alpha_{\max}$: The peak (maximum) learning rate.

* $\alpha_{\min}$: The target final (minimum) learning rate.

* $T_w$: The number of warm-up iterations.

* $T_c$: The final iteration of the cosine annealing decay.

---

### Mathematical Formulation

The learning rate $\alpha_t$ at any step $t$ is calculated as follows:

#### 1. Warm-up Phase ($t < T_w$)

During the first $T_w$ steps, the learning rate increases linearly from $0$ up to the peak value $\alpha_{\max}$. This prevents the gradients from exploding early in training when the model is highly unstable.

$$
\alpha_t = \frac{t}{T_w} \alpha_{\max}
$$

#### 2. Cosine Annealing Phase ($T_w \le t \le T_c$)

Once the warm-up is complete, the learning rate smoothly decays from $\alpha_{\max}$ down to $\alpha_{\min}$ following a half-period cosine curve. This smooth transition allows the model to explore the loss landscape effectively before slowing down.

$$
\alpha_t = \alpha_{\min} + \frac{1}{2} \left(1 + \cos\left(\frac{t - T_w}{T_c - T_w} \pi\right)\right) (\alpha_{\max} - \alpha_{\min})
$$

*Note: The term $\frac{t - T_w}{T_c - T_w}$ scales the progress through the decay phase to a range between $0$ and $1$, which, when multiplied by $\pi$, maps perfectly to the $[0, \pi]$ interval of the cosine function.*

#### 3. Post-annealing Phase ($t > T_c$)

After step $T_c$, the learning rate stops decaying and remains constant at the minimum value $\alpha_{\min}$ for any remaining training steps.

$$
\alpha_t = \alpha_{\min}
$$

---

### Summary of the Learning Rate Trajectory

1. **Iter $0 \to T_w$**: Linear ramp-up to $\alpha_{\max}$ (Warm-up).

2. **Iter $T_w \to T_c$**: Smooth, S-shaped decay down to $\alpha_{\min}$ (Cosine Annealing).

3. **Iter $> T_c$**: Constant flat-line at $\alpha_{\min}$ (Post-annealing).