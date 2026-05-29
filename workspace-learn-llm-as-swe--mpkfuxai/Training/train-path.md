# Training path

### 1. Initialization and Setup

Before entering the training loop, the script initializes the parameters, loads the dataset, and instantiates the model and optimizer.

#### Memory-Mapped Data Loading

To avoid loading the entire dataset into RAM, the training path uses memory-mapping (`numpy.memmap` or `numpy.load` with `mmap_mode="r"`) to load sequence tokens on-demand:

```
def load_token_array(path: Path, dtype: str) -> npt.NDArray:
    if path.suffix == ".npy":
        return np.load(path, mmap_mode="r")
    return np.memmap(path, dtype=np.dtype(dtype), mode="r")
```

#### Model and Optimizer Initialization

The script initializes the model weights and creates the `AdamWOptimizer`, configuring it with key hyperparameters like learning rate, betas, epsilon, and weight decay:

```
model = TransformerLM(
    **model_config,
    device=device,
)
optimizer = AdamWOptimizer(
    model.parameters(),
    lr=args.learning_rate,
    betas=(args.beta1, args.beta2),
    eps=args.eps,
    weight_decay=args.weight_decay,
)
```

#### Resuming Checkpoints

If the model was previously trained and saved, the script resumes the training progress (including the iteration counter, optimizer state, and weights) by loading the saved state:

```
start_iter = 0
if args.resume_from is not None:
    start_iter = load_checkpoint(args.resume_from, model, optimizer)
```

---

### 2. The Core Training Loop

At the heart of the script is the training loop that runs for `max_iters` iterations. Every iteration consists of the following steps:

#### Step A: Learning Rate Scheduling

At the start of step `it`, the script calculates the target learning rate using the **Cosine Annealing with Warm-up** schedule and applies it directly to the optimizer's parameter groups:

```
lr = get_lr_cosine_schedule(
    it=it,
    max_learning_rate=args.learning_rate,
    min_learning_rate=args.min_learning_rate,
    warmup_iters=args.warmup_iters,
    cosine_cycle_iters=args.cosine_cycle_iters,
)
for group in optimizer.param_groups:
    group["lr"] = lr
```

#### Step B: Data Mini-Batching

A batch of inputs `x` and targets `y` is sampled from the tokenized dataset. The target tokens `y` are shifted relative to `x` to construct a next-token prediction task:

```
x, y = get_batch(train_data, args.batch_size, args.context_length, args.device)
```

The helper function `get_batch` handles this selection process. It draws random starting offsets within the dataset, slices the sequences to build parallel input-target pairs, and loads the resulting PyTorch tensors directly onto the active device:

```
def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    starting_indices = np.random.randint(0, len(dataset) - context_length, size=batch_size)
    inputs = np.stack([dataset[start : start + context_length] for start in starting_indices])
    targets = np.stack([dataset[start + 1 : start + context_length + 1] for start in starting_indices])
    return (
        torch.tensor(inputs, dtype=torch.long, device=device),
        torch.tensor(targets, dtype=torch.long, device=device),
    )
```

#### Step C: Forward Pass and Loss Computation

The script zeros the gradients, passes the input through the model, and computes the stable **Cross-Entropy Loss**.

```
optimizer.zero_grad()
loss = loss_on_batch(model, x, y)
```

The sequence dimension is collapsed inside the helper function `loss_on_batch` to allow for clean vector execution in the loss calculation:

```
def loss_on_batch(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    logits = model(x)
    return cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
```

#### Step D: Backward Pass and Gradient Clipping

The gradients of the loss with respect to all parameters are calculated using backpropagation (`loss.backward()`). To prevent destabilizing parameter updates from massive gradients, global **gradient clipping** is applied to cap the collective $\ell_2$-norm:

```
loss.backward()
if args.max_grad_norm is not None:
    gradient_clipping(model.parameters(), args.max_grad_norm)
```

#### Step E: Optimization Step

With the clipped, stable gradients computed, the `AdamWOptimizer` updates the model's parameters:

```
optimizer.step()
```

---

### 3. Monitoring and Evaluation

To track training health and evaluate model generalization, the script periodically prints metrics, saves validation metrics, and pushes logs to **Weights & Biases (WandB)**.

#### Loss Evaluation on Validation Set

At specified intervals, the script disables gradient calculations and computes validation loss across several iterations using `estimate_loss`:

```
@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module,
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
    eval_iters: int,
) -> float:
    model.eval()
    losses = []
    for _ in range(eval_iters):
        x, y = get_batch(dataset, batch_size, context_length, device)
        losses.append(loss_on_batch(model, x, y).item())
    model.train()
    return sum(losses) / len(losses)
```

#### Logging and Metrics Reporting

```
if valid_data is not None and step % args.eval_interval == 0:
    val_loss = estimate_loss(
        model=model,
        dataset=valid_data,
        batch_size=args.batch_size,
        context_length=args.context_length,
        device=args.device,
        eval_iters=args.eval_iters,
    )
    metrics["valid/loss"] = val_loss
    print(f"iter {step}: valid_loss={val_loss:.4f}")

if wandb_run is not None:
    wandb_run.log(metrics, step=step)
```

---

### 4. Checkpointing

To safeguard long training sessions, the script saves complete progress state (weights, optimizer properties, and iteration count) to disk both periodically and when training concludes successfully:

```
if args.checkpoint_path is not None and step % args.checkpoint_interval == 0:
    os.makedirs(args.checkpoint_path.parent, exist_ok=True)
    save_checkpoint(model, optimizer, step, args.checkpoint_path, model_config=model_config)
```