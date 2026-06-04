# Inference Path

The inference path of a Large Language Model (LLM) describes the process of generating human-readable text from a given prompt. Because LLMs are **autoregressive**, text generation happens step-by-step: the model predicts one token at a time, appends it to the context, and feeds the updated sequence back into itself to predict the subsequent token.

Below is an explanation of each phase of the inference path, paired with code implementations.

---

### 1. Tokenization

Before text can be processed by the model, the raw input string (the prompt) must be converted into a sequence of integer IDs using a tokenizer.

```
# Convert prompt string to a list of token IDs
prompt_token_ids = tokenizer.encode(prompt)
```

Similarly, once the model finishes generating token IDs, they are converted back into a human-readable string:

```
# Convert token IDs back to a text string
generated_text = tokenizer.decode(generated_token_ids)
```

---

### 2. The Model Forward Pass

The tokenized input is passed through the `TransformerLM` model. Given an input sequence of length $t$, the model outputs a matrix of logits of size $(t, \text{vocab\_size})$.

To predict the next token (at position $t+1$), we only care about the logits corresponding to the final token in the current context (the $t$-th position).

```
# Extract the active context (constrained by the model's maximum context length)
context = generated[-context_length:]
input_ids = torch.tensor([context], dtype=torch.long, device=device)

# Forward pass: shape (1, context_len, vocab_size) -> extract the last token's logits
logits = model(input_ids)[0, -1]
```

---

### 3. Logit Post-processing & Sampling Tricks

Raw logits can yield low-quality or repetitive text if sampled directly. We apply two primary techniques to refine the logits before sampling.

#### Temperature Scaling

Temperature scaling adjusts the entropy of the probability distribution. A lower temperature $\tau \to 0$ makes the distribution sharper (approaching greedy search), while a higher temperature increases diversity.

$$
\text{softmax}(v, \tau)_i = \frac{\exp(v_i/\tau)}{\sum_{j=1}^{\text{vocab\_size}} \exp(v_j/\tau)}
$$

```
def temperature_scaled_softmax(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return soft_max(logits / temperature, dim=-1)
```

#### Nucleus (Top-$p$) Sampling

To prevent the model from selecting highly improbable "long-tail" tokens, we apply Top-$p$ sampling. This restricts our candidate pool to the smallest set of tokens $V(p)$ whose cumulative probability meets or exceeds a threshold $p$.

$$
P(x_{t+1} = i \mid q) = \begin{cases} \frac{q_i}{\sum_{j \in V(p)} q_j} & \text{if } i \in V(p) \\ 0 & \text{otherwise} \end{cases}
$$

```
def top_p_probs(probs: torch.Tensor, p: float) -> torch.Tensor:
    if not 0 < p <= 1:
        raise ValueError("top-p must be in (0, 1]")
    if p == 1:
        return probs

    # Sort probabilities in descending order
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # Identify which sorted indices to keep to reach cumulative probability p
    keep_sorted = cumulative_probs - sorted_probs < p
    filtered_sorted_probs = sorted_probs * keep_sorted

    # Reconstruct original probability tensor shape, zeroing out excluded tokens
    filtered_probs = torch.zeros_like(probs)
    filtered_probs.scatter_(dim=-1, index=sorted_indices, src=filtered_sorted_probs)
    
    # Re-normalize the remaining non-zero probabilities
    return filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
```

---

### 4. Token Sampling

Once the probabilities are scaled and filtered, the next token ID is sampled. If temperature is set to `0`, we skip sampling and perform greedy decoding (`argmax`). Otherwise, we perform categorical sampling using `torch.multinomial`.

```
def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
) -> int:
    # Greedy decoding
    if temperature == 0:
        return int(torch.argmax(logits).item())
    if temperature < 0:
        raise ValueError("temperature must be non-negative")

    # Apply temperature and top-p filtering
    probs = temperature_scaled_softmax(logits, temperature)
    if top_p is not None:
        probs = top_p_probs(probs, top_p)
        
    # Sample from the processed categorical distribution
    return int(torch.multinomial(probs, num_samples=1, generator=generator).item())
```

---

### 5. The Autoregressive Loop

Putting it all together, the generator runs a loop that continuously generates next-token predictions, appends them to the running sequence, and checks for termination conditions (such as hitting the `max_new_tokens` limit or generating an end-of-sequence `<|endoftext|>` token).

```
@torch.no_grad()
def generate_token_ids(
    model: TransformerLM,
    prompt_token_ids: list[int],
    max_new_tokens: int,
    eos_token_id: int | None = None,
    temperature: float = 1.0,
    top_p: float | None = None,
    device: str | torch.device | None = None,
    generator: torch.Generator | None = None,
) -> list[int]:
    model.eval()  # Put model in evaluation mode
    device = device or next(model.parameters()).device
    generated = list(prompt_token_ids)
    context_length = model.context_length

    for _ in range(max_new_tokens):
        # Truncate context to model's context capacity
        context = generated[-context_length:]
        input_ids = torch.tensor([context], dtype=torch.long, device=device)
        
        # Step 1: Forward pass to get final-token logits
        logits = model(input_ids)[0, -1]
        
        # Step 2: Post-process logits and sample next token ID
        next_token_id = sample_next_token(
            logits=logits,
            temperature=temperature,
            top_p=top_p,
            generator=generator,
        )
        
        # Step 3: Append generated token to sequence
        generated.append(next_token_id)
        
        # Step 4: Early exit check if EOS token is generated
        if eos_token_id is not None and next_token_id == eos_token_id:
            break

    return generated
```