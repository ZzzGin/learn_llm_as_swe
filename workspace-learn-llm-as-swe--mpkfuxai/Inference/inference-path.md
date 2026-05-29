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

---

### 6. Optimizing Inference: Key-Value (KV) Caching

In the naive autoregressive loop implemented above, generating $N$ tokens requires feeding a growing sequence of length $t, t+1, t+2, \dots, t+N$ back into the model at every step. This means the model repeatedly projects the entire history of tokens into queries, keys, and values ($Q, K, V$) and recalculates self-attention for past tokens, resulting in an $O(N^2)$ computational complexity.

Since LLMs are **causal** (future tokens cannot affect past tokens), the key ($K$) and value ($V$) representations of past tokens do not change during generation. We can store these representations in a **KV Cache** after they are computed for the first time. In subsequent steps, we only pass the single newly generated token into the model, project its individual $q_t, k_t, v_t$ vectors, append $k_t$ and $v_t$ to the cache, and calculate attention against the complete history.

#### The Prefill vs. Decode Phases

Using a KV Cache splits inference into two distinct operational phases:

1. **The Prefill Phase**: The model processes the entire prompt of length $L$ in parallel. It computes the initial key and value vectors for all $L$ tokens, stores them in the cache, and outputs the distribution for the first generated token.

2. **The Decode Phase**: For each subsequent token, the input sequence length is always exactly $1$. The model computes query, key, and value vectors only for this new token, appends the key and value to the KV Cache, and generates the next token using the consolidated history.

---

#### KV Cache Implementation

Below is a class to manage the KV Cache across multiple layers of a Transformer:

```
import torch

class KVCache:
    def __init__(self) -> None:
        # Stores cached keys and values as lists of tensors (one per layer)
        self.k_cache: list[torch.Tensor] = []
        self.v_cache: list[torch.Tensor] = []

    def update(
        self, 
        key_states: torch.Tensor, 
        value_states: torch.Tensor, 
        layer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Appends new key and value states to the cached states for a given layer.
        Shapes of states: (batch_size, num_heads, sequence_length, head_dim)
        """
        # If cache for this layer does not exist yet, initialize it
        if layer_idx >= len(self.k_cache):
            self.k_cache.append(key_states)
            self.v_cache.append(value_states)
        else:
            # Concatenate along the sequence dimension (dim=-2)
            self.k_cache[layer_idx] = torch.cat([self.k_cache[layer_idx], key_states], dim=-2)
            self.v_cache[layer_idx] = torch.cat([self.v_cache[layer_idx], value_states], dim=-2)
            
        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def clear(self) -> None:
        self.k_cache.clear()
        self.v_cache.clear()
```

---

#### Integrating KV Cache into Self-Attention

In the attention layer, we fetch, update, and use the cached keys and values instead of computing them exclusively from the current input.

This requires us to pass explicit `token_positions` to RoPE \[rope.md\] because the physical sequence length of the input tensor in the Decode phase is `1`, but its logical position in the sequence is $L$ (representing its actual index in the generated sequence).

```
# Inside CausalSelfAttentionWithRoPE.forward(...)
def forward(
    self,
    x: torch.Tensor,
    token_positions: torch.Tensor,
    kv_cache: KVCache | None = None,
    layer_idx: int | None = None,
) -> torch.Tensor:
    # 1. Project inputs to Q, K, V
    q = rearrange(self.w_q(x), "... seq (head d_head) -> ... head seq d_head", head=self.num_heads)
    k = rearrange(self.w_k(x), "... seq (head d_head) -> ... head seq d_head", head=self.num_heads)
    v = rearrange(self.w_v(x), "... seq (head d_head) -> ... head seq d_head", head=self.num_heads)

    # 2. Apply RoPE using true logical token positions
    q = self.rope(q, token_positions)
    k = self.rope(k, token_positions)

    # 3. Update and retrieve from KV Cache
    if kv_cache is not None:
        if layer_idx is None:
            raise ValueError("layer_idx must be provided when using a KV Cache")
        k, v = kv_cache.update(k, v, layer_idx)

    # 4. Compute attention (note: q has sequence length 1, k/v have full history sequence length)
    attention_output = scaled_dot_product_attention(q, k, v, mask=None)
    
    # 5. Reconstruct original dimension and project out
    attention_output = rearrange(attention_output, "... head seq d_head -> ... seq (head d_head)")
    return self.w_o(attention_output)
```

---

#### The Optimized Autoregressive Loop

By combining the prefill phase, the decode phase, and explicit logical position-tracking, we can implement an efficient text generator that avoids redundant computations:

```
@torch.no_grad()
def generate_token_ids_efficient(
    model: TransformerLM,
    prompt_token_ids: list[int],
    max_new_tokens: int,
    eos_token_id: int | None = None,
    temperature: float = 1.0,
    top_p: float | None = None,
    device: str | torch.device | None = None,
    generator: torch.Generator | None = None,
) -> list[int]:
    model.eval()
    device = device or next(model.parameters()).device
    generated = list(prompt_token_ids)
    
    # Initialize the KV Cache object
    kv_cache = KVCache()
    
    # ==========================================
    # PHASE 1: Prefill (Process entire prompt)
    # ==========================================
    input_ids = torch.tensor([prompt_token_ids], dtype=torch.long, device=device)
    
    # Positional indices for the prompt: [0, 1, ..., L-1]
    positions = torch.arange(len(prompt_token_ids), dtype=torch.long, device=device).unsqueeze(0)
    
    # Pass the empty kv_cache to be populated by the prefill pass
    logits = model(input_ids, positions=positions, kv_cache=kv_cache)
    
    # Sample the first new token using the logits of the final prompt token
    next_token_id = sample_next_token(
        logits=logits[0, -1],
        temperature=temperature,
        top_p=top_p,
        generator=generator,
    )
    generated.append(next_token_id)

    # ==========================================
    # PHASE 2: Decode (Process one token at a time)
    # ==========================================
    for _ in range(1, max_new_tokens):
        if eos_token_id is not None and next_token_id == eos_token_id:
            break
            
        # We pass ONLY the single most recently generated token as input
        input_ids = torch.tensor([[next_token_id]], dtype=torch.long, device=device)
        
        # The logical position of this new token is its index in the sequence
        current_position = len(generated) - 1
        positions = torch.tensor([[current_position]], dtype=torch.long, device=device)
        
        # Forward pass is O(1) in sequence length relative to KV projections
        logits = model(input_ids, positions=positions, kv_cache=kv_cache)
        
        next_token_id = sample_next_token(
            logits=logits[0, -1],
            temperature=temperature,
            top_p=top_p,
            generator=generator,
        )
        generated.append(next_token_id)
        
    return generated
```