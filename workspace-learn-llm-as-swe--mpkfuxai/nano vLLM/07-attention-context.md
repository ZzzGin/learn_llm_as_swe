# 07. Attention、上下文对象和 FlashAttention

Nano-VLLM 的 attention 实现分成两层：

- `Qwen3Attention`：负责 q/k/v 投影、RoPE、输出投影。
- `Attention`：负责 KV cache 写入和调用 FlashAttention。

`Attention` 所需的大部分运行时信息都来自 `nanovllm/utils/context.py` 中的全局 `Context`。

## Context 里有什么

`Context` 保存一次 forward 的临时信息：

| 字段 | 用途 |
| --- | --- |
| `is_prefill` | 当前是 prefill 还是 decode |
| `cu_seqlens_q` | varlen query 累计长度 |
| `cu_seqlens_k` | varlen key 累计长度 |
| `max_seqlen_q` | batch 内最大 query 长度 |
| `max_seqlen_k` | batch 内最大 key 长度 |
| `slot_mapping` | 本轮 token 写入 KV cache 的 slot |
| `context_lens` | decode 时每个序列上下文长度 |
| `block_tables` | 每个序列的物理 block table |

`ModelRunner.prepare_prefill` 或 `prepare_decode` 设置 context，模型 forward 结束后 `reset_context()` 清空。

## 写入 KV cache

在 `Attention.forward` 里，如果当前层已经绑定了 KV cache，就会调用：

```python
store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
```

`store_kvcache` 是一个 Triton kernel。它按 token 逐个读取 `slot_mapping`，把当前 token 的 k/v 写入：

```python
cache_offsets = slot * D + arange(0, D)
```

这里 `D = num_heads * head_dim`，所以一维 slot 实际对应物理 block 内某个 token 位置。

## prefill attention

prefill 使用：

```python
flash_attn_varlen_func(...)
```

普通 prefill 时，q/k/v 都来自本轮计算出的张量。

prefix cache 命中时，`context.block_tables is not None`，代码会改用整个 KV cache：

```python
if context.block_tables is not None:
    k, v = k_cache, v_cache
```

然后通过 FlashAttention 的 `block_table` 参数把逻辑序列映射到物理 KV block。

## decode attention

decode 使用：

```python
flash_attn_with_kvcache(
    q.unsqueeze(1),
    k_cache,
    v_cache,
    cache_seqlens=context.context_lens,
    block_table=context.block_tables,
    softmax_scale=self.scale,
    causal=True,
)
```

每个序列只有一个 query token，但 key/value 是历史上下文。FlashAttention 根据 `context_lens` 和 `block_tables` 去 paged KV cache 中读取历史 KV。

## RoPE 的实现

`RotaryEmbedding` 在初始化时预计算 `cos_sin_cache`：

```python
cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
```

forward 时按 `positions` 索引对应位置的 cos/sin，再对 query 和 key 应用旋转。`get_rope` 使用 `lru_cache(1)`，同一组参数下复用 RoPE 模块。

## 为什么使用全局 context

如果不使用全局 context，每一层 attention forward 都需要显式传入：

- cu_seqlens；
- max seqlen；
- slot mapping；
- block tables；
- context lens；
- 当前阶段标记。

这会污染模型层接口。Nano-VLLM 选择用一个进程内全局 context 作为执行期协议，让模型 forward 仍然接近普通 Hugging Face 模型：

```python
model(input_ids, positions)
```

代价是：这个实现天然假设同一进程同一时刻只跑一个 forward context。对这个离线推理引擎来说，这是合理约束。

上一章：[Qwen3 模型结构](./06-qwen3-model.md)  
下一章：[Tensor Parallel 和权重加载](./08-tensor-parallel-loading.md)

