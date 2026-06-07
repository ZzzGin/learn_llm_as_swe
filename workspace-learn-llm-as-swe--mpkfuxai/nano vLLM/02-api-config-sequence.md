# 02. 入口 API、配置和请求对象

本章看三个最靠近用户的对象：

- `LLM`：用户直接实例化的类。
- `Config`：引擎级配置。
- `Sequence`：单个请求在引擎内部的表示。

## LLM 是一层极薄的入口

`nanovllm/llm.py` 中只有：

```python
from nanovllm.engine.llm_engine import LLMEngine

class LLM(LLMEngine):
    pass
```

所以所有逻辑都在 `LLMEngine`。初始化时它做了几件事：

1. 从 `kwargs` 中筛选出 `Config` 支持的字段。
2. 创建 `Config(model, **config_kwargs)`。
3. 设置 `Sequence.block_size = config.kvcache_block_size`。
4. 根据 `tensor_parallel_size` 创建 worker 进程。
5. 创建 rank 0 的 `ModelRunner`。
6. 加载 tokenizer，读取 eos token。
7. 创建 `Scheduler`。

这意味着 `LLM` 的构造不只是轻量对象初始化，它会初始化分布式进程、加载模型、warmup、分配 KV cache。

## Config 决定资源上限

`Config` 在 `nanovllm/config.py` 中定义，关键字段包括：

| 字段 | 含义 |
| --- | --- |
| `model` | 本地模型目录，必须存在 |
| `max_num_batched_tokens` | 一个 prefill batch 最多处理多少 token |
| `max_num_seqs` | 同时调度的最大序列数 |
| `max_model_len` | 最大上下文长度，会被 HF config 的 `max_position_embeddings` 限制 |
| `gpu_memory_utilization` | 估算 KV cache 可用显存时使用 |
| `tensor_parallel_size` | 张量并行大小 |
| `enforce_eager` | 是否禁用 CUDA graph |
| `kvcache_block_size` | KV cache block 大小，要求是 256 的倍数 |
| `num_kvcache_blocks` | 实际可用 block 数，初始化后由 `ModelRunner` 计算 |

`__post_init__` 会读取 Hugging Face 的 `AutoConfig`，并把 `max_model_len` 截断到模型支持的最大位置。

## SamplingParams 很克制

`SamplingParams` 只包含：

- `temperature`
- `max_tokens`
- `ignore_eos`

它显式禁止 greedy sampling：

```python
assert self.temperature > 1e-10, "greedy sampling is not permitted"
```

所以当前实现只有温度采样，没有 top-k、top-p、presence penalty、frequency penalty 等策略。

## Sequence 是引擎内部的请求状态

`Sequence` 保存的信息可以分成四类。

请求身份和状态：

- `seq_id`
- `status`
- `is_prefill`

token 信息：

- `token_ids`
- `last_token`
- `num_tokens`
- `num_prompt_tokens`
- `num_completion_tokens`

调度进度：

- `num_cached_tokens`
- `num_scheduled_tokens`

KV cache 映射：

- `block_table`
- `num_blocks`
- `last_block_num_tokens`

## block_table 是理解项目的关键

`Sequence.block_table` 是逻辑 token block 到物理 KV cache block 的映射。例如一个序列长度为 600，`block_size=256`，它需要 3 个逻辑 block。`block_table` 可能是：

```python
[17, 42, 8]
```

意思是：

- 第 0 个逻辑 block 的 KV 存在物理 block 17；
- 第 1 个逻辑 block 的 KV 存在物理 block 42；
- 第 2 个逻辑 block 的 KV 存在物理 block 8。

模型执行时不会把 KV cache 复制成连续内存，而是通过 block table 告诉 FlashAttention 每个序列该去哪些 block 读历史 KV。

## 多进程传输中的 Sequence 简化

`Sequence.__getstate__` 和 `__setstate__` 用于 multiprocessing 传输。prefill 时需要传完整 token；decode 时只需要最后一个 token 和元数据：

```python
last_state = self.last_token if not self.is_prefill else self.token_ids
```

这是一个很实用的优化：decode 每步只追加一个 token，没有必要把整个 token 列表在进程间反复 pickle。

上一章：[项目地图和主流程](./01-project-map.md)  
下一章：[调度器：prefill、decode 和抢占](./03-scheduler.md)

