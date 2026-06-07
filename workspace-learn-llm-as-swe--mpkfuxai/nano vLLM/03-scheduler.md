# 03. 调度器：prefill、decode 和抢占

调度器位于 `nanovllm/engine/scheduler.py`。它维护两个队列：

- `waiting`：还没完成 prompt prefill，或被抢占后需要重新进入 prefill 的序列。
- `running`：已经完成 prompt prefill，正在逐 token decode 的序列。

`schedule()` 每次返回：

```python
tuple[list[Sequence], bool]
```

第二个值 `is_prefill` 表示这批是 prefill 还是 decode。

## prefill 优先

调度器每次先尝试从 `waiting` 里取序列做 prefill。核心限制是：

- batch 内序列数不能超过 `max_num_seqs`；
- batch 内 token 数不能超过 `max_num_batched_tokens`；
- KV cache block 必须能分配成功。

如果一个序列还没有 `block_table`，说明第一次调度它，调度器会调用：

```python
num_cached_blocks = self.block_manager.can_allocate(seq)
self.block_manager.allocate(seq, num_cached_blocks)
```

这里同时处理新 block 分配和 prefix cache 命中。

## chunked prefill 的简化策略

当剩余额度不足以放下完整 prompt 时，Nano-VLLM 允许对第一个序列做 chunked prefill：

```python
if remaining < num_tokens and scheduled_seqs:
    break
seq.num_scheduled_tokens = min(num_tokens, remaining)
```

含义是：

- 如果 batch 里还没有任何序列，可以把当前序列切一段进去；
- 如果 batch 里已经有别的序列，就不再切新的长 prompt。

这是一个简洁的折中：避免单个超长 prompt 永远堵住队列，同时避免 batch 内混入太多复杂切片。

## prefill 完成后进入 running

当一个序列的 cached token 加 scheduled token 等于总 token 数时，说明 prompt 已经全部 prefill 完成：

```python
seq.status = SequenceStatus.RUNNING
self.waiting.popleft()
self.running.append(seq)
```

注意，调度器随后仍会把这个序列放入当前 prefill batch。模型会在这个 prefill forward 的最后位置输出下一个 token。

## 没有 prefill 时才 decode

如果本轮没有任何 prefill 序列，调度器才进入 decode 分支。decode 每个 running 序列只调度一个 token：

```python
seq.num_scheduled_tokens = 1
seq.is_prefill = False
self.block_manager.may_append(seq)
scheduled_seqs.append(seq)
```

这对应 autoregressive decoding：每个序列用最后一个 token 和历史 KV cache 预测下一个 token。

## decode 前要确保能追加 KV cache

decode 会让序列长度增加 1。如果这个 token 正好落入新的 block，必须先有空闲 block。

```python
while not self.block_manager.can_append(seq):
    if self.running:
        self.preempt(self.running.pop())
    else:
        self.preempt(seq)
        break
```

当没有足够空闲 block 时，调度器会从 running 队列尾部抢占序列，释放它占用的 KV cache。

## 抢占的代价

`preempt(seq)` 会：

1. 把状态改回 `WAITING`。
2. 设置 `is_prefill=True`。
3. 释放 block table 对应的 KV cache。
4. 把序列放回 `waiting` 队首。

这意味着被抢占的序列之后需要重新 prefill。Nano-VLLM 没有做 swap 到 CPU 或更复杂的 recompute 策略，而是选择最直接的释放和重算。

## postprocess 的职责

模型返回 token 后，`postprocess` 会：

1. 调用 `block_manager.hash_blocks(seq)`，把刚写满的 block 登记进 prefix cache。
2. 更新 `num_cached_tokens`。
3. 如果 prefill 还没完成，跳过采样结果处理。
4. 否则追加新 token。
5. 如果遇到 eos 或达到 `max_tokens`，释放 KV cache 并从 running 移除。

这一点很重要：prefill 被切块时，中间 chunk 虽然也执行了模型，但不会把本轮采样结果当成生成 token。只有 prompt 全部 prefill 完成后，最后一个 prompt token 的 logits 才用于生成第一个 completion token。

上一章：[入口 API、配置和请求对象](./02-api-config-sequence.md)  
下一章：[KV cache、block table 和 prefix cache](./04-kv-cache-block-manager.md)

