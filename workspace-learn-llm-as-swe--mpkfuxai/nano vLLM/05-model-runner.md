# 05. ModelRunner：从序列到 GPU 张量

`ModelRunner` 是整个项目里最关键的桥接层。它一边面对 engine 世界里的 `Sequence`，一边面对 GPU 模型 forward 需要的张量和上下文。

## 初始化做了哪些重活

构造 `ModelRunner(config, rank, event)` 时会：

1. 初始化 NCCL process group。
2. 设置当前 CUDA device。
3. 设置默认 dtype 和默认 device。
4. 构造 `Qwen3ForCausalLM`。
5. 从 safetensors 加载权重。
6. 创建 `Sampler`。
7. warmup 模型。
8. 根据剩余显存分配 KV cache。
9. 如果不是 eager 模式，capture CUDA graph。

所以 `LLM(...)` 初始化慢是正常的，因为模型和推理运行时都在这里完成初始化。

## 多 rank 调用模型

当 `tensor_parallel_size > 1` 时，rank 0 是主进程，其它 rank 是 worker 进程。rank 0 调用：

```python
self.model_runner.call("run", seqs, is_prefill)
```

`call()` 会先把方法名和参数 pickle 到共享内存，然后 set event 通知其它 rank：

```python
self.write_shm(method_name, *args)
```

worker 在 `loop()` 中等待 event，读出共享内存，再调用同名方法。这样所有 rank 会对同一个 batch 执行 forward。

## prepare_prefill

prefill 阶段要把多个序列的 prompt chunk 打平成一个 token 列表，并构造 FlashAttention varlen 需要的信息。

核心张量包括：

- `input_ids`：本轮要计算的 token。
- `positions`：每个 token 在序列里的位置。
- `cu_seqlens_q`：query 的 cumulative sequence lengths。
- `cu_seqlens_k`：key 的 cumulative sequence lengths。
- `slot_mapping`：每个 token 的 K/V 应写入哪个 cache slot。
- `block_tables`：prefix cache 场景下，历史 KV cache 的 block table。

对于没有 prefix cache 的普通 prefill，`cu_seqlens_q` 和 `cu_seqlens_k` 等长，attention 直接使用本轮的 k/v。

当命中了 prefix cache 时：

```python
if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
    block_tables = self.prepare_block_tables(seqs)
```

此时 key/value 长度大于 query 长度，attention 需要从已有 KV cache 加上本轮新 token 一起计算。

## slot_mapping 的含义

`slot_mapping` 把本轮每个 token 映射到一维 cache slot：

```python
slot = block_id * block_size + offset_in_block
```

Attention 层里的 Triton kernel 会根据这个 slot，把本轮产生的 K/V 写入正确的物理 block。

## prepare_decode

decode 阶段每个序列只输入最后一个 token：

```python
input_ids.append(seq.last_token)
positions.append(len(seq) - 1)
context_lens.append(len(seq))
slot_mapping.append(seq.block_table[-1] * block_size + seq.last_block_num_tokens - 1)
```

这里 `context_lens` 告诉 FlashAttention 当前序列的上下文长度，`block_tables` 告诉它历史 K/V 分布在哪些物理 block。

## set_context 是模型和 runner 的协议

`prepare_prefill` 和 `prepare_decode` 最后都会调用 `set_context(...)`。模型 forward 本身只接收：

```python
model(input_ids, positions)
```

attention 层需要的额外信息都从全局 `get_context()` 读取。这是 Nano-VLLM 让模型层保持简洁的关键技巧。

## run_model 的三条路径

`run_model` 根据场景选择执行方式：

1. prefill：直接 eager forward，因为输入 token 数变化大。
2. eager decode：当 `enforce_eager=True` 时直接 forward。
3. CUDA graph decode：小 batch decode 时 replay 预先 capture 的图。

代码条件是：

```python
if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
    return self.model.compute_logits(self.model(input_ids, positions))
else:
    graph.replay()
```

decode 每步形状相对稳定，所以适合 CUDA graph；prefill 形状更动态，保留 eager 更简单。

上一章：[KV cache、block table 和 prefix cache](./04-kv-cache-block-manager.md)  
下一章：[Qwen3 模型结构](./06-qwen3-model.md)

