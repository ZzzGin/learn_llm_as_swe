# 08. Tensor Parallel 和权重加载

Nano-VLLM 的 tensor parallel 主要体现在三处：

1. `LLMEngine` 启动多个 `ModelRunner` 进程。
2. 模型层按 rank 切分权重。
3. forward 中用 `all_reduce` 或 `gather` 合并结果。

## 进程启动

`LLMEngine.__init__` 中，rank 0 留在主进程，其它 rank 用 multiprocessing spawn：

```python
for i in range(1, config.tensor_parallel_size):
    process = ctx.Process(target=ModelRunner, args=(config, i, event))
    process.start()
```

每个 `ModelRunner` 都会初始化：

```python
dist.init_process_group(
    "nccl",
    "tcp://localhost:2333",
    world_size=self.world_size,
    rank=rank,
)
```

这说明当前实现假设单机多 GPU，并使用固定的 localhost 端口。

## ColumnParallelLinear

Column parallel 切输出维度。假设完整权重形状是：

```python
[output_size, input_size]
```

每个 rank 只持有一段 output rows：

```python
loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
```

forward 时每个 rank 都能独立计算自己那部分输出，不需要马上通信。

典型用途：

- QKV projection；
- MLP gate/up projection。

## RowParallelLinear

Row parallel 切输入维度。每个 rank 计算部分输入对应的输出，然后用 all-reduce 求和：

```python
y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
if self.tp_size > 1:
    dist.all_reduce(y)
```

典型用途：

- attention output projection；
- MLP down projection。

## QKVParallelLinear

QKV projection 合并了 q/k/v 三个参数，并且每个 rank 只加载自己负责的 head：

```python
if loaded_shard_id == "q":
    shard_size = self.num_heads * self.head_size
elif loaded_shard_id == "k":
    shard_size = self.num_kv_heads * self.head_size
else:
    shard_size = self.num_kv_heads * self.head_size
```

这既减少模块数量，也让权重加载时能够直接把 HF 的 q/k/v 权重放入合并后的参数。

## MergedColumnParallelLinear

MLP 的 gate/up projection 合并在一个参数里。loader 传入 `loaded_shard_id` 为 `0` 或 `1`，表示写入 gate 部分还是 up 部分。

这样 forward 只做一次大矩阵乘法，然后 `SiluAndMul` 切成两半做激活。

## VocabParallelEmbedding 和 ParallelLMHead

词表维度也被切分：

```python
self.num_embeddings_per_partition = num_embeddings // tp_size
self.vocab_start_idx = ...
self.vocab_end_idx = ...
```

embedding forward 时，不属于当前 rank 词表范围的 token 会被 mask 掉，然后各 rank `all_reduce` 合并 embedding。

LM head forward 时，各 rank 只计算部分 vocab logits，rank 0 用 `dist.gather` 收集后拼接：

```python
dist.gather(logits, all_logits, 0)
logits = torch.cat(all_logits, -1)
```

只有 rank 0 需要完整 logits，因为采样只在 rank 0 执行。

## loader 如何配合并行层

`load_model` 遍历 safetensors 文件中的权重名。普通参数直接调用：

```python
weight_loader(param, loaded_weight)
```

合并参数则根据 `packed_modules_mapping` 找到目标参数和 shard id：

```python
param_name = weight_name.replace(k, v)
weight_loader(param, f.get_tensor(weight_name), shard_id)
```

每个参数对象在初始化时挂了自己的 `weight_loader`，所以 loader 不需要知道具体层类型。切分逻辑由层自己负责。

上一章：[Attention、上下文对象和 FlashAttention](./07-attention-context.md)  
下一章：[采样、生成循环和性能开关](./09-sampling-generation-performance.md)

