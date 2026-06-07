# 04. KV cache、block table 和 prefix cache

KV cache 是大语言模型推理的核心性能结构。没有 KV cache，decode 第 N 个 token 时就要重复计算前 N-1 个 token 的 key/value。Nano-VLLM 用固定大小 block 管理 KV cache，这就是 vLLM PagedAttention 思想的简化版。

## 物理 block 和逻辑 block

`BlockManager` 初始化时创建固定数量的 `Block`：

```python
self.blocks = [Block(i) for i in range(num_blocks)]
self.free_block_ids = deque(range(num_blocks))
self.used_block_ids = set()
```

每个 `Sequence` 按 token 顺序切成逻辑 block。逻辑 block 通过 `seq.block_table` 指向物理 block。

这种映射的好处是：序列增长、释放、prefix 共享时，不需要移动大块 KV 张量，只需要改 block table 和引用计数。

## Block 保存元数据，不保存真正 KV

`Block` 里有：

- `block_id`
- `ref_count`
- `hash`
- `token_ids`

真正的 KV cache 张量在 `ModelRunner.kv_cache` 中。`BlockManager` 管的是物理 block 的所有权、引用计数和 prefix cache 索引。

## KV cache 张量布局

`ModelRunner.allocate_kv_cache()` 创建：

```python
torch.empty(
    2,
    num_hidden_layers,
    num_kvcache_blocks,
    block_size,
    num_kv_heads,
    head_dim,
)
```

第一维 `2` 分别是 K 和 V。每层 attention 模块拿到自己那一层的：

```python
module.k_cache = self.kv_cache[0, layer_id]
module.v_cache = self.kv_cache[1, layer_id]
```

所以每个 attention 层只看到形如：

```python
[num_blocks, block_size, num_kv_heads, head_dim]
```

的 K/V cache。

## num_kvcache_blocks 如何估算

`ModelRunner` 根据显存估算可容纳多少 block：

```python
block_bytes = (
    2
    * num_hidden_layers
    * block_size
    * num_kv_heads
    * head_dim
    * dtype.itemsize
)
num_kvcache_blocks = available_bytes // block_bytes
```

其中 `available_bytes` 来自：

- GPU 总显存；
- 当前已用显存；
- warmup 时观察到的峰值模型内存；
- `gpu_memory_utilization`。

这个估算避免 KV cache 抢占模型权重和中间激活需要的显存。

## prefix cache 的 hash 链

`BlockManager.compute_hash(token_ids, prefix)` 会把当前 block 的 token 和前一个 block 的 hash 串起来：

```python
if prefix != -1:
    h.update(prefix.to_bytes(8, "little"))
h.update(np.array(token_ids).tobytes())
```

这意味着第 i 个 block 的 hash 不只取决于本 block token，也取决于前面所有 block。两个序列只有从开头到当前 block 完全相同，hash 才会一致。

## can_allocate 同时检查 prefix 命中和可用 block

`can_allocate(seq)` 做两件事：

1. 从头扫描完整 block，看看有多少 block 能命中 prefix cache。
2. 计算还需要多少新 block，如果空闲 block 不够则返回 `-1`。

它只扫描到 `seq.num_blocks - 1`，也就是不缓存最后一个可能未满的 block。原因是 prefix cache 只应该复用完整 block；未满 block 后续还会追加 token，直接共享会破坏隔离。

## allocate 会增加引用计数

命中的 block 如果仍在使用中，增加 `ref_count`；如果已经空闲但 hash 还保留在索引里，就从 free 队列取回并标记为 used。

没有命中的部分则调用 `_allocate_block()` 分配新 block。

## deallocate 不一定清除 hash

释放 block 时，`ref_count` 降到 0 后进入 free 队列，但 block 上的 `hash` 和 `token_ids` 不会立刻清除。这样即使某个 prefix 当前没有请求使用，它仍然可能被未来请求命中。

真正重新分配这个 free block 时，如果它的 hash 还指向自己，才从 `hash_to_block_id` 删除旧映射。

## hash_blocks 在 block 写满后登记 cache

`hash_blocks(seq)` 根据本次调度范围，找到刚刚写满的 block，把它们的 token 和 hash 记录下来：

```python
block.update(h, token_ids)
self.hash_to_block_id[h] = block.block_id
```

这一步发生在 `Scheduler.postprocess()`，也就是模型已经把本轮 K/V 写入 cache 之后。

上一章：[调度器：prefill、decode 和抢占](./03-scheduler.md)  
下一章：[ModelRunner：从序列到 GPU 张量](./05-model-runner.md)

