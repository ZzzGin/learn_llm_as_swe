# 09. 采样、生成循环和性能开关

本章把 `LLM.generate`、`Sampler`、CUDA graph 和 benchmark 串起来看。

## generate 的循环

`LLMEngine.generate` 做的是典型离线批量生成：

1. 把 prompt 和 sampling params 加入 scheduler。
2. 不断调用 `step()`。
3. 收集 finished sequence 的 completion token ids。
4. 最后按 `seq_id` 排序，decode 成文本。

`step()` 内部很短：

```python
seqs, is_prefill = self.scheduler.schedule()
token_ids = self.model_runner.call("run", seqs, is_prefill)
self.scheduler.postprocess(seqs, token_ids, is_prefill)
```

这就是整个引擎的主循环。

## prefill 和 decode throughput

`generate` 会根据 `num_tokens` 更新进度条：

- prefill 返回正数，表示本轮处理了多少 prompt token；
- decode 返回负数，绝对值表示本轮生成了多少 token。

```python
num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
```

这也是为什么进度条同时显示 Prefill 和 Decode 两个速度。

## Sampler 的采样技巧

`Sampler.forward` 做温度采样：

```python
logits = logits.float().div_(temperatures.unsqueeze(dim=1))
probs = torch.softmax(logits, dim=-1)
sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
```

最后一行使用的是 Gumbel-max / exponential race 的等价采样技巧。对每个 token 概率除以指数分布噪声，再取 argmax，可以从 categorical distribution 中采样。

这个实现没有 top-k/top-p，因此每个 vocab token 都参与 softmax 和采样。

## torch.compile

项目里几个小算子用了 `@torch.compile`：

- `Sampler.forward`
- `RotaryEmbedding.forward`
- `RMSNorm.rms_forward`
- `RMSNorm.add_rms_forward`
- `SiluAndMul.forward`

这些算子形状相对简单，适合让 PyTorch 编译优化。

## CUDA graph

当 `enforce_eager=False` 时，`ModelRunner.capture_cudagraph()` 会为多个 batch size 预先 capture decode graph：

```python
self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
```

decode 时选择第一个大于等于当前 batch size 的 graph：

```python
graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
```

然后把本轮实际数据拷贝进预分配张量，调用 `graph.replay()`。

CUDA graph 的收益来自减少 Python 和 CUDA launch overhead，尤其适合 decode 这种每步小 batch、小计算但高频调用的场景。

## 为什么 prefill 不走 CUDA graph

prefill 的 token 数和序列长度变化很大，capture 多种形状会让实现复杂很多。Nano-VLLM 保持简化：

- prefill 走 eager；
- 大 batch decode 也走 eager；
- 小 batch decode 走 CUDA graph。

这是代码可读性和性能之间的务实平衡。

## benchmark 在测什么

`bench.py` 随机生成 256 个请求：

- 输入长度 100 到 1024；
- 输出长度 100 到 1024；
- `ignore_eos=True`，确保每个请求生成指定长度；
- 先跑一次短 generate 作为 warmup；
- 再统计总输出 token / 耗时。

README 中给出的测试结果显示，在指定硬件和模型上 Nano-VLLM 与 vLLM 同级，甚至略高。但这个 benchmark 是离线批量吞吐场景，不代表所有服务场景。

上一章：[Tensor Parallel 和权重加载](./08-tensor-parallel-loading.md)  
下一章：[按代码调试的推荐路线](./10-debugging-path.md)

