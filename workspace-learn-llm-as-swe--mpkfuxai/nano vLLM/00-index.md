# Nano-vLLM 实现导读

这一组笔记按 ohmy.md 的 Markdown 工作流组织：每章一个 Markdown 文件，文件之间使用相对链接，适合按顺序阅读，也方便后续继续扩展。

Nano-vLLM 的目标不是覆盖 vLLM 的全部工程复杂度，而是用约一千多行 Python 代码复现离线推理引擎最核心的路径：

- 请求进入引擎后被封装成 `Sequence`。
- `Scheduler` 把请求分成 prefill 和 decode 两类批次。
- `BlockManager` 以固定大小的 block 管理 KV cache，并支持 prefix cache。
- `ModelRunner` 把调度结果转换为模型 forward 需要的张量。
- `Qwen3ForCausalLM`、并行线性层、FlashAttention、Triton kernel 共同完成推理。
- `Sampler` 对 logits 做温度采样，生成下一个 token。

## 阅读顺序

1. [项目地图和主流程](./01-project-map.md)
2. [入口 API、配置和请求对象](./02-api-config-sequence.md)
3. [调度器：prefill、decode 和抢占](./03-scheduler.md)
4. [KV cache、block table 和 prefix cache](./04-kv-cache-block-manager.md)
5. [ModelRunner：从序列到 GPU 张量](./05-model-runner.md)
6. [Qwen3 模型结构](./06-qwen3-model.md)
7. [Attention、上下文对象和 FlashAttention](./07-attention-context.md)
8. [Tensor Parallel 和权重加载](./08-tensor-parallel-loading.md)
9. [采样、生成循环和性能开关](./09-sampling-generation-performance.md)
10. [按代码调试的推荐路线](./10-debugging-path.md)

## 代码入口速查

| 主题 | 文件 |
| --- | --- |
| 用户入口 | [`../../nanovllm/llm.py`](https://github.com/GeeeekExplorer/nano-vllm/blob/main/nanovllm/llm.py), [`../../nanovllm/engine/llm_engine.py`](https://github.com/GeeeekExplorer/nano-vllm/blob/main/nanovllm/engine/llm_engine.py) |
| 配置 | [`../../nanovllm/config.py`](https://github.com/GeeeekExplorer/nano-vllm/blob/main/nanovllm/config.py) |
| 请求状态 | [`../../nanovllm/engine/sequence.py`](https://github.com/GeeeekExplorer/nano-vllm/blob/main/nanovllm/engine/sequence.py) |
| 调度 | [`../../nanovllm/engine/scheduler.py`](https://github.com/GeeeekExplorer/nano-vllm/blob/main/nanovllm/engine/scheduler.py) |
| KV cache block 管理 | [`../../nanovllm/engine/block_manager.py`](https://github.com/GeeeekExplorer/nano-vllm/blob/main/nanovllm/engine/block_manager.py) |
| GPU 执行 | [`../../nanovllm/engine/model_runner.py`](https://github.com/GeeeekExplorer/nano-vllm/blob/main/nanovllm/engine/model_runner.py) |
| 模型 | [`../../nanovllm/models/qwen3.py`](https://github.com/GeeeekExplorer/nano-vllm/blob/main/nanovllm/models/qwen3.py) |
| Attention | [`../../nanovllm/layers/attention.py`](https://github.com/GeeeekExplorer/nano-vllm/blob/main/nanovllm/layers/attention.py) |
| 并行线性层 | [`../../nanovllm/layers/linear.py`](https://github.com/GeeeekExplorer/nano-vllm/blob/main/nanovllm/layers/linear.py) |
| 权重加载 | [`../../nanovllm/utils/loader.py`](https://github.com/GeeeekExplorer/nano-vllm/blob/main/nanovllm/utils/loader.py) |

