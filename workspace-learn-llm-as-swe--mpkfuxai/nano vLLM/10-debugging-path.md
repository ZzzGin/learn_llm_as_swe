# 10. 按代码调试的推荐路线

这章给出一个实际读代码和打断点的路线。建议先用 `tensor_parallel_size=1` 和 `enforce_eager=True`，把多进程和 CUDA graph 的复杂性暂时关掉。

## 1. 从 example.py 开始

先看 `example.py`：

```python
llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)
outputs = llm.generate(prompts, sampling_params)
```

这能确认三件事：

- prompt 如何经 tokenizer 变成 token；
- sampling params 如何传入；
- 输出格式是 `{"text": ..., "token_ids": ...}`。

## 2. 跟进 LLMEngine.__init__

重点观察：

- `Config` 如何读取 HF config；
- `Sequence.block_size` 如何被设置；
- `ModelRunner` 何时初始化；
- `Scheduler` 何时创建。

如果只想理解调度，可以先不深入模型加载，直接读 `Scheduler` 和 `Sequence`。

## 3. 跟一次 add_request

`add_request` 会把 prompt 变成：

```python
seq = Sequence(prompt, sampling_params)
self.scheduler.add(seq)
```

这里可以观察：

- `seq_id`
- `num_prompt_tokens`
- `num_tokens`
- `block_table` 初始为空
- `status=WAITING`

## 4. 跟第一次 step

第一次 `step()` 通常进入 prefill：

```python
seqs, is_prefill = self.scheduler.schedule()
```

观察 `schedule()` 中：

- `waiting` 是否弹出；
- `block_manager.can_allocate(seq)` 返回多少 cached blocks；
- `seq.block_table` 被分配成什么；
- `seq.num_scheduled_tokens` 是多少。

## 5. 跟 prepare_prefill

进入 `ModelRunner.prepare_prefill` 后，重点看这些张量：

- `input_ids`
- `positions`
- `cu_seqlens_q`
- `cu_seqlens_k`
- `slot_mapping`
- `block_tables`

如果没有 prefix cache，`block_tables` 通常是 `None`。如果构造两个共享长前缀的 prompt，第二个请求可能看到 prefix cache 相关路径。

## 6. 跟 Attention.forward

在 `nanovllm/layers/attention.py` 里观察：

- `store_kvcache` 是否被调用；
- `context.is_prefill` 是 True 还是 False；
- prefill 是否走 `flash_attn_varlen_func`；
- decode 是否走 `flash_attn_with_kvcache`。

这一步能把 block table、slot mapping 和真正 attention 计算联系起来。

## 7. 跟 postprocess

模型返回 token 后，`Scheduler.postprocess` 会更新序列状态。重点观察：

- `hash_blocks` 是否登记了完整 block；
- `num_cached_tokens` 如何增加；
- 生成 token 是否 append 到 `seq.token_ids`；
- eos 或 max_tokens 是否触发释放。

## 8. 再看 decode

第二次或后续 `step()` 在没有 waiting 请求时会进入 decode。重点比较：

| 项目 | prefill | decode |
| --- | --- | --- |
| 每个序列输入 token | prompt chunk | last token |
| attention 函数 | `flash_attn_varlen_func` | `flash_attn_with_kvcache` |
| 是否可能写多个 cache slot | 是 | 通常每序列一个 |
| batch 形状 | token 级 varlen | sequence 级 one-token |
| CUDA graph | 不使用 | 可使用 |

## 9. 最后打开性能开关

理解主流程后，再把：

```python
enforce_eager=False
```

这样 decode 会尝试走 CUDA graph。此时再读：

- `capture_cudagraph`
- `run_model` 中 graph replay 路径
- `graph_vars` 如何复用固定张量

会更容易。

## 推荐读法总结

最短路径是：

```text
example.py
-> LLMEngine.generate
-> Scheduler.schedule
-> ModelRunner.prepare_prefill / prepare_decode
-> Qwen3ForCausalLM.forward
-> Attention.forward
-> Sampler.forward
-> Scheduler.postprocess
```

如果这条线打通了，再回头看 tensor parallel、loader、CUDA graph 和 prefix cache，复杂点会自然落到对应位置。

上一章：[采样、生成循环和性能开关](./09-sampling-generation-performance.md)  
返回：[目录](./00-index.md)

