# Qwen3.5-2B iGPU：OpenVINO 2026.3 全量重测

- 重测日期：2026-08-07
- Runtime：`OpenVINO 2026.3.0-22452-c2638477a62-qwen-ttft-optimizations`
- GenAI：与 `benchmark/openvino.genai` 2026.3 build 配套
- 设备：`GPU.0` Intel iGPU
- 模型：Qwen3.5-2B VLM，FP16 / INT8 / INT4 及 g128、AWQ、混合精度变体
- 评测方式：HF tokenizer 预渲染 chat template，`enable_thinking=False`，`apply_chat_template=False`，贪心生成 1 token，PA + prefix caching 开启
- 结果目录：[results_20263_qwen35_igpu](../results_20263_qwen35_igpu)
- 评测脚本：[eval_qwen35_20263_igpu.py](../eval_qwen35_20263_igpu.py)

## 结论

1. 2026.3 下 Qwen3.5-2B FP16 在 PA + prefix caching 开启时，`cpu_routing` 为 **93.1% (54/58)**。此前 2026.2.1 的 PA prefix-caching 精度问题（60.3%）在 2026.3 未复现。
2. CLINC 和 BANKING 的 FP16 准确率分别为 **77.7%** 和 **73.2%**，与历史量级一致。
3. INT4-g128 在 CLINC/BANKING 上仍接近 FP16，但 `cpu_routing` 只有 51.7%，说明中文路由任务的主要瓶颈仍是量化敏感性，不是 PA/prefix caching。
4. `AWQ(mixed)+ratio=0.8` 是当前纯任务矩阵中最能恢复中文路由的 INT4 变体：`cpu_routing=79.3%`；但仍低于 FP16，且英文任务不如基础 g128/r08 稳定。
5. 下面的逐条 TTFT 是完整数据集单条 `generate()` 墙钟时间，包含 prompt 处理、prefill 和首 token，不等同于 1k→512 的 model_bench 纯性能口径。

## 数据集和 token 统计

Token 使用 Qwen3.5-2B tokenizer，`query_tokens` 是原始 query 的 token 数，`prompt_tokens` 是最终 chat-template 输入（含 system prompt、user 消息和 generation prompt），`shared_prefix_tokens` 是当前 query 与前一条 query 的完整 prompt 公共前缀长度。

| 数据集 | 样本数 | system prompt | query tokens min/p50/mean/max | prompt tokens min/p50/mean/max | shared prefix min/p50/mean/max |
|---|---:|---:|---:|---:|---:|
| cpu_routing | 58 | 623 | 6 / 21 / 24.52 / 65 | 648 / 663 / 666.52 / 707 | 633 / 633 / 635.26 / 651 |
| CLINC150 | 4500 | 841 | 1 / 8 / 8.96 / 32 | 860 / 868 / 868.76 / 892 | 850 / 851 / 851.20 / 862 |
| BANKING cluster10 | 3080 | 637 | 2 / 11 / 12.59 / 78 | 658 / 666 / 668.27 / 734 | 646 / 647 / 646.70 / 655 |

## 准确率和逐条 TTFT

TTFT 统计为去掉第一条冷启动样本后的 p50/p95；每条结果 JSON 还保存了 query、prompt、预测、raw 输出和 token 数。

| 模型 | cpu acc | CLINC acc | BANKING acc | cpu TTFT p50/p95 | CLINC TTFT p50/p95 | BANKING TTFT p50/p95 |
|---|---:|---:|---:|---:|---:|---:|
| FP16 | 93.1% | 77.7% | 73.2% | 336.0 / 355.5 ms | 107.7 / 110.4 ms | 326.3 / 398.0 ms |
| INT8 | 74.1% | 75.8% | 71.5% | 230.8 / 240.3 ms | 290.5 / 292.7 ms | 231.5 / 235.5 ms |
| INT4 | 37.9% | 53.8% | 51.4% | 194.1 / 203.2 ms | 245.9 / 247.8 ms | 195.4 / 199.3 ms |
| INT4-g128 | 51.7% | 76.2% | 70.0% | 227.8 / 237.1 ms | 292.8 / 294.7 ms | 228.8 / 232.5 ms |
| INT4-g128-awqmix | 62.1% | 72.6% | 65.9% | 229.4 / 239.3 ms | 71.4 / 72.6 ms | 230.6 / 234.3 ms |
| INT4-g128-awqmix-r08 | 79.3% | 74.2% | 68.6% | 227.6 / 236.4 ms | 72.8 / 75.1 ms | 229.6 / 238.3 ms |
| INT4-g128-awqzh | 67.2% | 75.8% | 63.4% | 231.4 / 241.9 ms | 70.3 / 73.5 ms | 230.2 / 235.4 ms |
| INT4-g128-r08 | 56.9% | 77.8% | 73.5% | 229.4 / 239.9 ms | 72.2 / 74.6 ms | 226.7 / 231.2 ms |

> 2026.3 的结果以 `results_20263_qwen35_igpu/` 中按模型×数据集的最新 JSON 为准。目录中早期因 prompt 拼接错误产生的重复结果不纳入上表。

## 1k 输入 / 512 输出性能

使用 `model_bench` C++ VLM benchmark，1k prompt、512 个输出 token、10 次测量加 1 次 warm-up、prompt permutation 开启、GPU 串行运行。输入实际统计为 `in=1002`（最后一轮为 1003），prompt 原始文本约 988 token。PA 使用 `PERFORMANCE_HINT=LATENCY` 和 `ATTENTION_BACKEND=PA`；SDPA 使用相同性能提示但切换为 `ATTENTION_BACKEND=SDPA`。完整 CSV/JSON/log 在 [perf_results_20263_qwen35_igpu](../perf_results_20263_qwen35_igpu)。

### PA

| 模型 | TTFT ms | TPOT ms | Decode TPS |
|---|---:|---:|---:|
| FP16 | 438.0 | 93.2 | 10.7 |
| INT8 | 335.9 | 49.0 | 20.4 |
| INT4 | 289.8 | 32.6 | 30.7 |
| INT4-g128 | 348.8 | 33.9 | 29.5 |
| INT4-g128-awqmix | 353.3 | 34.5 | 29.0 |
| INT4-g128-awqmix-r08 | 341.8 | 36.9 | 27.1 |
| INT4-g128-awqzh | 349.1 | 33.7 | 29.7 |
| INT4-g128-r08 | 344.8 | 37.1 | 27.0 |

### SDPA 对照

| 模型 | TTFT ms | TPOT ms | Decode TPS |
|---|---:|---:|---:|
| FP16 | 572.9 | 97.1 | 10.3 |
| INT8 | 485.8 | 54.1 | 18.5 |
| INT4 | 434.0 | 37.4 | 26.7 |
| INT4-g128 | 495.2 | 38.8 | 25.8 |
| INT4-g128-awqmix | 492.1 | 39.2 | 25.5 |
| INT4-g128-awqmix-r08 | 481.6 | 42.4 | 23.6 |
| INT4-g128-awqzh | 491.5 | 39.0 | 25.6 |
| INT4-g128-r08 | 481.7 | 42.2 | 23.7 |

### PA / SDPA 对应关系

下表将同一模型的两种 attention backend 直接并列，所有数据均来自 2026.3 runtime 的同一批 1k→512 测试。

| 模型 | PA TTFT | SDPA TTFT | PA TPOT | SDPA TPOT | PA TPS | SDPA TPS |
|---|---:|---:|---:|---:|---:|---:|
| FP16 | 438.0 | 572.9 | 93.2 | 97.1 | 10.7 | 10.3 |
| INT8 | 335.9 | 485.8 | 49.0 | 54.1 | 20.4 | 18.5 |
| INT4 | 289.8 | 434.0 | 32.6 | 37.4 | 30.7 | 26.7 |
| INT4-g128 | 348.8 | 495.2 | 33.9 | 38.8 | 29.5 | 25.8 |
| INT4-g128-awqmix | 353.3 | 492.1 | 34.5 | 39.2 | 29.0 | 25.5 |
| INT4-g128-awqmix-r08 | 341.8 | 481.6 | 36.9 | 42.4 | 27.1 | 23.6 |
| INT4-g128-awqzh | 349.1 | 491.5 | 33.7 | 39.0 | 29.7 | 25.6 |
| INT4-g128-r08 | 344.8 | 481.7 | 37.1 | 42.2 | 27.0 | 23.7 |

### 与 2026.2.1 性能比较

旧 2026.2.1 结果来自 `perf_results/logs/2B_*.log`，测试口径同为 1k→512、10 次测量加 warm-up：

| 配置 | 2026.2.1 TTFT / TPOT / TPS | 2026.3 TTFT / TPOT / TPS | 变化 |
|---|---:|---:|---|
| PA FP16 | 356.3 / 50.1 / 19.7 | 438.0 / 93.2 / 10.7 | **回退**：TTFT +23.0%，TPOT +86.0%，TPS -45.7% |
| PA INT8 | 281.1 / 20.8 / 47.0 | 335.9 / 49.0 / 20.4 | **回退**：TTFT +19.5%，TPOT +135.6%，TPS -56.6% |
| PA INT4-g128 | 304.9 / 14.6 / 65.7 | 348.8 / 33.9 / 29.5 | **回退**：TTFT +14.4%，TPOT +132.2%，TPS -55.1% |
| SDPA FP16 | 496.4 / 52.9 / 18.6 | 572.9 / 97.1 / 10.3 | **回退**：TTFT +15.4%，TPOT +83.6%，TPS -44.6% |
| SDPA INT4-g128 | 448.3 / 16.6 / 57.7 | 495.2 / 38.8 / 25.8 | **回退**：TTFT +10.5%，TPOT +133.7%，TPS -55.3% |

**性能结论**：本次 2026.3 runtime/build、同一 iGPU 和同一测试口径下，实测没有性能改善；PA/SDPA 的 TTFT、TPOT、TPS 均低于对应的 2026.2.1 记录，尤其是 decode TPOT/TPS。与此同时，准确率和 PA prefix-caching correctness 明显改善：旧版 PA FP16 cpu_routing 为 60.3%，2026.3 为 93.1%。因此本次升级实测收益体现在 correctness，不体现在性能。

## 与 2026.2.1 的变化

旧报告中 2026.2.1 的 Qwen3.5-2B iGPU PA 参考值为：FP16 `cpu_routing=60.3%`、CLINC `78.0%`、BANKING `72.9%`。2026.3 重测为 `93.1% / 77.7% / 73.2%`。因此最重要的改善是 PA + prefix caching 的 correctness issue 不再复现；CLINC/BANKING 属于测试噪声范围内的小幅变化。

1k→512 性能重测已经完成，详细结果见上节。旧日志中的 2B 参考值为：PA FP16 TTFT 356.3 ms、TPOT 50.1 ms；PA INT4-g128 TTFT 304.9 ms、TPOT 14.6 ms；SDPA FP16 TTFT 496.4 ms、TPOT 52.9 ms；SDPA INT4-g128 TTFT 448.3 ms、TPOT 16.6 ms。8 个模型的完整 PA/SDPA 结果见上面的对应表。

## 复现

```bash
source /home/chaofan/envs/qwen_env/bin/activate
export PYTHONPATH=/home/chaofan/model_benchmark/benchmark/openvino/install/python:/home/chaofan/model_benchmark/benchmark/openvino.genai/build:$PYTHONPATH
export LD_LIBRARY_PATH=/home/chaofan/model_benchmark/benchmark/openvino/install/runtime/lib/intel64:/home/chaofan/model_benchmark/benchmark/openvino.genai/build/openvino_genai:$LD_LIBRARY_PATH
cd /home/chaofan/model_benchmark/query_classifier_handoff
python eval_qwen35_20263_igpu.py
```
