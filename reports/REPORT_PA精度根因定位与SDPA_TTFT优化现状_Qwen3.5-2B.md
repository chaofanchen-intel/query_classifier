# Qwen3.5-2B VLM：PA 精度 bug 根因定位 + SDPA/PA TTFT 优化现状复核

> **2026.3 重测订正（2026-08-07）**：基于 OpenVINO/OpenVINO GenAI 2026.3 runtime 的 Qwen3.5-2B iGPU 全量重测已完成。PA + prefix caching 开启时，FP16 `cpu_routing=93.1% (54/58)`，与 PA 关闭 prefix caching 的结果一致；旧版 2026.2.1 的 `60.3%` 精度问题在 2026.3 未复现。新的完整准确率、query/prompt token 统计和逐条 TTFT 见 [REPORT_Qwen3.5-2B_iGPU_2026.3重测.md](REPORT_Qwen3.5-2B_iGPU_2026.3重测.md)。本文 §2 的 2026.2.1 实验结论保留为历史对照，不能直接用于判断 2026.3。

> **性能补充**：2026.3 的 1k→512 iGPU 压测已对全部 8 个 Qwen3.5-2B 变体分别测试 PA/SDPA；TTFT、TPOT、TPS 均不优于 2026.2.1。correctness 已改善，但性能出现回退。具体数字和口径见专用重测报告。

> 任务背景：`README.md` 注意事项 6 / `pipe_config.py` docstring 已记录 **`ATTENTION_BACKEND=PA` 会把 Qwen3.5-2B（VLMPipeline）在中文 `cpu_routing` 任务上的精度从 93.1%（SDPA）打到 60.3%**，但 CLINC/BANKING（同模型、同 VLM 管线）不受影响，根因当时标注为"疑与 Qwen3.5 hybrid/Gated-DeltaNet 注意力在 paged 实现下的掩码/位置处理有关"（推测，未坐实）。
>
> 本文目的：① 把这个"疑似"根因坐实到具体机制（用真实 GPU 做受控实验，而非只读代码猜测）；② 复核现有 SDPA/PA TTFT 优化文档（`benchmark/tests/OV源码优化降低TTFT_{SDPA,PA}.md`、`benchmark/openvino/qwen3.5_patch/`）是否还有进一步降 TTFT 的空间。
>
> 实验环境：本机 Intel iGPU（32 EU，`clinfo` 识别为 `Intel(R) Graphics`，与既有文档里的"356H/32-XVE"机器同量级）、`openvino=2026.2.1-ede283a88e3`、`openvino_genai=2026.2.1.0-7dea0459b2a`（`/home/chaofan/envs/qwen_env`）、模型 `benchmark/models/Qwen3.5-2B-FP16`。**当前 `benchmark/openvino` 工作树已应用 opt-1~opt-5 全部 5 个 GPU 插件优化 patch**（`git status` 确认，见 §4），所以本文下面的所有实测数字都是"优化后"的现状基线，不是 pristine OpenVINO。
>
> 复现脚本：`diag_pa_prefix_caching.py`、`diag_pa_latency_compare.py`、`diag_pa_clinc_control.py`（本目录，均可直接跑）。

---

## 0. 结论先行

1. **根因已从"疑似"坐实为可复现的受控实验结果：PA 的精度 bug 不在 PagedAttention 本身，而在"前缀缓存（prefix caching）与 Qwen3.5 hybrid 线性注意力（Gated-DeltaNet）状态的检查点/复用机制"的交互上。** 同一个 PA 后端，只把 `enable_prefix_caching` 从默认的 `True` 改成 `False`，cpu_routing 精度从 **60.3% → 93.1%**（与 SDPA、与文档 93% 完全吻合），其余所有配置（模型、量化、attention backend=PA、prompt）不变。
2. **代价：关掉 prefix caching 也顺带关掉了 PA 在这个任务上最大的速度优势。** `PA+prefix_caching=OFF` 的 p50 延迟（323ms）和 SDPA（425ms，本机口径）同量级，**不比 SDPA 快多少**——因为 PA 对这类"58 条查询共享同一条长 system prompt"的分类器工作负载，其低延迟本来就是靠前缀缓存跳过重复 prefill 拿到的（`PA+prefix_caching=ON` 只要 108ms）。所以"关掉前缀缓存"不是一个能同时拿到"又快又对"的免费修复，本质是在拿"prefix caching 收益"换"正确性"，跟直接切 SDPA 是同一笔账。
3. **CLINC/BANKING 为什么没事：用同样方法做了对照实验，确认 PA 在 CLINC 上与 SDPA 逐条预测完全一致（60/60 相同）**，而 cpu_routing 上 PA 与"正确值"逐条对比有约 40% 的条目不同、且不同的条目在数据里**不连续、看起来与具体查询内容无强相关**（见 §2.3），更像是"缓存命中/状态复用时序"相关的 bug，而不是"某种特定中文 token 触发数值错误"。因此**任何同样具备"长 system prompt + 大量近似重复请求"特征的部署场景，即便是英文/其它任务，都有中招风险，不能只按"中文任务才要小心"来划安全边界**——CLINC/BANKING 目前测不出来，不代表这个机制对它们免疫，只是没被现在的数据触发。
4. **SDPA 的 TTFT 优化已经做到位、且已经全部合入本地工作树**：`benchmark/tests/OV源码优化降低TTFT_{SDPA,PA}.md` + `benchmark/openvino/qwen3.5_patch/` 记录的 opt-1~opt-5（消除 Reorder / 修 Transpose LWS 派发 / 复用 DynamicQuantize / conv1d token-tiled / activation_opt 走向量化 kernel）**都已经 apply 到当前 `benchmark/openvino` 源码里**（`git status` 可见 8 个文件的 diff）。这些文档自己的结论是"3-opt/5-opt 态已无可安全落地的新局部优化点，唯一还没做的方向是把 SSM conv 的 Transpose+Concat+Conv+Slice+Transpose 融成一个有状态 fused kernel（高风险，需要新 op/新 pass，且要保正确性）"。**本次复核未发现比这更好的新方向**——因此"能否进一步优化 SDPA TTFT"这件事，答案是"该做的都做了，工作树里已经生效，剩下的路径是一个大工程量的图融合，不建议在当前任务里现做"。
5. **给 leader 的最终建议**：维持现状（GPU 默认 SDPA，`pipe_config.py` 已经这么做）。**不要**把 PA 的"opt-3/4/5 已经很快"当成"可以放心切回 PA"的理由——那些优化只动 kernel 派发/图结构，不碰这个 correctness bug。**唯一安全地拿回一部分 PA 速度优势的路是升级 OpenVINO 到修复此 bug 之后的版本（上游需要修 `LinearAttentionCacheManager`/前缀缓存对 hybrid 模型的检查点复用逻辑）**，或者业务层面接受"每次都重算前缀、不用 KV 复用"这条不划算的折中路。

---

## 1. 问题重述与既有材料

- `README.md` §9 注意事项 6 / `pipe_config.py` docstring：cpu_routing 上 PA 60.3% vs SDPA 93.1%，CLINC/BANKING 及全部 1.7B（dense、无线性注意力层）不受影响；根因写的是"疑与 Qwen3.5 hybrid/Gated-DeltaNet 注意力在 paged 实现下的掩码/位置处理有关"——**这是推测，没有实验坐实到具体机制**。
- `benchmark/openvino.genai`（本机 openvino_genai 的完整源码树）里能确认到的相关事实：
  - Qwen3.5 是 24 层里 18 层 `linear_attention`（Gated-DeltaNet，带 causal conv1d）+ 6 层 `full_attention` 的 hybrid 架构（`benchmark/tests/TTFT加速方案分析报告.md` §6 已用 `config.json` 核实）。
  - PA 对 Qwen3.5 的支持是较新加入的：`git log` 显示 commit `1550089e [VLM] Enable PA for Qwen3.5 (#3801)`（2026-05-12）**从 `requires_sdpa()` 里删掉了 `QWEN3_5`/`QWEN3_5_MOE`**（此前这两个模型类型被强制 SDPA，注释是 `// TODO: remove Qwen3.5 limitation once ticket-183791 is fixed`），同一个 commit 还把 M-RoPE 的 position_ids 维度断言从硬编码的 `3` 改成了通用的 `shape[0]`（因为 Qwen3.5 用的是 **4 维**位置编码：`InputsEmbedderQwen3_5::create_position_ids` 在标准 Qwen2VL 的 3 维 t/h/w 之外，额外拼了一个"纯文本顺序位置"的第 0 维，见 `src/cpp/src/visual_language/qwen3_5/classes.cpp:52`）。这说明 PA 对 Qwen3.5 的支持本身就是"最近才打开、且明确带着一个未完全验证的 ticket-183791"，不是打磨很久的成熟路径。
  - `src/cpp/src/continuous_batching/cache/linear_attention_cache_manager.hpp` + `cache_type.hpp` + `cache_orchestrator.hpp`：PA/continuous-batching 给 hybrid 模型专门加了一套"线性注意力状态缓存"（`conv_state_table.N` / `gated_delta_state_table.N`，每个模型有独立的 conv1d 状态和 SSM 递归状态，按"每个序列一块"的粒度管理，与标准 KV cache 的按 token-block 粒度完全不同）。
  - `SchedulerConfig` 有一个专门字段 `cache_interval_multiplier`（`src/cpp/include/openvino/genai/scheduler_config.hpp`）：**"用于推导前缀缓存场景下线性注意力检查点间隔的乘数……仅对开启前缀缓存的 hybrid 模型有意义"**。这是 PA 为了让"前缀缓存"和"hybrid 模型的递归状态"能配合工作而专门加的机制——递归状态本质上不能像标准 KV cache 那样按 token 块粒度直接复用/拼接，所以需要在若干 token 间隔处存一份"检查点"，命中前缀缓存时从最近的检查点回放，而不是回到 t=0 全量重算。**这正是我们怀疑的机制，也是下面受控实验要验证的对象。**

给定这套材料，"疑似根因"已经指向了一个很具体的候选：**`enable_prefix_caching`（VLMPipeline/LLMPipeline 便捷接口默认通过 `get_latency_oriented_scheduler_config()` 开启）叠加 hybrid 模型的线性注意力检查点复用逻辑，两者结合时有 bug。** 下面直接用真实 GPU 做实验验证，而不是继续读代码猜测。

---

## 2. 受控实验：坐实根因

### 2.1 核心实验：只翻转 `enable_prefix_caching`

固定模型（Qwen3.5-2B-FP16）、固定 `ATTENTION_BACKEND=PA`、固定 prompt（`cpu_routing_qwen3.5`，58 条中文查询）、固定 `max_num_batched_tokens=100000`（排除 `dynamic_split_fuse` 分块 prefill 的干扰），**唯一变量是 `SchedulerConfig.enable_prefix_caching`**：

```python
sc = ov_genai.SchedulerConfig()
sc.enable_prefix_caching = False   # 唯一改动
sc.max_num_batched_tokens = 100000
pipe = ov_genai.VLMPipeline(MODEL, "GPU", ATTENTION_BACKEND="PA", scheduler_config=sc)
```

（完整脚本见 `diag_pa_prefix_caching.py`，直接 `python diag_pa_prefix_caching.py` 复现。）

| 配置 | cpu_routing 精度 |
|---|---:|
| PA + `enable_prefix_caching=True`（VLMPipeline 便捷接口的默认值） | **60.3%**（35/58）——与既有报告数字完全吻合 |
| **PA + `enable_prefix_caching=False`** | **93.1%**（54/58）——与 SDPA、与文档 93% 完全吻合 |

**结论：精度差异 100% 由 `enable_prefix_caching` 决定，与"PA vs SDPA"这个 attention backend 选择本身无关。** 也就是说，此前"PA 有精度 bug"这个说法不够精确——精确的说法是"**PA 的前缀缓存路径**在 Qwen3.5 这种 hybrid 模型上有精度 bug"。

### 2.2 延迟对照：这个"修复"值不值

同样 58 条 cpu_routing，测 p50/p95（python 层 `pipe.generate()` 墙钟时间，`diag_pa_latency_compare.py`）：

| 配置 | 精度 | p50 | p95 | 首条（冷启动） |
|---|---:|---:|---:|---:|
| PA + prefix_caching **ON**（默认） | 60.3%（错） | **107.9ms** | 127.6ms | 477.7ms |
| PA + prefix_caching **OFF** | 93.1%（对） | 323.0ms | 332.0ms | 469.9ms |
| SDPA（现默认，无前缀缓存概念） | 93.1%（对） | 425.4ms | 449.7ms | 590.7ms |

**PA 关掉前缀缓存后（323ms）和 SDPA（425ms）在同一个量级，比 PA 开着前缀缓存（108ms）慢了 3 倍。** 这印证了一个此前文档没直接点出的机制：**对"58 条查询共享同一条 ~630-token 中文 system prompt"这种分类器工作负载，PA 相对 SDPA 的 5 倍速度优势，本质上就是前缀缓存省掉的那部分重复 prefill**，不是 PagedAttention 计算本身比 SDPA 快 5 倍。一旦前缀缓存不能用（因为它有 bug），PA 就没有比 SDPA 明显的优势了——**"关掉前缀缓存"不是一条免费的正确性修复路径，而是把"pv 缓存收益 vs 正确性"这笔账重新算了一遍，结论跟"直接用 SDPA"是同一个数量级。**

> 侧记：本机（32-EU iGPU，工作树已合入 opt-1~5 全部优化）测出的 SDPA p50 是 425ms（python 墙钟口径），比 `OV源码优化降低TTFT_SDPA.md` 报的 322ms（`model_bench` C++ 纯推理口径、358H/大 iGPU）更慢——这是预期之内的：measurement 口径不同（python 端到端 vs C++ 纯 infer）、硬件不同（本机是 32-EU 的小 iGPU，量级更接近文档里的 356H）。两份数字不能直接比大小，但"PA 关掉前缀缓存后约等于 SDPA"这个相对结论在两次测量里是一致的。

### 2.3 逐条比对：这个 bug 长什么样

把 `enable_prefix_caching=True` 那组 58 条逐条打印 gold/pred（见对话记录，也可重跑 `diag_pa_prefix_caching.py`），观察到的模式：

- **不是"第一条对、后面全错"，也不是"每隔 N 条错一次"这种规律模式**：错误分布是 #1-3 连错、#4-14 连对 11 条、#15-17 连错、#18-19 对、#20-22 连错……看起来像是"突发簇"而非均匀分布或简单周期。
- 错误的方向也不固定：A 类被误判成 B，B 类被误判成 A，C/D 类被误判成 A/B 都有，不是"总是偏向某一类"。

这个模式**更像是"KV 块 / 线性注意力状态检查点被分配、复用、回收时的时序/索引类 bug"**（比如：多个共享同一前缀的并发/连续请求竞争同一个检查点块，或某个块被复用前没有正确失效），**而不是"某种中文 token 组合触发确定性的数值错误"**。这与 §2.4 的对照实验（同样的前缀缓存机制在 CLINC 上完全无损）是一致的——如果是"内容触发"的 bug，CLINC 反复出现的英文前缀没道理完全免疫；如果是"缓存块生命周期管理"类的 bug，则确实可能对不同 prompt 长度/复用节奏表现出不同的触发概率。**受限于这是一个纯 Python/黑盒层面的诊断（没有对 C++ 端重新编译加日志/断言去定位具体是哪一次 `copy_blocks`/`zero_blocks`/检查点回放调用出的错），无法进一步定位到具体代码行——这需要在 `LinearAttentionCacheManager` 和 `cache_orchestrator.hpp` 的检查点回放路径里加调试日志、重新编译 GPU/CPU 插件后复测，工作量与本次任务的"先看能不能定位、定位不了就转向 SDPA 优化"的既定预算不匹配，留作后续（见 §5 建议）。**

### 2.4 对照实验：CLINC 上 PA 是否真的"无损"

用同样的方法在 CLINC150（英文，10 类）上做 60 条随机抽样（`seed=42`，`diag_pa_clinc_control.py`）：

| 配置 | 精度（60 条随机子集） |
|---|---:|
| PA（默认前缀缓存 ON） | 71.7%（43/60） |
| SDPA | 71.7%（43/60） |

**逐条比对 60 条预测，PA 与 SDPA 完全一致（0 条不同）**，与 README/报告里"CLINC/BANKING 逐条一致"的说法吻合。**这次实测确认了这个"无损"结论在当前工作树（已合入 opt-1~5）下依然成立**——所以 CLINC/BANKING 的免疫不是历史数据过期，是真实可复现的。

### 2.5 排除"prompt 长度阈值"这个最直观的假设

一个自然的猜测是"cpu_routing 的共享前缀比 CLINC 短/长，跨过了 `cache_interval_multiplier` 定义的检查点间隔边界，CLINC 没跨过"。用 tokenizer 直接量出两条 prompt 之间的共享前缀长度（渲染完整 chat template 后逐 token 比较到第一个分叉点）：

| 任务 | 单条 prompt 总 token 数 | 两条不同 query 之间的共享前缀 token 数 |
|---|---:|---:|
| cpu_routing（中文，4 类） | 668 | **633** |
| CLINC（英文，10 类） | 867 | **851** |

**CLINC 的共享前缀（851 token）比 cpu_routing（633 token）更长**，如果假设是"共享前缀越长越容易触发"，CLINC 应该更容易中招而不是完全免疫——**数据不支持这个假设**。以 OpenVINO 默认 KV block size（16 或 32）× `cache_interval_multiplier` 默认值 8 计算，两个任务的共享前缀都会跨越不止一个检查点间隔（128 或 256 token 一个检查点），**跨越检查点边界的次数本身也不是决定性因素**。

**结论：触发条件目前无法简化成"prompt 长度"或"共享前缀长度"这类单一标量**，更可能与"中文 tokenizer 产出的具体 token id 序列"、"请求到达的具体时序/并发调度顺序"或某个尚未定位的索引 off-by-one 有关。**这意味着不能简单地用"prompt 短就安全"这类经验法则来规避这个 bug**——安全的做法只有"不用前缀缓存"或"不用 PA"。

---

## 3. 根因小结（给 leader 的一段话）

> PA 在 Qwen3.5-2B 这个 hybrid（18 层线性注意力 GatedDeltaNet + 6 层全注意力）VLM 上的精度问题，根因不是 PagedAttention 的注意力计算本身，而是**前缀缓存（prefix caching）命中时，线性注意力的递归状态（conv1d 状态 + SSM 状态）检查点复用机制有 bug**——这是 OpenVINO GenAI 为了让 hybrid 模型也能享受前缀缓存收益而专门加的新机制（`SchedulerConfig.cache_interval_multiplier` + `LinearAttentionCacheManager`），PA 对 Qwen3.5 的支持本身也是本季度才打开的（5 月的 PR，注释里挂着一个未验证完的 ticket）。用受控实验（只翻转 `enable_prefix_caching`）把"疑似"坐实为可复现的结论：**关掉前缀缓存，PA 精度立即恢复到 93.1%（与 SDPA/文档一致）**；但这么做也让 PA 在这类"重复相似前缀"工作负载上的速度优势基本消失（从 108ms 退回到 323ms，跟 SDPA 同量级）。CLINC/BANKING 目前测不出问题，但排除了"prompt 短/长"这类简单解释，所以不能假设其它任务天然安全。**结论：这不是一个"调参就能两全"的 bug，是一个需要上游修复的正确性问题；在修复之前，唯一安全的选择是不使用前缀缓存（等价于不用 PA 的速度优势，效果约等于直接用 SDPA）。**

---

## 4. SDPA / PA TTFT 优化现状复核

### 4.1 已有文档结论（未过期，本次未推翻）

- `benchmark/tests/OV源码优化降低TTFT_SDPA.md`：SDPA 路径的 opt-1（消除 depthwise conv 的 Reorder）+ opt-2（修 permute_ref 的 LWS 派发）+ opt-3（共享激活的 FC 复用 DynamicQuantize），infer-TTFT 总降幅 **−10~21%**（模型越小降幅越大），全部 md5 逐字节无损验证。文档 §9 明确列出"3-opt 态已无可安全落地的新局部优化点"，唯一未做的方向是"把 conv 前后的 Transpose+Concat+Conv+Slice+Transpose 融成一个有状态、layout-aware 的 fused kernel"，标注为"高风险（需要新 internal op + 新 pass + 新 kernel，且要同时保 prefill/decode 数值不变），唯一值得立项的性能方向，但需要先做 POC"。
- `benchmark/tests/OV源码优化降低TTFT_PA.md`：PA 路径额外的 opt-4（`paged_causal_conv1d` token-tiled 派发，把占用率从 ~20-60% 提到 100% 而不放大内存读）+ opt-5/opt-6/opt-7（356H 上取代 opt-4 的 tiled 版本 + kernel SLM 分配修正 + GenAI e2e 小优化），在小 iGPU（356H/32-XVE）上 infer-TTFT 总降幅 **−4.8~9.0%**。文档同样明确排除了"GDN chunkwise 重写"（会改变浮点累加顺序、几乎必然破坏 md5 字节一致性，且工作量极高）等更激进的方向。
- `benchmark/openvino/qwen3.5_patch/README.md`：把上述 5 个优化整理成了可直接 `git apply` 的独立 patch，并给出了"PA 用 opt3+4+5、SDPA 用 opt1+2+3"的落地建议。

### 4.2 本次复核：这些优化在当前工作树里的状态

`git status`/`git diff --stat`（`benchmark/openvino`）确认：**opt-1 ~ opt-5 全部 5 个 patch 当前都已经应用在工作树里**（`layout_optimizer.cpp`、`permute_kernel_ref.cpp`、`dynamic_quantize.cpp`、`dynamic_quantize_fully_connected.cpp`、`paged_causal_conv1d_ref.{cl,cpp}`、`activation_opt.cl`、`activation_kernel_opt.cpp` 均有本地改动，且改动内容与两份优化文档描述的 opt-1~5 逐一对应）。也就是说：**"进一步优化 SDPA/PA 的 TTFT"这件事，能做的都已经做了、也已经在跑的代码里生效了**，本次没有找到比 opt-1~5 更好的新优化点（既有文档的"已到 floor"判断在重新审视代码后依然成立）。

### 4.3 一个新发现：opt-1~5 都不能修复 §2 的精度 bug

需要特别指出：**§2 的实验（`PA + prefix_caching=True` 精度只有 60.3%）是在"已经应用 opt-1~5"的工作树上跑出来的**——即便所有已知的 GPU kernel 派发/图结构优化都在，前缀缓存的精度 bug 依然复现，说明 **opt-1~5 是纯粹的性能优化（kernel 派发方式、图结构去重），完全不触碰前缀缓存/线性注意力状态管理这条逻辑路径**，两者是正交的问题，不能指望"性能优化顺便修了正确性 bug"，也不用担心"以后修正确性 bug 会影响这些性能优化"。

---

## 5. 建议

1. **维持 `pipe_config.py` 现状（GPU 默认 `ATTENTION_BACKEND=SDPA`）**，这是目前对 cpu_routing 这类"长中文 system prompt + 大量结构相似查询"的分类任务唯一已知安全的配置。
2. **不要尝试"PA + 关闭前缀缓存"作为折中方案**：§2.2 已证明这条路速度上不比 SDPA 有优势，却要多维护一份非默认的 `SchedulerConfig`，性价比为负。
3. **如果需要更快的 TTFT**：当前 SDPA 路径能做的 kernel 级优化（opt-1/2/3）已经在工作树里，无需也无法再从这个方向进一步压缩；唯一还没做的方向是 opt-3/4/5 文档提到的"conv-sandwich 融合成有状态 fused kernel"，这是一个高风险、高工作量（需要新 OpenVINO internal op + 新 transformation pass + 新 GPU kernel，还要保证不破坏 prefill/decode 的数值一致性）的项目，**建议单独立项评估，不建议顺带做**。
4. **把 §2 的复现实验和结论上报/提交给 OpenVINO GenAI 上游**（`LinearAttentionCacheManager` + `SchedulerConfig.cache_interval_multiplier` 相关逻辑），因为：
   - 这是一个明确的正确性 bug，影响所有"hybrid 线性注意力模型 + 前缀缓存"的组合，不止 Qwen3.5-2B-VLM 一个模型；
   - 本文档的三个诊断脚本（`diag_pa_prefix_caching.py`/`diag_pa_latency_compare.py`/`diag_pa_clinc_control.py`）已经是一个干净、可直接复现的最小 repro，可以直接作为 issue 附件；
   - 只有上游修复后，PA 的前缀缓存速度优势（108ms vs 现在被迫放弃后的 323-425ms）才能安全拿回来。
5. **后续如果要精确定位到具体代码行**（而不是停在"前缀缓存 + 线性注意力检查点复用"这个机制层面），需要在 `LinearAttentionCacheManager`/`cache_orchestrator.hpp` 的检查点回放路径里加调试日志、重新编译 GPU/CPU 插件后用 §2.3 的"逐条打印 + 找突发错误簇"方法继续二分定位。这部分工作量较大（涉及重新编译 OpenVINO 主库，不只是 GPU 插件的 ~9 秒增量重建），本次任务按"先看能否定位，定位不了就转向 SDPA 优化现状复核"的既定预算没有展开，留作后续任务。

---

## 附：复现清单

```bash
source /home/chaofan/envs/qwen_env/bin/activate
export LD_LIBRARY_PATH="/home/chaofan/model_benchmark/benchmark/openvino/install/runtime/lib/intel64:${LD_LIBRARY_PATH:-}"
cd /home/chaofan/model_benchmark/query_classifier_handoff

# §2.1/2.2：核心根因实验 + 延迟对照
python diag_pa_prefix_caching.py     # PA prefix_caching True/False 精度对比
python diag_pa_latency_compare.py    # PA(on/off) vs SDPA 的 p50/p95 对比

# §2.4：CLINC 对照实验（确认 PA==SDPA 在 CLINC 上逐条一致）
python diag_pa_clinc_control.py
```

| 文件 | 说明 |
|---|---|
| `diag_pa_prefix_caching.py` | 核心实验：固定 PA，只翻转 `enable_prefix_caching`，测 cpu_routing 精度 |
| `diag_pa_latency_compare.py` | PA(prefix on/off) vs SDPA 的精度 + p50/p95 延迟对比 |
| `diag_pa_clinc_control.py` | CLINC 150 60 条子集上 PA vs SDPA 逐条预测对比（确认无差异） |
| `benchmark/tests/OV源码优化降低TTFT_SDPA.md` | SDPA opt-1/2/3 完整方法学与数据（本文档 §4 复核对象） |
| `benchmark/tests/OV源码优化降低TTFT_PA.md` | PA opt-3/4/5/6/7 完整方法学与数据（本文档 §4 复核对象） |
| `benchmark/openvino/qwen3.5_patch/` | 上述优化的独立 git patch，当前已全部 apply 到 `benchmark/openvino` 工作树 |
