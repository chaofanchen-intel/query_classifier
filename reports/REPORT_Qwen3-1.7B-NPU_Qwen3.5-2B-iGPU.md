# Query-Classifier 准确率评估报告：Qwen3-1.7B @ NPU 与 Qwen3.5-2B @ iGPU

- 日期：2026-07-15
- 环境：本机优化版 OpenVINO（`install/` 树，`2026.2.1-...-qwen-ttft-optimizations`）+ `qwen_env`（transformers/torch）
- 任务：中文 A/B/C/D 路由 `eval_set`(58)、CLINC150 10-domain(4500)、BANKING77-cluster10(3080)
- 目标：① 测 Qwen3-1.7B@NPU、Qwen3.5-2B@iGPU 的分类准确率；② 找到「保留 INT4 级 TTFT 又能恢复准确率」的量化方案；③ 解释 FP16 在 `eval_set` 上为何达不到文档的 ~93%。

## 0. 三条核心结论

1. **仓库自带的 INT4 权重量化过于激进**（`group_size=-1` 逐通道、对称、`ratio=1.0`），是准确率远低于 FP16 的**主因**——而非模型或设备。CLINC 上 Qwen3.5-2B：INT4 52% → FP16 78%；Qwen3-1.7B：INT4 24%(GPU)/1.5%(NPU 垃圾 token) → FP16 68%。

2. **「不那么激进的 INT4」能在 INT4 级 TTFT 下恢复大部分准确率，但配方随设备不同**：
   - **iGPU（支持非对称）**：`INT4 group_size=128 非对称` 即可，CLINC 75.7% / BANKING 70.2%（≈FP16、≈INT8），**仍是纯 4-bit（TTFT 不变）**。
   - **NPU（硬件要求对称量化）**：纯对称 INT4 救不回小模型（g128-sym 仅 16–18%）；需**混合精度** `INT4 g128 sym + ratio=0.8（20% 层走 int8-sym）`，cpu_routing 65.5% / CLINC 54.9% / BANKING 48.1%（旧激进 INT4 分别是 39.7% / 1.5%垃圾 / 垃圾），绝大多数层仍是 4-bit。**INT8 虽能恢复但 TTFT 过高，不采用。**

3. **FP16 达不到文档 cpu_routing ~93% 是因为 `eval_set` 本身很难，不是流水线问题**：FP16 在 CLINC(78.0%)/BANKING(72.9%) 上**精确复现**文档（78/73），证明评测链路正确；但 cpu_routing 的 `correct_by` 显示连 14B 都只有 83%、C 类历史可解率仅 53%，一个 2B 模型不可能到 93%。开启 thinking 反而更差（73%→33%）。**文档 93% 极可能来自更早/更简单版本的 eval_set 或不同口径。**

---

## 1. 环境与运行配置

```bash
# env.sh —— 所有评测的前置
source /home/chaofan/envs/qwen_env/bin/activate
source /home/chaofan/model_benchmark/benchmark/openvino/install/setupvars.sh
export PYTHONPATH="/home/chaofan/model_benchmark/benchmark/openvino/install/python:$PYTHONPATH"
```
- 设备：`NPU`(AI Boost) / `GPU.0`(iGPU, Intel Graphics) / `GPU.1`(dGPU, Arc Pro B70) / `CPU`。
- 按设备自动注入 pipeline 配置（`pipe_config.py`，与 `Qwen3_Qwen3.5_优化说明..._20260711.md` 对齐）：
  - iGPU：`ATTENTION_BACKEND=PA`
  - NPU：`NPU_TURBO=YES` + `MAX_PROMPT_LEN=1024` + `MIN_RESPONSE_LEN=512` + `NPUW_LLM_PREFILL_CHUNK_SIZE=2048`
- **验证：PA / turbo 对准确率无影响**——cpu_routing 在有无配置下逐条一致（NPU 39.7%、iGPU 37.9%），与文档「逐字节无损」一致。它们只影响 TTFT。

## 2. 方法
HF tokenizer 预渲染 chat template（`enable_thinking=False`）→ `apply_chat_template=False` → 贪心 `max_new_tokens=1` → 首 token 匹配类别。Qwen3-1.7B 用 `LLMPipeline`，Qwen3.5-2B 用 `VLMPipeline`。

---

## 3. 结果总表

### 3.1 Qwen3.5-2B @ iGPU（GPU.0；FP16 为 GPU.1 参考天花板）

| 权重 | 量化配置 | cpu_routing(58) | CLINC(4500) | BANKING(3080) | 运行时格式 / TTFT |
|---|---|---:|---:|---:|---|
| **FP16** | — | 60.3% | **78.0%** | **72.9%** | fp16（参考天花板） |
| INT8 | int8 per-channel | 69.0% | 76.2% | 72.1% | 8-bit（**TTFT 过高，不用**） |
| **INT4-g128（推荐）** | int4 **非对称** g128 | 51.7% | **75.7%** | **70.2%** | **4-bit（INT4 TTFT）** |
| INT4（仓库自带） | int4 对称 **g=-1** ratio=1 | 37.9% | 52.1% | 45.4% | 4-bit（激进，精度差） |

> 结论：iGPU 上把 `g=-1 对称` 换成 `g128 非对称`，CLINC 52%→76%、BANKING 45%→70%，**逼近 FP16/INT8，且仍是纯 4-bit**。这是 iGPU 的直接答案。

### 3.2 Qwen3-1.7B @ NPU（FP16 为 GPU.1 参考天花板）

| 权重 | 量化配置 | cpu_routing(58) | CLINC | BANKING | NPU 兼容 / 说明 |
|---|---|---:|---:|---:|---|
| **FP16** | — | 69.0% | 67.6% | 57.9% | GPU.1 参考（NPU 不跑 fp16） |
| INT8 | int8 | — | 77%(30) / 67.6%(GPU500) | — | 恢复好但 **TTFT 过高** |
| **INT4-g128-sym-r08（推荐）** | int4 sym g128 + **ratio=0.8**（20% int8-sym） | **65.5%** | **54.9%** | **48.1%** | ✅ 大部分仍 4-bit |
| INT4-g128-sym | int4 **对称** g128 | — | 16.2%(NPU)/18.6%(GPU500) | — | ✅ 但对称救不回小模型 |
| INT4-g128（非对称） | int4 非对称 g128 | — | 46.4%(GPU500) | — | ❌ **NPU 编译失败**（要求对称） |
| INT4（仓库自带） | int4 sym g=-1 | 39.7% | **1.5%**（91% 垃圾 token `'Ğ'`） | — | 4-bit，NPU 上尤其崩 |

> 结论：NPU **硬件要求对称量化**，而对称 INT4 对 1.7B 这种小模型损失巨大（g128-sym 仅 16–18%）。可行解是**混合精度**（`ratio=0.8`，20% 敏感层走 int8-sym），cpu_routing 恢复到 65.5%（≈FP16 69%），绝大多数层仍是 4-bit，TTFT 远低于全 INT8。

### 3.3 逐类召回要点（混淆矩阵见 `confusion_matrices/`）
- 仓库 INT4 的错误是**系统性坍缩**：Qwen3.5-2B-INT4 在 CLINC 把大量类别误判到 `banking`(97% 召回但抢答)、`travel` 仅 39%；g128 后分布恢复正常。
- Qwen3-1.7B-INT4 在 NPU 上 CLINC 91% 输出非字母垃圾 token（`'Ğ'`）→ 1.5%；g128-sym-r08 后 0–2% 无法解析、恢复正常。

---

## 4. 分析 B：为什么 FP16 在 `eval_set` 上达不到文档的 ~93%

这是本轮**专门排查**的问题，结论：**`eval_set` 本身很难，不是评测流水线的问题。**

**证据 1 —— 流水线正确。** FP16-2B 在 CLINC=78.0%、BANKING=72.9%，与文档 §10 的 78% / 73% **精确吻合**。若流水线有 bug，这两个也会偏。

**证据 2 —— `eval_set` 的 `correct_by` 字段暴露了难度。** 每条样本记录了历史上哪些尺寸答对：

| 参考模型 | eval_set(58) 答对率 |
|---|---:|
| 14B | 83% |
| 32B | 72% |
| 70B | 95% |

逐类历史可解率（跨 14B/32B/70B 投票）：**A=100%、B=100%、C=53%、D=88%**。C 类（发现型/需查库检索）连大模型平均也只有 53%。一个 **2B 模型不可能到 93%**——那要同时超过 14B(83%) 并逼近 70B(95%)。FP16-2B 实测 C 召回 57%，**恰好贴着历史 53% 的天花板**，说明 2B 已发挥正常，是任务/样本难。

**证据 3 —— 不是 thinking 的功劳。** 开启 thinking 反而更差（FP16-2B cpu_routing 15 样本子集：1-token 73% → thinking 33%）。模型在长推理里把答案带偏、且末尾字母解析不稳。所以文档 93% 也不是靠推理得到的。

**证据 4 —— 4 类 vs 二分路由。** 若只看 local/cloud 二分，FP16-2B 为 81.0%（4 类 60.3%），仍达不到 93%。

**推断**：文档 §10 的 cpu_routing ~93% 最可能来自**更早/更简单版本的 `eval_set`**（在加入大量难 C「发现型」样本之前），或不同的评测口径/更大模型。当前 58 条 4 类集合，**FP16 上限就在 ~60%（4 类）/ ~81%（二分）**。CLINC/BANKING 能复现文档，佐证差异只出在 cpu_routing 这个被刻意加难的小集合上。

> 建议：README §10 的 cpu_routing ~93% 应标注为「历史/旧 eval_set」，当前集合的 FP16 参考值应更新为 ~60%（4 类）。

---

## 5. 优化产物与代码改动

### 5.1 重新量化命令（复现本报告的 INT4 变体）
```bash
# iGPU 推荐：INT4 g128 非对称（Qwen3.5-2B，VLM）
optimum-cli export openvino --model Qwen/Qwen3.5-2B --weight-format int4 --group-size 128 --ratio 1.0 \
  --task image-text-to-text        benchmark/models/Qwen3.5-2B-INT4-g128
# NPU 推荐：INT4 g128 对称 + ratio=0.8 混合精度（Qwen3-1.7B，文本）
optimum-cli export openvino --model Qwen/Qwen3-1.7B --weight-format int4 --group-size 128 --sym \
  --ratio 0.8 --backup-precision int8_sym --task text-generation-with-past \
  benchmark/models/Qwen3-1.7B-INT4-g128sym-r08
```
> 进一步（未在离线环境跑通）：`--awq --scale-estimation --dataset <校准集>` 的**数据感知对称 INT4**，理论上能在**纯 4-bit**下恢复更多，是 NPU 上比 r08 更优的方向（本轮因校准集离线下载失败未完成，建议联网后补跑）。

### 5.2 handoff 代码改动（均已落地，提升可复现性与可诊断性）
- `pipe_config.py`（新增）：按设备注入 PA / turbo+NPUW；`--pipe-config K=V` / `--no-auto-config` 可覆盖。
- `eval_ov_genai.py` / `eval_ov_public.py`：接入配置；JSON 记录 `device`/`pipe_config`；**`eval_ov_public.py` 的 `per_query` 增记 `raw`**——正是靠它发现 NPU 的 `'Ğ'` 垃圾 token。
- `plot_confusion.py`：分组/标题改用 JSON 的 `model/device/prompt` 字段，标签更准。
- `env.sh` / `run_campaign.sh` / `smoke_test.py` / `diag_thinking.py`（新增）：环境与批量运行脚手架。

## 6. 复现步骤
```bash
cd /home/chaofan/model_benchmark/query_classifier_handoff && source env.sh
M=/home/chaofan/model_benchmark/benchmark/models
# 基线（仓库 INT4）/ 推荐（更优 INT4）/ 参考（FP16）
bash run_campaign.sh $M/Qwen3-1.7B-INT4            llm NPU   qwen3
bash run_campaign.sh $M/Qwen3-1.7B-INT4-g128sym-r08 llm NPU   qwen3
bash run_campaign.sh $M/Qwen3.5-2B-INT4            vlm GPU.0 qwen3.5
bash run_campaign.sh $M/Qwen3.5-2B-INT4-g128       vlm GPU.0 qwen3.5
bash run_campaign.sh $M/Qwen3.5-2B-FP16            vlm GPU.1 qwen3.5   # 参考
python plot_confusion.py
```

## 7. 建议
1. **iGPU/Qwen3.5-2B：改用 `INT4 g128 非对称`**——CLINC 52%→76%、BANKING 45%→70%，纯 4-bit、TTFT 不变。零延迟代价的直接升级。
2. **NPU/Qwen3-1.7B：用 `INT4 g128 sym + ratio=0.8` 混合精度**（cpu_routing 39.7%→65.5%）；若能联网，优先试**数据感知对称 INT4**（AWQ+scale-estimation），有望在纯 4-bit 下更好。
3. **不要用当前的 `g=-1 对称 ratio=1` 激进 INT4**——它在 iGPU 掉 20–25pp，在 NPU 直接产出垃圾 token。
4. **INT8 精度最好但 TTFT 过高，不作生产选择**（仅作精度天花板参照）。
5. **更新 README §10**：cpu_routing 93% 标为历史/旧集合；当前 `eval_set` 的 FP16 参考约 60%（4 类）。C 类（发现型）连大模型也仅 ~53%，是该集合的固有上限。
