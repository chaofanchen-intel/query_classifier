# 量化全量对比评测报告：Qwen3-1.7B / Qwen3.5-2B × FP16 / INT8 / INT4（多设备）

- 日期：2026-07-18（准确率）/ 2026-07-20（TTFT/TPS/TPOT 性能压测）
- 环境：本机源码优化版 OpenVINO（`benchmark/openvino/install/`，`2026.2.1-…-qwen-ttft-optimizations`）+ `openvino_genai` + `qwen_env`
- 设备：`GPU.0`(iGPU, Intel Graphics) / `GPU.1`(dGPU, Arc Pro B70) / `NPU`(AI Boost)
- 覆盖：
  - **准确率**：23 个 (模型×设备×量化) 组合 × 3 任务 = **69 个评测单元**；
  - **性能（TTFT/TPOT/TPS）**：14 个 (模型×设备×量化) 组合，1k 输入/512 输出，n=10（1.7B 测 iGPU+NPU，2B 测 iGPU）；
  - 全部**严格串行**（一次只跑一个，避免相互干扰延迟）。
- 任务：中文 A/B/C/D 路由 `eval_set`(58) / CLINC150 10-domain(4500) / BANKING77-cluster10(3080)。
- 目标：① 全量对比各量化类型在各设备的准确率与延迟；② 找到「保留 INT4 级 TTFT 又能恢复准确率」的量化配方；③ 解释各量化类型与文档记录性能的差异成因；④ 给出「INT4 如何达到/接近 FP16」「如何达到/接近文档性能」的可执行结论；⑤ **综合 TTFT 与 accuracy 给出选型建议（§3.5）**。

---

## 0. 核心结论（TL;DR）

1. **仓库自带的激进 INT4（`group_size=-1` 逐通道、对称、`ratio=1.0`）是准确率崩塌的主因**，与模型/设备无关。它在 GPU 上让 2B 掉 20–25pp、让 1.7B 掉 40–48pp；在 **NPU 上直接产出 91% 垃圾 token**（CLINC 4107/4500 无法解析，准确率 1.5%）。

2. **恢复 INT4 精度的配方随设备的对称性约束而不同，且小模型更难救**：
   - **iGPU / dGPU（支持非对称）**：
     - **Qwen3.5-2B** → `INT4 g128 非对称`即可，CLINC 52%→**75.7%**（≈FP16 78%），**纯 4-bit、TTFT 不变**。AWQ 非必需。
     - **Qwen3-1.7B**（更小、更敏感）→ 仅 g128 不够（CLINC 只 44.9%），需 `INT4 g128 + AWQ`（数据感知），CLINC 44.9%→**54.0%**、BANKING 46.6%→**55.1%**，仍是纯 4-bit。
   - **NPU（硬件要求对称）**：纯对称 INT4 救不回 1.7B（g128sym 仅 CLINC 16.4%）；两条可行路线——
     - **`INT4 g128sym + AWQ`（数据感知对称，推荐大任务）**：CLINC **59.8%**（NPU 4-bit 最佳）、BANKING **57.4%**（≈FP16 57.9%）；
     - **`INT4 g128sym + ratio=0.8`（混合精度，推荐小/特异任务）**：cpu_routing **65.5%**、CLINC 54.8%、BANKING 48.2%，对 58 条小集合更稳。
     - 两者绝大多数层仍是 4-bit，TTFT ≈350ms，**远低于 INT8 的 670ms**。

3. **INT8 的延迟结论按设备分裂（标准化 1k→512 压测，见 §3.5）**：
   - **NPU 上 INT8 不可用**——TTFT 711ms、**TPOT 407ms/token、吞吐仅 2.5 tok/s**（512 token 需 ~208s）；NPU 快路径只对 INT4 权重优化，INT8 掉出快路径，是 4-bit（~18ms TPOT）的 ~20 倍。
   - **iGPU 上 INT8 可用且不慢**——TTFT 192ms（1.7B，甚至低于 INT4-g128）、TPOT ~20ms，以 FP16 级精度换 ~2.5× FP16 的 decode 速度，是 1.7B 精度优先场景的合理选择。

4. **TTFT × accuracy 综合选型（决策图 `charts/perf_tradeoff_*.png`）**：
   - **iGPU 综合最佳 = Qwen3.5-2B INT4-g128**（CLINC 75.7 ≈ FP16，TTFT 305ms@1k / ~40ms 短 prompt，66 tok/s）；
   - **iGPU 小模型**：精度优先 → 1.7B INT8（67.9%）；吞吐优先 → 1.7B INT4-g128+AWQ（54.0%，71 tok/s）；
   - **NPU**：大任务 → 1.7B g128sym+AWQ（59.8/57.4，TTFT 401/TPOT 17.9）；小任务 → g128sym r0.8（cpu 65.5）。

5. **FP16 达不到文档 cpu_routing ~93% 是 `eval_set` 被加难所致，不是流水线问题**：FP16 在 CLINC(78.0%)/BANKING(72.9%) 上**精确复现**文档，证明链路正确；cpu_routing 的 `correct_by` 显示连 14B 也仅 83%、C「发现型」类历史可解率仅 53%，2B 不可能到 93%。

6. **NPU「很慢」的原因**：静态流水线把每次 prefill 补齐到 `MAX_PROMPT_LEN`；且 decode 对 INT8 极慢（见结论 3）。准确率评测（1-token）逐条 p50——激进 INT4 159ms、可用 4-bit ~330–350ms、INT8 670ms，4500 条 ≈ 30 分钟。用户记忆中的「~200ms」正是**激进 INT4（快但输出垃圾）**。

---

## 1. 操作步骤（可复现）

### 1.1 环境初始化（`env.sh`）
```bash
source /home/chaofan/envs/qwen_env/bin/activate
source /home/chaofan/model_benchmark/benchmark/openvino/install/setupvars.sh
export PYTHONPATH="/home/chaofan/model_benchmark/benchmark/openvino/install/python:$PYTHONPATH"
```

### 1.2 设备配置自动注入（`pipe_config.py`，与统一优化说明文档对齐）
- **iGPU/dGPU**：`ATTENTION_BACKEND=PA`
- **NPU**：`NPU_TURBO=YES` + `MAX_PROMPT_LEN` + `MIN_RESPONSE_LEN=512` + `NPUW_LLM_PREFILL_CHUNK_SIZE=2048`
- 验证：PA / turbo **对准确率无影响**（逐条一致，与文档「逐字节无损」一致），只影响 TTFT。
- NPU 加速：prompt 均 ≤521 token，故把 `MAX_PROMPT_LEN` 从 1024 收到 **640**（覆盖全部样本），NPU 逐条延迟 ~减半。

### 1.3 量化命令（复现各 INT4 变体）
```bash
M=/home/chaofan/model_benchmark/benchmark/models
# —— iGPU/dGPU 用（非对称）——
# 2B：g128 非对称即可达标
optimum-cli export openvino --model Qwen/Qwen3.5-2B --weight-format int4 --group-size 128 --ratio 1.0 \
  --task image-text-to-text  $M/Qwen3.5-2B-INT4-g128
# 1.7B：g128 + AWQ（数据感知，通用校准集 wikitext-2）
python requant_aware.py --model-id Qwen/Qwen3-1.7B --ref-ir $M/Qwen3-1.7B-FP16 \
  --out $M/Qwen3-1.7B-INT4-g128-aware --group-size 128 --ratio 1.0 --num-samples 128

# —— NPU 用（硬件要求对称）——
# 混合精度：20% 敏感层走 int8_sym
optimum-cli export openvino --model Qwen/Qwen3-1.7B --weight-format int4 --group-size 128 --sym \
  --ratio 0.8 --backup-precision int8_sym --task text-generation-with-past \
  $M/Qwen3-1.7B-INT4-g128sym-r08
# 数据感知对称：AWQ + scale-estimation（通用校准集）
python requant_aware.py --model-id Qwen/Qwen3-1.7B --ref-ir $M/Qwen3-1.7B-FP16 \
  --out $M/Qwen3-1.7B-INT4-g128sym-aware --sym --group-size 128 --ratio 1.0 --num-samples 128
```
> **校准集说明**：AWQ / scale-estimation 使用**通用语料 wikitext-2**（`data/wikitext2_calib.txt`，经 hf-mirror 拉取），**不使用本任务自身提示词**，以保证量化的通用性而非过拟合到评测集。

### 1.4 批量评测（严格串行）
```bash
cd /home/chaofan/model_benchmark/query_classifier_handoff && source env.sh
bash run_all_sequential.sh        # 完整 23 组合矩阵，一次只跑一个
# 单组合：run_campaign.sh <模型目录> <llm|vlm> <设备> <qwen3|qwen3.5>
bash run_campaign.sh $M/Qwen3-1.7B-INT4-g128sym-aware llm NPU qwen3
```
评测方法：HF tokenizer 预渲染 chat template（`enable_thinking=False`）→ `apply_chat_template=False` → 贪心 `max_new_tokens=1` → 首 token 匹配类别。1.7B 用 `LLMPipeline`，2B 用 `VLMPipeline`。

### 1.5 出图与逐类分析
```bash
python make_charts.py       # -> charts/acc_*.png, charts/lat_*.png, charts/summary_table.md
python plot_confusion.py    # -> confusion_matrices/*.png + 逐类召回表
```

---

## 2. 测试结果（全量矩阵）

> `cpu(58)` 为 58 条小集合、方差大仅供参考；CLINC(4500)/BANKING(3080) 为主结论依据。`p50` 为暖态逐条延迟（`max_new_tokens=1`，近似 TTFT）。`unpars` = CLINC 上无法解析成类别的输出数（越大表示模型输出越乱）。

### 2.1 Qwen3-1.7B（LLM）

| 设备 | 量化 | cpu(58) | CLINC(4500) | BANKING(3080) | p50 ms | unpars(clinc) | 运行时 |
|---|---|---:|---:|---:|---:|---:|---|
| **iGPU** | FP16 | 69.0 | **67.6** | **57.9** | 52 | 0 | fp16 参考 |
| iGPU | INT8 | 70.7 | 67.9 | 57.2 | 30 | 0 | 8-bit（TTFT 高） |
| iGPU | INT4 激进(g=-1) | 34.5 | 19.9 | 16.8 | 20 | 16 | 4-bit（崩） |
| iGPU | INT4 g128(非对称) | 46.6 | 44.9 | 46.6 | 23 | 1 | 4-bit |
| **iGPU** | **INT4 g128+AWQ(非对称) ✅** | 46.6 | **54.0** | **55.1** | 23 | 0 | **4-bit（推荐）** |
| **dGPU** | FP16 | 69.0 | 67.6 | 57.9 | **10** | 0 | fp16 参考 |
| dGPU | INT8 | 70.7 | 67.9 | 57.2 | 13 | 0 | 8-bit |
| dGPU | INT4 激进(g=-1) | 34.5 | 19.9 | 16.8 | 12 | 15 | 4-bit（崩） |
| dGPU | INT4 g128(非对称) | 46.6 | 44.8 | 46.6 | 12 | 0 | 4-bit |
| **dGPU** | **INT4 g128+AWQ(非对称) ✅** | 46.6 | 54.1 | 55.1 | 12 | 0 | **4-bit（推荐）** |
| NPU | INT8 | 70.7 | 67.3 | 58.3 | 670 | 0 | 8-bit（TTFT 最高） |
| NPU | INT4 激进(g=-1) | 39.7 | **1.5** | 12.6 | 159 | **4107** | 4-bit（垃圾 token） |
| NPU | INT4 g128sym | 56.9 | 16.4 | 29.6 | 350 | 213 | 4-bit（对称救不回） |
| **NPU** | **INT4 g128sym r0.8（混合）✅** | **65.5** | 54.8 | 48.2 | 330 | 12 | **多为 4-bit（小任务稳）** |
| **NPU** | **INT4 g128sym+AWQ（数据感知）✅** | 36.2 | **59.8** | **57.4** | 352 | 1 | **4-bit（大任务最佳）** |

### 2.2 Qwen3.5-2B（VLM）

| 设备 | 量化 | cpu(58) | CLINC(4500) | BANKING(3080) | p50 ms | unpars(clinc) | 运行时 |
|---|---|---:|---:|---:|---:|---:|---|
| **iGPU** | FP16 | 60.3 | **78.0** | **72.9** | 70 | 0 | fp16 参考 |
| iGPU | INT8 | 69.0 | 76.2 | 72.1 | 41 | 0 | 8-bit |
| iGPU | INT4 激进(g=-1) | 37.9 | 52.1 | 45.4 | 35 | 2 | 4-bit（崩） |
| **iGPU** | **INT4 g128(非对称) ✅** | 51.7 | **75.7** | **70.2** | 40 | 0 | **4-bit（推荐）** |
| **dGPU** | FP16 | 60.3 | 78.0 | 72.9 | **21** | 0 | fp16 参考 |
| dGPU | INT8 | 67.2 | 76.0 | 72.2 | 28 | 0 | 8-bit |
| dGPU | INT4 激进(g=-1) | 37.9 | 52.8 | 45.9 | 25 | 0 | 4-bit（崩） |
| **dGPU** | **INT4 g128(非对称) ✅** | 51.7 | 75.6 | 70.3 | 28 | 0 | **4-bit（推荐）** |

> 注：Qwen3.5-2B 因硬件档期未跑 NPU（仅 1.7B 在 NPU）；2B 用 `image-text-to-text`（VLM）导出。

---

## 3. 延迟分析：为什么 NPU「很慢」

- **本质是静态流水线**：NPU 用固定形状图，每次 prefill 补齐到 `MAX_PROMPT_LEN`，无论真实 prompt 多长。故逐条延迟由 `MAX_PROMPT_LEN` 决定，而非样本长度。
- **逐条 p50（1-token，近似 TTFT）**：
  - dGPU 最快（10–28ms）> iGPU（20–70ms）≫ NPU（159–670ms）。
  - NPU 内部：激进 INT4 **159ms** < 可用 4-bit（g128sym/r0.8/AWQ）**~330–352ms** < INT8 **670ms**。
- **「30 分钟」从何而来**：4500 条 × ~350–670ms ≈ 26–50 分钟。用户记忆的「~200ms」对应**激进 INT4（159ms）**——它快，但输出 91% 垃圾 token 不可用；可用的对称 4-bit 因分组/混合精度反量化更重，约 350ms。
- **INT8 vs 4-bit 的 TTFT 差**：NPU 上 INT8（670ms）是可用 4-bit（~350ms）的 ~1.9 倍 —— 这正是本项目不能用 INT8、必须把 4-bit 救回来的根本原因。
- 加速手段：把 `MAX_PROMPT_LEN` 1024→640（覆盖全部样本）即可 ~减半 NPU 逐条延迟，不影响准确率。

> 注：§3 的 p50 是**准确率评测**里的 1-token 暖态延迟（短 prompt ≤521 token），近似分类器真实负载（生成 1 token）。下面 §3.5 是**标准化性能压测**（1k 输入 / 512 输出），用于全面刻画 TTFT/TPOT/TPS。

---

## 3.5 标准化性能测试：TTFT / TPOT / TPS（1k 输入 → 512 输出）

用 `model_bench`（C++ `ov::genai` 压测套件）标准化测量：**1k 输入 / 512 输出，10 次测量迭代（+1 预热），prompt-permutation 开启**；**iGPU 用 PA（`gpu_pa.json`：`PERFORMANCE_HINT=LATENCY` + `ATTENTION_BACKEND=PA`），NPU 用 best-perf turbo（`npu_best_perf_turbo.json`：`NPU_TURBO=YES` + `BEST_PERF` + `MAX_PROMPT_LEN=1024`）**。全部**严格串行**（一次一个 + 30s 冷却，iGPU/NPU 共用 SoC 封装需散热）。指标：TTFT=首 token 时间（prefill）、TPOT=每输出 token 时间（decode）、TPS=吞吐。命令示例见 `run_perf.sh`。

> 口径提示：本项目分类器实际只生成 **1 个 token**，故**决策看 TTFT**；TPOT/TPS 反映长文本生成能力，供全面参考。TTFT 在 1k 长 prompt（本压测）与短 prompt（§3 的 ≤521 token）下**排序会变**，见下「关键结论 5」。

### 3.5.1 结果总表（TTFT/TPOT/TPS + 对齐的 accuracy）

**Qwen3-1.7B @ iGPU（GPU.0）**

| 量化 | TTFT ms | TTFT p90 | TPOT ms | TPS | Decode TPS | CLINC | BANK | cpu |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FP16 | 275 | 277 | 49.6 | 20.0 | 20.2 | 67.6 | 57.9 | 69.0 |
| **INT8** | **192** | 193 | 19.7 | 49.9 | 50.8 | **67.9** | 57.2 | 70.7 |
| INT4 激进(g=-1) | 175 | 178 | 13.1 | 74.4 | 76.2 | 19.9 | 16.8 | 34.5 |
| INT4 g128 | 214 | 217 | 13.6 | 71.2 | 73.3 | 44.9 | 46.6 | 46.6 |
| **INT4 g128+AWQ ✅** | 215 | 217 | **13.6** | **71.5** | 73.6 | 54.0 | 55.1 | 46.6 |

**Qwen3-1.7B @ NPU（best-perf turbo）**

| 量化 | TTFT ms | TTFT p90 | TPOT ms | TPS | Decode TPS | CLINC | BANK | cpu |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **INT8** | **711** | 713 | **407.1** | **2.5** | 2.5 | 67.3 | 58.3 | 70.7 |
| INT4 激进(g=-1) | 264 | 264 | 14.6 | 66.3 | 68.6 | 1.5 | 12.6 | 39.7 |
| INT4 g128sym | 400 | 400 | 18.0 | 53.4 | 55.6 | 16.4 | 29.6 | 56.9 |
| **INT4 g128sym r0.8 ✅** | 413 | 415 | 19.5 | 49.4 | 51.4 | 54.8 | 48.2 | **65.5** |
| **INT4 g128sym+AWQ ✅** | **401** | 403 | **17.9** | **53.5** | 55.7 | **59.8** | **57.4** | 36.2 |

**Qwen3.5-2B @ iGPU（GPU.0）**

| 量化 | TTFT ms | TTFT p90 | TPOT ms | TPS | Decode TPS | CLINC | BANK | cpu |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FP16 | 356 | 358 | 50.2 | 19.7 | 19.9 | 78.0 | 72.9 | 60.3 |
| INT8 | 281 | 283 | 20.8 | 47.0 | 48.1 | 76.2 | 72.1 | 69.0 |
| INT4 激进(g=-1) | 263 | 267 | 14.2 | 68.2 | 70.6 | 52.1 | 45.4 | 37.9 |
| **INT4 g128 ✅** | 305 | 310 | **14.7** | **65.7** | 68.2 | **75.7** | 70.2 | 51.7 |

### 3.5.2 关键结论

1. **NPU 上 INT8 被 decode 判了死刑，不只是 prefill 慢**：TPOT **407ms/token（2.5 tok/s）**，是 NPU 可用 4-bit（~18ms）的 **~20 倍**；512 token 生成需 **~208 秒**，TTFT 也最高（711ms）。原因：NPU 的快路径为 **INT4 权重解压**专门优化，INT8 权重掉出快路径。**这是「NPU 不能用 INT8」的量化铁证**（此前只有 1-token 代理 670ms，现看到 decode 崩塌）。

2. **NPU 可用 4-bit ≈ TTFT 400ms / TPOT 18ms / 53 tok/s**。其中 **`g128sym+AWQ` 在精度和速度上双双最优**（CLINC 59.8 / BANK 57.4，TTFT 401 / TPOT 17.9），全面压过 r0.8（413 / 19.5）——**唯一例外是 58 条的 cpu_routing**（AWQ 通用校准把小任务边界带偏，见 §5.8），该任务选 r0.8（65.5）。激进 INT4 最快（264ms）但输出垃圾（CLINC 1.5）。

3. **iGPU 上 INT8 并不慢**：TTFT 192ms（1.7B）/281ms（2B），甚至**低于 INT4-g128 的 TTFT**，TPOT 仅 ~20ms。**「INT8 太慢」是 NPU 专属结论，不适用于 iGPU**——在 iGPU 上 INT8 以 FP16 级精度换来 ~2.5× FP16 的 decode 速度。

4. **iGPU 上 TTFT 与 TPOT 分道扬镳**：1k prefill 时 INT8 的 TTFT < INT4-g128（prefill 是**计算密集**，INT8 反量化更简单）；但 INT4-g128 的 TPOT 远低（decode 是**访存密集**，INT4 权重更小）。故**长文本生成看吞吐 → INT4 胜；长 prompt 首字延迟 → INT8 胜**。

5. **TTFT 排序随 prompt 长度反转**（对「用 TTFT 选型」很关键）：
   - 短 prompt（分类器真实负载 ≤521 token，见 §3 的 1-token p50）：1.7B iGPU **INT4-g128 ≈23ms < INT8 30ms < FP16 52ms** —— INT4 首字更快。
   - 长 prompt（1k，本压测）：**INT8 192 < INT4-g128 214 < FP16 275** —— INT8 首字更快。
   - 交叉点成因：短 prompt 偏访存（INT4 小权重占优），长 prompt 偏计算（INT8 简单反量化占优）。

### 3.5.3 图表

- **TTFT / TPOT / 吞吐 分组柱**（1k→512）：
  - ![Qwen3-1.7B 性能](../charts/perf_ttft_Qwen3-1.7B.png)
  - ![Qwen3.5-2B 性能](../charts/perf_ttft_Qwen3.5-2B.png)
  - 读图：NPU 的 INT8 TTFT/TPOT 柱**爆表**（407ms TPOT）；iGPU 上 INT4 的 TPOT 柱最矮、吞吐柱最高。
- **TTFT × accuracy 权衡散点（决策图，越靠左上越好）**：
  - ![Qwen3-1.7B 权衡](../charts/perf_tradeoff_Qwen3-1.7B.png)
  - ![Qwen3.5-2B 权衡](../charts/perf_tradeoff_Qwen3.5-2B.png)
  - 读图：**2B INT4-g128 稳居左上**（CLINC 75.7 ≈ FP16，TTFT 305 < FP16 356）；1.7B iGPU 上 INT8 与 INT4-g128+AWQ 形成「精度 vs 吞吐」的选择；NPU 三点整体靠右（TTFT 高），AWQ/r0.8 明显高于 g128sym。

### 3.5.4 综合决策（TTFT + accuracy 一起看）

| 场景 | 推荐 | 依据 |
|---|---|---|
| **iGPU · 综合最佳** | **Qwen3.5-2B INT4-g128** | CLINC 75.7%（≈FP16 78），TTFT 305ms@1k（短 prompt ~40ms），TPOT 14.7ms/66 tok/s——散点左上角 |
| **iGPU · 小模型 · 精度优先** | **Qwen3-1.7B INT8** | CLINC 67.9%（=FP16），TTFT 192ms（iGPU 上最低），TPOT 19.7ms。iGPU 上 INT8 不慢 |
| **iGPU · 小模型 · 吞吐/长文本优先** | **Qwen3-1.7B INT4-g128+AWQ** | TPOT 13.6ms/71 tok/s（最高吞吐），CLINC 54.0%（纯 4-bit 上限） |
| **NPU · 大规模分类** | **Qwen3-1.7B INT4 g128sym+AWQ** | CLINC 59.8/BANK 57.4，TTFT 401ms、TPOT 17.9ms（NPU 4-bit 最优）。**绝不用 INT8**（TTFT 711/TPOT 407） |
| **NPU · 小/特异任务** | **Qwen3-1.7B INT4 g128sym r0.8** | cpu_routing 65.5%（AWQ 会把小任务带偏），TTFT 413ms |

> 对**分类器**（1 token 输出）而言：延迟即 TTFT。短 prompt 下 4-bit 首字更快、1k prompt 下 INT8 首字更快，两者都够用；**最终由精度定夺**——iGPU 首选 2B INT4-g128；小模型精度优先选 1.7B INT8，吞吐优先选 1.7B INT4-g128+AWQ；NPU 选 1.7B g128sym+AWQ。

---

## 4. 图表分析

### 4.1 准确率对比（量化 × 任务 × 设备）
- ![Qwen3-1.7B 准确率](../charts/acc_Qwen3-1.7B.png)
- ![Qwen3.5-2B 准确率](../charts/acc_Qwen3.5-2B.png)

读图要点：
- **激进 INT4 柱最矮**（NPU 上 CLINC 几乎贴地——垃圾 token），是所有配置里最差的一档。
- **g128 系列显著抬升**；2B 的 g128 柱几乎顶到 FP16 虚线（CLINC），1.7B 的 g128 柱明显还有缺口，**加 AWQ 后再抬一截**。
- NPU 三根可用 4-bit 柱（g128sym / r0.8 / AWQ）里，**AWQ 在 CLINC/BANKING 最高**，但 **cpu_routing 上 AWQ 反而塌**（见 §5.8 成因）。

### 4.2 延迟对比（CLINC 暖态 p50）
- ![Qwen3-1.7B 延迟](../charts/lat_Qwen3-1.7B.png)
- ![Qwen3.5-2B 延迟](../charts/lat_Qwen3.5-2B.png)

读图要点：NPU 柱远高于 GPU；NPU 内部 INT8 柱约为 4-bit 柱的两倍，直观印证「INT8 TTFT 过高」。

### 4.3 逐类混淆矩阵
- 全部 69 张见 [`confusion_matrices/`](../confusion_matrices/)。关键对比（CLINC）：
  - `cm_pub_clinc_Qwen3-1.7B-INT4_NPU_*.png`：激进 INT4 几乎全部误判/垃圾。
  - `cm_pub_clinc_Qwen3-1.7B-INT4-g128sym-aware_NPU_*.png`：AWQ 后对角线恢复。
  - `cm_pub_clinc_Qwen3.5-2B-INT4-g128_*.png`：2B g128 对角线接近 FP16。

---

## 5. 逐量化类型分析：与文档/FP16 差异的成因

### 5.1 FP16（参考天花板）
CLINC 78.0%(2B)/67.6%(1.7B)、BANKING 72.9%/57.9% —— **CLINC/BANKING 精确复现历史文档**（≈78/≈73），证明评测链路无误。唯一「差异」在 cpu_routing（见 §9）。

### 5.2 INT8 —— 精度对标 FP16，延迟「NPU 崩、iGPU 可用」
各任务与 FP16 相差 ≤2pp（个别更高，如 2B cpu 60.3→69.0 属小集合方差）。**延迟结论按设备分裂**（详见 §3.5）：
- **NPU：INT8 不可用**——TTFT 711ms、**TPOT 407ms/token（吞吐仅 2.5 tok/s）**，512 token 需 ~208 秒。NPU 快路径只对 INT4 权重优化，INT8 掉出快路径。
- **iGPU：INT8 可用且不慢**——TTFT 192ms（1.7B，甚至低于 INT4-g128）、TPOT ~20ms。以 FP16 级精度换 ~2.5× FP16 的 decode 速度。
- 故 INT8 在 **iGPU 上是 1.7B 精度优先场景的合理生产选择**；只有在 **NPU 上才因延迟被否决**。

### 5.3 INT4 激进（`g=-1` 对称 ratio=1，仓库默认）—— 崩塌主因
- 逐通道单一 scale + 无零点 + 无离群值处理：一个通道内的大幅值权重迫使 scale 变粗，绝大多数权重被压成同一档 → 信息坍缩。
- **小模型更惨**（冗余少）：2B CLINC 掉到 52%，1.7B 掉到 19.9%（GPU）/1.5%（NPU，91% 垃圾 token）。
- 结论：这是「性能远低于文档」的**直接元凶**，务必弃用。

### 5.4 INT4 g128（非对称，GPU）—— 2B 达标、1.7B 不够
- 每 128 权重一组独立 scale+zero-point → 恢复动态范围。
- **2B**：CLINC 52%→**75.7%**、BANKING 45%→**70.2%**，≈FP16、纯 4-bit。**2B 到此即可**。
- **1.7B**：CLINC 只到 44.9% —— 小模型仅靠分组还差 ~23pp，需再上 AWQ。

### 5.5 INT4 g128+AWQ（非对称，数据感知，1.7B GPU）—— 小模型的关键补丁
- AWQ 按激活重要性缩放权重 + scale-estimation 细调 → 保护重要通道。
- **1.7B**：CLINC 44.9%→**54.0%**、BANKING 46.6%→**55.1%**（+9/+8.5pp），**纯 4-bit**。补回了「g128 之后仍存在的一半缺口」。
- cpu_routing 未变（46.6%）：通用语料校准对这个 58 条、口径特异的小任务无增益。

### 5.6 INT4 g128sym（对称，NPU）—— 对称救不回小模型
- NPU 硬件要求对称（非对称在 NPU 编译失败 `ZE_RESULT_ERROR_INVALID_ARGUMENT`）。
- 去掉零点后，1.7B 的偏斜权重分布无法重新居中：CLINC 仅 16.4%、BANKING 29.6%（还有 213 条无法解析）。**不可单独用**。

### 5.7 INT4 g128sym + ratio=0.8（混合精度，NPU）—— 稳健路线
- 20% 最敏感层回退 int8_sym，其余仍 4-bit。CLINC 54.8%、BANKING 48.2%、**cpu_routing 65.5%（≈FP16 69%）**。
- 对**小/特异任务（cpu_routing）最稳**，且绝大多数层是 4-bit，TTFT ~330ms。

### 5.8 INT4 g128sym + AWQ（数据感知对称，NPU）—— 大任务最佳，但小任务需注意
- CLINC **59.8%**（NPU 4-bit 之最）、BANKING **57.4%（≈FP16 57.9%，几乎无损）**，纯 4-bit。
- **但 cpu_routing 塌到 36.2%**：AWQ 用通用 wikitext 校准，优化的是「平均下一 token 保真度」，利好 CLINC/BANKING 这类大而多样的任务；却把 58 条、决策边界特异的 4 类小任务的边界带偏（混淆矩阵显示 A 类召回被压到 0%、集中误判到 B）。
- **取舍**：面向大规模真实分类（CLINC/BANKING 口径）→ 选 AWQ；面向 cpu_routing 这类小而特异集合 → 选 r0.8。

---

## 6. 结论一：INT4 要达到 / 接近 FP16，应如何量化

| 模型 / 设备 | 达到/接近 FP16 的 INT4 配方 | 效果 | 是否纯 4-bit |
|---|---|---|---|
| **Qwen3.5-2B / iGPU·dGPU** | `INT4 g128 非对称` | CLINC 75.7 vs FP16 78（−2.3pp），BANKING 70.2 vs 72.9 | ✅ 纯 4-bit |
| **Qwen3-1.7B / iGPU·dGPU** | `INT4 g128 + AWQ 非对称` | CLINC 54.0 vs 67.6，BANKING 55.1 vs 57.9（BANKING 近平） | ✅ 纯 4-bit |
| **Qwen3-1.7B / NPU（大任务）** | `INT4 g128sym + AWQ` | CLINC 59.8 vs 67.6，**BANKING 57.4 vs 57.9（几乎无损）** | ✅ 纯 4-bit |
| **Qwen3-1.7B / NPU（小/特异任务）** | `INT4 g128sym + ratio=0.8` | **cpu_routing 65.5 vs 69.0** | 多为 4-bit |

**要点**：
- 模型越大越容易靠 4-bit 恢复：**2B 只需 g128**；**1.7B 必须叠 AWQ / 混合精度**。
- **纯 4-bit 无法在 1.7B 的 CLINC 上完全追平 FP16**（最好 54 vs 67.6，差 ~13pp）；BANKING 可近平。若要 CLINC 也追平，只有 INT8（67.9，但 TTFT 高）或换 2B。
- 决定性动作永远是**先把 `g=-1` 换成 `g128`**；这一步就拿回大部分精度且 TTFT 不变。

---

## 7. 结论二：要达到 / 接近文档记录的性能，应怎么做

1. **CLINC / BANKING 的文档值已达成**：FP16 精确复现（78/73），**2B 的 INT4-g128 也已逼近**（75.7/70.2）。生产用 2B + INT4-g128 即可对标文档且享 INT4 TTFT。
2. **cpu_routing 的历史 ~93% 在当前 `eval_set` 上不可达**（见 §9）——这是数据集被加难，而非量化或流水线问题。当前集合上：
   - 最佳实测为 **INT8/1.7B 的 70.7%**、FP16/1.7B 69.0%；2B FP16 60.3%。
   - 若必须提高该指标：优先**换更大模型**或**放宽到 local/cloud 二分**（2B 二分 81%），并把文档 93% 标注为「历史/旧集合」。
3. **设备选择**：追求最低 TTFT 用 **dGPU（10–28ms）**；NPU 适合功耗受限场景但需对称量化 + 上述配方；iGPU 居中。
4. **不要用**仓库默认的激进 INT4——它是文档差距的主因。

---

## 8. FP16 为何达不到文档 cpu_routing ~93%（诊断复述）

**结论：`eval_set` 本身很难，不是流水线问题。**
- **证据 1（链路正确）**：FP16-2B 在 CLINC=78.0%、BANKING=72.9%，与文档 78/73 精确吻合。
- **证据 2（数据集变难）**：`eval_set` 的 `correct_by` 字段显示历史答对率——14B=83%、32B=72%、70B=95%；逐类可解率 A=100/B=100/**C=53**/D=88。C「发现型」类连大模型平均也只 53%，2B 不可能到 93%（要同时超 14B 并逼近 70B）。FP16-2B 实测 C 召回 57%，**恰贴历史 53% 天花板**。
- **证据 3（非 thinking 之功）**：开 thinking 反而更差（1-token 73%→thinking 33%）。
- **证据 4（二分口径）**：只看 local/cloud 二分，FP16-2B 为 81.0%，仍达不到 93%。
- **推断**：文档 93% 最可能来自更早/更简单版本的 `eval_set` 或不同口径/更大模型。当前 58 条 4 类集合 FP16 上限约 60%(4 类)/81%(二分)。

---

## 9. 建议

1. **iGPU/dGPU + Qwen3.5-2B**：用 `INT4 g128 非对称`——零延迟代价逼近 FP16，直接升级。
2. **iGPU/dGPU + Qwen3-1.7B**：用 `INT4 g128 + AWQ 非对称`——小模型的最佳纯 4-bit。
3. **NPU + Qwen3-1.7B**：大规模分类用 `g128sym + AWQ`（BANKING 近无损）；小/特异任务用 `g128sym + ratio=0.8`（cpu_routing 最稳）。可两者都导出，按任务选。
4. **弃用**仓库默认 `INT4 g=-1 对称 ratio=1`（GPU 掉 20–48pp，NPU 出垃圾 token）。
5. **INT8 分设备用**：**NPU 上禁用**（TTFT 711/TPOT 407ms，见 §3.5）；**iGPU 上可用**——1.7B 精度优先时 INT8（CLINC 67.9、TTFT 192、TPOT 20ms）优于纯 4-bit。
6. **NPU 提速**：`MAX_PROMPT_LEN` 按实际最长 prompt 收紧（分类器 1024→640），逐条延迟 ~减半，准确率不变。
7. **AWQ/scale-estimation 校准用通用语料**（wikitext-2），不用任务自身提示词——更通用；但注意对极小/特异任务（cpu_routing）可能带偏边界，此类任务改用混合精度更稳。
8. **性能压测复现**：`bash run_perf.sh`（1k→512，n=10，串行 + 30s 冷却，GPU=PA / NPU=best-perf turbo）；出图 `python make_perf_charts.py`。TTFT×accuracy 决策图见 `charts/perf_tradeoff_*.png`。

---

## 附录：产物与文件
- 准确率结果 JSON：`results/*.json`（69 单元，含逐条 gold/pred/raw、p50/p95）
- **性能压测结果**：`perf_results/perf.json`（14 单元 TTFT/TPOT/TPS 全分位）、`perf_results/perf.csv`、`perf_results/logs/*.log`
- 准确率图表：`charts/acc_*.png`、`charts/lat_*.png`、`charts/summary_table.md`
- **性能图表**：`charts/perf_ttft_*.png`（TTFT/TPOT/吞吐）、`charts/perf_tradeoff_*.png`（TTFT×accuracy 决策散点）、`charts/perf_summary.md`
- 混淆矩阵：`confusion_matrices/*.png`（69 张）+ 逐类召回表（`plot_confusion.py` stdout）
- 脚本：`run_all_sequential.sh`（准确率全量串行）、`run_campaign.sh`（单组合）、**`run_perf.sh`（性能压测串行）**、`requant_aware.py`（数据感知量化）、`make_charts.py`、**`make_perf_charts.py`**、`plot_confusion.py`、`pipe_config.py`（设备配置）
- 性能配置：`benchmark/tests/repro_rerun_20260710/gpu_pa.json`（iGPU PA）、`npu_best_perf_turbo.json`（NPU turbo）；提示词 `model_bench/prompts/qwen3_1k.jsonl` / `qwen3.5_1k.jsonl`
- 校准集：`data/wikitext2_calib.txt`（通用 wikitext-2，256 条）
