# Query Classifier —— 评估复现包（handoff）

用 **OpenVINO + LLM** 做「查询分类」（query routing / intent classification）的离线评估。
本目录是核心复现子集，只含跑通评估所需的脚本、数据集和 prompt。

> 数据集已内置在 `data/`。模型是外部依赖，**由你替换成自己的模型**（见第 5 节）。

---

## 1. 目录结构

```
query_classifier_handoff/
├── README.md
├── eval_ov_genai.py         # 中文 A/B/C/D 路由分类（openvino_genai 生成）
├── eval_ov_public.py        # CLINC150 / BANKING-cluster10（openvino_genai 生成）
├── plot_confusion.py        # 读 results/ 生成混淆矩阵 PNG + 汇总表
├── data/                    # 数据集（内置，见第 4 节）
├── prompts/                 # system prompt（见第 6 节）
├── results/                 # 运行后自动生成的 JSON 结果（脚本自动创建）
└── confusion_matrices/      # plot_confusion.py 输出的 PNG（脚本自动创建）
```

## 2. 评估方式

用 `openvino_genai`（LLMPipeline / VLMPipeline）实际生成文本，再做字符串匹配得到预测类别，
统计准确率与延迟。这是接近生产的推理路径。

## 3. 构建运行环境

代码会被拷到你自己的机器上运行。需要一个装了 **OpenVINO 2026.x + openvino_genai + optimum-intel** 的 Python 环境。
下面是本包实测通过的版本（conda + pip，Python 3.10+）：

```bash
conda create -n qc-eval python=3.10 -y
conda activate qc-eval

pip install \
  "openvino==2026.2.0" \
  "openvino-genai==2026.2.0.0" \
  "optimum-intel[openvino]" \
  transformers torch numpy \
  matplotlib seaborn          # 仅 plot_confusion.py 需要
```

说明：
- 两个 `eval_*.py` 只依赖 `openvino_genai` / `transformers`（`optimum-intel` 用于导出模型，见第 5 节）。
- `plot_confusion.py` 额外需要 `matplotlib` + `seaborn`。
- 实测版本组合：OV 2026.2.0 / openvino-genai 2026.2.0.0 / optimum-intel 1.27 / transformers 5.2 / torch 2.12。其它相近版本一般也可，以能 `import openvino_genai` 为准。

之后所有命令里的 `python` 都指这个环境的解释器。

## 4. 数据集（本目录 `data/`，无需外部路径）

| 文件 | 内容 | 条数 | gold 字段 |
|---|---|---|---|
| `eval_set.jsonl` | 中文查询，4 类 A/B/C/D（A/B→local，C/D→cloud 路由） | 58 | `category` |
| `clinc150_domain_full.jsonl` | CLINC150 10-domain | 4500 | `domain` |
| `banking77_cluster10_full.jsonl` | BANKING77 经 Ward 聚类成 10 类（labels A-J，gold 内联 `cluster` 字段） | 3080 | `cluster` |

## 5. 在哪里替换成你自己的模型

脚本不绑定任何具体模型，通过两个参数注入：

| 参数 | 含义 |
|---|---|
| `--ov-model` | **OpenVINO IR 目录**（转换后的模型，含 `openvino_model.xml/.bin`） |
| `--hf-model` | **HuggingFace 模型目录**，只用来加载 tokenizer + chat template |
| `--pipeline` | 模型类型：文本 LLM 用 `llm`；多模态（如 Qwen3.5 VLM 架构）用 `vlm` |
| `--device` | `GPU.1`（Intel Arc dGPU）或 `CPU` |

如果你手上只有 HF 权重、还没有 OV IR，用 `optimum-cli` 转一次（装了 `optimum-intel[openvino]` 后可用）：

```bash
# 普通文本 LLM（对应 --pipeline llm）
optimum-cli export openvino \
  --model /path/to/your/hf-model \
  --weight-format fp16 \
  --task text-generation-with-past \
  /path/to/output/OV-IR-FP16
```

要点：
- `--hf-model` 指向的目录要能被 `AutoTokenizer.from_pretrained` 加载，并支持 `apply_chat_template(..., enable_thinking=False)`（Qwen 系列支持；换其它模型时确认 chat template 兼容）。
- **绝大多数模型用 `--pipeline llm`**；`vlm` 仅用于多模态架构的模型。

下文命令里的 `--ov-model` / `--hf-model` 路径请替换成你自己的。

## 6. Prompt（本目录 `prompts/`）

全部为 **zero-shot**（无 few-shot 示例）。命名规则 `{任务}_{调优所用模型}`：

| 文件 | 任务 | 面向模型 |
|---|---|---|
| `cpu_routing_qwen3.txt` / `cpu_routing_qwen3.5.txt` | 中文 A/B/C/D 路由 | 分别为 Qwen3 / Qwen3.5 调优 |
| `clinc_domain_qwen3.txt` / `clinc_domain_qwen3.5.txt` | CLINC150 10-domain（letter A-J） | 同上 |
| `banking_cluster10_qwen3.txt` / `banking_cluster10_qwen3.5.txt` | BANKING cluster10（letter A-J） | 同上 |

`_qwen3.5` 版是后续针对 Qwen3.5 调优、准确率更高的一版；`_qwen3` 版是较早的基线。
换成你自己的模型时，建议先用 `_qwen3.5` 版作起点，再按需微调（prompt 对准确率影响很大）。

## 7. 怎么运行

下面命令中的模型路径为占位符，替换成你的模型。`python` 指第 3 节的环境。

### 7.1 中文 A/B/C/D 路由 —— `eval_ov_genai.py`

```bash
python eval_ov_genai.py \
  --ov-model /path/to/OV-IR-FP16 \
  --hf-model /path/to/hf-model \
  --pipeline vlm --device GPU.1 --prompt cpu_routing_qwen3.5
```
输出：准确率、under-routing（C/D 被误判成 local 的漏判数）、每类 recall、p50/p95 延迟。
`--prompt` 默认 `cpu_routing_qwen3.5`。文本 LLM 用 `--pipeline llm`。

### 7.2 CLINC150 / BANKING-cluster10 —— `eval_ov_public.py`

```bash
# CLINC150（letter A-J）
python eval_ov_public.py \
  --ov-model /path/to/OV-IR-FP16 --hf-model /path/to/hf-model \
  --pipeline vlm --dataset clinc --prompt clinc_domain_qwen3.5 --device GPU.1

# BANKING 聚类 10 类（letter A-J）
python eval_ov_public.py \
  --ov-model /path/to/OV-IR-FP16 --hf-model /path/to/hf-model \
  --pipeline vlm --dataset banking_cluster10 --prompt banking_cluster10_qwen3.5 --device GPU.1
```
`--dataset` 可选 `clinc` / `banking_cluster10`。
快速迭代加 `--limit N --seed 42`（随机抽样；注意抽样比全量乐观约 4pp）。

### 7.3 混淆矩阵 —— `plot_confusion.py`

跑完若干评估后（需先装 matplotlib+seaborn）：
```bash
python plot_confusion.py
```
自动扫 `results/` 下每个 (dataset, tag) 的**最新** JSON（`ovg_*` / `pub_*`），
输出行归一化（recall %）的 PNG 到 `confusion_matrices/`，并打印每类 recall 汇总表。

## 8. 输出格式

两个 `eval_*.py` 把结果写到 `results/`，文件名带时间戳：
- `ovg_{model}_{device}_{prompt}_{ts}.json` —— eval_ov_genai
- `pub_{dataset}_{tag}_{prompt}_{ts}.json` —— eval_ov_public

每个文件含总体准确率 + `per_query`（逐条 gold/pred，供 plot_confusion 用）。

## 9. 注意事项 / 踩坑

1. **`apply_chat_template=False` 必设**（脚本已设）：先用 HF tokenizer 预渲染模板并 `enable_thinking=False`，若让 pipeline 再套一次模板会重开 thinking、破坏 `max_new_tokens=1`，导致数字虚高。
2. **聚类降维有效**：BANKING77 从 77 类聚成 10 类，准确率大幅提升（历史上 ~41% → ~73%）。
3. **prompt 影响很大**：换模型后建议在小样本（`--limit`）上先调 prompt 再跑全量。
4. **量化配置对准确率影响极大（2026-07-15 实测）**：`INT4 group_size=-1 逐通道对称`（激进）会大掉 20–25pp，NPU 上甚至输出垃圾 token。请用 `INT4 group_size=128`（iGPU 用非对称，NPU 因硬件要求对称需配 `--ratio 0.8` 混合精度）。详见 [reports/REPORT_Qwen3-1.7B-NPU_Qwen3.5-2B-iGPU.md](reports/REPORT_Qwen3-1.7B-NPU_Qwen3.5-2B-iGPU.md)。
5. **NPU 需要对称量化**：非对称 INT4 在 NPU 编译失败（`ZE_RESULT_ERROR_INVALID_ARGUMENT`）。NPU 运行还需 `NPU_TURBO=YES` + `MAX_PROMPT_LEN`/`MIN_RESPONSE_LEN`/`NPUW_LLM_PREFILL_CHUNK_SIZE`（脚本按设备自动注入，见 `pipe_config.py`）。
6. **🔴 GPU 用 SDPA，勿用 PA on 2B-VLM（2026-07-20）**：`ATTENTION_BACKEND=PA` 会**静默**把 Qwen3.5-2B（VLMPipeline）在 cpu_routing 上的精度从 **93.1%（SDPA）打到 60.3%**。范围仅 2B-VLM×cpu_routing（CLINC/BANKING 及全部 1.7B-LLM 的 PA/SDPA 一致）。`pipe_config.py` 已默认 SDPA；PA 仅对 1.7B-LLM 正确且更快，可显式 opt-in。详见报告 §8。
   **根因已定位（2026-08-03）**：不是 PA 本身，是"前缀缓存（prefix caching）+ Qwen3.5 hybrid 线性注意力（Gated-DeltaNet）状态检查点复用"的交互 bug——只翻转 `enable_prefix_caching` 即可复现/修复（关掉后 PA 精度回到 93.1%，但也失去 PA 在此类工作负载上的主要速度优势，效果约等于 SDPA）。CLINC/BANKING 经对照实验确认真的不受影响，但排除了"prompt 长度"这类简单解释，机制上不能假设其它任务天然安全。SDPA/PA 的 TTFT 优化（opt-1~opt-5）已全部合入 `benchmark/openvino` 工作树，已到现有方法的 floor。详见 [reports/REPORT_PA精度根因定位与SDPA_TTFT优化现状_Qwen3.5-2B.md](reports/REPORT_PA精度根因定位与SDPA_TTFT优化现状_Qwen3.5-2B.md)。

## 10. 参考准确率

> ✅ **2026-07-20 修正（撤回 07-15 的错误结论）**：cpu_routing 的文档 ~93% **是真实可复现的**——Qwen3.5-2B FP16 + **SDPA** = **93.1%**。此前测得 ~60% 是 `ATTENTION_BACKEND=PA` 的精度 bug（见注意事项 6 / 报告 §8），**不是** eval_set 太难。CLINC/BANKING 无论 PA/SDPA 都精确复现文档。

**Qwen3.5-2B 实测（cpu_routing 为 SDPA 正确值，2026-07-20）：**

| 任务 | 数据集 | prompt | FP16 | INT8 | INT4-g128 | 旧 INT4(g=-1) | 历史文档值 |
|---|---|---|---:|---:|---:|---:|---:|
| 中文 4 类路由 | eval_set (58) | `cpu_routing_qwen3.5` | **93.1%** | 75.9% | 51.7% | 34.5% | ≈93% ✓ |
| CLINC150 10-domain | clinc (4500) | `clinc_domain_qwen3.5` | 78.0% | 76.2% | 75.7% | 52.1% | ≈78% ✓ |
| BANKING cluster10 | banking_cluster10 (3080) | `banking_cluster10_qwen3.5` | 72.9% | 72.1% | 70.2% | 45.4% | ≈73% ✓ |

> ⚠️ cpu_routing 对量化极敏感（FP16 93→INT4-g128 51），与 CLINC/BANKING（INT4-g128≈FP16）不同。
>
> ✅ **补充实验（2026-07-21）：中文/混合校准 AWQ 救回 cpu_routing。** cpu_routing 是中文、CLINC/BANKING 是英文；用通用中英混合语料做 AWQ 并叠加 20% INT8 混合精度（`INT4 g128 +AWQ(mixed)+ratio0.8`）→ **cpu 82.8 / CLINC 74.3 / BANKING 69.2**（三任务均值 75.4≈INT8），模型仍 1.3G≈纯 4-bit。这是「INT4 又不在三个数据集上明显掉精度」的最佳方案。脚本 `requant_nncf_vlm.py`，校准集 `data/{zh,mixed}_calib.txt`，详见报告 §8.5。

| INT4 变体（2B） | cpu_routing(zh) | CLINC(en) | BANKING(en) | 均值 |
|---|---:|---:|---:|---:|
| g128（原纯 4-bit） | 51.7 | 75.7 | 70.2 | 65.9 |
| g128 +AWQ(mixed) +ratio0.8 ★ | **82.8** | 74.3 | 69.2 | **75.4** |

Qwen3-1.7B（FP16 参考，GPU）：cpu_routing 69.0% / CLINC 67.6% / BANKING 57.9%。
- **GPU（非对称）推荐 `INT4-g128 + AWQ`**：CLINC 54.0% / BANKING 55.1%（仅 g128 只有 44.9% / 46.6%，小模型需叠 AWQ）。
- **NPU（对称）**：大任务用 `INT4-g128sym + AWQ`（CLINC 59.8% / BANKING 57.4% ≈ FP16），小/特异任务用 `INT4-g128sym + ratio=0.8`（cpu_routing 65.5%）。

**全量 23 组合对比（FP16/INT8/INT4 各变体 × iGPU/dGPU/NPU × 3 任务）、图表与逐类混淆矩阵见 [reports/REPORT_量化全量对比_Qwen3-1.7B_Qwen3.5-2B.md](reports/REPORT_量化全量对比_Qwen3-1.7B_Qwen3.5-2B.md)（2026-07-18）。**
