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

## 10. 参考准确率（历史，Qwen3.5-2B / FP16 / 全量）

仅供对照量级，你换模型后以实跑为准：

| 任务 | 数据集 | 推荐 prompt | 准确率 |
|---|---|---|---|
| 中文 4 类路由 | eval_set (58) | `cpu_routing_qwen3.5` | ≈ 93% |
| CLINC150 10-domain | clinc (4500) | `clinc_domain_qwen3.5` | ≈ 78% |
| BANKING cluster10 | banking_cluster10 (3080) | `banking_cluster10_qwen3.5` | ≈ 73% |
