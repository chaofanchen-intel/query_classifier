#!/usr/bin/env bash
# Resume only the matrix cells missing after the reboot (sequential, one at a time).
cd /home/chaofan/model_benchmark/query_classifier_handoff
M=/home/chaofan/model_benchmark/benchmark/models

C() { local model="$1"
  if [ ! -d "$model" ]; then echo ">>> SKIP missing dir: $(basename "$model") @ $3"; return; fi
  echo ">>> RUN: $(basename "$model") $2 $3 $4 [$(date +%H:%M:%S)]"; bash run_campaign.sh "$@"; }

echo "########## finish interrupted: 1.7B g128-aware banking on iGPU ##########"
source env.sh
echo ">>> banking-only g128-aware GPU.0 [$(date +%H:%M:%S)]"
python eval_ov_public.py --ov-model $M/Qwen3-1.7B-INT4-g128-aware --hf-model $M/Qwen3-1.7B-INT4-g128-aware \
    --pipeline llm --device GPU.0 --dataset banking_cluster10 --prompt banking_cluster10_qwen3; echo "exit=$?"

echo "########## Qwen3-1.7B (LLM) on dGPU (GPU.1) ##########"
for Q in FP16 INT8 INT4 INT4-g128 INT4-g128-aware; do
  C $M/Qwen3-1.7B-$Q llm GPU.1 qwen3
done

echo "########## Qwen3-1.7B (LLM) on NPU — LAST ##########"
C $M/Qwen3-1.7B-INT8               llm NPU qwen3
C $M/Qwen3-1.7B-INT4               llm NPU qwen3
C $M/Qwen3-1.7B-INT4-g128sym       llm NPU qwen3
C $M/Qwen3-1.7B-INT4-g128sym-r08   llm NPU qwen3
C $M/Qwen3-1.7B-INT4-g128sym-aware llm NPU qwen3

echo "########## RESUME DONE [$(date +%H:%M:%S)] ##########"
