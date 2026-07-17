#!/usr/bin/env bash
# Full comparison matrix, STRICTLY SEQUENTIAL (one campaign at a time) so latency
# measurements are not affected by contention. GPUs first, NPU last (slower).
# Each campaign = 3 tasks (cpu_routing, clinc, banking). Results -> results/.
cd /home/chaofan/model_benchmark/query_classifier_handoff
M=/home/chaofan/model_benchmark/benchmark/models

C() {  # <model_dir> <pipeline> <device> <prompt_suffix>
  local model="$1"
  if [ ! -d "$model" ]; then echo ">>> SKIP (missing): $(basename "$model") @ $3"; return; fi
  echo ">>> RUN: $(basename "$model") $2 $3 $4  [$(date +%H:%M:%S)]"
  bash run_campaign.sh "$@"
}

echo "########## Qwen3.5-2B (VLM) on iGPU then dGPU ##########"
for DEV in GPU.0 GPU.1; do
  C $M/Qwen3.5-2B-FP16       vlm $DEV qwen3.5
  C $M/Qwen3.5-2B-INT8       vlm $DEV qwen3.5
  C $M/Qwen3.5-2B-INT4       vlm $DEV qwen3.5
  C $M/Qwen3.5-2B-INT4-g128  vlm $DEV qwen3.5
done

echo "########## Qwen3-1.7B (LLM) on iGPU then dGPU ##########"
for DEV in GPU.0 GPU.1; do
  C $M/Qwen3-1.7B-FP16            llm $DEV qwen3
  C $M/Qwen3-1.7B-INT8            llm $DEV qwen3
  C $M/Qwen3-1.7B-INT4            llm $DEV qwen3
  C $M/Qwen3-1.7B-INT4-g128       llm $DEV qwen3
  C $M/Qwen3-1.7B-INT4-g128-aware llm $DEV qwen3
done

echo "########## Qwen3-1.7B (LLM) on NPU — LAST (sym-compatible quants) ##########"
C $M/Qwen3-1.7B-INT8               llm NPU qwen3
C $M/Qwen3-1.7B-INT4               llm NPU qwen3
C $M/Qwen3-1.7B-INT4-g128sym       llm NPU qwen3
C $M/Qwen3-1.7B-INT4-g128sym-r08   llm NPU qwen3
C $M/Qwen3-1.7B-INT4-g128sym-aware llm NPU qwen3

echo "########## ALL DONE [$(date +%H:%M:%S)] ##########"
