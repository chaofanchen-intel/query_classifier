#!/usr/bin/env bash
# Sequential TTFT/TPS/TPOT benchmark for the query-classifier candidate models.
# STRICTLY one run at a time (latency must not be affected by contention); 30s
# cooldown between runs (iGPU/NPU share the SoC package -> let it settle).
# Standard workload: 1k input / 512 output, 10 measured iters (+1 warmup),
# prompt-permutation ON. GPU uses PA (gpu_pa.json); NPU uses best-perf turbo.
# Results append to perf_results/perf.json (+ per-run logs).
set -o pipefail
ROOT=/home/chaofan/model_benchmark
cd "$ROOT"
source benchmark/openvino/install/setupvars.sh >/dev/null 2>&1

MB=./model_bench/build-local/model_bench
M=./benchmark/models
CFG=benchmark/tests/repro_rerun_20260710
GPU_CFG=$CFG/gpu_pa.json                    # PERFORMANCE_HINT=LATENCY + ATTENTION_BACKEND=PA
NPU_CFG=$CFG/npu_best_perf_turbo.json       # NPU_TURBO=YES + BEST_PERF + MAX_PROMPT_LEN=1024 (fits 1k)
P_QWEN3=./model_bench/prompts/qwen3_1k.jsonl
P_QWEN35=./model_bench/prompts/qwen3.5_1k.jsonl

OUT=query_classifier_handoff/perf_results
mkdir -p "$OUT/logs"
JSON=$OUT/perf.json
CSV=$OUT/perf.csv
rm -f "$JSON" "$CSV"                          # fresh (append-mode files)

N=10; IC=512; COOLDOWN=30

run() {  # <task> <model_basename> <device> <prompt_file> <config> <label>
  local task=$1 model=$2 dev=$3 pf=$4 cfg=$5 label=$6
  local mdir=$M/$model
  if [ ! -d "$mdir" ]; then echo ">>> SKIP missing: $model"; return; fi
  local perm=(--enable-prompt-permutation)
  local log=$OUT/logs/${label}.log
  echo ">>> RUN [$(date +%H:%M:%S)] $label : $task $model @ $dev (n=$N ic=$IC 1k-in)"
  $MB "$task" -m "$mdir" -d "$dev" -n "$N" -ic "$IC" -pf "$pf" \
      "${perm[@]}" --no-monitor -lc "$cfg" -rj "$JSON" -r "$CSV" > "$log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then
    grep -E "^  (TTFT|TPOT|Throughput|Decode TPS) " "$log" | sed 's/^/    /'
  else
    echo "    !! FAILED rc=$rc (see $log)"; tail -5 "$log" | sed 's/^/    /'
  fi
  echo "    cooldown ${COOLDOWN}s..."; sleep "$COOLDOWN"
}

echo "########## Qwen3.5-2B (VLM) @ iGPU (GPU.0) ##########"
for Q in FP16 INT8 INT4 INT4-g128; do
  run vlm Qwen3.5-2B-$Q GPU.0 "$P_QWEN35" "$GPU_CFG" "2B_${Q}_iGPU"
done

echo "########## Qwen3-1.7B (LLM) @ iGPU (GPU.0) ##########"
for Q in FP16 INT8 INT4 INT4-g128 INT4-g128-aware; do
  run llm Qwen3-1.7B-$Q GPU.0 "$P_QWEN3" "$GPU_CFG" "1.7B_${Q}_iGPU"
done

echo "########## Qwen3-1.7B (LLM) @ NPU — LAST (best-perf turbo) ##########"
for Q in INT8 INT4 INT4-g128sym INT4-g128sym-r08 INT4-g128sym-aware; do
  run llm Qwen3-1.7B-$Q NPU "$P_QWEN3" "$NPU_CFG" "1.7B_${Q}_NPU"
done

echo "########## PERF DONE [$(date +%H:%M:%S)] ##########"
