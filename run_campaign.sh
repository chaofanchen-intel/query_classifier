#!/usr/bin/env bash
# Run all 3 classification tasks for one (model, device, pipeline, prompt-family).
# usage: run_campaign.sh <MODEL_DIR> <PIPELINE:llm|vlm> <DEVICE> <PROMPT_SUFFIX:qwen3|qwen3.5>
# Results land in results/ ; each task's stdout is echoed.
# NOTE: do NOT use `set -u` here — sourcing venv activate / setupvars.sh trips on
# unbound vars and aborts silently.
set -o pipefail
cd /home/chaofan/model_benchmark/query_classifier_handoff

# Capture args BEFORE sourcing env.sh: setupvars.sh clobbers $1..$N (it parses its own args).
MODEL="$1"; PIPE="$2"; DEV="$3"; PS="$4"

source env.sh

# NPU static pipeline prefills to MAX_PROMPT_LEN regardless of actual prompt length.
# Our prompts are <=521 tokens, so shrink 1024 -> 640 (fits all) to ~halve NPU TTFT.
EXTRA=()
case "$DEV" in NPU*) EXTRA=(--pipe-config MAX_PROMPT_LEN=640) ;; esac

run() { echo; echo "########## $* ##########"; "$@"; echo "exit=$?"; }

echo "=== CAMPAIGN: model=$(basename "$MODEL") pipeline=$PIPE device=$DEV prompts=*_$PS extra=${EXTRA[*]} ==="

run python eval_ov_genai.py  --ov-model "$MODEL" --hf-model "$MODEL" \
    --pipeline "$PIPE" --device "$DEV" --prompt "cpu_routing_$PS" "${EXTRA[@]}"

run python eval_ov_public.py --ov-model "$MODEL" --hf-model "$MODEL" \
    --pipeline "$PIPE" --device "$DEV" --dataset clinc --prompt "clinc_domain_$PS" "${EXTRA[@]}"

run python eval_ov_public.py --ov-model "$MODEL" --hf-model "$MODEL" \
    --pipeline "$PIPE" --device "$DEV" --dataset banking_cluster10 --prompt "banking_cluster10_$PS" "${EXTRA[@]}"

echo; echo "=== CAMPAIGN DONE: $(basename "$MODEL") @ $DEV ==="
