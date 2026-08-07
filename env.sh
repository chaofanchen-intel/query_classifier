# Canonical environment for the query-classifier accuracy evals.
# Source this before running any eval script:  source env.sh
#
# Combines:
#   - qwen_env  : python deps (transformers, torch, numpy, matplotlib, seaborn)
#   - setupvars : the source-built *optimized* OpenVINO (2026.3 qwen-ttft-optimizations)
#                 + matching openvino_genai, both from install/python (self-consistent, no ABI mix)
source /home/chaofan/envs/qwen_env/bin/activate
source /home/chaofan/model_benchmark/benchmark/openvino/install/setupvars.sh >/dev/null 2>&1
# Explicitly prepend install/python so the optimized openvino + genai win over qwen_env's pip copies.
export PYTHONPATH="/home/chaofan/model_benchmark/benchmark/openvino/install/python:$PYTHONPATH"
