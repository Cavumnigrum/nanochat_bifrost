#!/bin/bash
# Сравнение Baseline vs Bifrost на одинаковой модели

set -e # Exit on error

DEPTH=${1:-12} # Размер модели (по умолчанию 12)
ITERS=${2:-5000} # Количество итераций (по умолчанию 5000)
GPUS=${3:-8} # Количество GPU (по умолчанию 8)

echo "==================================================================="
echo "Bifrost vs Baseline Comparison"
echo "==================================================================="
echo "Depth: $DEPTH"
echo "Iterations: $ITERS"
echo "GPUs: $GPUS"
echo "==================================================================="

# 1. Обучаем BASELINE (Bifrost ВЫКЛЮЧЕН)
echo ""
echo ">>> STEP 1/4: Training BASELINE model (bifrost_enabled=False)"
echo ""
torchrun --standalone --nproc_per_node=$GPUS -m scripts.bifrost_train -- \
    --depth=$DEPTH \
    --num_iterations=$ITERS \
    --bifrost_enabled=False \
    --run=baseline_d${DEPTH} \
    --model_tag=d${DEPTH}_baseline

# 2. Обучаем BIFROST (Bifrost ВКЛЮЧЕН)
echo ""
echo ">>> STEP 2/4: Training BIFROST model (bifrost_enabled=True)"
echo ""
torchrun --standalone --nproc_per_node=$GPUS -m scripts.bifrost_train -- \
    --depth=$DEPTH \
    --num_iterations=$ITERS \
    --bifrost_enabled=True \
    --bifrost_alpha=0.2 \
    --bifrost_lambda=0.2 \
    --bifrost_beta=0.6 \
    --run=bifrost_d${DEPTH} \
    --model_tag=d${DEPTH}_bifrost

# 3. Оцениваем BASELINE на CORE
echo ""
echo ">>> STEP 3/4: Evaluating BASELINE on CORE benchmark"
echo ""
python -m scripts.base_eval bifrost_checkpoints/d${DEPTH}_baseline

# 4. Оцениваем BIFROST на CORE
echo ""
echo ">>> STEP 4/4: Evaluating BIFROST on CORE benchmark"
echo ""
python -m scripts.base_eval bifrost_checkpoints/d${DEPTH}_bifrost

echo ""
echo "==================================================================="
echo "✅ Comparison complete!"
echo "==================================================================="
echo ""
echo "Results are in:"
echo "- base_eval/d${DEPTH}_baseline_*.csv"
echo "- base_eval/d${DEPTH}_bifrost_*.csv"
echo ""
echo "To generate comparison report:"
echo "python scripts/analyze_bifrost_results.py"