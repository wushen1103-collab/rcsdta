#!/usr/bin/env bash
set -u

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p logs/pmm
export HF_ENDPOINT=https://hf-mirror.com
export TOKENIZERS_PARALLELISM=false

summary_path() {
  local dataset="$1" split="$2" seed="$3"
  echo "reports/deployment_upgrade_experiments/pmmr_training/${dataset}/${split}_seed${seed}/summary.json"
}

is_running_task() {
  local dataset="$1" split="$2" seed="$3"
  pgrep -af "train_pmmr_external.py" \
    | grep -F -- "--dataset-name ${dataset}" \
    | grep -F -- "--split-name ${split}" \
    | grep -F -- "--seed ${seed}" >/dev/null 2>&1
}

launch() {
  local gpu="$1" dataset="$2" split="$3" seed="$4"
  [ -f "$(summary_path "$dataset" "$split" "$seed")" ] && return 0
  is_running_task "$dataset" "$split" "$seed" && return 0
  local log="logs/pmmr/${dataset}_${split}_seed${seed}_gapfill.log"
  nohup env CUDA_VISIBLE_DEVICES="${gpu}" HF_ENDPOINT=https://hf-mirror.com TOKENIZERS_PARALLELISM=false \
    ./.venv/bin/python scripts/train_pmmr_external.py \
    --workspace . \
    --external-root ./external/PMMR \
    --dataset-name "${dataset}" \
    --split-name "${split}" \
    --seed "${seed}" \
    --prepare-assets \
    --batch-size 64 \
    --max-epochs 20 \
    --num-workers 2 \
    > "${log}" 2>&1 < /dev/null &
  echo "$(date '+%F %T') launch gpu=${gpu} ${dataset} ${split} seed=${seed}"
}

launch 0 davis random 43
launch 1 davis random 44
launch 2 kiba random 43
launch 3 kiba random 44
launch 4 bindingdb random 43
launch 5 bindingdb random 44

