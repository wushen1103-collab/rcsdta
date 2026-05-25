#!/usr/bin/env bash
set -u

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p logs/pmmr_queue logs/pmmr
export HF_ENDPOINT=https://hf-mirror.com
export TOKENIZERS_PARALLELISM=false

# dataset split seed epochs batch_size workers prepare_assets
TASKS=(
  "davis random 42 20 64 2 1"
  "davis similarity_aware_target_scores 42 20 64 2 1"
  "davis temporal_proxy 42 20 64 2 1"
  "kiba all_unseen 42 20 64 2 1"
  "kiba random 42 20 64 2 1"
  "kiba similarity_aware_target_scores 42 20 64 2 1"
  "kiba temporal_proxy 42 20 64 2 1"
  "kiba unseen_drug 42 20 64 2 1"
  "bindingdb similarity_aware_target_scores 42 20 64 2 1"
  "bindingdb true_temporal 43 20 64 2 1"
  "bindingdb true_temporal 44 20 64 2 1"
  "davis unseen_target 43 20 64 2 1"
  "davis unseen_target 44 20 64 2 1"
  "davis similarity_aware_unseen_target 43 20 64 2 1"
  "davis similarity_aware_unseen_target 44 20 64 2 1"
  "davis all_unseen 43 20 64 2 1"
  "davis all_unseen 44 20 64 2 1"
  "davis unseen_drug 43 20 64 2 1"
  "davis unseen_drug 44 20 64 2 1"
  "kiba unseen_target 43 20 64 2 1"
  "kiba unseen_target 44 20 64 2 1"
  "kiba similarity_aware_unseen_target 43 20 64 2 1"
  "kiba similarity_aware_unseen_target 44 20 64 2 1"
  "kiba all_unseen 43 20 64 2 1"
  "kiba all_unseen 44 20 64 2 1"
  "kiba unseen_drug 43 20 64 2 1"
  "kiba unseen_drug 44 20 64 2 1"
  "bindingdb unseen_target 43 20 64 2 1"
  "bindingdb unseen_target 44 20 64 2 1"
  "bindingdb similarity_aware_unseen_target 43 20 64 2 1"
  "bindingdb similarity_aware_unseen_target 44 20 64 2 1"
  "bindingdb all_unseen 43 20 64 2 1"
  "bindingdb all_unseen 44 20 64 2 1"
  "bindingdb unseen_drug 43 20 64 2 1"
  "bindingdb unseen_drug 44 20 64 2 1"
)

summary_path() {
  local dataset="$1" split="$2" seed="$3"
  echo "reports/deployment_upgrade_experiments/pmmr_training/${dataset}/${split}_seed${seed}/summary.json"
}

split_path() {
  local dataset="$1" split="$2" seed="$3"
  echo "data/processed/${dataset}/splits/${split}_seed${seed}.csv"
}

is_running_task() {
  local dataset="$1" split="$2" seed="$3"
  pgrep -af "train_pmmr_external.py" \
    | grep -F -- "--dataset-name ${dataset}" \
    | grep -F -- "--split-name ${split}" \
    | grep -F -- "--seed ${seed}" >/dev/null 2>&1
}

free_gpu() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F', ' '$1 >= 1 && $1 <= 7 && $2 < 500 {print $1; exit}'
}

launch_task() {
  local gpu="$1" dataset="$2" split="$3" seed="$4" epochs="$5" batch="$6" workers="$7" prepare="$8"
  local log="logs/pmmr/${dataset}_${split}_seed${seed}_ep${epochs}_bs${batch}_queue.log"
  local prepare_flag=""
  if [ "$prepare" = "1" ]; then
    prepare_flag="--prepare-assets"
  fi
  echo "$(date '+%F %T') LAUNCH gpu=${gpu} ${dataset} ${split} seed=${seed} epochs=${epochs} batch=${batch}" \
    | tee -a logs/pmmr_queue/queue.log
  nohup bash -lc "export CUDA_VISIBLE_DEVICES=${gpu}; export HF_ENDPOINT=https://hf-mirror.com; export TOKENIZERS_PARALLELISM=false; ./.venv/bin/python scripts/train_pmmr_external.py --workspace . --external-root ./external/PMMR --dataset-name ${dataset} --split-name ${split} --seed ${seed} ${prepare_flag} --batch-size ${batch} --max-epochs ${epochs} --num-workers ${workers} > ${log} 2>&1" >/dev/null 2>&1 &
}

echo "$(date '+%F %T') queue daemon started with ${#TASKS[@]} candidate tasks" | tee -a logs/pmmr_queue/queue.log
while true; do
  for task in "${TASKS[@]}"; do
    read -r dataset split seed epochs batch workers prepare <<< "$task"
    [ -f "$(summary_path "$dataset" "$split" "$seed")" ] && continue
    [ -f "$(split_path "$dataset" "$split" "$seed")" ] || continue
    is_running_task "$dataset" "$split" "$seed" && continue
    gpu="$(free_gpu || true)"
    [ -n "$gpu" ] || break
    launch_task "$gpu" "$dataset" "$split" "$seed" "$epochs" "$batch" "$workers" "$prepare"
    sleep 10
  done

  remaining=0
  for task in "${TASKS[@]}"; do
    read -r dataset split seed epochs batch workers prepare <<< "$task"
    [ -f "$(summary_path "$dataset" "$split" "$seed")" ] && continue
    [ -f "$(split_path "$dataset" "$split" "$seed")" ] || continue
    remaining=$((remaining + 1))
  done
  if [ "$remaining" -eq 0 ]; then
    echo "$(date '+%F %T') queue daemon finished all available tasks" | tee -a logs/pmmr_queue/queue.log
    exit 0
  fi
  sleep 60
done

