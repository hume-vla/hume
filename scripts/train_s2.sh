DEBUG=false
if [ "$DEBUG" = true ]; then
  GPUS=1
  PER_DEVICE_BATCH_SIZE=8
  wandb_enable=false
  ACCELERATE_ARGS="--num_machines 1 --num_processes ${GPUS} --mixed_precision=bf16 --dynamo_backend=no"
  num_workers=0
  save_freq=5
  steps=10
fi

# distributed settings
GPUS=${GPUS:-2}
GPUS_PER_NODE=${GPUS_PER_NODE:-2}
NODES=$((GPUS / GPUS_PER_NODE))
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-32}
wandb_enable=${wandb_enable:-true}
num_workers=${num_workers:-0}
save_freq=${save_freq:-5000}

# set environments
source scripts/env.sh
# distritubed training
find_free_port() {
    while true; do
        port=$(shuf -i 20000-65535 -n 1)
        if ! netstat -tna | grep -q ":${port}.*LISTEN"; then
            echo $port
            break
        fi
    done
}
MASTER_ADDR="127.0.0.1"
MASTER_PORT=$(find_free_port)
ACCELERATE_ARGS=${ACCELERATE_ARGS:-"--main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT \
  --num_machines ${NODES} --num_processes=${GPUS} --multi_gpu \
  --mixed_precision=no --dynamo_backend=no"}

# dataset mapping
declare -A data_map
data_map["libero_spatial"]=libero_spatial_no_noops_1.0.0_lerobot
data_map["libero_object"]=libero_object_no_noops_1.0.0_lerobot
data_map["libero_goal"]=libero_goal_no_noops_1.0.0_lerobot
data_map["libero_10"]=libero_10_no_noops_1.0.0_lerobot


data_name=libero_goal
dataset=${data_map[$data_name]}
echo "dataset: ${dataset}"
cfg=$(echo $dataset | sed 's/^\([a-zA-Z]\+\).*/\1/')

lr=5e-5
steps=$((GPUS * 200000))
chunk_size=4
pretrained_policy="pretrianed_s2"

job_name=hume_s2_${dataset}_ck${chunk_size}_gpu${GPUS}_lr${lr}_bs${PER_DEVICE_BATCH_SIZE}_s$((steps / 1000))k
accelerate launch $ACCELERATE_ARGS src/hume/training/train_s2.py \
  --config_path=config/${cfg}.json \
  --dataset.repo_id=${dataset} \
  --dataset.image_transforms.enable=false \
  --num_workers=${num_workers} \
  --policy_optimizer_lr=${lr} \
  --chunk_size=${chunk_size} \
  --steps=${steps} \
  --batch=${PER_DEVICE_BATCH_SIZE} \
  --save_freq=$save_freq \
  --log_freq=100 \
  --job_name=${job_name} \
  --wandb.enable=${wandb_enable} \
  --wandb.disable_artifact=true \
  --wandb.project=${WANDB_PROJECT} \
  --wandb.entity=${WANDB_ENTITY} \
  --checkpoints_total_limit=0 \
  --dataset.video_backend="pyav" \
  --policy.path=$pretrained_policy \

