# set environments
# set your own path to LEROBOT_DATASET
export HF_LEROBOT_HOME="/cpfs01/shared/optimal/vla_next/LEROBOT_DATASET"
# set your own path to TRITON_CACHE_DIR
export TRITON_CACHE_DIR="/cpfs01/shared/optimal/vla_ptm/.triton"
export TOKENIZERS_PARALLELISM=false

# set your own WANDB_API_KEY
export WANDB_API_KEY="6952544f9d1cdf5de4ec3a9bc35722fd2e1b0eae"
export WANDB_PROJECT=Hume

# set your own WANDB_ENTITY
export WANDB_ENTITY="qudelin"

source "$(pwd)/.venv/bin/activate"
