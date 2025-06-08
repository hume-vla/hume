# set environments
export HF_LEROBOT_HOME="set your own path to LEROBOT_DATASET"
export TRITON_CACHE_DIR="set your own path to LEROBOT_DATASET"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$(pwd):$PYTHONPATH
export WANDB_API_KEY="set your own WANDB_API_KEY"
export WANDB_PROJECT=Hume
export WANDB_ENTITY="set your own WANDB_ENTITY"

conda activate hume

