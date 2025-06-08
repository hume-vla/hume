# set environments
export HF_LEROBOT_HOME=/cpfs01/shared/optimal/vla_next/LEROBOT_DATASET
# export HF_LEROBOT_HOME=/cpfs01/shared/optimal/vla_next/openss2/data_cpfs/lerobot
export TRITON_CACHE_DIR=/cpfs01/shared/optimal/vla_ptm/.triton
export TOKENIZERS_PARALLELISM=false
export PATH="/cpfs01/shared/optimal/vla_next/ffmpeg-master-latest-linux64-gpl/bin:${PATH}"
export PYTHONPATH="$(pwd)/src/hume/models":$(pwd):$PYTHONPATH
export WANDB_API_KEY=6952544f9d1cdf5de4ec3a9bc35722fd2e1b0eae
export WANDB_PROJECT=ss2-o1
export WANDB_ENTITY=qudelin

# set conda
export PATH="/cpfs01/shared/optimal/vla_ptm/miniconda3/bin:$PATH"
. /cpfs01/shared/optimal/vla_ptm/miniconda3/etc/profile.d/conda.sh
export LD_LIBRARY_PATH="/cpfs01/shared/optimal/vla_ptm/miniconda3/lib:${LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH="/cpfs01/shared/optimal/vla_ptm/miniconda3/envs/vla_next/lib/python3.10/site-packages/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH="/cpfs01/shared/optimal/vla_ptm/miniconda3/envs/vla_next/lib/python3.10/site-packages/nvidia/cusparse/lib:${LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"
CFLAGS="/cpfs01/shared/optimal/vla_ptm/miniconda3/include"
LDFLAGS="/cpfs01/shared/optimal/vla_ptm/miniconda3/lib"
conda activate hume

