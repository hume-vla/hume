# set conda
export TZ=UTC-8

export PATH="/cpfs01/shared/optimal/vla_ptm/miniconda3/bin:$PATH"
. /cpfs01/shared/optimal/vla_ptm/miniconda3/etc/profile.d/conda.sh
export LD_LIBRARY_PATH="/cpfs01/shared/optimal/vla_ptm/miniconda3/lib:${LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH="/cpfs01/shared/optimal/vla_ptm/miniconda3/envs/vla_next/lib/python3.10/site-packages/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH="/cpfs01/shared/optimal/vla_ptm/miniconda3/envs/vla_next/lib/python3.10/site-packages/nvidia/cusparse/lib:${LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"
CFLAGS="/cpfs01/shared/optimal/vla_ptm/miniconda3/include"
LDFLAGS="/cpfs01/shared/optimal/vla_ptm/miniconda3/lib"
conda activate libero

ckpts_paths=(
    /cpfs01/shared/optimal/vla_next/Hume/exported/libero/goal_2
)   

# TTS args
s2_candidates_num=5
noise_temp_lower_bound=1.0
noise_temp_upper_bound=1.0
time_temp_lower_bound=0.9
time_temp_upper_bound=1.0

# s1s2 args
s1_replan_steps=8
s2_replan_steps=16
output=s1_action

task_suite_name=libero_goal
run_ckpt_name=test


for i in ${!ckpts_paths[@]}; do
    ckpt_path=${ckpts_paths[$i]}

    test_name=s1-${s1_replan_steps}_s2-${s2_replan_steps}_s2cand-${s2_candidates_num}_ntl-${noise_temp_lower_bound}_ntu-${noise_temp_upper_bound}_ttl-${time_temp_lower_bound}_ttu-${time_temp_upper_bound}
    job_name=${run_ckpt_name}_${test_name}
    
    echo "$ckpt_path"
    echo $job_name
    cuda=0
    # tmux new-session -d -s ${cuda}_${job_name} "CUDA_VISIBLE_DEVICES=${cuda} python test/eval_libero.py \
    CUDA_VISIBLE_DEVICES=$cuda python experiments/libero/eval_libero.py \
        --args.pretrained-path $ckpt_path \
        --args.task-suite-name ${task_suite_name} \
        --args.job-name ${job_name} \
        --args.post-process-action \
        --args.num-trials-per-task 10 \
        --args.replan-steps ${s1_replan_steps} \
        --args.s2-replan-steps ${s2_replan_steps} \
        --args.s2-candidates-num ${s2_candidates_num} \
        --args.noise-temp-lower-bound ${noise_temp_lower_bound} \
        --args.noise-temp-upper-bound ${noise_temp_upper_bound} \
        --args.time-temp-lower-bound ${time_temp_lower_bound} \
        --args.time-temp-upper-bound ${time_temp_upper_bound}
done