# set conda
export TZ=UTC-8
conda activate libero

ckpts_paths=(
)   

# TTS args - 1
s2_candidates_num=5
noise_temp_lower_bound=1.0
noise_temp_upper_bound=1.0
time_temp_lower_bound=0.9
time_temp_upper_bound=1.0

# TTS args - 2
# s2_candidates_num=5
# noise_temp_lower_bound=1.0
# noise_temp_upper_bound=1.2
# time_temp_lower_bound=1.0
# time_temp_upper_bound=1.0

# TTS args - 3
# s2_candidates_num=5
# noise_temp_lower_bound=1.0
# noise_temp_upper_bound=2.0
# time_temp_lower_bound=1.0
# time_temp_upper_bound=1.0

# s1s2 args
s1_replan_steps=8
s2_replan_steps=16
output=s1_action

task_suite_name=libero_spatial
run_ckpt_name=test


for i in ${!ckpts_paths[@]}; do
    ckpt_path=${ckpts_paths[$i]}

    test_name=s1-${s1_replan_steps}_s2-${s2_replan_steps}_s2cand-${s2_candidates_num}_ntl-${noise_temp_lower_bound}_ntu-${noise_temp_upper_bound}_ttl-${time_temp_lower_bound}_ttu-${time_temp_upper_bound}
    job_name=${run_ckpt_name}_${test_name}
    
    echo "$ckpt_path"
    echo $job_name
    cuda=0
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
        --args.time-temp-upper-bound ${time_temp_upper_bound} \
        --args.num_trials_per_task 5
done