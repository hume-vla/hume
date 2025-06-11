# set conda
export TZ=UTC-8
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
source experiments/libero/.venv/bin/activate

# Server Info
policy_ip="0.0.0.0"
policy_port=8000   

# TTS args
s2_candidates_num=5
noise_temp_lower_bound=1.0
noise_temp_upper_bound=1.0
time_temp_lower_bound=0.9
time_temp_upper_bound=1.0

# s1s2 args
s1_replan_steps=8
s2_replan_steps=16

task_suite_name=libero_goal
run_ckpt_name=test

test_name="s1-${s1_replan_steps}_"\
"s2-${s2_replan_steps}_"\
"s2cand-${s2_candidates_num}_"\
"ntl-${noise_temp_lower_bound}_"\
"ntu-${noise_temp_upper_bound}_"\
"ttl-${time_temp_lower_bound}_"\
"ttu-${time_temp_upper_bound}"

job_name=${run_ckpt_name}_${test_name}

echo $job_name
python experiments/libero/eval_libero.py \
    --args.policy_ip $policy_ip \
    --args.policy_port $policy_port \
    --args.task-suite-name ${task_suite_name} \
    --args.job-name ${job_name} \
    --args.post-process-action \
    --args.num-trials-per-task 5 \
    --args.replan-steps ${s1_replan_steps} \
    --args.s2-replan-steps ${s2_replan_steps} \
    --args.s2-candidates-num ${s2_candidates_num} \
    --args.noise-temp-lower-bound ${noise_temp_lower_bound} \
    --args.noise-temp-upper-bound ${noise_temp_upper_bound} \
    --args.time-temp-lower-bound ${time_temp_lower_bound} \
    --args.time-temp-upper-bound ${time_temp_upper_bound} \
