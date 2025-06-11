source scripts/env.sh
port=8000

ckpts=(
    exported/libero/2025-05-02/08-10-44_libero_goal_ck8-16-1_sh-4_gpu8_lr5e-5_1e-5_1e-5_2e-5_bs16_s1600k/0090000
    # exported/libero/2025-05-02/08-10-44_libero_goal_ck8-16-1_sh-4_gpu8_lr5e-5_1e-5_1e-5_2e-5_bs16_s1600k/0120000
    # exported/libero/2025-05-02/08-10-44_libero_goal_ck8-16-1_sh-4_gpu8_lr5e-5_1e-5_1e-5_2e-5_bs16_s1600k/0150000
    # exported/libero/2025-05-02/08-10-44_libero_goal_ck8-16-1_sh-4_gpu8_lr5e-5_1e-5_1e-5_2e-5_bs16_s1600k/0190000
)
for ckpt in ${ckpts[@]}; do
    echo $ckpt
    python src/hume/serve_policy.py \
        --ckpt_path $ckpt \
        --port $port
done