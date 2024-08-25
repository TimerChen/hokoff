#!/bin/bash

# 定义参数
root_path="/NAS2020/Share/jxchen/hokoff/1v1_logs/"
run_prefix="run_1v1qmix_5v5_1"
levels="5,5"
cpu_num=20
eval_num=1
final_test=0
tensorflow_oppo=1
dataset_name="level-5-5"

# 生成均匀分布的 max_steps 值
for i in $(seq 0 9); do
    max_steps=$((50000 + i * 50000))  # 从 50000 到 500000 均匀分布
    echo "Running command with max_steps=${max_steps}..."
    
    # 执行命令
    python offline_eval/evaluation.py \
        --root_path=${root_path} \
        --run_prefix=${run_prefix} \
        --levels=${levels} \
        --cpu_num=${cpu_num} \
        --eval_num=${eval_num} \
        --final_test=${final_test} \
        --tensorflow_oppo=${tensorflow_oppo} \
        --max_steps=${max_steps} \
        --dataset_name=${dataset_name}
    
    # 检查命令是否成功
    if [ $? -ne 0 ]; then
        echo "Error occurred while running command with max_steps=${max_steps}."
    fi
done
    