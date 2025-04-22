#!/bin/sh
env="async_rul_schedule"
scenario="rul_schedule"
num_agents=12
algo="mappo"
exp="async_rul_schedule"

echo "env is ${env}"

CUDA_VISIBLE_DEVICES=0 python onpolicy/scripts/train/train_schedule.py\
    --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} \
    --num_agents ${num_agents} \
    --n_training_threads 1 --n_rollout_threads 1\
    --cnn_layers_params '16,7,2,1 32,5,2,1 16,3,1,1' \
    --model_dir "./results/rul_schedule/async_mappo/" \
    --max_steps 200 --use_complete_reward --agent_view_size 7 --local_step_num 1 --use_random_pos \
    --astar_cost_mode utility --grid_goal --goal_grid_size 5 --cnn_trans_layer 1,3,1,1 \
    --use_stack --grid_size 25 --use_recurrent_policy --use_stack --use_global_goal --use_overlap_penalty --use_eval --wandb_name "async_mappo" --asynch & 
