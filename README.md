# Fast-Quadruped

unset all_proxy 
export http_proxy="http://127.0.0.1:7897" 

export https_proxy="http://127.0.0.1:7897"


curl -v https://www.google.com

# 激活正确的环境
conda activate isaac-lab

# 设置显卡驱动修复（每次新终端都需要）
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json

置环境变量（必须执行）
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-zmli

# 启动tmux 会话并训练
tmux new -s training

# 查看tmux
tmux attach -t training

tmux kill-session -t training2

tmux kill-session -t session-name


# 设置使用指定GPU训练
export CUDA_VISIBLE_DEVICES=0





# 开启训练
python scripts/reinforcement_learning/rsl_rl/train.py --task RobotLab-Isaac-Velocity-Flat-Run-Himmy-Mark2-v0 --headless --num_envs 4096 --run_name=no_spine_training



python scripts/reinforcement_learning/rsl_rl/train.py --task RobotLab-Isaac-Velocity-Flat-Run-NoSpine-Himmy-Mark2-v0 --headless --num_envs 4096 --run_name=no_spine_running_training




python scripts/reinforcement_learning/rsl_rl/train.py --task RobotLab-Isaac-Velocity-Flat-Twist-NoSpine-Himmy-Mark2-v0 --headless --num_envs 4096 --run_name=no_spine_twist


python scripts/reinforcement_learning/rsl_rl/train.py --task RobotLab-Isaac-Velocity-Flat-Run-Turn-NoSpine-Himmy-Mark2-v0 --headless --num_envs 4096 --run_name=no_spine_turn


python scripts/reinforcement_learning/rsl_rl/train.py --task RobotLab-Isaac-Velocity-Flat-Run-Turn-Himmy-Mark2-v0 --headless --num_envs 4096 --run_name=Turning_training_turn_reward_small_scale


python scripts/reinforcement_learning/rsl_rl/train.py --task RobotLab-Isaac-Velocity-Flat-Himmy-Mark2-v0 --headless --num_envs 4096 --run_name=twist_training


python scripts/reinforcement_learning/rsl_rl/train.py --task RobotLab-Isaac-Velocity-Flat-Twist-Himmy-Mark2-v0 --headless --num_envs 4096 --run_name=twist_training1

# resume
python scripts/reinforcement_learning/rsl_rl/train.py --task=RobotLab-Isaac-Velocity-Flat-Run-Himmy-Mark2-v0 --headless --num_envs 4096 --resume --load_run 2026-02-07_22-34-55_resum--resum--GallopGait_reward_rotary_gallop--spine_training_no_average_all_leg_bias2.5 --run_name=resum--resume_form__resum--resum--GallopGait_reward_rotary_gallop--spine_training_no_average_all_leg_bias2.5--bias0_jingu0.2




python scripts/reinforcement_learning/rsl_rl/train.py --task=RobotLab-Isaac-Velocity-Flat-Run-Turn-Himmy-Mark2-v0 --headless --num_envs 4096 --resume --load_run 2026-02-11_15-07-45_small_scal_fast --run_name=small_scal_faster







python scripts/reinforcement_learning/rsl_rl/train.py --task=RobotLab-Isaac-Velocity-Flat-Run-NoSpine-Himmy-Mark2-v0 --headless --num_envs 4096 --resume --load_run 2026-02-09_10-04-11_no_spine_running_training --run_name=resum--no_spine_running_training--more_training




python scripts/reinforcement_learning/rsl_rl/train.py --task=RobotLab-Isaac-Velocity-Flat-Run-Turn-NoSpine-Himmy-Mark2-v0 --headless --num_envs 4096 --resume --load_run 2026-02-10_23-59-23_no_spine_turn_ang --run_name=no_spine_turn_fast_turn





# play and video
python scripts/reinforcement_learning/rsl_rl/play.py --task=RobotLab-Isaac-Velocity-Flat-Run-Himmy-Mark2-v0 --headless --load_run 2026-01-24_16-06-00_resume_form__new_start_dynamic_gait3__new_gait_reward --checkpoint /root/gpufree-data/Fast-Quadruped/logs/rsl_rl/himmy_mark2_flat/2026-01-24_16-06-00_resume_form__new_start_dynamic_gait3__new_gait_reward/model_5000.pt --video --video_length 1000 --num_envs 32 --debug_vis



python scripts/reinforcement_learning/rsl_rl/play.py --task=RobotLab-Isaac-Velocity-Flat-Run-Himmy-Mark2-v0 --load_run 2026-01-27_19-48-58_sim2sim_test --checkpoint /root/gpufree-data/Fast-Quadruped/logs/rsl_rl/himmy_mark2_flat/2026-01-27_19-48-58_sim2sim_test/model_6700.pt --num_envs 1 --video --video_length 500


python scripts/reinforcement_learning/rsl_rl/play.py --task=RobotLab-Isaac-Velocity-Flat-Run-Himmy-Mark2-v0 --load_run 2026-01-22_17-53-43_resume_form__resume_form__new_start_dynamic_gait3___new_spine_reward_powerful --checkpoint /root/gpufree-data/Fast-Quadruped/logs/rsl_rl/himmy_mark2_flat/2026-01-22_17-53-43_resume_form__resume_form__new_start_dynamic_gait3___new_spine_reward_powerful/model_35600.pt --num_envs 1



python scripts/reinforcement_learning/rsl_rl/play.py --task=RobotLab-Isaac-Velocity-Flat-Run-Himmy-Mark2-v0 --load_run 2026-02-09_09-34-37_resum--resume_form__resum--resum--GallopGait_reward_rotary_gallop--spine_training_no_average_all_leg_bias2.5--bias0_jingu0.2 --checkpoint /root/gpufree-data/Fast-Quadruped/logs/rsl_rl/himmy_mark2_flat/2026-02-09_09-34-37_resum--resume_form__resum--resum--GallopGait_reward_rotary_gallop--spine_training_no_average_all_leg_bias2.5--bias0_jingu0.2/model_14800.pt --num_envs 1 --video --video_length 1000 --log





python scripts/reinforcement_learning/rsl_rl/play.py --task=RobotLab-Isaac-Velocity-Flat-Run-Himmy-Mark2-v0 --load_run 2026-02-09_11-37-49_Turning_training_turn_reward_new --checkpoint /root/gpufree-data/Fast-Quadruped/logs/rsl_rl/himmy_mark2_flat/2026-02-09_11-37-49_Turning_training_turn_reward_new/model_5200.pt --num_envs 1






python scripts/reinforcement_learning/rsl_rl/play.py --task=RobotLab-Isaac-Velocity-Flat-Run-Himmy-Mark2-v0 --load_run 2026-02-11_10-35-19_Turning_training_turn_reward_small_scal --checkpoint /root/gpufree-data/Fast-Quadruped/logs/rsl_rl/himmy_mark2_flat/2026-02-11_10-35-19_Turning_training_turn_reward_small_scal/model_9100.pt --num_envs 1 --headless --video --video_length 2000 --log



python scripts/reinforcement_learning/rsl_rl/play.py --task=RobotLab-Isaac-Velocity-Flat-Run-Himmy-Mark2-v0 --load_run 2026-02-11_17-44-24_small_scal_faster --checkpoint /root/gpufree-data/Fast-Quadruped/logs/rsl_rl/himmy_mark2_flat/2026-02-11_17-44-24_small_scal_faster/model_23000.pt --num_envs 1 


# turning play and video

python scripts/reinforcement_learning/rsl_rl/play.py --task=RobotLab-Isaac-Velocity-Flat-Run-Turn-Himmy-Mark2-v0 --load_run 2026-02-05_21-20-34_Turning_training_turn_reward --checkpoint /root/gpufree-data/Fast-Quadruped/logs/rsl_rl/himmy_mark2_flat/2026-01-28_17-34-32_resum--resum--try2_more_spine_power--foot_hight_penalty--more/model_1400.pt --num_envs 1




python scripts/reinforcement_learning/rsl_rl/play.py --task=RobotLab-Isaac-Velocity-Flat-Run-Himmy-Mark2-v0 --load_run 2026-02-10_15-46-48_resum--resum--resum--Turning_training_turn_reward_new--more_training--more_training --checkpoint /root/gpufree-data/Fast-Quadruped/logs/rsl_rl/himmy_mark2_flat/2026-02-10_15-46-48_resum--resum--resum--Turning_training_turn_reward_new--more_training--more_training/model_23200.pt --num_envs 1





# air twist
python scripts/reinforcement_learning/rsl_rl/play.py --task=RobotLab-Isaac-Velocity-Flat-Twist-Himmy-Mark2-v0 --load_run 2026-01-31_17-27-30_twist_training --checkpoint /root/gpufree-data/Fast-Quadruped/logs/rsl_rl/himmy_mark2_flat/2026-01-31_17-27-30_twist_training/model_6900.pt --num_envs 1 --headless --video --video_length 1000 --log

python scripts/reinforcement_learning/rsl_rl/play.py --task=RobotLab-Isaac-Velocity-Flat-Twist-Himmy-Mark2-v0 --load_run 2026-02-09_09-34-37_resum--resume_form__resum--resum--GallopGait_reward_rotary_gallop--spine_training_no_average_all_leg_bias2.5--bias0_jingu0.2 --checkpoint /root/gpufree-data/Fast-Quadruped/logs/rsl_rl/himmy_mark2_flat/2026-02-09_09-34-37_resum--resume_form__resum--resum--GallopGait_reward_rotary_gallop--spine_training_no_average_all_leg_bias2.5--bias0_jingu0.2/model_14800.pt --num_envs 1 




 --headless --video --video_length 1000


python scripts/reinforcement_learning/rsl_rl/play.py --task=RobotLab-Isaac-Velocity-Flat-Twist-NoSpine-Himmy-Mark2-v0 --load_run 2026-02-09_18-03-32_no_spine_twist --checkpoint /root/gpufree-data/Fast-Quadruped/logs/rsl_rl/himmy_mark2_flat/2026-02-09_18-03-32_no_spine_twist/model_6900.pt --num_envs 1



# no spine

running:

python scripts/reinforcement_learning/rsl_rl/play.py --task=RobotLab-Isaac-Velocity-Flat-Run-NoSpine-Himmy-Mark2-v0 --load_run 2026-02-09_14-13-34_resum--no_spine_running_training--more_training --checkpoint /root/gpufree-data/Fast-Quadruped/logs/rsl_rl/himmy_mark2_flat/2026-02-09_14-13-34_resum--no_spine_running_training--more_training/model_14800.pt --num_envs 1



python scripts/reinforcement_learning/rsl_rl/play.py --task=RobotLab-Isaac-Velocity-Flat-Run-NoSpine-Himmy-Mark2-v0 --load_run 2026-02-10_23-59-23_no_spine_turn_ang --checkpoint /root/gpufree-data/Fast-Quadruped/logs/rsl_rl/himmy_mark2_flat/2026-02-10_23-59-23_no_spine_turn_ang/model_8400.pt --num_envs 1











# 查看显卡占用
nvidia-smi

# 查看tersorboard
tensorboard --logdir=/root/gpufree-data/Fast-Quadruped/logs/rsl_rl/himmy_mark2_flat --port 6007

# 可视化AMP运动轨迹
python scripts/tools/visualize_amp_motion_simple.py --motion_file pace1.txt





# 可视化Himmy运动轨迹
cd /data/zmli/Fast-Quadruped && conda activate isaac-lab && python scripts/tools/visualize_himmy_motion.py --motion_file trot_forward1.txt --output_dir /data/zmli/Fast-Quadruped/logs/







--log
--log_no_spine



默认为机器人视角
--viewer_world








