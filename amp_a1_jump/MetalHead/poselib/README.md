## poselib

该`poselib` 是基于IssacGymEnvs中AMP Humanoid motion retargeting代码进行二次开发，完成了基于IK和Key points的a1 motion retargeting

## 原始动捕数据
使用AI4Animation提供的动捕数据: https://github.com/sebastianstarke/AI4Animation/tree/master/AI4Animation/SIGGRAPH_2018, 原始动捕数据为bvh格式，需要在windows下安装motion builder，然后将其导出为fbx格式。另外需要注意，AI4Animation中的数据除了第一帧外，其余帧均重复一次。



## a1 motion retargeting 过程
- 准备好fbx动捕文件放在`./data/amp_hardware_a1/fbx文件名/`目录下
- 在`load_config.json`中修改`file_name`(fbx文件名), `clip`(裁剪片段区间), `remarks`(输出文件名)
- 运行`fbx_importer.py`读取fbx文件, 生成npy文件, 去除相邻重复帧
- 运行`key_points_retargetting.py`生成retargeting后的npy文件
- 运行`json_exporter.py`导出训练所使用的json文件



python himmy_key_points_retargetting.py
python visualize_himmy_motion.py 


cd /data/zmli/Fast-Quadruped/amp_a1_jump/MetalHead/poselib && python himmy_generate_amp_himmy_mark2_tpose.py


cd /data/zmli/Fast-Quadruped/amp_a1_jump/MetalHead/poselib && python himmy_key_points_retargetting.py


cd /data/zmli/Fast-Quadruped/amp_a1_jump/MetalHead/poselib && python visualize_himmy_motion.py --max_frames 30

python ./himmy_json_exporter.py

gallop_forward0
gallop_forward1
trot_forward0   124-273
turn_left0      177-260
turn_left1      376-419
turn_right0     273-355
turn_right1     435-490