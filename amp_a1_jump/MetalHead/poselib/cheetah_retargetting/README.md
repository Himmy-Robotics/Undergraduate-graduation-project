cd /data/zmli/Fast-Quadruped/amp_a1_jump/MetalHead && python poselib/cheetah_retargetting/visualize_himmy_motion.py --npy_file poselib/data/cheetah_data_retarget_npy/fte19-2-27-romeo-flick_gallop_to_stop_amp_0_199_fte19-2-27-romeo-flick_gallop_to_stop.npy

cd /data/zmli/Fast-Quadruped/amp_a1_jump/MetalHead/poselib/cheetah_retargetting && python cheetah_key_points_retargetting.py


cd /data/zmli/Fast-Quadruped/amp_a1_jump/MetalHead && python poselib/cheetah_retargetting/cheetah_json_exporter.py


cd /data/zmli/Fast-Quadruped/amp_a1_jump/MetalHead && python poselib/convert_json_to_txt.py