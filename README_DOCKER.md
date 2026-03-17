# 首次创建容器（只做一次）

docker run --name isaac-lab-dev \
  -it \
  --entrypoint bash \
  --gpus '"device=2"' \
  --ipc=host \
  --shm-size=32g \
  -e "ACCEPT_EULA=Y" \
  -e "PRIVACY_CONSENT=Y" \
  -e "OMNI_KIT_ALLOW_ROOT=1" \
  --network=host \
  -v ~/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
  -v ~/isaac-sim/cache/ov:/root/.cache/ov:rw \
  -v ~/isaac-sim/cache/pip:/root/.cache/pip:rw \
  -v ~/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
  -v ~/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
  -v ~/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
  -v ~/isaac-sim/data:/root/.local/share/ov/data:rw \
  -v ~/isaac-sim/documents:/root/Documents:rw \
  -v ~/Undergraduate-graduation-project:/workspace/Undergraduate-graduation-project:rw \
  nvcr.io/nvidia/isaac-lab:2.3.0

# 进入后安装依赖：

cd /workspace/Undergraduate-graduation-project
/workspace/isaaclab/_isaac_sim/python.sh -m pip install -e source/robot_lab
/workspace/isaaclab/_isaac_sim/python.sh -m pip install pybullet


# 在一个容器中开启多个终端
docker exec -it isaac-lab-dev bash

# 启动已有容器（不需要重新 pip install！）
docker start -i isaac-lab-dev

# 进入后直接训练
cd /workspace/Undergraduate-graduation-project
/workspace/isaaclab/_isaac_sim/python.sh \
    scripts/reinforcement_learning/rsl_rl/train.py \
    --task=RobotLab-Isaac-Velocity-Flat-Run-Himmy-Mark2-v0 \
    --headless \
    --num_envs=2048






# 如果需要换 GPU

# 删除旧容器
docker stop isaac-lab-dev
docker rm isaac-lab-dev

# 用新 GPU 重新创建（比如换成 GPU 3）
docker run --name isaac-lab-dev \
  -it \
  --entrypoint bash \
  --gpus '"device=3"' \
  ... （其他参数不变）

# 进入后重新安装依赖（只需这一次）
cd /workspace/Undergraduate-graduation-project
/workspace/isaaclab/_isaac_sim/python.sh -m pip install -e source/robot_lab
/workspace/isaaclab/_isaac_sim/python.sh -m pip install pybullet