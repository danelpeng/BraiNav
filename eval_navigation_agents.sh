# Script to evaluate navigation agents
for i in 1 2 3
do
agent=YOUR_PATH/checkpoints/Pointnav-RoboTHOR-Vanilla-RGB-ResNet-DDPPO/rnav_pointnav_vanilla_rgb_resnet_ddppo_clean/xxx.pt
date=2025-7-6

# ================================================================================
# PointNav RGB agents
# ================================================================================

# Clean
python main.py \
    -o storage/robothor-pointnav-rgb-resnetgru-ddppo-eval \
    -b projects/robustnav_baselines/experiments/robustnav_eval pointnav_robothor_vanilla_rgb_resnet_ddppo \
    -c $agent \
    -t $date \
    -et rnav_pointnav_vanilla_rgb_resnet_ddppo_clean \
    -s 12345 \
    -e \
    -tsg 0


# Visual Corruptions
# ********************************************************************************

# (Defocus Blur, Motion Blur, Spatter, Low Lighting, Speckle Noise)
# for CORR in Defocus_Blur
for CORR in Defocus_Blur Lighting Speckle_Noise Spatter Motion_Blur
do
    python main.py \
        -o storage/robothor-pointnav-rgb-resnetgru-ddppo-eval \
        -b projects/robustnav_baselines/experiments/robustnav_eval pointnav_robothor_vanilla_rgb_resnet_ddppo \
        -c $agent \
        -t $date \
        -et rnav_pointnav_vanilla_rgb_resnet_ddppo_"$CORR"_s5 \
        -s 12345 \
        -e \
        -tsg 0 \
        -vc $CORR \
        -vs 5
done

# Lower-FOV
python main.py \
    -o storage/robothor-pointnav-rgb-resnetgru-ddppo-eval \
    -b projects/robustnav_baselines/experiments/robustnav_eval pointnav_robothor_vanilla_rgb_resnet_ddppo_fov \
    -c $agent \
    -t $date \
    -et rnav_pointnav_vanilla_rgb_resnet_ddppo_fov \
    -s 12345 \
    -e \
    -tsg 0

# Camera-Crack
python main.py \
    -o storage/robothor-pointnav-rgb-resnetgru-ddppo-eval \
    -b projects/robustnav_baselines/experiments/robustnav_eval pointnav_robothor_vanilla_rgb_resnet_ddppo_cam_crack \
    -c $agent \
    -t $date \
    -et rnav_pointnav_vanilla_rgb_resnet_ddppo_cam_crack \
    -s 12345 \
    -e \
    -tsg 0

done
