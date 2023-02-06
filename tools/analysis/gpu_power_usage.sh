#!/bin/bash

# Start monitoring GPU power usage
nvidia-smi --loop-ms=100 --id=0 --format=csv --query-gpu=power.draw --query-gpu-uuid > power_usage.log & POWER_MONITOR_PID=$!

# Run your program here
# C3D
# python demo/demo.py configs/recognition/c3d/c3d_sports1m_16x1x1_45e_ucf101_rgb_v2.py checkpoints/c3d/c3d_sports1m_16x1x1_45e_ucf101_rgb_20201021-26655025.pth demo/demo.mp4 tools/data/ucf101/label_map.txt
# X3D-M
# python demo/demo.py configs/recognition/x3d/x3d_m_16x5x1_facebook_kinetics400_rgb.py checkpoints/x3d/x3d_m_facebook_16x5x1_kinetics400_rgb_20201027-3f42382a.pth demo/demo.mp4 tools/data/kinetics/label_map_k400.txt
# Slowonly
# python demo/demo.py configs/recognition/slowonly/slowonly_imagenet_pretrained_r50_8x8x1_150e_kinetics400_rgb.py checkpoints/slowonly/slowonly_imagenet_pretrained_r50_8x8x1_150e_kinetics400_rgb_20200912-3f9ce182.pth demo/demo.mp4 tools/data/kinetics/label_map_k400.txt
# R(2+1)D-34
python demo/demo.py configs/recognition/r2plus1d/r2plus1d_r34_8x8x1_180e_kinetics400_rgb.py checkpoints/r2plus1d/r2plus1d_r34_8x8x1_180e_kinetics400_rgb_20200618-3fce5629.pth demo/demo.mp4 tools/data/kinetics/label_map_k400.txt

# Stop monitoring GPU power usage
kill $POWER_MONITOR_PID

# Calculate average power usage
python get_power_usage.py --file power_usage.log

