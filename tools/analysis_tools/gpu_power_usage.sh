#!/bin/bash

# Start monitoring GPU power usage
nvidia-smi --loop-ms=100 --id=0 --format=csv --query-gpu=power.draw --query-gpu-uuid > power_usage.log & POWER_MONITOR_PID=$!

# Run your program here
# C3D
python tools/test.py configs/recognition/c3d/c3d_sports1m-pretrained_8xb30-16x1x1-45e_ucf101-rgb.py checkpoints/c3d/c3d_sports1m_16x1x1_45e_ucf101_rgb_20201021-26655025.pth
# X3D-M
# python tools/test.py configs/recognition/x3d/x3d_m_16x5x1_facebook-kinetics400-rgb.py checkpoints/x3d/x3d_m_facebook_16x5x1_kinetics400_rgb_20201027-3f42382a.pth

# Stop monitoring GPU power usage
kill $POWER_MONITOR_PID

# Calculate average power usage
python get_power_usage.py --file power_usage.log