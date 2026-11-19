#!/bin/bash
# Reload the NVIDIA driver so /etc/modprobe.d/nvidia-profiling.conf (non-admin Nsight Compute) takes effect without a
# reboot. Run:  sudo /home/turbotage/Documents/GitHub/HastyCompute/research/experiments/sample_kernels/reload_nvidia.sh
# Holds the session GPU lock for the whole reload (waits for a running GPU step of the agents' queues, blocks new ones).
# Stop your own GPU programs (e.g. the mnist visualiser) first; the script lists and refuses if any compute process is
# still on the GPU.
set -u
LOCK=/tmp/claude-1000/-home-turbotage-Documents-GitHub-HastyCompute/c25caa1d-d864-44cf-9019-66f29ac9cb24/scratchpad/gpu.lock
exec 9>>$LOCK
echo "waiting for the GPU lock ..."; flock 9; echo "GPU lock held"
apps=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader)
if [ -n "$apps" ]; then echo "compute processes still on the GPU, stop them first:"; echo "$apps"; exit 1; fi
systemctl stop gdm3 nvidia-persistenced coolercontrold   # coolercontrold holds /dev/nvidia*; fans fall back to firmware control for the few seconds
sleep 2
if ! modprobe -r nvidia_drm nvidia_modeset nvidia_uvm nvidia; then
  echo "unload failed; holders:"; fuser -v /dev/nvidia* 2>&1; lsmod | grep -E '^nvidia'
  modprobe nvidia_modeset; modprobe nvidia_drm; systemctl start nvidia-persistenced coolercontrold gdm3; exit 1
fi
modprobe nvidia && modprobe nvidia_uvm && modprobe nvidia_modeset && modprobe nvidia_drm
systemctl start nvidia-persistenced coolercontrold gdm3
sleep 2
nvidia-smi -pl 450
grep RmProfilingAdminOnly /proc/driver/nvidia/params
