#!/bin/bash
exec gnome-terminal -- /bin/bash -c "
cd /mnt/Storage/RVU
source ~/miniconda3/etc/profile.d/conda.sh
conda activate video-upscaler

while true; do
    echo '[COMFY] Pornesc serverul de autentificare pe 0.0.0.0:8188...'
    python real_video_webui.py
    echo '[COMFY] S-a oprit. Repornesc în 2s...'
    sleep 2
done
exec bash"
