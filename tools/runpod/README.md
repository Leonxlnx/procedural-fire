# Rendering the final video on a RunPod GPU

The renderer is headless Chrome + WebGPU, so it runs unchanged on a Linux NVIDIA pod. On an RTX 4090 the
1080×1920 hero render should take roughly 15–25 min, compared with about 3 h on the local integrated GPU.

## 1. Start the pod (RunPod web UI)

- GPU: RTX 4090 (L40S / A6000 also fine), any Ubuntu 22.04/24.04 template (e.g. RunPod PyTorch).
- Environment variable: `NVIDIA_DRIVER_CAPABILITIES=all` — required. Without it the container has no Vulkan
  driver and Chrome's WebGPU falls back to the CPU.
- Enable SSH with your public key, and give it ~20 GB of container disk.

## 2. Upload the project (from this PC, in `D:\fire`)

```powershell
ssh -p PORT -i ~/.ssh/KEY root@HOST "mkdir -p /workspace/fire"
scp -P PORT -i ~/.ssh/KEY -r index.html package.json package-lock.json src tools root@HOST:/workspace/fire/
```

## 3. Set up and render (on the pod)

```bash
cd /workspace/fire && bash tools/runpod/setup.sh
```

`setup.sh` installs Node 22, Chrome and ffmpeg, checks that a Vulkan GPU is visible, and renders one test frame.
Its output must show an NVIDIA GPU on the `GPU:` line. Then:

```bash
cd /workspace/fire && node tools/render_final.mjs
```

This writes `out/video/final_raw.mp4` (picture) and then `out/video/final_final.mp4` (picture + soundtrack).

## 4. Download

The pod's ffmpeg has no MediaFoundation AAC encoder, so mux the sound on this PC (it avoids AAC peak overshoot):

```powershell
scp -P PORT -i ~/.ssh/KEY root@HOST:/workspace/fire/out/video/final_raw.mp4 root@HOST:/workspace/fire/out/video/final_raw_frames.json out/video/
node tools/finish_video.mjs --video out/video/final_raw.mp4 --crf 17
```

Stop the pod afterwards; it bills by the minute.
