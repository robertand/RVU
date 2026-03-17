import os
import sys
import argparse
import cv2
import torch
import numpy as np
import subprocess
from pathlib import Path

# Try to import spandrel for model loading
try:
    from spandrel import ImageModelDescriptor, ModelLoader
    SPANDREL_AVAILABLE = True
except ImportError:
    SPANDREL_AVAILABLE = False

def get_video_info(input_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return None
    info = {
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    }
    cap.release()
    return info

def process_frame(model, frame, device):
    if model is None:
        return frame

    # Convert BGR to RGB and to tensor
    img = frame.astype(np.float32) / 255.0
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img = img.unsqueeze(0).to(device)

    with torch.no_grad():
        # spandrel model descriptor can be called directly or via .model
        if hasattr(model, 'model'):
            output = model.model(img)
        else:
            output = model(img)

    # Convert back to numpy BGR
    output = output.squeeze(0).cpu().numpy()
    output = np.clip(np.transpose(output, (1, 2, 0)) * 255.0, 0, 255).astype(np.uint8)
    output = output[:, :, [2, 1, 0]]
    return output

def main():
    parser = argparse.ArgumentParser(description="Simple Stable RVE Backend")
    parser.add_argument("-i", "--input", help="Input video path")
    parser.add_argument("-o", "--output", help="Output video path")
    parser.add_argument("--upscale_model", help="Path to upscale model")
    parser.add_argument("--interpolate_model", help="Path to interpolation model (currently placeholder)")
    parser.add_argument("--interpolate_factor", type=float, help="Interpolation multiplier")
    parser.add_argument("--extra_restoration_models", action='append', help="Paths to extra restoration models (denoise, etc.)")
    parser.add_argument("--override_upscale_scale", type=int, help="Scale factor")
    parser.add_argument("--backend", default="pytorch", help="Processing backend")
    parser.add_argument("--pytorch_gpu_id", type=int, default=0, help="GPU ID")
    parser.add_argument("--crf", default="18", help="CRF for encoding")
    parser.add_argument("--version", action="store_true", help="Print version")
    parser.add_argument("--list_backends", action="store_true", help="List backends")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output")

    args, unknown = parser.parse_known_args()

    if args.version:
        print("Simple-Backend-1.1.0")
        return

    # Set GPU environment variable if specified and not already set
    if args.pytorch_gpu_id is not None and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.pytorch_gpu_id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.list_backends:
        print(f"Available Backends: ['pytorch']")
        print(f"Device: {device}")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                print(f"PyTorch GPU {i}: {torch.cuda.get_device_name(i)}")
        return

    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found")
        sys.exit(1)

    info = get_video_info(args.input)
    if not info:
        print("Error: Could not open video")
        sys.exit(1)

    # Load models
    upscale_model = None
    restoration_models = []

    if SPANDREL_AVAILABLE:
        if args.upscale_model and os.path.exists(args.upscale_model):
            print(f"Loading upscale model: {args.upscale_model}")
            try:
                upscale_model = ModelLoader().load_from_file(args.upscale_model).to(device).eval()
            except Exception as e:
                print(f"Error loading upscale model: {e}")

        if args.extra_restoration_models:
            for m_path in args.extra_restoration_models:
                if os.path.exists(m_path):
                    print(f"Loading restoration model: {m_path}")
                    try:
                        restoration_models.append(ModelLoader().load_from_file(m_path).to(device).eval())
                    except Exception as e:
                        print(f"Error loading restoration model {m_path}: {e}")
    else:
        print("Warning: spandrel not available. AI models will not be used. Falling back to resizing.")

    # Determine output resolution
    # Try to get scale from model if possible
    scale = args.override_upscale_scale or 2
    if upscale_model and hasattr(upscale_model, 'scale'):
        scale = upscale_model.scale

    output_w = info['width'] * scale
    output_h = info['height'] * scale

    print(f"Processing: {args.input}")
    print(f"Resolution: {info['width']}x{info['height']} -> {output_w}x{output_h}, FPS: {info['fps']}, Frames: {info['total_frames']}")

    if args.upscale_model:
        print(f"Using Upscale Model: {args.upscale_model}")

    if args.interpolate_model:
        print(f"Interpolation requested with: {args.interpolate_model} (Factor: {args.interpolate_factor or 2.0}x)")
        print("Note: Frame interpolation is currently handled as a pass-through in this simplified backend.")

    # First, create a temporary video file for the processed frames
    temp_video = args.output + ".temp.mp4"

    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f"{output_w}x{output_h}",
        '-pix_fmt', 'bgr24', '-r', str(info['fps']),
        '-i', '-',
        '-c:v', 'libx264', '-crf', args.crf, '-pix_fmt', 'yuv420p',
        temp_video
    ]

    process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
    cap = cv2.VideoCapture(args.input)
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Apply restoration models (denoise, etc.) first
            for r_model in restoration_models:
                frame = process_frame(r_model, frame, device)

            # Apply upscale model
            if upscale_model:
                processed_frame = process_frame(upscale_model, frame, device)
                # If model output resolution doesn't match expected output resolution, resize it
                if processed_frame.shape[1] != output_w or processed_frame.shape[0] != output_h:
                    processed_frame = cv2.resize(processed_frame, (output_w, output_h), interpolation=cv2.INTER_LANCZOS4)
            else:
                processed_frame = cv2.resize(frame, (output_w, output_h), interpolation=cv2.INTER_CUBIC)

            process.stdin.write(processed_frame.tobytes())

            frame_idx += 1
            if frame_idx % 10 == 0:
                print(f"Current Frame: {frame_idx}")
                sys.stdout.flush()

    finally:
        cap.release()
        process.stdin.close()
        process.wait()

    # Merge audio from original video
    print("Merging audio...")
    merge_cmd = [
        'ffmpeg', '-y',
        '-i', temp_video,
        '-i', args.input,
        '-map', '0:v:0',
        '-map', '1:a:0?',
        '-c:v', 'copy',
        '-c:a', 'copy',
        args.output
    ]
    subprocess.run(merge_cmd)

    # Clean up temp file
    if os.path.exists(temp_video):
        os.remove(temp_video)

    print("Processing completed successfully.")

if __name__ == "__main__":
    main()
