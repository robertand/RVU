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

def process_frame(model, frame, device, tilesize=0, overlap=16, precision="auto", tta=False):
    if model is None:
        return frame

    # Convert BGR to RGB
    img = frame.astype(np.float32) / 255.0
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img = img.unsqueeze(0).to(device)

    # Set precision
    if precision == "float16" or (precision == "auto" and device.type == "cuda"):
        img = img.half()
        model = model.half()
    else:
        img = img.float()
        model = model.float()

    # Handle Tiling
    b, c, h, w = img.shape
    scale = model.scale if hasattr(model, 'scale') else 1

    if tilesize > 0:
        # Tiling Implementation
        stride = tilesize - overlap
        output_h, output_w = h * scale, w * scale
        output = torch.zeros((b, c, output_h, output_w), device=device, dtype=img.dtype)
        weight = torch.zeros((b, c, output_h, output_w), device=device, dtype=img.dtype)

        for y in range(0, h, stride):
            for x in range(0, w, stride):
                # Extract tile
                y1, x1 = y, x
                y2, x2 = min(y + tilesize, h), min(x + tilesize, w)
                tile = img[:, :, y1:y2, x1:x2]

                # Process tile
                with torch.no_grad():
                    if hasattr(model, 'model'):
                        tile_out = model.model(tile)
                    else:
                        tile_out = model(tile)

                # Place tile back
                oy1, ox1 = y1 * scale, x1 * scale
                oy2, ox2 = oy1 + tile_out.shape[2], ox1 + tile_out.shape[3]
                output[:, :, oy1:oy2, ox1:ox2] += tile_out
                weight[:, :, oy1:oy2, ox1:ox2] += 1.0

        output /= weight
        return output

    with torch.no_grad():
        # Handle TTA if requested
        if tta:
            outputs = []
            for flip in [False, True]:
                curr_img = torch.flip(img, [3]) if flip else img
                curr_out = model.model(curr_img) if hasattr(model, 'model') else model(curr_img)
                if flip: curr_out = torch.flip(curr_out, [3])
                outputs.append(curr_out)
            output = torch.stack(outputs).mean(0)
        else:
            output = model.model(img) if hasattr(model, 'model') else model(img)

    # Convert back to numpy BGR
    output = output.squeeze(0).float().cpu().numpy()
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
    parser.add_argument("--tilesize", type=int, default=0, help="Tile size for processing (0 = auto)")
    parser.add_argument("--overlap", type=int, default=16, help="Overlap between tiles")
    parser.add_argument("--precision", choices=["float16", "float32", "auto"], default="auto", help="Inference precision")
    parser.add_argument("--tta", action="store_true", help="Enable Test Time Augmentation")
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
    model_scale = None

    if SPANDREL_AVAILABLE:
        if args.upscale_model and os.path.exists(args.upscale_model):
            print(f"Loading upscale model: {args.upscale_model}")
            try:
                # Load model using spandrel
                descriptor = ModelLoader().load_from_file(args.upscale_model)
                upscale_model = descriptor.to(device).eval()

                # Check for scale in descriptor
                if hasattr(descriptor, 'scale'):
                    model_scale = descriptor.scale
                    print(f"Model architecture: {descriptor.architecture.name if hasattr(descriptor, 'architecture') else 'Unknown'}, Native scale: {model_scale}x")
                elif hasattr(upscale_model, 'scale'):
                    model_scale = upscale_model.scale
                    print(f"Detected scale from model attribute: {model_scale}x")
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
    if args.override_upscale_scale:
        if args.override_upscale_scale > 10:  # Assume it's a target width in pixels
            output_w = args.override_upscale_scale
            output_h = int(info['height'] * (output_w / info['width']))
            scale = output_w / info['width']
            print(f"Target resolution override: {output_w}x{output_h} (Scale: {scale:.2f}x)")
        else:  # Assume it's a scale factor
            scale = args.override_upscale_scale
            output_w = info['width'] * scale
            output_h = info['height'] * scale
            print(f"Scale factor override: {scale}x ({output_w}x{output_h})")
    else:
        scale = model_scale or 2
        output_w = info['width'] * scale
        output_h = info['height'] * scale
        print(f"Using {'model native' if model_scale else 'default'} scale: {scale}x ({output_w}x{output_h})")

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
                frame = process_frame(r_model, frame, device, args.tilesize, args.overlap, args.precision, args.tta)

            # Apply upscale model
            if upscale_model:
                processed_frame = process_frame(upscale_model, frame, device, args.tilesize, args.overlap, args.precision, args.tta)
                # Only resize if the user requested a specific resolution that differs from the model's native output
                if processed_frame.shape[1] != output_w or processed_frame.shape[0] != output_h:
                    processed_frame = cv2.resize(processed_frame, (output_w, output_h), interpolation=cv2.INTER_LANCZOS4)
            else:
                # No AI model, use traditional upscaling
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
