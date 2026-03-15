"""
REAL Video Enhancer WebUI - Cu selecție GPU și comparație video
"""

import os
import sys
import json
import time
import shutil
import threading
import subprocess
import tempfile
import base64
import signal
import psutil
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import queue

# Third-party imports
try:
    import cv2
    import numpy as np
    from flask import Flask, render_template, request, jsonify, send_file, Response
    from flask_socketio import SocketIO, emit
    from werkzeug.utils import secure_filename
    import ffmpeg
except ImportError:
    print("Installing required packages...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", 
                          "opencv-python", "numpy", "flask", "flask-socketio",
                          "python-socketio", "werkzeug", "ffmpeg-python",
                          "psutil", "eventlet"])
    import cv2
    import numpy as np
    from flask import Flask, render_template, request, jsonify, send_file, Response
    from flask_socketio import SocketIO, emit
    from werkzeug.utils import secure_filename
    import ffmpeg

# ============================================================================
# CONFIGURATION AND CONSTANTS
# ============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@dataclass
class Config:
    """Configuration for the application"""
    BASE_DIR = BASE_DIR
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
    OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs")
    MODELS_FOLDER = "/media/prouser/Storage/PlayGround/RVU/models/"
    TEMP_FOLDER = os.path.join(BASE_DIR, "temp")
    TEMPLATE_FOLDER = os.path.join(BASE_DIR, "templates")
    RVE_BACKEND_PATH = "/media/prouser/Storage/PlayGround/RVU/backend"
    RVE_PYTHON_PATH = "/media/prouser/Storage/PlayGround/RVU/python/python/bin/python3"
    ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv', 'webm', 'flv', 'wmv'}
    HOST = "127.0.0.1"
    PORT = 5000
    DEBUG = False
    SECRET_KEY = "real-video-enhancer-secret-key-2024"
    MAX_UPLOAD_SIZE = 2 * 1024 * 1024 * 1024
    PREVIEW_FPS = 10
    PREVIEW_MAX_WIDTH = 640
    AVAILABLE_BACKENDS = ["pytorch", "ncnn", "tensorrt"]
    DEFAULT_BACKEND = "pytorch"
    RVE_MAX_CONCURRENT_JOBS = 1
    MODEL_DOWNLOAD_URLS = {
        # Upscale Models
        "realesr-animevideov3": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth",
        "realesr-general-x4v3": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
        "4x_Nomos8k_32000G": "https://github.com/BlueAmulet/Real-ESRGAN-ncnn-vulkan/releases/download/2024.11.04/4x_Nomos8k_32000G.pth",
        "2x_AnimeJaNai_V2_Compact": "https://github.com/BlueAmulet/Real-ESRGAN-ncnn-vulkan/releases/download/2024.11.04/2x_AnimeJaNai_V2_Compact.pth",
        "2xLollipopAnimeSharpV2": "https://github.com/BlueAmulet/Real-ESRGAN-ncnn-vulkan/releases/download/2024.11.04/2xLollipopAnimeSharpV2.pth",
        "4xLoyalPromo_240000": "https://github.com/BlueAmulet/Real-ESRGAN-ncnn-vulkan/releases/download/2024.11.04/4xLoyalPromo_240000.pth",
        "2x_DF2K": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        "4x_DF2K": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "4x_UniversalUpscalerV2-Sharp": "https://github.com/BlueAmulet/Real-ESRGAN-ncnn-vulkan/releases/download/2024.11.04/4x_UniversalUpscalerV2-Sharp.pth",
        "2x_Astronomical": "https://github.com/BlueAmulet/Real-ESRGAN-ncnn-vulkan/releases/download/2024.11.04/2x_Astronomical.pth",
        
        # Interpolation Models
        "rife4.7": "https://github.com/BlueAmulet/Real-ESRGAN-ncnn-vulkan/releases/download/2024.11.04/rife4.7.pkl",
        "rife4.26": "https://github.com/BlueAmulet/Real-ESRGAN-ncnn-vulkan/releases/download/2024.11.04/rife4.26.pkl",
        "gmfss_fortuna": "https://github.com/BlueAmulet/Real-ESRGAN-ncnn-vulkan/releases/download/2024.11.04/gmfss_fortuna.pkl",
        "gmfss_fortuna_v2": "https://github.com/BlueAmulet/Real-ESRGAN-ncnn-vulkan/releases/download/2024.11.04/gmfss_fortuna_v2.pkl",
        "RIFE4.14": "https://github.com/hzwer/Practical-RIFE/releases/download/v4.14/flownet.pkl",
        
        # Denoise Models
        "drunet_deblocking_qual": "https://github.com/cszn/KAIR/releases/download/v1.0/deblocking_drunet_color.pth",
        "drunet_gray": "https://github.com/cszn/KAIR/releases/download/v1.0/dncnn_gray_blind.pth",
        "drunet_color": "https://github.com/cszn/KAIR/releases/download/v1.0/dncnn_color_blind.pth",
        "drunet_deblocking_compression": "https://github.com/cszn/KAIR/releases/download/v1.0/deblocking_compact.pth",
        
        # Decompress Models
        "deh264-compact": "https://github.com/BlueAmulet/Real-ESRGAN-ncnn-vulkan/releases/download/2024.11.04/deh264-compact.pth",
        "deh264": "https://github.com/BlueAmulet/Real-ESRGAN-ncnn-vulkan/releases/download/2024.11.04/deh264.pth",
    }

config = Config()

# ============================================================================
# APPLICATION SETUP
# ============================================================================

app = Flask(__name__, template_folder=config.TEMPLATE_FOLDER)
app.config['SECRET_KEY'] = config.SECRET_KEY
app.config['UPLOAD_FOLDER'] = config.UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = config.MAX_UPLOAD_SIZE

# Use eventlet for WebSocket support
try:
    import eventlet
    eventlet.monkey_patch()
    async_mode = 'eventlet'
    print("Using eventlet for WebSocket support")
except ImportError:
    async_mode = 'threading'
    print("eventlet not installed, falling back to threading")

socketio = SocketIO(
    app, 
    cors_allowed_origins="*", 
    async_mode=async_mode,
    ping_timeout=60,
    ping_interval=25,
    logger=False,
    engineio_logger=False,
    manage_session=False
)

# Create necessary directories
for folder in [config.UPLOAD_FOLDER, config.OUTPUT_FOLDER, config.TEMP_FOLDER, config.TEMPLATE_FOLDER]:
    Path(folder).mkdir(exist_ok=True)

print(f"Base directory: {config.BASE_DIR}")
print(f"Models folder: {config.MODELS_FOLDER}")
print(f"Backend path: {config.RVE_BACKEND_PATH}")
print(f"Python path: {config.RVE_PYTHON_PATH}")

# ============================================================================
# VIDEO INFO CLASS
# ============================================================================

@dataclass
class VideoInfo:
    """Video file information"""
    filename: str
    width: int
    height: int
    fps: float
    duration: float
    total_frames: int
    format: str
    codec: str
    size_bytes: int
    bit_depth: int = 8
    is_hdr: bool = False
    color_space: str = "unknown"
    pixel_format: str = "unknown"
    bitrate: str = "unknown"
    
    @classmethod
    def from_file(cls, filepath: str) -> 'VideoInfo':
        """Extract video information using OpenCV"""
        try:
            # Get file size
            size_bytes = os.path.getsize(filepath)
            
            # Get video info using OpenCV
            cap = cv2.VideoCapture(filepath)
            
            if not cap.isOpened():
                raise Exception("Could not open video file")
            
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if fps > 0:
                duration = total_frames / fps
            else:
                duration = 0
            
            # Get filename and extension
            filename = os.path.basename(filepath)
            format_name = os.path.splitext(filename)[1][1:].upper()
            
            # Try to get codec info
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)]) if fourcc != 0 else "unknown"
            
            # Get additional info using ffprobe
            bit_depth = 8
            is_hdr = False
            color_space = "unknown"
            pixel_format = "unknown"
            bitrate = "unknown"
            
            try:
                probe = ffmpeg.probe(filepath)
                video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
                
                if video_stream:
                    bitrate = video_stream.get('bit_rate', 'unknown')
                    pix_fmt = video_stream.get('pix_fmt', 'unknown')
                    pixel_format = pix_fmt
                    
                    # Detect HDR
                    color_transfer = video_stream.get('color_transfer', 'unknown')
                    color_space_val = video_stream.get('color_space', 'unknown')
                    color_space = color_space_val
                    
                    if 'bt2020' in color_space_val.lower() or 'smpte2084' in str(color_transfer).lower():
                        is_hdr = True
                    
                    # Detect bit depth
                    if '10le' in pix_fmt or '10be' in pix_fmt or 'p010le' in pix_fmt:
                        bit_depth = 10
                    elif '12le' in pix_fmt or '12be' in pix_fmt:
                        bit_depth = 12
                    elif '16le' in pix_fmt or '16be' in pix_fmt:
                        bit_depth = 16
            except:
                pass
            
            cap.release()
            
            return cls(
                filename=filename,
                width=width,
                height=height,
                fps=fps,
                duration=duration,
                total_frames=total_frames,
                format=format_name,
                codec=codec,
                size_bytes=size_bytes,
                bit_depth=bit_depth,
                is_hdr=is_hdr,
                color_space=color_space,
                pixel_format=pixel_format,
                bitrate=bitrate
            )
            
        except Exception as e:
            print(f"Error analyzing video {filepath}: {e}")
            
            # Fallback: return basic info
            return cls(
                filename=os.path.basename(filepath),
                width=0,
                height=0,
                fps=0,
                duration=0,
                total_frames=0,
                format=os.path.splitext(filepath)[1][1:].upper(),
                codec="unknown",
                size_bytes=os.path.getsize(filepath),
                bit_depth=8,
                is_hdr=False,
                color_space="unknown",
                pixel_format="unknown",
                bitrate="unknown"
            )

# ============================================================================
# VERIFICARE BACKEND ȘI GPU-URI
# ============================================================================

RVE_BACKEND_FILE = os.path.join(config.RVE_BACKEND_PATH, "rve-backend.py")
RVE_BACKEND_AVAILABLE = os.path.exists(RVE_BACKEND_FILE)

# Detect available GPUs
def detect_gpus():
    """Detect available GPUs in the system"""
    gpus = []
    
    try:
        # Try to use nvidia-smi
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,name', '--format=csv,noheader'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if ',' in line:
                    idx, name = line.split(',', 1)
                    gpus.append({
                        'id': int(idx.strip()),
                        'name': name.strip(),
                        'type': 'nvidia'
                    })
                else:
                    # Try parsing without comma
                    parts = line.strip().split()
                    if parts:
                        gpus.append({
                            'id': len(gpus),
                            'name': ' '.join(parts),
                            'type': 'nvidia'
                        })
    except:
        pass
    
    # If no NVIDIA GPUs found, add placeholder
    if not gpus:
        gpus = [
            {'id': 0, 'name': 'GPU 0 (Default)', 'type': 'default'},
            {'id': 1, 'name': 'GPU 1', 'type': 'default'},
            {'id': 2, 'name': 'GPU 2', 'type': 'default'},
            {'id': 3, 'name': 'GPU 3', 'type': 'default'}
        ]
    
    return gpus

AVAILABLE_GPUS = detect_gpus()

print(f"RVE backend available: {RVE_BACKEND_AVAILABLE}")
if RVE_BACKEND_AVAILABLE:
    print(f"RVE backend path: {RVE_BACKEND_FILE}")
    print(f"Python executable: {config.RVE_PYTHON_PATH}")
    
    if not os.path.exists(config.RVE_PYTHON_PATH):
        print(f"WARNING: Python not found at {config.RVE_PYTHON_PATH}")
        RVE_BACKEND_AVAILABLE = False
else:
    print(f"RVE backend file not found at: {RVE_BACKEND_FILE}")

print(f"Available GPUs: {[gpu['name'] for gpu in AVAILABLE_GPUS]}")

# ============================================================================
# CREATE HTML TEMPLATE WITH GPU SELECTION AND COMPARISON
# ============================================================================

def create_html_template():
    """Create the HTML template file with GPU selection and video comparison"""
    template_path = os.path.join(config.TEMPLATE_FOLDER, "index.html")
    
    # Generate GPU options HTML
    gpu_options_html = ""
    for gpu in AVAILABLE_GPUS:
        selected = "selected" if gpu['id'] == 0 else ""
        gpu_options_html += f'<option value="{gpu["id"]}" {selected}>{gpu["name"]}</option>'
    
    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>REAL Video Enhancer</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: #0f172a;
            color: #f8fafc;
            line-height: 1.6;
            height: 100vh;
            display: flex;
            flex-direction: column;
        }}
        
        .container {{
            display: flex;
            flex-direction: column;
            height: 100vh;
            padding: 0;
            margin: 0;
        }}
        
        header {{
            background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
            padding: 15px 20px;
            flex-shrink: 0;
        }}
        
        .header-content {{
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .logo {{
            display: flex;
            align-items: center;
            gap: 15px;
        }}
        
        .logo i {{
            font-size: 28px;
            color: white;
        }}
        
        .logo-text h1 {{
            font-size: 22px;
            font-weight: 700;
            color: white;
        }}
        
        .logo-text p {{
            color: rgba(255, 255, 255, 0.8);
            font-size: 13px;
        }}
        
        .status-indicator {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 13px;
            font-weight: 600;
        }}
        
        .status-online {{
            background: rgba(16, 185, 129, 0.2);
            color: #10b981;
        }}
        
        .status-offline {{
            background: rgba(239, 68, 68, 0.2);
            color: #ef4444;
        }}
        
        .tab-container {{
            display: flex;
            gap: 5px;
            background: #334155;
            padding: 5px;
            flex-shrink: 0;
        }}
        
        .tab {{
            flex: 1;
            padding: 10px;
            text-align: center;
            background: transparent;
            border: none;
            color: #cbd5e1;
            cursor: pointer;
            border-radius: 6px;
            transition: all 0.2s;
            font-size: 14px;
        }}
        
        .tab.active {{
            background: #6366f1;
            color: white;
        }}
        
        .main-layout {{
            display: flex;
            flex: 1;
            overflow: hidden;
            position: relative;
        }}
        
        .sidebar {{
            background: #1e293b;
            width: 300px;
            transition: all 0.3s ease;
            box-shadow: 2px 0 10px rgba(0, 0, 0, 0.2);
            z-index: 100;
            position: absolute;
            left: 0;
            top: 0;
            bottom: 0;
            transform: translateX(0);
            overflow-y: auto;
        }}
        
        .sidebar.closed {{
            transform: translateX(-100%);
        }}
        
        .sidebar-toggle {{
            position: absolute;
            left: 300px;
            top: 20px;
            background: #6366f1;
            color: white;
            border: none;
            border-radius: 0 4px 4px 0;
            padding: 10px 6px;
            cursor: pointer;
            z-index: 101;
            transition: all 0.3s;
        }}
        
        .sidebar.closed + .sidebar-toggle {{
            left: 0;
        }}
        
        .sidebar-toggle:hover {{
            background: #4f46e5;
        }}
        
        .sidebar-toggle i {{
            font-size: 16px;
        }}
        
        .sidebar-content {{
            padding: 20px;
            height: 100%;
        }}
        
        .video-preview-area {{
            flex: 1;
            display: flex;
            flex-direction: column;
            background: #000;
            position: relative;
            margin-left: 300px;
            transition: margin-left 0.3s ease;
        }}
        
        .sidebar.closed + .sidebar-toggle + .video-preview-area {{
            margin-left: 0;
        }}
        
        .video-preview {{
            flex: 1;
            background: #000;
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
            overflow: hidden;
        }}
        
        .video-controls {{
            background: rgba(0, 0, 0, 0.8);
            padding: 10px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-shrink: 0;
        }}
        
        h2 {{
            color: #f8fafc;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #334155;
            font-size: 18px;
        }}
        
        h3 {{
            color: #cbd5e1;
            margin-bottom: 12px;
            font-size: 15px;
        }}
        
        .upload-area {{
            border: 2px dashed #475569;
            border-radius: 8px;
            padding: 30px 20px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
            margin-bottom: 20px;
        }}
        
        .upload-area:hover {{
            border-color: #6366f1;
            background: rgba(99, 102, 241, 0.05);
        }}
        
        .upload-area i {{
            font-size: 40px;
            color: #6366f1;
            margin-bottom: 10px;
        }}
        
        .btn {{
            padding: 10px 20px;
            border-radius: 8px;
            border: none;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
            display: inline-flex;
            align-items: center;
            gap: 6px;
            font-size: 14px;
        }}
        
        .btn-primary {{
            background: #6366f1;
            color: white;
        }}
        
        .btn-primary:hover {{
            background: #4f46e5;
        }}
        
        .btn-block {{
            width: 100%;
            justify-content: center;
        }}
        
        .video-placeholder {{
            text-align: center;
            color: #64748b;
        }}
        
        .video-placeholder i {{
            font-size: 64px;
            margin-bottom: 15px;
            opacity: 0.5;
        }}
        
        .video-player {{
            width: 100%;
            height: 100%;
            object-fit: contain;
            display: none;
        }}
        
        .video-frame {{
            width: 100%;
            height: 100%;
            object-fit: contain;
            display: none;
        }}
        
        .form-group {{
            margin-bottom: 15px;
        }}
        
        label {{
            display: block;
            margin-bottom: 6px;
            color: #cbd5e1;
            font-weight: 500;
            font-size: 14px;
        }}
        
        select, input[type="range"] {{
            width: 100%;
            padding: 8px;
            background: #334155;
            border: 1px solid #475569;
            border-radius: 6px;
            color: #f8fafc;
            font-size: 14px;
        }}
        
        .checkbox-group {{
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 8px;
        }}
        
        input[type="checkbox"] {{
            width: 16px;
            height: 16px;
            accent-color: #6366f1;
        }}
        
        .gpu-info {{
            background: rgba(99, 102, 241, 0.1);
            border-radius: 6px;
            padding: 8px;
            margin-top: 5px;
            font-size: 12px;
            color: #94a3b8;
        }}
        
        .tab-content {{
            display: none;
            padding: 20px;
            overflow-y: auto;
            height: calc(100% - 80px);
        }}
        
        .tab-content.active {{
            display: block;
        }}
        
        #videoInfo {{
            background: rgba(255, 255, 255, 0.05);
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
        }}
        
        #videoInfo h3 {{
            margin-top: 0;
        }}
        
        .info-row {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 5px;
        }}
        
        .info-label {{
            color: #94a3b8;
            font-size: 13px;
        }}
        
        .info-value {{
            color: #f8fafc;
            font-weight: 500;
            font-size: 13px;
        }}
        
        /* COMPARISON STYLES */
        .comparison-container {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: #000;
            display: none;
        }}
        
        .comparison-video {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            object-fit: contain;
        }}
        
        #enhancedVideo {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            object-fit: contain;
            z-index: 1;
        }}
        
        .original-video-container {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: 2;
            clip-path: inset(0 50% 0 0);
        }}
        
        .comparison-divider {{
            position: absolute;
            top: 0;
            left: 50%;
            transform: translateX(-50%);
            height: 100%;
            width: 4px;
            background: #6366f1;
            cursor: col-resize;
            z-index: 3;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        
        .divider-handle {{
            position: absolute;
            width: 36px;
            height: 36px;
            background: #6366f1;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 18px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        }}
        
        .comparison-controls {{
            position: absolute;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            display: flex;
            gap: 8px;
            z-index: 4;
            background: rgba(0,0,0,0.7);
            padding: 8px;
            border-radius: 8px;
        }}
        
        .comparison-btn {{
            padding: 6px 12px;
            background: #6366f1;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 13px;
            display: flex;
            align-items: center;
            gap: 5px;
        }}
        
        .comparison-btn:hover {{
            background: #4f46e5;
        }}
        
        .exit-comparison {{
            position: absolute;
            top: 20px;
            right: 20px;
            z-index: 4;
            background: rgba(0,0,0,0.7);
            color: white;
            border: none;
            border-radius: 4px;
            padding: 6px 12px;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 5px;
            font-size: 13px;
        }}
        
        .comparison-mode .video-placeholder,
        .comparison-mode #videoPlayer,
        .comparison-mode #videoFrame {{
            display: none;
        }}
        
        .comparison-mode .comparison-container {{
            display: block;
        }}
        
        .comparison-label {{
            position: absolute;
            top: 20px;
            padding: 4px 8px;
            background: rgba(0,0,0,0.7);
            color: white;
            border-radius: 4px;
            font-size: 12px;
            z-index: 4;
        }}
        
        .comparison-label.original {{
            left: 20px;
        }}
        
        .comparison-label.enhanced {{
            right: 20px;
        }}
        
        .comparison-fps-display {{
            position: absolute;
            top: 50px;
            left: 20px;
            padding: 4px 8px;
            background: rgba(0,0,0,0.7);
            color: #10b981;
            border-radius: 4px;
            font-size: 12px;
            z-index: 4;
        }}
        
        .comparison-time-display {{
            position: absolute;
            top: 80px;
            left: 20px;
            padding: 4px 8px;
            background: rgba(0,0,0,0.7);
            color: #f8fafc;
            border-radius: 4px;
            font-size: 12px;
            z-index: 4;
        }}
        
        .range-value {{
            display: inline-block;
            min-width: 40px;
            text-align: center;
            color: #6366f1;
            font-weight: 600;
        }}
        
        .jobs-list {{
            margin-top: 15px;
        }}
        
        .job-item {{
            background: #334155;
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 8px;
            border-left: 4px solid #6366f1;
        }}
        
        .job-item.completed {{
            border-left-color: #10b981;
        }}
        
        .job-item.comparing {{
            border-left: 4px solid #f59e0b;
            background: rgba(245, 158, 11, 0.1);
        }}
        
        .job-header {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 6px;
        }}
        
        .job-status {{
            font-size: 11px;
            padding: 3px 6px;
            border-radius: 10px;
            font-weight: 600;
        }}
        
        .status-processing {{
            background: rgba(99, 102, 241, 0.2);
            color: #6366f1;
        }}
        
        .status-completed {{
            background: rgba(16, 185, 129, 0.2);
            color: #10b981;
        }}
        
        .progress-bar {{
            height: 4px;
            background: #475569;
            border-radius: 2px;
            overflow: hidden;
            margin: 8px 0;
        }}
        
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #6366f1, #10b981);
            transition: width 0.3s;
        }}
        
        .toast {{
            position: fixed;
            top: 20px;
            right: 20px;
            background: #1e293b;
            color: white;
            padding: 12px;
            border-radius: 8px;
            border-left: 4px solid #6366f1;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
            z-index: 1000;
            animation: slideIn 0.3s ease;
        }}
        
        @keyframes slideIn {{
            from {{ transform: translateX(100%); }}
            to {{ transform: translateX(0); }}
        }}
        
        .hidden {{
            display: none !important;
        }}
        
        #fileInput {{
            display: none;
        }}
        
        .model-category {{
            margin-bottom: 20px;
        }}
        
        .model-category h4 {{
            color: #cbd5e1;
            margin-bottom: 10px;
            padding-bottom: 5px;
            border-bottom: 1px solid #475569;
            font-size: 14px;
        }}
        
        .model-item {{
            background: #334155;
            padding: 10px;
            border-radius: 6px;
            margin-bottom: 5px;
            transition: all 0.2s;
        }}
        
        .model-item:hover {{
            background: #3c4a6b;
        }}
        
        .model-name {{
            font-weight: bold;
            font-size: 13px;
            color: #f8fafc;
            margin-bottom: 3px;
        }}
        
        .model-details {{
            font-size: 11px;
            color: #94a3b8;
            display: flex;
            justify-content: space-between;
        }}
        
        .model-badge {{
            background: #475569;
            padding: 2px 6px;
            border-radius: 10px;
            font-size: 10px;
        }}
        
        .model-badge.scale {{
            background: #6366f1;
            color: white;
        }}
        
        .model-badge.backend {{
            background: #10b981;
            color: white;
        }}
        
        .model-badge.engine {{
            background: #f59e0b;
            color: white;
        }}
        
        .model-download-item {{
            background: #334155;
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 8px;
            border-left: 4px solid #6366f1;
            cursor: pointer;
            transition: all 0.2s;
        }}
        
        .model-download-item:hover {{
            background: #3c4a6b;
            transform: translateY(-2px);
        }}
        
        .model-download-item.downloading {{
            border-left-color: #f59e0b;
        }}
        
        .model-download-item.installed {{
            border-left-color: #10b981;
        }}
        
        .download-progress {{
            height: 4px;
            background: #475569;
            border-radius: 2px;
            overflow: hidden;
            margin: 8px 0;
        }}
        
        .download-progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #f59e0b, #10b981);
            transition: width 0.3s;
        }}
        
        .download-button {{
            padding: 6px 12px;
            background: #6366f1;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
            display: inline-flex;
            align-items: center;
            gap: 4px;
        }}
        
        .download-button:hover {{
            background: #4f46e5;
        }}
        
        .download-button.installed {{
            background: #10b981;
        }}
        
        .download-button.installed:hover {{
            background: #0da271;
        }}
        
        .batch-mode-container {{
            background: rgba(99, 102, 241, 0.1);
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
        }}
        
        .batch-mode-container h4 {{
            color: #cbd5e1;
            margin-bottom: 10px;
            font-size: 14px;
        }}
        
        .batch-file-list {{
            max-height: 150px;
            overflow-y: auto;
            background: #334155;
            border-radius: 6px;
            padding: 10px;
            margin-bottom: 10px;
        }}
        
        .batch-file-item {{
            padding: 5px;
            border-bottom: 1px solid #475569;
            font-size: 12px;
            color: #cbd5e1;
        }}
        
        .batch-file-item:last-child {{
            border-bottom: none;
        }}
        
        .loading-spinner {{
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 2px solid #f3f3f3;
            border-top: 2px solid #6366f1;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }}
        
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        
        /* Error states */
        .error-message {{
            background: rgba(239, 68, 68, 0.2);
            border: 1px solid #ef4444;
            color: #fca5a5;
            padding: 10px;
            border-radius: 6px;
            margin: 10px 0;
            font-size: 13px;
        }}
    </style>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
</head>
<body>
    <div class="container">
        <header>
            <div class="header-content">
                <div class="logo">
                    <i class="fas fa-film"></i>
                    <div class="logo-text">
                        <h1>REAL Video Enhancer</h1>
                        <p>Professional AI Video Enhancement</p>
                    </div>
                </div>
                <div id="backendStatus" class="status-indicator status-offline">
                    <i class="fas fa-circle"></i>
                    <span>Checking backend...</span>
                </div>
            </div>
        </header>
        
        <div class="tab-container">
            <button class="tab active" data-tab="editor">Editor</button>
            <button class="tab" data-tab="models">Models</button>
            <button class="tab" data-tab="download">Download</button>
            <button class="tab" data-tab="jobs">Jobs</button>
            <button class="tab" data-tab="settings">Settings</button>
        </div>
        
        <div class="main-layout">
            <div class="sidebar" id="sidebar">
                <div class="sidebar-content">
                    <div id="editor-tab" class="tab-content active">
                        <h2>Upload Video</h2>
                        <div class="upload-area" id="uploadArea">
                            <i class="fas fa-cloud-upload-alt"></i>
                            <h3>Drop video file here</h3>
                            <p>or click to browse (MP4, MOV, AVI, MKV)</p>
                            <input type="file" id="fileInput" accept=".mp4,.mov,.avi,.mkv,.webm">
                        </div>
                        
                        <div class="batch-mode-container" id="batchModeContainer" style="display: none;">
                            <h4>Batch Mode</h4>
                            <div class="batch-file-list" id="batchFileList">
                                <!-- Batch files will appear here -->
                            </div>
                            <div style="display: flex; gap: 10px;">
                                <button class="btn btn-primary" onclick="openBatchFolder()" style="flex: 1;">
                                    <i class="fas fa-folder-plus"></i> Add Folder
                                </button>
                                <button class="btn" onclick="clearBatchFiles()" style="background: #475569; color: white; flex: 1;">
                                    <i class="fas fa-trash"></i> Clear
                                </button>
                            </div>
                        </div>
                        
                        <div id="videoInfo" class="hidden">
                            <h3>Video Info</h3>
                            <div class="info-row">
                                <span class="info-label">Filename:</span>
                                <span class="info-value" id="infoFilename">-</span>
                            </div>
                            <div class="info-row">
                                <span class="info-label">Resolution:</span>
                                <span class="info-value" id="infoResolution">-</span>
                            </div>
                            <div class="info-row">
                                <span class="info-label">Duration:</span>
                                <span class="info-value" id="infoDuration">-</span>
                            </div>
                            <div class="info-row">
                                <span class="info-label">FPS:</span>
                                <span class="info-value" id="infoFps">-</span>
                            </div>
                            <div class="info-row">
                                <span class="info-label">Codec:</span>
                                <span class="info-value" id="infoCodec">-</span>
                            </div>
                            <div class="info-row">
                                <span class="info-label">Bit Depth:</span>
                                <span class="info-value" id="infoBitDepth">-</span>
                            </div>
                            <div class="info-row">
                                <span class="info-label">HDR:</span>
                                <span class="info-value" id="infoHDR">-</span>
                            </div>
                        </div>
                        
                        <h3>Quick Actions</h3>
                        <button class="btn btn-primary btn-block" onclick="startProcessing()">
                            <i class="fas fa-bolt"></i> Start Enhancement
                        </button>
                        <button class="btn btn-block" onclick="addToRenderQueue()" style="background: #475569; color: white; margin-top: 10px;">
                            <i class="fas fa-plus"></i> Add to Queue
                        </button>
                    </div>
                    
                    <div id="models-tab" class="tab-content">
                        <h2>AI Models</h2>
                        <div id="modelsList">
                            <p>Loading models...</p>
                        </div>
                    </div>
                    
                    <div id="download-tab" class="tab-content">
                        <h2>Download Models</h2>
                        <div id="downloadModelsList">
                            <p>Loading available models for download...</p>
                        </div>
                    </div>
                    
                    <div id="jobs-tab" class="tab-content">
                        <h2>Processing Jobs</h2>
                        <button class="btn btn-primary" onclick="refreshJobs()" style="margin-bottom: 15px;">
                            <i class="fas fa-sync"></i> Refresh
                        </button>
                        <div id="jobsList" class="jobs-list">
                            <p>No jobs yet</p>
                        </div>
                    </div>
                    
                    <div id="settings-tab" class="tab-content">
                        <h2>Settings</h2>
                        <div class="form-group">
                            <label>Theme:</label>
                            <select id="theme">
                                <option value="dark" selected>Dark</option>
                                <option value="light">Light</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label>Max Concurrent Jobs:</label>
                            <select id="maxJobs">
                                <option value="1">1 Job</option>
                                <option value="2" selected>2 Jobs</option>
                                <option value="3">3 Jobs</option>
                                <option value="4">4 Jobs</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label>Comparison Settings:</label>
                            <div class="checkbox-group">
                                <input type="checkbox" id="autoSyncVideos" checked>
                                <label for="autoSyncVideos">Auto-sync videos in comparison</label>
                            </div>
                            <div class="checkbox-group">
                                <input type="checkbox" id="showComparisonInfo" checked>
                                <label for="showComparisonInfo">Show comparison info overlay</label>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <button class="sidebar-toggle" id="sidebarToggle">
                <i class="fas fa-chevron-left"></i>
            </button>
            
            <div class="video-preview-area" id="videoPreviewArea">
                <div class="video-preview">
                    <div id="videoPlaceholder" class="video-placeholder">
                        <i class="fas fa-video"></i>
                        <p>Video preview will appear here</p>
                    </div>
                    <video id="videoPlayer" class="video-player" controls>
                        Your browser does not support the video tag.
                    </video>
                    <img id="videoFrame" class="video-frame">
                    
                    <!-- COMPARISON MODE -->
                    <div class="comparison-container">
                        <button class="exit-comparison" onclick="exitComparisonMode()">
                            <i class="fas fa-times"></i> Exit Comparison
                        </button>
                        
                        <!-- Comparison labels -->
                        <div class="comparison-label original">
                            <i class="fas fa-film"></i> Original
                        </div>
                        <div class="comparison-label enhanced">
                            <i class="fas fa-magic"></i> Enhanced
                        </div>
                        
                        <!-- Info displays -->
                        <div class="comparison-fps-display" id="comparisonFps">
                            <i class="fas fa-tachometer-alt"></i> Speed: --
                        </div>
                        <div class="comparison-time-display" id="comparisonTime">
                            <i class="fas fa-clock"></i> Time: 00:00
                        </div>
                        
                        <!-- Enhanced Video -->
                        <video id="enhancedVideo" class="comparison-video" controls></video>
                        
                        <!-- Original Video -->
                        <div class="original-video-container" id="originalVideoContainer">
                            <video id="originalVideo" class="comparison-video" controls></video>
                        </div>
                        
                        <!-- Divider line with handle -->
                        <div class="comparison-divider" id="comparisonDivider">
                            <div class="divider-handle">
                                <i class="fas fa-arrows-left-right"></i>
                            </div>
                        </div>
                        
                        <div class="comparison-controls">
                            <button class="comparison-btn" onclick="resetComparison()">
                                <i class="fas fa-redo"></i> Reset
                            </button>
                            <button class="comparison-btn" onclick="toggleSync()">
                                <i class="fas fa-link"></i> Sync: <span id="syncStatus">On</span>
                            </button>
                            <button class="comparison-btn" onclick="togglePlayPause()">
                                <i class="fas fa-play" id="playPauseIcon"></i> <span id="playPauseText">Play</span>
                            </button>
                            <button class="comparison-btn" onclick="toggleComparisonLabels()">
                                <i class="fas fa-eye" id="labelsIcon"></i> Labels
                            </button>
                        </div>
                    </div>
                </div>
                
                <div class="video-controls">
                    <div style="flex: 1;">
                        <h2>Enhancement Settings</h2>
                    </div>
                    <div style="display: flex; gap: 10px;">
                        <div class="form-group" style="min-width: 150px;">
                            <label>GPU:</label>
                            <select id="gpuSelection">
                                {gpu_options_html}
                            </select>
                        </div>
                        <div class="form-group" style="min-width: 150px;">
                            <label>Backend:</label>
                            <select id="backend">
                                <option value="pytorch" selected>PyTorch</option>
                                <option value="ncnn">NCNN</option>
                                <option value="tensorrt">TensorRT</option>
                            </select>
                        </div>
                    </div>
                </div>
                
                <div style="background: #1e293b; padding: 20px;">
                    <!-- Upscale Settings -->
                    <div class="form-group">
                        <div class="checkbox-group">
                            <input type="checkbox" id="enableUpscale" checked>
                            <label for="enableUpscale">Enable Upscaling</label>
                        </div>
                        <div id="upscaleOptions" style="margin-left: 20px; margin-top: 10px;">
                            <label>Upscale Model:</label>
                            <select id="upscaleModel">
                                <option value="">Select model...</option>
                            </select>
                            <label>Scale Factor: <span class="range-value" id="upscaleValue">2x</span></label>
                            <input type="range" id="upscaleFactor" min="1" max="4" step="1" value="2">
                        </div>
                    </div>
                    
                    <!-- Interpolation Settings -->
                    <div class="form-group">
                        <div class="checkbox-group">
                            <input type="checkbox" id="enableInterpolate">
                            <label for="enableInterpolate">Enable Frame Interpolation</label>
                        </div>
                        <div id="interpolateOptions" class="hidden" style="margin-left: 20px; margin-top: 10px;">
                            <label>Interpolation Model:</label>
                            <select id="interpolateModel">
                                <option value="">Select model...</option>
                            </select>
                            <label>Interpolation Factor: <span class="range-value" id="interpolateValue">2x</span></label>
                            <input type="range" id="interpolateFactor" min="1" max="4" step="0.5" value="2">
                        </div>
                    </div>
                    
                    <!-- Denoise Settings -->
                    <div class="form-group">
                        <div class="checkbox-group">
                            <input type="checkbox" id="enableDenoise">
                            <label for="enableDenoise">Enable Denoising</label>
                        </div>
                        <div id="denoiseOptions" class="hidden" style="margin-left: 20px; margin-top: 10px;">
                            <label>Denoise Model:</label>
                            <select id="denoiseModel">
                                <option value="">Select model...</option>
                            </select>
                        </div>
                    </div>
                    
                    <!-- Decompress Settings -->
                    <div class="form-group">
                        <div class="checkbox-group">
                            <input type="checkbox" id="enableDecompress">
                            <label for="enableDecompress">Enable Compression Artifact Removal</label>
                        </div>
                        <div id="decompressOptions" class="hidden" style="margin-left: 20px; margin-top: 10px;">
                            <label>Decompress Model:</label>
                            <select id="decompressModel">
                                <option value="">Select model...</option>
                            </select>
                        </div>
                    </div>
                    
                    <!-- Advanced Settings -->
                    <div class="form-group" style="display: flex; gap: 20px; flex-wrap: wrap; margin-top: 20px;">
                        <div style="min-width: 150px;">
                            <label>Precision:</label>
                            <select id="precision">
                                <option value="auto" selected>Auto</option>
                                <option value="float16">FP16 (Fast)</option>
                                <option value="float32">FP32 (Accurate)</option>
                            </select>
                        </div>
                        <div style="min-width: 150px;">
                            <label>Output Format:</label>
                            <select id="outputFormat">
                                <option value="mp4" selected>MP4</option>
                                <option value="mkv">MKV</option>
                                <option value="webm">WebM</option>
                            </select>
                        </div>
                        <div style="min-width: 200px;">
                            <label>Video Quality (CRF): <span class="range-value" id="crfValue">18</span></label>
                            <input type="range" id="crf" min="0" max="51" value="18">
                            <div style="font-size: 12px; color: #94a3b8; margin-top: 5px;">
                                Lower = Better Quality, Larger File
                            </div>
                        </div>
                        <div style="min-width: 150px;">
                            <label>Tile Size:</label>
                            <select id="tileSize">
                                <option value="0" selected>Auto</option>
                                <option value="32">32</option>
                                <option value="64">64</option>
                                <option value="128">128</option>
                                <option value="256">256</option>
                                <option value="512">512</option>
                            </select>
                        </div>
                    </div>
                    
                    <!-- Encoder Settings -->
                    <div class="form-group" style="margin-top: 20px;">
                        <label>Encoder Command:</label>
                        <input type="text" id="encoderCommand" style="width: 100%;" placeholder="-c:v libx264 -preset medium -crf 18 -c:a copy">
                        <div style="font-size: 12px; color: #94a3b8; margin-top: 5px;">
                            Advanced FFmpeg encoder settings (optional)
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <div id="toastContainer"></div>
    
    <script src="https://cdn.socket.io/4.5.0/socket.io.min.js"></script>
    <script>
        // Global variables
        let socket = null;
        let currentVideo = null;
        let currentVideoInfo = null;
        let backendAvailable = false;
        let videoComparator = null;
        let isComparisonMode = false;
        let isSyncEnabled = true;
        let showComparisonLabels = true;
        let showComparisonInfo = true;
        let batchVideos = [];
        
        // Initialize SocketIO connection
        function initSocket() {{
            console.log('Initializing SocketIO connection...');
            
            // Connect to server
            socket = io('http://' + window.location.hostname + ':{config.PORT}', {{
                transports: ['websocket', 'polling'],
                reconnection: true,
                reconnectionAttempts: 5,
                reconnectionDelay: 1000
            }});
            
            socket.on('connect', () => {{
                console.log('Connected to server');
                updateBackendStatusElement('Connected to server', 'success');
                checkBackendStatus();
            }});
            
            socket.on('disconnect', () => {{
                console.log('Disconnected from server');
                updateBackendStatusElement('Disconnected from server', 'error');
            }});
            
            socket.on('connect_error', (error) => {{
                console.error('Connection error:', error);
                updateBackendStatusElement('Connection error: ' + error.message, 'error');
            }});
            
            socket.on('job_update', (job) => {{
                console.log('Job update received:', job.id, job.status);
                updateJobInList(job);
            }});
            
            socket.on('backend_status', (status) => {{
                console.log('Backend status received:', status);
                updateBackendStatus(status);
            }});
            
            socket.on('model_download_progress', (data) => {{
                console.log('Download progress:', data);
                updateDownloadProgress(data);
            }});
            
            socket.on('model_download_complete', (data) => {{
                console.log('Download complete:', data);
                downloadComplete(data);
            }});
            
            socket.on('model_download_error', (data) => {{
                console.error('Download error:', data);
                showToast('Download failed: ' + data.error, 'error');
                resetDownloadButton(data.model_name);
            }});
        }}
        
        function updateBackendStatusElement(message, type) {{
            const statusElement = document.getElementById('backendStatus');
            if (!statusElement) return;
            
            const icon = type === 'success' ? 'fas fa-check-circle' : 
                        type === 'error' ? 'fas fa-exclamation-circle' : 'fas fa-circle';
            const className = type === 'success' ? 'status-online' : 
                             type === 'error' ? 'status-offline' : '';
            
            statusElement.className = 'status-indicator ' + className;
            statusElement.innerHTML = `<i class="${{icon}}"></i><span>${{message}}</span>`;
        }}
        
        function checkBackendStatus() {{
            fetch('/backend/status')
                .then(response => {{
                    if (!response.ok) throw new Error('Network response was not ok');
                    return response.json();
                }})
                .then(status => updateBackendStatus(status))
                .catch(error => {{
                    console.error('Error checking backend status:', error);
                    updateBackendStatus({{ 
                        available: false, 
                        error: error.message 
                    }});
                }});
        }}
        
        function updateBackendStatus(status) {{
            backendAvailable = status.available;
            const statusElement = document.getElementById('backendStatus');
            
            if (backendAvailable) {{
                statusElement.className = 'status-indicator status-online';
                statusElement.innerHTML = '<i class="fas fa-check-circle"></i> Backend Ready';
            }} else {{
                statusElement.className = 'status-indicator status-offline';
                statusElement.innerHTML = '<i class="fas fa-exclamation-circle"></i> Backend Unavailable';
                if (status.error) {{
                    statusElement.innerHTML += '<br><small>' + status.error + '</small>';
                }}
            }}
        }}
        
        function showToast(message, type = 'info') {{
            const container = document.getElementById('toastContainer');
            if (!container) return;
            
            const toast = document.createElement('div');
            toast.className = 'toast';
            
            const icons = {{
                'info': 'fas fa-info-circle',
                'success': 'fas fa-check-circle',
                'error': 'fas fa-exclamation-circle',
                'warning': 'fas fa-exclamation-triangle'
            }};
            
            toast.innerHTML = '<i class="' + (icons[type] || icons.info) + '"></i>' +
                            '<span>' + message + '</span>';
            
            container.appendChild(toast);
            
            // Remove toast after 3 seconds
            setTimeout(() => {{
                if (toast.parentElement) {{
                    toast.remove();
                }}
            }}, 3000);
        }}
        
        // Tab switching
        document.querySelectorAll('.tab').forEach(tab => {{
            tab.addEventListener('click', () => {{
                document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                
                const tabId = tab.getAttribute('data-tab');
                document.querySelectorAll('.tab-content').forEach(content => {{
                    content.classList.remove('active');
                }});
                
                const targetTab = document.getElementById(tabId + '-tab');
                if (targetTab) {{
                    targetTab.classList.add('active');
                }}
                
                // Exit comparison when switching tabs
                if (isComparisonMode) {{
                    exitComparisonMode();
                }}
                
                // Load content for specific tabs
                if (tabId === 'models') {{
                    loadModels();
                }} else if (tabId === 'download') {{
                    loadDownloadableModels();
                }} else if (tabId === 'jobs') {{
                    loadJobs();
                }}
            }});
        }});
        
        // Sidebar toggle
        const sidebar = document.getElementById('sidebar');
        const sidebarToggle = document.getElementById('sidebarToggle');
        
        if (sidebarToggle) {{
            sidebarToggle.addEventListener('click', () => {{
                if (sidebar) {{
                    sidebar.classList.toggle('closed');
                    const icon = sidebarToggle.querySelector('i');
                    if (sidebar.classList.contains('closed')) {{
                        icon.className = 'fas fa-chevron-right';
                    }} else {{
                        icon.className = 'fas fa-chevron-left';
                    }}
                }}
            }});
        }}
        
        // File upload
        document.getElementById('uploadArea').addEventListener('click', () => {{
            document.getElementById('fileInput').click();
        }});
        
        document.getElementById('fileInput').addEventListener('change', async (e) => {{
            const file = e.target.files[0];
            if (!file) return;
            
            const formData = new FormData();
            formData.append('file', file);
            
            try {{
                showToast('Uploading video...', 'info');
                const response = await fetch('/upload', {{
                    method: 'POST',
                    body: formData
                }});
                
                const result = await response.json();
                
                if (result.success) {{
                    currentVideo = result.filename;
                    currentVideoInfo = result.video_info;
                    updateVideoInfo();
                    
                    // Display video in preview area
                    displayVideoPreview(result.filename);
                    
                    showToast('Video uploaded successfully', 'success');
                    
                    // Hide batch mode if active
                    document.getElementById('batchModeContainer').style.display = 'none';
                    batchVideos = [];
                }} else {{
                    showToast(result.error || 'Upload failed', 'error');
                }}
            }} catch (error) {{
                showToast('Upload failed: ' + error.message, 'error');
                console.error('Upload error:', error);
            }}
        }});
        
        function openBatchFolder() {{
            // Create a file input for multiple file selection
            const fileInput = document.createElement('input');
            fileInput.type = 'file';
            fileInput.multiple = true;
            fileInput.accept = '.mp4,.mov,.avi,.mkv,.webm';
            
            fileInput.addEventListener('change', async (e) => {{
                const files = Array.from(e.target.files);
                if (files.length === 0) return;
                
                showToast(`Processing ${{files.length}} files...`, 'info');
                
                // Process each file
                const uploadPromises = [];
                for (const file of files) {{
                    const formData = new FormData();
                    formData.append('file', file);
                    
                    uploadPromises.push(
                        fetch('/upload', {{
                            method: 'POST',
                            body: formData
                        }}).then(response => response.json())
                    );
                }}
                
                try {{
                    const results = await Promise.all(uploadPromises);
                    batchVideos = results
                        .filter(result => result.success)
                        .map(result => ({{
                            filename: result.filename,
                            info: result.video_info
                        }}));
                    
                    // Update UI
                    updateBatchUI();
                    showToast(`Added ${{batchVideos.length}} files to batch`, 'success');
                }} catch (error) {{
                    console.error('Error uploading batch files:', error);
                    showToast('Error uploading batch files', 'error');
                }}
            }});
            
            fileInput.click();
        }}
        
        function updateBatchUI() {{
            const batchContainer = document.getElementById('batchModeContainer');
            const batchFileList = document.getElementById('batchFileList');
            
            if (batchVideos.length > 0) {{
                batchContainer.style.display = 'block';
                batchFileList.innerHTML = '';
                
                batchVideos.forEach((video, index) => {{
                    const fileItem = document.createElement('div');
                    fileItem.className = 'batch-file-item';
                    fileItem.textContent = video.filename;
                    batchFileList.appendChild(fileItem);
                }});
                
                // Clear single video
                currentVideo = null;
                currentVideoInfo = null;
                document.getElementById('videoInfo').classList.add('hidden');
                document.getElementById('videoPlaceholder').style.display = 'flex';
                document.getElementById('videoPlayer').style.display = 'none';
            }} else {{
                batchContainer.style.display = 'none';
            }}
        }}
        
        function clearBatchFiles() {{
            batchVideos = [];
            updateBatchUI();
            showToast('Batch files cleared', 'info');
        }}
        
        function updateVideoInfo() {{
            if (!currentVideoInfo) return;
            
            const videoInfoElement = document.getElementById('videoInfo');
            if (!videoInfoElement) return;
            
            videoInfoElement.classList.remove('hidden');
            document.getElementById('infoFilename').textContent = currentVideoInfo.filename || '-';
            document.getElementById('infoResolution').textContent = 
                (currentVideoInfo.width || 0) + ' × ' + (currentVideoInfo.height || 0);
            document.getElementById('infoDuration').textContent = 
                formatDuration(currentVideoInfo.duration || 0);
            document.getElementById('infoFps').textContent = 
                (currentVideoInfo.fps || 0).toFixed(2) + ' fps';
            document.getElementById('infoCodec').textContent = currentVideoInfo.codec || '-';
            document.getElementById('infoBitDepth').textContent = 
                (currentVideoInfo.bit_depth || 8) + ' bit';
            document.getElementById('infoHDR').textContent = 
                currentVideoInfo.is_hdr ? 'Yes' : 'No';
        }}
        
        function displayVideoPreview(filename) {{
            const videoPlayer = document.getElementById('videoPlayer');
            const videoFrame = document.getElementById('videoFrame');
            const videoPlaceholder = document.getElementById('videoPlaceholder');
            
            if (!videoPlayer || !videoPlaceholder) return;
            
            const videoUrl = '/uploads/' + encodeURIComponent(filename);
            
            videoPlayer.src = videoUrl;
            videoPlayer.style.display = 'block';
            if (videoFrame) videoFrame.style.display = 'none';
            videoPlaceholder.style.display = 'none';
            videoPlayer.load();
        }}
        
        function formatDuration(seconds) {{
            if (!seconds) return '0:00';
            
            const hours = Math.floor(seconds / 3600);
            const minutes = Math.floor((seconds % 3600) / 60);
            const secs = Math.floor(seconds % 60);
            
            if (hours > 0) {{
                return hours + ':' + minutes.toString().padStart(2, '0') + ':' + secs.toString().padStart(2, '0');
            }} else {{
                return minutes + ':' + secs.toString().padStart(2, '0');
            }}
        }}
        
        // Range updates
        document.getElementById('upscaleFactor')?.addEventListener('input', (e) => {{
            document.getElementById('upscaleValue').textContent = e.target.value + 'x';
        }});
        
        document.getElementById('interpolateFactor')?.addEventListener('input', (e) => {{
            document.getElementById('interpolateValue').textContent = e.target.value + 'x';
        }});
        
        document.getElementById('crf')?.addEventListener('input', (e) => {{
            document.getElementById('crfValue').textContent = e.target.value;
        }});
        
        // Checkbox toggles
        document.getElementById('enableInterpolate')?.addEventListener('change', (e) => {{
            const options = document.getElementById('interpolateOptions');
            if (options) options.classList.toggle('hidden', !e.target.checked);
        }});
        
        document.getElementById('enableDenoise')?.addEventListener('change', (e) => {{
            const options = document.getElementById('denoiseOptions');
            if (options) options.classList.toggle('hidden', !e.target.checked);
        }});
        
        document.getElementById('enableDecompress')?.addEventListener('change', (e) => {{
            const options = document.getElementById('decompressOptions');
            if (options) options.classList.toggle('hidden', !e.target.checked);
        }});
        
        // Load models
        async function loadModels() {{
            try {{
                const response = await fetch('/models');
                if (!response.ok) throw new Error('Failed to load models');
                const models = await response.json();
                
                // Populate model dropdowns
                populateModelDropdown('upscaleModel', models.upscale || {{}});
                populateModelDropdown('interpolateModel', models.interpolate || {{}});
                populateModelDropdown('denoiseModel', models.denoise || {{}});
                populateModelDropdown('decompressModel', models.decompress || {{}});
                
                // Update models list display
                const modelsList = document.getElementById('modelsList');
                if (!modelsList) return;
                
                modelsList.innerHTML = '';
                
                for (const [category, categoryModels] of Object.entries(models)) {{
                    if (!categoryModels || Object.keys(categoryModels).length === 0) continue;
                    
                    const categoryDiv = document.createElement('div');
                    categoryDiv.className = 'model-category';
                    categoryDiv.innerHTML = '<h4>' + category.charAt(0).toUpperCase() + category.slice(1) + ' Models (' + Object.keys(categoryModels).length + ')</h4>';
                    
                    for (const [modelName, modelInfo] of Object.entries(categoryModels)) {{
                        // Clean model name for display
                        const displayName = modelInfo.original_name || modelName;
                        
                        // Create model item for display
                        const modelDiv = document.createElement('div');
                        modelDiv.className = 'model-item';
                        
                        const scaleBadge = modelInfo.scale ? 
                            '<span class="model-badge scale">' + modelInfo.scale + 'x</span>' : '';
                        const backendBadge = '<span class="model-badge backend">' + 
                            (modelInfo.is_engine ? 'TensorRT' : (modelInfo.backend ? modelInfo.backend.join(', ') : 'PyTorch')) + 
                            '</span>';
                        const engineBadge = modelInfo.is_engine ? 
                            '<span class="model-badge engine">Engine</span>' : '';
                        
                        modelDiv.innerHTML = 
                            '<div class="model-name">' + displayName + '</div>' +
                            '<div class="model-details">' +
                                '<div>' + (modelInfo.filename || '') + '</div>' +
                                '<div>' + scaleBadge + backendBadge + engineBadge + '</div>' +
                            '</div>';
                        
                        categoryDiv.appendChild(modelDiv);
                    }}
                    
                    modelsList.appendChild(categoryDiv);
                }}
                
            }} catch (error) {{
                console.error('Error loading models:', error);
                const modelsList = document.getElementById('modelsList');
                if (modelsList) {{
                    modelsList.innerHTML = 
                        '<div class="error-message">' +
                        '<p>Error loading models. Please check console.</p>' +
                        '<p>Error: ' + error.message + '</p>' +
                        '</div>';
                }}
            }}
        }}
        
        function populateModelDropdown(dropdownId, models) {{
            const select = document.getElementById(dropdownId);
            if (!select) return;
            
            select.innerHTML = '<option value="">Select model...</option>';
            
            for (const [modelName, modelInfo] of Object.entries(models)) {{
                const displayName = modelInfo.original_name || modelName;
                const option = document.createElement('option');
                option.value = displayName;
                option.textContent = displayName + (modelInfo.is_engine ? ' (TensorRT)' : '');
                option.dataset.fullName = modelName;
                select.appendChild(option);
            }}
        }}
        
        // Load downloadable models
        async function loadDownloadableModels() {{
            try {{
                const response = await fetch('/downloadable_models');
                if (!response.ok) throw new Error('Failed to load downloadable models');
                const models = await response.json();
                
                const downloadList = document.getElementById('downloadModelsList');
                if (!downloadList) return;
                
                downloadList.innerHTML = '';
                
                for (const [category, categoryModels] of Object.entries(models)) {{
                    const categoryDiv = document.createElement('div');
                    categoryDiv.className = 'model-category';
                    categoryDiv.innerHTML = '<h4>' + category.charAt(0).toUpperCase() + category.slice(1) + ' Models</h4>';
                    
                    for (const [modelName, modelInfo]