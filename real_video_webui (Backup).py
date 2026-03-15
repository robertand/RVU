"""
PRO AI Video UPSCALER - Cu selecție GPU și comparație video
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
from dataclasses import dataclass, asdict, field
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
                          "psutil"])
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
    HOST = "0.0.0.0"
    PORT = 5000
    DEBUG = True
    SECRET_KEY = "real-video-enhancer-secret-key-2024"
    MAX_UPLOAD_SIZE = 2 * 1024 * 1024 * 1024
    PREVIEW_FPS = 10
    PREVIEW_MAX_WIDTH = 640
    AVAILABLE_BACKENDS = ["pytorch", "ncnn", "tensorrt"]
    DEFAULT_BACKEND = "pytorch"
    RVE_MAX_CONCURRENT_JOBS = 1
    CUSTOM_MODELS_PATH = os.path.join(BASE_DIR, "custom_models")
    TEMP_DOWNLOAD_PATH = os.path.join(BASE_DIR, "temp_download")

config = Config()

# ============================================================================
# APPLICATION SETUP
# ============================================================================

app = Flask(__name__, template_folder=config.TEMPLATE_FOLDER)
app.config['SECRET_KEY'] = config.SECRET_KEY
app.config['UPLOAD_FOLDER'] = config.UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = config.MAX_UPLOAD_SIZE

socketio = SocketIO(
    app, 
    cors_allowed_origins="*", 
    async_mode='threading',
    ping_timeout=60,
    ping_interval=25,
    logger=False,
    engineio_logger=False
)

# Create necessary directories
for folder in [config.UPLOAD_FOLDER, config.OUTPUT_FOLDER, config.TEMP_FOLDER, 
               config.TEMPLATE_FOLDER, config.CUSTOM_MODELS_PATH, config.TEMP_DOWNLOAD_PATH]:
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
            
            # Try to get codec info (might not be available in all cases)
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)]) if fourcc != 0 else "unknown"
            
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
                size_bytes=size_bytes
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
                size_bytes=os.path.getsize(filepath)
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
# CREATE HTML TEMPLATE WITH ALL FEATURES
# ============================================================================

def create_html_template():
    """Create the HTML template file with all features"""
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
    <title>PRO AI Video UPSCALER</title>
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
        
        .comparison-container.fullscreen {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            z-index: 9999;
            background: #000;
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
        
        .download-model-item {{
            background: #334155;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 10px;
            border-left: 4px solid #6366f1;
        }}
        
        .download-model-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }}
        
        .download-model-name {{
            font-weight: bold;
            font-size: 14px;
            color: #f8fafc;
        }}
        
        .download-model-desc {{
            font-size: 12px;
            color: #94a3b8;
            margin-bottom: 10px;
        }}
        
        .model-download-progress {{
            height: 6px;
            background: #475569;
            border-radius: 3px;
            overflow: hidden;
            margin-bottom: 5px;
        }}
        
        .model-download-progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #6366f1, #10b981);
            transition: width 0.3s;
        }}
        
        .model-download-size {{
            font-size: 11px;
            color: #94a3b8;
            text-align: right;
        }}
        
        .import-model-btn {{
            margin-top: 10px;
            display: flex;
            gap: 10px;
        }}
        
        .batch-file-info {{
            background: rgba(99, 102, 241, 0.1);
            border-radius: 6px;
            padding: 10px;
            margin-top: 10px;
            font-size: 12px;
            color: #94a3b8;
        }}
        
        .job-timestamp {{
            font-size: 10px;
            color: #94a3b8;
            margin-top: 3px;
        }}
        
        .job-display-name {{
            font-weight: bold;
            color: #f8fafc;
            margin-bottom: 2px;
        }}
        
        .job-video-info {{
            font-size: 11px;
            color: #cbd5e1;
            display: flex;
            gap: 10px;
        }}
        
        .fullscreen-btn {{
            position: absolute;
            top: 20px;
            right: 120px;
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
        
        .exit-fullscreen-btn {{
            position: absolute;
            top: 20px;
            right: 120px;
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
            display: none;
        }}
        
        .fullscreen-btn:hover {{
            background: rgba(0,0,0,0.9);
        }}
        
        .exit-fullscreen-btn:hover {{
            background: rgba(0,0,0,0.9);
        }}
        
        .fullscreen-btn i {{
            font-size: 12px;
        }}
        
        .exit-fullscreen-btn i {{
            font-size: 12px;
        }}
        
        body.fullscreen-mode {{
            overflow: hidden;
        }}
        
        body.fullscreen-mode .container {{
            display: none;
        }}
        
        body.fullscreen-mode .comparison-container.fullscreen {{
            display: block;
        }}
        
        /* Fullscreen specific styles */
        :fullscreen .comparison-container {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background: #000;
            z-index: 9999;
        }}
        
        :-webkit-full-screen .comparison-container {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background: #000;
            z-index: 9999;
        }}
        
        :-moz-full-screen .comparison-container {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background: #000;
            z-index: 9999;
        }}
        
        :-ms-fullscreen .comparison-container {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background: #000;
            z-index: 9999;
        }}
        
        /* Queue styles */
        .queue-item {{
            background: #334155;
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 8px;
            border-left: 4px solid #f59e0b;
        }}
        
        .queue-controls {{
            margin-top: 15px;
            display: flex;
            gap: 10px;
        }}
        
        .queue-video-info {{
            font-size: 12px;
            color: #cbd5e1;
            margin-top: 5px;
        }}
        
        .queue-batch-info {{
            background: rgba(245, 158, 11, 0.1);
            border-radius: 6px;
            padding: 10px;
            margin-top: 15px;
        }}
        
        /* New start button position */
        .enhancement-header {{
            display: flex;
            align-items: center;
            gap: 15px;
            margin-bottom: 15px;
        }}
        
        .enhancement-title {{
            flex: 1;
        }}
        
        .start-button-container {{
            margin-left: 20px;
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
                        <h1>PRO AI Video UPSCALER</h1>
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
            <button class="tab active" data-tab="load-video">Load Video</button>
            <button class="tab" data-tab="queue">Queue</button>
            <button class="tab" data-tab="models">Models</button>
            <button class="tab" data-tab="download">Download</button>
            <button class="tab" data-tab="jobs">Jobs</button>
            <button class="tab" data-tab="settings">Settings</button>
        </div>
        
        <div class="main-layout">
            <div class="sidebar" id="sidebar">
                <div class="sidebar-content">
                    <div id="load-video-tab" class="tab-content active">
                        <h2>Upload Video</h2>
                        <div class="upload-area" id="uploadArea">
                            <i class="fas fa-cloud-upload-alt"></i>
                            <h3>Drop video file here</h3>
                            <p>or click to browse (MP4, MOV, AVI, MKV)</p>
                            <input type="file" id="fileInput" accept=".mp4,.mov,.avi,.mkv,.webm">
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
                        </div>
                    </div>
                    
                    <div id="queue-tab" class="tab-content">
                        <h2>Processing Queue</h2>
                        <div class="queue-controls">
                            <button class="btn btn-primary" onclick="processQueue()">
                                <i class="fas fa-play"></i> Process Queue
                            </button>
                            <button class="btn" onclick="clearQueue()" style="background: #475569; color: white;">
                                <i class="fas fa-trash"></i> Clear
                            </button>
                        </div>
                        <div id="queueList" class="jobs-list">
                            <p>Queue is empty</p>
                        </div>
                        <div class="queue-batch-info">
                            <p><i class="fas fa-info-circle"></i> Add videos to queue for batch processing</p>
                            <p style="font-size: 11px; margin-top: 5px;">Each video will use the current enhancement settings</p>
                        </div>
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
                            <p>Loading available models...</p>
                        </div>
                        
                        <div class="import-model-btn">
                            <button class="btn btn-primary" onclick="importPyTorchModel()">
                                <i class="fas fa-upload"></i> Import PyTorch Model
                            </button>
                            <button class="btn btn-primary" onclick="importNCNNModel()">
                                <i class="fas fa-upload"></i> Import NCNN Model
                            </button>
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
                        <div class="form-group">
                            <label>Output Settings:</label>
                            <div class="checkbox-group">
                                <input type="checkbox" id="autoHDRMode" checked>
                                <label for="autoHDRMode">Auto HDR Mode</label>
                            </div>
                            <div class="checkbox-group">
                                <input type="checkbox" id="tilingEnabled">
                                <label for="tilingEnabled">Enable Tiling</label>
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
                    <div class="comparison-container" id="comparisonContainer">
                        <button class="fullscreen-btn" onclick="toggleFullscreen()" id="fullscreenBtn">
                            <i class="fas fa-expand"></i> Fullscreen
                        </button>
                        <button class="exit-fullscreen-btn" onclick="exitFullscreen()" id="exitFullscreenBtn">
                            <i class="fas fa-compress"></i> Exit Fullscreen
                        </button>
                        
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
                    <div class="enhancement-header">
                        <div class="enhancement-title">
                            <h2>Enhance Video</h2>
                        </div>
                        <div class="start-button-container">
                            <button class="btn btn-primary" onclick="startProcessing()">
                                <i class="fas fa-bolt"></i> Start Enhancement
                            </button>
                            <button class="btn" onclick="addToQueue()" style="background: #f59e0b; color: white; margin-left: 10px;">
                                <i class="fas fa-plus"></i> Add to Queue
                            </button>
                        </div>
                    </div>
                </div>
                
                <div style="background: #1e293b; padding: 20px;">
                    <div style="display: flex; gap: 20px; margin-bottom: 20px; flex-wrap: wrap;">
                        <div class="form-group" style="min-width: 150px;">
                            <label>GPU:</label>
                            <select id="gpuSelection">
                                {gpu_options_html}
                            </select>
                            <div class="gpu-info" id="gpuTensorrtInfo" style="display: none;">
                                <i class="fas fa-info-circle"></i> TensorRT requires GPU 0
                            </div>
                        </div>
                        <div class="form-group" style="min-width: 150px;">
                            <label>Backend:</label>
                            <select id="backend">
                                <option value="pytorch" selected>PyTorch</option>
                                <option value="ncnn">NCNN</option>
                                <option value="tensorrt">TensorRT (PyTorch fallback)</option>
                            </select>
                        </div>
                    </div>
                    
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
                    
                    <div class="form-group" style="display: flex; gap: 20px; flex-wrap: wrap;">
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
                            </select>
                        </div>
                        <div style="min-width: 200px;">
                            <label>Video Quality (CRF): <span class="range-value" id="crfValue">18</span></label>
                            <input type="range" id="crf" min="0" max="51" value="18">
                            <div style="font-size: 12px; color: #94a3b8; margin-top: 5px;">
                                Lower = Better Quality, Larger File
                            </div>
                        </div>
                    </div>
                    
                    <div class="form-group">
                        <div class="checkbox-group">
                            <input type="checkbox" id="benchmarkMode">
                            <label for="benchmarkMode">Benchmark Mode</label>
                        </div>
                        <div class="checkbox-group">
                            <input type="checkbox" id="ensembleMode">
                            <label for="ensembleMode">Ensemble Mode (Better Quality)</label>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <div id="toastContainer"></div>
    
    <script src="https://cdn.socket.io/4.5.0/socket.io.min.js"></script>
    <script>
        let socket = null;
        let currentVideo = null;
        let currentVideoInfo = null;
        let backendAvailable = false;
        let videoComparator = null;
        let isComparisonMode = false;
        let isSyncEnabled = true;
        let showComparisonLabels = true;
        let showComparisonInfo = true;
        let downloadingModels = {{}};
        let isFullscreen = false;
        let processingQueue = [];
        
        function initSocket() {{
            socket = io();
            
            socket.on('connect', () => {{
                console.log('Connected to server');
                showToast('Connected to server', 'success');
                checkBackendStatus();
            }});
            
            socket.on('job_update', (job) => {{
                updateJobInList(job);
            }});
            
            socket.on('backend_status', (status) => {{
                updateBackendStatus(status);
            }});
            
            socket.on('download_progress', (data) => {{
                updateDownloadProgress(data);
            }});
            
            socket.on('download_complete', (data) => {{
                downloadComplete(data);
            }});
        }}
        
        function checkBackendStatus() {{
            fetch('/backend/status')
                .then(response => response.json())
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
            }}
        }}
        
        function showToast(message, type = 'info') {{
            const container = document.getElementById('toastContainer');
            const toast = document.createElement('div');
            toast.className = 'toast';
            
            const icons = {{
                'info': 'fas fa-info-circle',
                'success': 'fas fa-check-circle',
                'error': 'fas fa-exclamation-circle'
            }};
            
            toast.innerHTML = '<i class="' + (icons[type] || icons.info) + '"></i>' +
                            '<span>' + message + '</span>';
            
            container.appendChild(toast);
            
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
                document.getElementById(tabId + '-tab').classList.add('active');
                
                // Load content based on tab
                if (tabId === 'download') {{
                    loadDownloadableModels();
                }} else if (tabId === 'models') {{
                    loadModels();
                }} else if (tabId === 'jobs') {{
                    loadJobs();
                }} else if (tabId === 'queue') {{
                    loadQueue();
                }}
                
                // Exit comparison when switching tabs
                if (isComparisonMode) {{
                    exitComparisonMode();
                }}
            }});
        }});
        
        // Sidebar toggle
        const sidebar = document.getElementById('sidebar');
        const sidebarToggle = document.getElementById('sidebarToggle');
        const videoPreviewArea = document.getElementById('videoPreviewArea');
        
        sidebarToggle.addEventListener('click', () => {{
            sidebar.classList.toggle('closed');
            const icon = sidebarToggle.querySelector('i');
            if (sidebar.classList.contains('closed')) {{
                icon.className = 'fas fa-chevron-right';
            }} else {{
                icon.className = 'fas fa-chevron-left';
            }}
        }});
        
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
                }} else {{
                    showToast(result.error || 'Upload failed', 'error');
                }}
            }} catch (error) {{
                showToast('Upload failed: ' + error.message, 'error');
            }}
        }});
        
        function updateVideoInfo() {{
            if (!currentVideoInfo) return;
            
            document.getElementById('videoInfo').classList.remove('hidden');
            document.getElementById('infoFilename').textContent = currentVideoInfo.filename;
            document.getElementById('infoResolution').textContent = 
                currentVideoInfo.width + ' × ' + currentVideoInfo.height;
            document.getElementById('infoDuration').textContent = 
                formatDuration(currentVideoInfo.duration);
        }}
        
        function displayVideoPreview(filename) {{
            const videoPlayer = document.getElementById('videoPlayer');
            const videoFrame = document.getElementById('videoFrame');
            const videoPlaceholder = document.getElementById('videoPlaceholder');
            const videoUrl = '/uploads/' + encodeURIComponent(filename);
            
            videoPlayer.src = videoUrl;
            videoPlayer.classList.add('video-player');
            videoPlayer.style.display = 'block';
            videoFrame.style.display = 'none';
            videoPlaceholder.style.display = 'none';
            videoPlayer.load();
        }}
        
        function formatDuration(seconds) {{
            const hours = Math.floor(seconds / 3600);
            const minutes = Math.floor((seconds % 3600) / 60);
            const secs = Math.floor(seconds % 60);
            
            if (hours > 0) {{
                return hours + ':' + minutes.toString().padStart(2, '0') + ':' + secs.toString().padStart(2, '0');
            }} else {{
                return minutes + ':' + secs.toString().padStart(2, '0');
            }}
        }}
        
        function formatTimestamp(timestamp) {{
            if (!timestamp) return '';
            const date = new Date(timestamp);
            return date.toLocaleTimeString([], {{ hour: '2-digit', minute: '2-digit' }}) + 
                   ' ' + date.toLocaleDateString();
        }}
        
        // Range updates
        document.getElementById('upscaleFactor').addEventListener('input', (e) => {{
            document.getElementById('upscaleValue').textContent = e.target.value + 'x';
        }});
        
        document.getElementById('interpolateFactor').addEventListener('input', (e) => {{
            document.getElementById('interpolateValue').textContent = e.target.value + 'x';
        }});
        
        document.getElementById('crf').addEventListener('input', (e) => {{
            document.getElementById('crfValue').textContent = e.target.value;
        }});
        
        // Checkbox toggles
        document.getElementById('enableInterpolate').addEventListener('change', (e) => {{
            document.getElementById('interpolateOptions').classList.toggle('hidden', !e.target.checked);
        }});
        
        document.getElementById('enableDenoise').addEventListener('change', (e) => {{
            document.getElementById('denoiseOptions').classList.toggle('hidden', !e.target.checked);
        }});
        
        document.getElementById('enableDecompress').addEventListener('change', (e) => {{
            document.getElementById('decompressOptions').classList.toggle('hidden', !e.target.checked);
        }});
        
        // Backend change listener for TensorRT GPU restriction
        document.getElementById('backend').addEventListener('change', function() {{
            loadModels();
            updateGPUForBackend(this.value);
        }});
        
        function updateGPUForBackend(backend) {{
            const gpuSelect = document.getElementById('gpuSelection');
            const gpuInfo = document.getElementById('gpuTensorrtInfo');
            
            if (backend === 'tensorrt') {{
                // Force GPU 0 for TensorRT
                gpuSelect.value = '0';
                gpuSelect.disabled = true;
                gpuInfo.style.display = 'block';
                showToast('TensorRT requires GPU 0. GPU selection locked.', 'info');
            }} else {{
                gpuSelect.disabled = false;
                gpuInfo.style.display = 'none';
            }}
        }}
        
        // Initialize GPU restriction
        document.addEventListener('DOMContentLoaded', () => {{
            updateGPUForBackend(document.getElementById('backend').value);
        }});
        
        // Load models
        async function loadModels() {{
            try {{
                const backend = document.getElementById('backend').value;
                const response = await fetch('/models?backend=' + backend);
                const models = await response.json();
                
                // Populate all model dropdowns
                populateModelDropdown('upscaleModel', models.upscale || {{}});
                populateModelDropdown('interpolateModel', models.interpolate || {{}});
                populateModelDropdown('denoiseModel', models.denoise || {{}});
                populateModelDropdown('decompressModel', models.decompress || {{}});
                
                // Update models list display
                const modelsList = document.getElementById('modelsList');
                modelsList.innerHTML = '';
                
                for (const [category, categoryModels] of Object.entries(models)) {{
                    if (Object.keys(categoryModels).length === 0) continue;
                    
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
                                '<div>' + modelInfo.filename + '</div>' +
                                '<div>' + scaleBadge + backendBadge + engineBadge + '</div>' +
                            '</div>';
                        
                        categoryDiv.appendChild(modelDiv);
                    }}
                    
                    modelsList.appendChild(categoryDiv);
                }}
                
                // Set default selections
                setDefaultModelSelection('upscaleModel', ['Nomos8k', 'realesr']);
                setDefaultModelSelection('interpolateModel', ['rife']);
                setDefaultModelSelection('denoiseModel', ['drunet']);
                setDefaultModelSelection('decompressModel', ['deh264']);
                
            }} catch (error) {{
                console.error('Error loading models:', error);
                document.getElementById('modelsList').innerHTML = 
                    '<p>Error loading models. Please check console.</p>' +
                    '<p>Error: ' + error.message + '</p>';
            }}
        }}
        
        function populateModelDropdown(dropdownId, models) {{
            const select = document.getElementById(dropdownId);
            select.innerHTML = '<option value="">Select model...</option>';
            
            for (const [modelName, modelInfo] of Object.entries(models)) {{
                const option = document.createElement('option');
                option.value = modelInfo.original_name || modelName;
                option.textContent = (modelInfo.original_name || modelName) + (modelInfo.is_engine ? ' (TensorRT)' : '');
                option.dataset.fullName = modelName;
                select.appendChild(option);
            }}
        }}
        
        function setDefaultModelSelection(dropdownId, keywords) {{
            const select = document.getElementById(dropdownId);
            if (select.options.length > 1) {{
                for (let i = 0; i < select.options.length; i++) {{
                    for (const keyword of keywords) {{
                        if (select.options[i].value.includes(keyword)) {{
                            select.selectedIndex = i;
                            return;
                        }}
                    }}
                }}
            }}
        }}
        
        // Load downloadable models
        async function loadDownloadableModels() {{
            try {{
                const response = await fetch('/downloadable_models');
                const models = await response.json();
                
                const downloadList = document.getElementById('downloadModelsList');
                downloadList.innerHTML = '';
                
                if (models.length === 0) {{
                    downloadList.innerHTML = '<p>No downloadable models available</p>';
                    return;
                }}
                
                for (const model of models) {{
                    const modelDiv = document.createElement('div');
                    modelDiv.className = 'download-model-item';
                    modelDiv.id = 'download-model-' + model.id;
                    
                    const isDownloaded = model.status === 'downloaded';
                    const isDownloading = downloadingModels[model.id];
                    
                    modelDiv.innerHTML = '<div class="download-model-header">' +
                        '<div class="download-model-name">' + model.name + '</div>' +
                        '<span class="job-status ' + (isDownloaded ? 'status-completed' : 'status-processing') + '">' + 
                            (isDownloaded ? 'DOWNLOADED' : (isDownloading ? 'DOWNLOADING' : 'AVAILABLE')) + 
                        '</span>' +
                        '</div>' +
                        '<div class="download-model-desc">' + model.description + '</div>' +
                        (isDownloading ? 
                            '<div class="model-download-progress">' +
                                '<div class="model-download-progress-fill" id="progress-' + model.id + '" style="width: 0%"></div>' +
                            '</div>' +
                            '<div class="model-download-size" id="size-' + model.id + '">0%</div>' : '') +
                        '<div style="display: flex; gap: 10px;">' +
                        (!isDownloaded && !isDownloading ? 
                            '<button class="btn btn-primary" onclick="downloadModel(\\'' + model.id + '\\')" style="padding: 6px 12px; font-size: 12px;">' +
                                '<i class="fas fa-download"></i> Download (' + model.size + ')' +
                            '</button>' : '') +
                        (isDownloaded ? 
                            '<button class="btn" onclick="deleteModel(\\'' + model.id + '\\')" style="background: #475569; color: white; padding: 6px 12px; font-size: 12px;">' +
                                '<i class="fas fa-trash"></i> Delete' +
                            '</button>' : '') +
                        '</div>';
                    
                    downloadList.appendChild(modelDiv);
                }}
                
            }} catch (error) {{
                console.error('Error loading downloadable models:', error);
                document.getElementById('downloadModelsList').innerHTML = 
                    '<p>Error loading models. Please check console.</p>' +
                    '<p>Error: ' + error.message + '</p>';
            }}
        }}
        
        async function downloadModel(modelId) {{
            downloadingModels[modelId] = true;
            
            try {{
                const response = await fetch('/download_model/' + modelId, {{
                    method: 'POST'
                }});
                
                const result = await response.json();
                if (result.success) {{
                    showToast('Download started: ' + modelId, 'success');
                    // Update UI
                    loadDownloadableModels();
                }} else {{
                    showToast('Download failed: ' + result.error, 'error');
                    delete downloadingModels[modelId];
                }}
            }} catch (error) {{
                showToast('Download failed: ' + error.message, 'error');
                delete downloadingModels[modelId];
            }}
        }}
        
        function updateDownloadProgress(data) {{
            const progressBar = document.getElementById('progress-' + data.model_id);
            const sizeText = document.getElementById('size-' + data.model_id);
            
            if (progressBar) {{
                progressBar.style.width = data.progress + '%';
            }}
            if (sizeText) {{
                sizeText.textContent = data.progress + '% (' + data.downloaded + '/' + data.total + ')';
            }}
        }}
        
        function downloadComplete(data) {{
            delete downloadingModels[data.model_id];
            showToast('Download complete: ' + data.model_name, 'success');
            loadDownloadableModels();
            loadModels(); // Reload installed models
        }}
        
        async function deleteModel(modelId) {{
            if (!confirm('Are you sure you want to delete this model?')) return;
            
            try {{
                const response = await fetch('/delete_model/' + modelId, {{
                    method: 'DELETE'
                }});
                
                const result = await response.json();
                if (result.success) {{
                    showToast('Model deleted', 'success');
                    loadDownloadableModels();
                    loadModels();
                }} else {{
                    showToast('Failed to delete model: ' + result.error, 'error');
                }}
            }} catch (error) {{
                showToast('Failed to delete model: ' + error.message, 'error');
            }}
        }}
        
        async function importPyTorchModel() {{
            const input = document.createElement('input');
            input.type = 'file';
            input.accept = '.pth,.safetensors,.pt';
            input.onchange = async (e) => {{
                const file = e.target.files[0];
                if (!file) return;
                
                const formData = new FormData();
                formData.append('file', file);
                formData.append('format', 'pytorch');
                
                try {{
                    const response = await fetch('/import_model', {{
                        method: 'POST',
                        body: formData
                    }});
                    
                    const result = await response.json();
                    if (result.success) {{
                        showToast('Model imported successfully!', 'success');
                        loadModels();
                    }} else {{
                        showToast('Import failed: ' + result.error, 'error');
                    }}
                }} catch (error) {{
                    showToast('Import failed: ' + error.message, 'error');
                }}
            }};
            input.click();
        }}
        
        async function importNCNNModel() {{
            showToast('Please select the .bin file first, then the .param file', 'info');
            
            const binInput = document.createElement('input');
            binInput.type = 'file';
            binInput.accept = '.bin';
            binInput.onchange = async (e) => {{
                const binFile = e.target.files[0];
                if (!binFile) return;
                
                const paramInput = document.createElement('input');
                paramInput.type = 'file';
                paramInput.accept = '.param';
                paramInput.onchange = async (e2) => {{
                    const paramFile = e2.target.files[0];
                    if (!paramFile) return;
                    
                    const formData = new FormData();
                    formData.append('bin_file', binFile);
                    formData.append('param_file', paramFile);
                    formData.append('format', 'ncnn');
                    
                    try {{
                        const response = await fetch('/import_model', {{
                            method: 'POST',
                            body: formData
                        }});
                        
                        const result = await response.json();
                        if (result.success) {{
                            showToast('Model imported successfully!', 'success');
                            loadModels();
                        }} else {{
                            showToast('Import failed: ' + result.error, 'error');
                        }}
                    }} catch (error) {{
                        showToast('Import failed: ' + error.message, 'error');
                    }}
                }};
                paramInput.click();
            }};
            binInput.click();
        }}
        
        // Queue management
        function loadQueue() {{
            const queueList = document.getElementById('queueList');
            
            if (processingQueue.length === 0) {{
                queueList.innerHTML = '<p>Queue is empty</p>';
                return;
            }}
            
            queueList.innerHTML = '';
            
            processingQueue.forEach((item, index) => {{
                const queueDiv = document.createElement('div');
                queueDiv.className = 'queue-item';
                queueDiv.innerHTML = '<div class="job-header">' +
                    '<div>' +
                        '<div class="job-display-name">' + item.filename + '</div>' +
                        '<div class="queue-video-info">' +
                            '<span>Position: ' + (index + 1) + '</span>' +
                            '<span>Backend: ' + item.settings.backend + '</span>' +
                            '<span>GPU: ' + item.settings.gpu_id + '</span>' +
                        '</div>' +
                    '</div>' +
                    '<span class="job-status status-processing">QUEUED</span>' +
                    '</div>' +
                    '<div style="margin-top: 10px; display: flex; gap: 5px;">' +
                    '<button class="btn" onclick="removeFromQueue(' + index + ')" style="background: #475569; color: white; padding: 6px 12px; font-size: 12px;">' +
                        '<i class="fas fa-times"></i> Remove' +
                    '</button>' +
                    '<button class="btn" onclick="moveUpInQueue(' + index + ')" style="background: #475569; color: white; padding: 6px 12px; font-size: 12px;" ' + (index === 0 ? 'disabled' : '') + '>' +
                        '<i class="fas fa-arrow-up"></i> Up' +
                    '</button>' +
                    '<button class="btn" onclick="moveDownInQueue(' + index + ')" style="background: #475569; color: white; padding: 6px 12px; font-size: 12px;" ' + (index === processingQueue.length - 1 ? 'disabled' : '') + '>' +
                        '<i class="fas fa-arrow-down"></i> Down' +
                    '</button>' +
                    '</div>';
                
                queueList.appendChild(queueDiv);
            }});
        }}
        
        function addToQueue() {{
            if (!currentVideo) {{
                showToast('Please upload a video first', 'error');
                return;
            }}
            
            const settings = getCurrentSettings();
            
            // Validate required settings
            if (settings.upscale_enabled && !settings.upscale_model) {{
                showToast('Please select an upscale model', 'error');
                return;
            }}
            
            if (settings.interpolate_enabled && !settings.interpolate_model) {{
                showToast('Please select an interpolation model', 'error');
                return;
            }}
            
            if (settings.denoise_enabled && !settings.denoise_model) {{
                showToast('Please select a denoise model', 'error');
                return;
            }}
            
            if (settings.decompress_enabled && !settings.decompress_model) {{
                showToast('Please select a decompress model', 'error');
                return;
            }}
            
            const queueItem = {{
                filename: currentVideo,
                settings: settings,
                timestamp: new Date().toISOString()
            }};
            
            processingQueue.push(queueItem);
            loadQueue();
            showToast('Video added to queue', 'success');
        }}
        
        function removeFromQueue(index) {{
            processingQueue.splice(index, 1);
            loadQueue();
            showToast('Video removed from queue', 'info');
        }}
        
        function moveUpInQueue(index) {{
            if (index > 0) {{
                const temp = processingQueue[index];
                processingQueue[index] = processingQueue[index - 1];
                processingQueue[index - 1] = temp;
                loadQueue();
            }}
        }}
        
        function moveDownInQueue(index) {{
            if (index < processingQueue.length - 1) {{
                const temp = processingQueue[index];
                processingQueue[index] = processingQueue[index + 1];
                processingQueue[index + 1] = temp;
                loadQueue();
            }}
        }}
        
        async function processQueue() {{
            if (processingQueue.length === 0) {{
                showToast('Queue is empty', 'error');
                return;
            }}
            
            showToast('Processing ' + processingQueue.length + ' videos from queue...', 'info');
            
            for (let i = 0; i < processingQueue.length; i++) {{
                const item = processingQueue[i];
                
                try {{
                    const response = await fetch('/process', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify({{
                            filename: item.filename,
                            settings: item.settings,
                            use_real_backend: backendAvailable
                        }})
                    }});
                    
                    const result = await response.json();
                    
                    if (result.success) {{
                        showToast('Started processing: ' + item.filename, 'success');
                    }} else {{
                        showToast('Failed to start: ' + item.filename, 'error');
                    }}
                    
                    // Small delay between starting jobs
                    await new Promise(resolve => setTimeout(resolve, 1000));
                    
                }} catch (error) {{
                    showToast('Error starting job: ' + error.message, 'error');
                }}
            }}
            
            // Clear queue after processing
            processingQueue = [];
            loadQueue();
            
            // Switch to jobs tab
            document.querySelector('[data-tab="jobs"]').click();
            loadJobs();
            
            showToast('All videos from queue have been submitted for processing', 'success');
        }}
        
        function clearQueue() {{
            if (processingQueue.length === 0) return;
            
            if (confirm('Clear all videos from queue?')) {{
                processingQueue = [];
                loadQueue();
                showToast('Queue cleared', 'info');
            }}
        }}
        
        // Job management
        async function loadJobs() {{
            try {{
                const response = await fetch('/jobs');
                const jobs = await response.json();
                
                const jobsList = document.getElementById('jobsList');
                jobsList.innerHTML = '';
                
                if (jobs.length === 0) {{
                    jobsList.innerHTML = '<p>No jobs yet</p>';
                    return;
                }}
                
                // Sort jobs by timestamp (newest first)
                jobs.sort((a, b) => {{
                    const timeA = a.start_time ? new Date(a.start_time).getTime() : 0;
                    const timeB = b.start_time ? new Date(b.start_time).getTime() : 0;
                    return timeB - timeA;
                }});
                
                jobs.forEach(job => {{
                    const jobDiv = document.createElement('div');
                    jobDiv.className = 'job-item ' + job.status;
                    jobDiv.id = 'job-' + job.id;
                    
                    const statusClass = 'status-' + job.status;
                    
                    // Extract original filename and timestamp
                    const originalFilename = job.input_path ? job.input_path.split('/').pop() : 'Unknown';
                    const displayName = job.display_name || originalFilename;
                    const timestamp = job.start_time ? formatTimestamp(job.start_time) : '';
                    
                    jobDiv.innerHTML = '<div class="job-header">' +
                        '<div>' +
                            '<div class="job-display-name">' + displayName + '</div>' +
                            '<div class="job-video-info">' +
                                '<span>GPU ' + (job.settings?.gpu_id || 0) + '</span>' +
                                '<span>' + (job.settings?.backend || 'pytorch').toUpperCase() + '</span>' +
                            '</div>' +
                        '</div>' +
                        '<span class="job-status ' + statusClass + '">' + job.status.toUpperCase() + '</span>' +
                        '</div>' +
                        '<div class="job-timestamp">Started: ' + timestamp + '</div>' +
                        (job.status === 'processing' ? 
                            '<div>Progress: ' + job.progress.toFixed(1) + '%</div>' +
                            '<div class="progress-bar">' +
                                '<div class="progress-fill" style="width: ' + job.progress + '%"></div>' +
                            '</div>' : '') +
                        '<div style="margin-top: 10px; display: flex; gap: 5px; flex-wrap: wrap;">' +
                        (job.status === 'completed' ? 
                            '<button class="btn btn-primary" onclick="showComparison(\\'' + job.id + '\\')" style="padding: 6px 12px; font-size: 12px;">' +
                                '<i class="fas fa-compress-arrows-alt"></i> Compare' +
                            '</button>' +
                            '<button class="btn btn-primary" onclick="showFullscreenComparison(\\'' + job.id + '\\')" style="padding: 6px 12px; font-size: 12px;">' +
                                '<i class="fas fa-expand"></i> Fullscreen' +
                            '</button>' +
                            '<button class="btn btn-primary" onclick="showVideoPreview(\\'' + job.id + '\\')" style="padding: 6px 12px; font-size: 12px;">' +
                                '<i class="fas fa-play"></i> Preview' +
                            '</button>' +
                            '<button class="btn btn-primary" onclick="downloadResult(\\'' + job.id + '\\')" style="padding: 6px 12px; font-size: 12px;">' +
                                '<i class="fas fa-download"></i> Download' +
                            '</button>' : '') +
                        '<button class="btn" onclick="deleteJob(\\'' + job.id + '\\')" style="background: #475569; color: white; padding: 6px 12px; font-size: 12px;">' +
                            '<i class="fas fa-trash"></i>' +
                        '</button>' +
                        '</div>';
                    
                    jobsList.appendChild(jobDiv);
                }});
            }} catch (error) {{
                console.error('Error loading jobs:', error);
            }}
        }}
        
        function refreshJobs() {{
            loadJobs();
            showToast('Jobs refreshed', 'success');
        }}
        
        function updateJobInList(job) {{
            const jobElement = document.getElementById('job-' + job.id);
            if (jobElement) {{
                jobElement.outerHTML = '';
            }}
            loadJobs();
            
            if (job.status === 'completed') {{
                showToast('Job completed: ' + (job.display_name || 'Unknown'), 'success');
            }}
        }}
        
        async function deleteJob(jobId) {{
            try {{
                const response = await fetch('/jobs/' + jobId + '/delete', {{
                    method: 'DELETE'
                }});
                
                const result = await response.json();
                if (result.success) {{
                    showToast('Job deleted', 'success');
                    loadJobs();
                    
                    // Exit comparison if we're comparing this job
                    if (isComparisonMode && currentComparisonJobId === jobId) {{
                        exitComparisonMode();
                    }}
                }}
            }} catch (error) {{
                showToast('Failed to delete job', 'error');
            }}
        }}
        
        function showVideoPreview(jobId) {{
            // Exit comparison mode if active
            if (isComparisonMode) {{
                exitComparisonMode();
            }}
            
            // Find job
            fetch('/jobs')
                .then(response => response.json())
                .then(jobs => {{
                    const job = jobs.find(j => j.id === jobId);
                    if (job && job.output_url) {{
                        const videoPlayer = document.getElementById('videoPlayer');
                        const videoFrame = document.getElementById('videoFrame');
                        const videoPlaceholder = document.getElementById('videoPlaceholder');
                        
                        videoPlayer.src = job.output_url;
                        videoPlayer.style.display = 'block';
                        videoFrame.style.display = 'none';
                        videoPlaceholder.style.display = 'none';
                        videoPlayer.load();
                        videoPlayer.play();
                        
                        showToast('Playing enhanced video', 'success');
                    }} else if (job && job.preview) {{
                        // Fallback to thumbnail
                        const videoFrame = document.getElementById('videoFrame');
                        const videoPlayer = document.getElementById('videoPlayer');
                        const videoPlaceholder = document.getElementById('videoPlaceholder');
                        
                        videoFrame.src = job.preview;
                        videoFrame.style.display = 'block';
                        videoPlayer.style.display = 'none';
                        videoPlaceholder.style.display = 'none';
                        
                        showToast('Showing preview image', 'info');
                    }}
                }});
        }}
        
        async function downloadResult(jobId) {{
            window.open('/download/' + jobId, '_blank');
        }}
        
        // Processing
        function getCurrentSettings() {{
            // Get full model name from dropdowns
            const upscaleSelect = document.getElementById('upscaleModel');
            const interpolateSelect = document.getElementById('interpolateModel');
            const denoiseSelect = document.getElementById('denoiseModel');
            const decompressSelect = document.getElementById('decompressModel');
            
            const upscaleModelValue = upscaleSelect.value;
            const upscaleFullName = upscaleSelect.options[upscaleSelect.selectedIndex]?.dataset.fullName || upscaleModelValue;
            
            const interpolateModelValue = interpolateSelect.value;
            const interpolateFullName = interpolateSelect.options[interpolateSelect.selectedIndex]?.dataset.fullName || interpolateModelValue;
            
            const denoiseModelValue = denoiseSelect.value;
            const denoiseFullName = denoiseSelect.options[denoiseSelect.selectedIndex]?.dataset.fullName || denoiseModelValue;
            
            const decompressModelValue = decompressSelect.value;
            const decompressFullName = decompressSelect.options[decompressSelect.selectedIndex]?.dataset.fullName || decompressModelValue;
            
            return {{
                upscale_enabled: document.getElementById('enableUpscale').checked,
                upscale_model: upscaleFullName,
                upscale_factor: parseInt(document.getElementById('upscaleFactor').value),
                interpolate_enabled: document.getElementById('enableInterpolate').checked,
                interpolate_model: interpolateFullName,
                interpolate_factor: parseFloat(document.getElementById('interpolateFactor').value),
                denoise_enabled: document.getElementById('enableDenoise').checked,
                denoise_model: denoiseFullName,
                decompress_enabled: document.getElementById('enableDecompress').checked,
                decompress_model: decompressFullName,
                backend: document.getElementById('backend').value,
                gpu_id: parseInt(document.getElementById('gpuSelection').value),
                precision: document.getElementById('precision').value,
                output_format: document.getElementById('outputFormat').value,
                crf: parseInt(document.getElementById('crf').value),
                tiling_enabled: document.getElementById('tilingEnabled').checked,
                benchmark_mode: document.getElementById('benchmarkMode').checked,
                ensemble_mode: document.getElementById('ensembleMode').checked,
                auto_hdr_mode: document.getElementById('autoHDRMode').checked,
                use_real_backend: backendAvailable
            }};
        }}
        
        async function startProcessing() {{
            if (!currentVideo) {{
                showToast('Please upload a video first', 'error');
                return;
            }}
            
            const settings = getCurrentSettings();
            
            // Validate required settings
            if (settings.upscale_enabled && !settings.upscale_model) {{
                showToast('Please select an upscale model', 'error');
                return;
            }}
            
            if (settings.interpolate_enabled && !settings.interpolate_model) {{
                showToast('Please select an interpolation model', 'error');
                return;
            }}
            
            if (settings.denoise_enabled && !settings.denoise_model) {{
                showToast('Please select a denoise model', 'error');
                return;
            }}
            
            if (settings.decompress_enabled && !settings.decompress_model) {{
                showToast('Please select a decompress model', 'error');
                return;
            }}
            
            // Warn if backend is not available
            if (!backendAvailable) {{
                const useDemo = confirm('RVE backend is not available. Would you like to run in demo mode? (Processing will be simulated)');
                if (!useDemo) return;
            }}
            
            // Show info if TensorRT is selected
            if (settings.backend === 'tensorrt') {{
                showToast('TensorRT selected - will use PyTorch fallback with GPU acceleration', 'info');
            }}
            
            try {{
                const response = await fetch('/process', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{
                        filename: currentVideo,
                        settings: settings,
                        use_real_backend: backendAvailable
                    }})
                }});
                
                const result = await response.json();
                
                if (result.success) {{
                    showToast('Processing started! Check Jobs tab.', 'success');
                    document.querySelector('[data-tab="jobs"]').click();
                    loadJobs();
                }} else {{
                    showToast(result.error || 'Failed to start processing', 'error');
                }}
            }} catch (error) {{
                showToast('Failed to start processing: ' + error.message, 'error');
            }}
        }}
        
        // COMPARISON FUNCTIONS
        let currentComparisonJobId = null;
        
        function showComparison(jobId) {{
            fetch('/jobs')
                .then(response => response.json())
                .then(jobs => {{
                    const job = jobs.find(j => j.id === jobId);
                    if (job && job.output_url) {{
                        // Get original video URL
                        const originalFilename = job.input_path ? job.input_path.split('/').pop() : '';
                        const originalUrl = originalFilename ? '/uploads/' + encodeURIComponent(originalFilename) : '';
                        
                        if (!originalUrl) {{
                            showToast('Original video not found', 'error');
                            return;
                        }}
                        
                        // Store current job ID for highlighting
                        currentComparisonJobId = jobId;
                        
                        // Enter comparison mode
                        enterComparisonMode(originalUrl, job.output_url, job.id);
                        showToast('Comparison mode activated. Drag the center handle to compare.', 'success');
                    }}
                }});
        }}
        
        function showFullscreenComparison(jobId) {{
            fetch('/jobs')
                .then(response => response.json())
                .then(jobs => {{
                    const job = jobs.find(j => j.id === jobId);
                    if (job && job.output_url) {{
                        // Get original video URL
                        const originalFilename = job.input_path ? job.input_path.split('/').pop() : '';
                        const originalUrl = originalFilename ? '/uploads/' + encodeURIComponent(originalFilename) : '';
                        
                        if (!originalUrl) {{
                            showToast('Original video not found', 'error');
                            return;
                        }}
                        
                        // Store current job ID for highlighting
                        currentComparisonJobId = jobId;
                        
                        // Enter comparison mode with fullscreen option
                        enterComparisonMode(originalUrl, job.output_url, job.id);
                        // Trigger fullscreen after a short delay
                        setTimeout(() => {{
                            if (isComparisonMode && videoComparator) {{
                                enterFullscreen();
                            }}
                        }}, 500);
                    }}
                }});
        }}
        
        function enterComparisonMode(originalUrl, enhancedUrl, jobId) {{
            isComparisonMode = true;
            document.querySelector('.video-preview').classList.add('comparison-mode');
            
            // Load comparison settings
            showComparisonLabels = document.getElementById('showComparisonInfo')?.checked ?? true;
            isSyncEnabled = document.getElementById('autoSyncVideos')?.checked ?? true;
            
            // Initialize video comparator
            initVideoComparator(originalUrl, enhancedUrl);
            
            // Update job listing to show we're in comparison
            document.querySelectorAll('.job-item').forEach(item => {{
                if (item.id === 'job-' + jobId) {{
                    item.classList.add('comparing');
                }}
            }});
            
            // Update UI
            document.getElementById('syncStatus').textContent = isSyncEnabled ? 'On' : 'Off';
            updateComparisonUI();
        }}
        
        // FULLSCREEN FUNCTIONS
        function toggleFullscreen() {{
            if (!isComparisonMode) return;
            
            if (!isFullscreen) {{
                enterFullscreen();
            }} else {{
                exitFullscreen();
            }}
        }}
        
        function enterFullscreen() {{
            if (!isComparisonMode) return;
            
            const comparisonContainer = document.getElementById('comparisonContainer');
            const fullscreenBtn = document.getElementById('fullscreenBtn');
            const exitFullscreenBtn = document.getElementById('exitFullscreenBtn');
            
            if (comparisonContainer.requestFullscreen) {{
                comparisonContainer.requestFullscreen();
            }} else if (comparisonContainer.webkitRequestFullscreen) {{
                comparisonContainer.webkitRequestFullscreen();
            }} else if (comparisonContainer.msRequestFullscreen) {{
                comparisonContainer.msRequestFullscreen();
            }}
            
            // Update buttons
            fullscreenBtn.style.display = 'none';
            if (exitFullscreenBtn) exitFullscreenBtn.style.display = 'block';
            
            isFullscreen = true;
            
            // Resize videos for fullscreen
            if (videoComparator) {{
                videoComparator.resizeVideos();
            }}
            
            showToast('Entered fullscreen mode. Press ESC to exit.', 'success');
        }}
        
        function exitFullscreen() {{
            if (document.exitFullscreen) {{
                document.exitFullscreen();
            }} else if (document.webkitExitFullscreen) {{
                document.webkitExitFullscreen();
            }} else if (document.msExitFullscreen) {{
                document.msExitFullscreen();
            }}
            
            const fullscreenBtn = document.getElementById('fullscreenBtn');
            const exitFullscreenBtn = document.getElementById('exitFullscreenBtn');
            
            // Update buttons
            fullscreenBtn.style.display = 'block';
            if (exitFullscreenBtn) exitFullscreenBtn.style.display = 'none';
            
            isFullscreen = false;
            
            // Resize videos back to normal
            if (videoComparator) {{
                videoComparator.resizeVideos();
            }}
            
            showToast('Exited fullscreen mode', 'info');
        }}
        
        // Add fullscreen change listeners
        document.addEventListener('fullscreenchange', handleFullscreenChange);
        document.addEventListener('webkitfullscreenchange', handleFullscreenChange);
        document.addEventListener('msfullscreenchange', handleFullscreenChange);
        
        function handleFullscreenChange() {{
            const isFullscreenNow = !!(document.fullscreenElement || 
                                      document.webkitFullscreenElement || 
                                      document.msFullscreenElement);
            
            if (isFullscreen !== isFullscreenNow) {{
                isFullscreen = isFullscreenNow;
                const fullscreenBtn = document.getElementById('fullscreenBtn');
                const exitFullscreenBtn = document.getElementById('exitFullscreenBtn');
                
                if (isFullscreen) {{
                    fullscreenBtn.style.display = 'none';
                    if (exitFullscreenBtn) exitFullscreenBtn.style.display = 'block';
                }} else {{
                    fullscreenBtn.style.display = 'block';
                    if (exitFullscreenBtn) exitFullscreenBtn.style.display = 'none';
                }}
            }}
        }}
        
        function exitComparisonMode() {{
            // Exit fullscreen if active
            if (isFullscreen) {{
                exitFullscreen();
            }}
            
            isComparisonMode = false;
            document.querySelector('.video-preview').classList.remove('comparison-mode');
            
            // Stop and cleanup videos
            if (videoComparator) {{
                videoComparator.destroy();
                videoComparator = null;
            }}
            
            // Reset job highlighting
            document.querySelectorAll('.job-item').forEach(item => {{
                item.classList.remove('comparing');
            }});
            
            currentComparisonJobId = null;
            showToast('Exited comparison mode', 'info');
        }}
        
        function initVideoComparator(originalUrl, enhancedUrl) {{
            // Destroy existing comparator
            if (videoComparator) {{
                videoComparator.destroy();
            }}
            
            videoComparator = new VideoComparator(originalUrl, enhancedUrl);
        }}
        
        function resetComparison() {{
            if (videoComparator) {{
                videoComparator.reset();
                showToast('Comparison reset to 50%', 'info');
            }}
        }}
        
        function toggleSync() {{
            isSyncEnabled = !isSyncEnabled;
            document.getElementById('syncStatus').textContent = isSyncEnabled ? 'On' : 'Off';
            
            if (videoComparator) {{
                videoComparator.setSync(isSyncEnabled);
            }}
            
            showToast('Sync ' + (isSyncEnabled ? 'enabled' : 'disabled'), 'info');
        }}
        
        function togglePlayPause() {{
            if (videoComparator) {{
                videoComparator.togglePlayPause();
                const isPlaying = videoComparator.isPlaying;
                document.getElementById('playPauseIcon').className = isPlaying ? 'fas fa-pause' : 'fas fa-play';
                document.getElementById('playPauseText').textContent = isPlaying ? 'Pause' : 'Play';
            }}
        }}
        
        function toggleComparisonLabels() {{
            showComparisonLabels = !showComparisonLabels;
            const icon = document.getElementById('labelsIcon');
            icon.className = showComparisonLabels ? 'fas fa-eye' : 'fas fa-eye-slash';
            
            if (videoComparator) {{
                videoComparator.toggleLabels(showComparisonLabels);
            }}
            
            showToast('Comparison labels ' + (showComparisonLabels ? 'shown' : 'hidden'), 'info');
        }}
        
        function updateComparisonUI() {{
            document.getElementById('labelsIcon').className = showComparisonLabels ? 'fas fa-eye' : 'fas fa-eye-slash';
        }}
        
        // Video Comparator Class
        class VideoComparator {{
            constructor(originalUrl, enhancedUrl) {{
                this.originalVideo = document.getElementById('originalVideo');
                this.enhancedVideo = document.getElementById('enhancedVideo');
                this.originalVideoContainer = document.getElementById('originalVideoContainer');
                this.comparisonDivider = document.getElementById('comparisonDivider');
                this.fpsDisplay = document.getElementById('comparisonFps');
                this.timeDisplay = document.getElementById('comparisonTime');
                this.labels = document.querySelectorAll('.comparison-label');
                
                this.isPlaying = false;
                this.isDragging = false;
                this.isSynced = true;
                this.showLabels = true;
                this.showInfo = true;
                
                // Set video sources
                this.originalVideo.src = originalUrl;
                this.enhancedVideo.src = enhancedUrl;
                
                this.setupEvents();
                this.setupSync();
                this.setupDrag();
                this.setupInfoUpdates();
                this.resizeVideos();
            }}
            
            setupEvents() {{
                // Wait for videos to load
                const videosLoaded = Promise.all([
                    new Promise(resolve => this.originalVideo.addEventListener('loadedmetadata', resolve)),
                    new Promise(resolve => this.enhancedVideo.addEventListener('loadedmetadata', resolve))
                ]);
                
                videosLoaded.then(() => {{
                    // Set both videos to start
                    this.originalVideo.currentTime = 0;
                    this.enhancedVideo.currentTime = 0;
                    
                    // Set initial position (50% - half of original video visible)
                    this.reset();
                    
                    // Update FPS display
                    this.updateFPSDisplay();
                    
                    // Show videos
                    this.originalVideo.style.display = 'block';
                    this.enhancedVideo.style.display = 'block';
                    
                    // Adjust for initial size
                    this.resizeVideos();
                }});
                
                // Handle window resize
                window.addEventListener('resize', () => {{
                    this.resizeVideos();
                }});
                
                // Handle fullscreen changes
                document.addEventListener('fullscreenchange', () => this.resizeVideos());
                document.addEventListener('webkitfullscreenchange', () => this.resizeVideos());
                document.addEventListener('msfullscreenchange', () => this.resizeVideos());
            }}
            
            resizeVideos() {{
                // Videos will automatically resize due to CSS object-fit: contain
                // Force redraw for smooth transitions
                if (this.originalVideo) {{
                    this.originalVideo.style.width = '100%';
                    this.originalVideo.style.height = '100%';
                }}
                if (this.enhancedVideo) {{
                    this.enhancedVideo.style.width = '100%';
                    this.enhancedVideo.style.height = '100%';
                }}
            }}
            
            setupDrag() {{
                const divider = this.comparisonDivider;
                let isDragging = false;
                let startX = 0;
                let startClip = 50;
                
                // Mouse events
                divider.addEventListener('mousedown', (e) => {{
                    e.preventDefault();
                    e.stopPropagation();
                    
                    isDragging = true;
                    startX = e.clientX;
                    startClip = this.getCurrentClipPercentage();
                    
                    const onMouseMove = (e) => {{
                        if (!isDragging) return;
                        
                        const container = document.querySelector('.comparison-container');
                        const containerRect = container.getBoundingClientRect();
                        const deltaX = e.clientX - startX;
                        const deltaPercent = (deltaX / containerRect.width) * 100;
                        
                        let newClip = startClip - deltaPercent;
                        newClip = Math.max(10, Math.min(90, newClip));
                        
                        // Update clip mask
                        this.originalVideoContainer.style.clipPath = `inset(0 ${{newClip}}% 0 0)`;
                        
                        // Update divider position
                        this.comparisonDivider.style.left = `${{100 - newClip}}%`;
                    }};
                    
                    const onMouseUp = () => {{
                        isDragging = false;
                        document.removeEventListener('mousemove', onMouseMove);
                        document.removeEventListener('mouseup', onMouseUp);
                    }};
                    
                    document.addEventListener('mousemove', onMouseMove);
                    document.addEventListener('mouseup', onMouseUp);
                }});
                
                // Touch events for mobile
                divider.addEventListener('touchstart', (e) => {{
                    e.preventDefault();
                    e.stopPropagation();
                    
                    isDragging = true;
                    startX = e.touches[0].clientX;
                    startClip = this.getCurrentClipPercentage();
                    
                    const onTouchMove = (e) => {{
                        if (!isDragging) return;
                        
                        const container = document.querySelector('.comparison-container');
                        const containerRect = container.getBoundingClientRect();
                        const deltaX = e.touches[0].clientX - startX;
                        const deltaPercent = (deltaX / containerRect.width) * 100;
                        
                        let newClip = startClip - deltaPercent;
                        newClip = Math.max(10, Math.min(90, newClip));
                        
                        // Update clip mask
                        this.originalVideoContainer.style.clipPath = `inset(0 ${{newClip}}% 0 0)`;
                        
                        // Update divider position
                        this.comparisonDivider.style.left = `${{100 - newClip}}%`;
                    }};
                    
                    const onTouchEnd = () => {{
                        isDragging = false;
                        document.removeEventListener('touchmove', onTouchMove);
                        document.removeEventListener('touchend', onTouchEnd);
                    }};
                    
                    document.addEventListener('touchmove', onTouchMove);
                    document.addEventListener('touchend', onTouchEnd);
                }});
                
                // Double click to reset
                divider.addEventListener('dblclick', (e) => {{
                    e.preventDefault();
                    this.reset();
                }});
            }}
            
            getCurrentClipPercentage() {{
                const clipPath = this.originalVideoContainer.style.clipPath;
                if (!clipPath) return 50;
                
                const match = clipPath.match(/inset\\(0\\s+(\\d+)%\\s+0\\s+0\\)/);
                return match ? parseFloat(match[1]) : 50;
            }}
            
            setupSync() {{
                // Sync play events
                this.originalVideo.addEventListener('play', () => {{
                    if (this.isSynced && !this.enhancedVideo.paused) return;
                    if (this.isSynced) this.enhancedVideo.play();
                    this.isPlaying = true;
                }});
                
                this.enhancedVideo.addEventListener('play', () => {{
                    if (this.isSynced && !this.originalVideo.paused) return;
                    if (this.isSynced) this.originalVideo.play();
                    this.isPlaying = true;
                }});
                
                // Sync pause events
                this.originalVideo.addEventListener('pause', () => {{
                    if (this.isSynced) this.enhancedVideo.pause();
                    this.isPlaying = false;
                }});
                
                this.enhancedVideo.addEventListener('pause', () => {{
                    if (this.isSynced) this.originalVideo.pause();
                    this.isPlaying = false;
                }});
                
                // Sync time updates
                this.originalVideo.addEventListener('timeupdate', () => {{
                    if (!this.isSynced) return;
                    const diff = Math.abs(this.originalVideo.currentTime - this.enhancedVideo.currentTime);
                    if (diff > 0.1) {{
                        this.enhancedVideo.currentTime = this.originalVideo.currentTime;
                    }}
                }});
            }}
            
            setupInfoUpdates() {{
                // Update time display
                this.originalVideo.addEventListener('timeupdate', () => {{
                    this.updateTimeDisplay();
                }});
                
                // Update FPS periodically
                setInterval(() => {{
                    this.updateFPSDisplay();
                }}, 1000);
            }}
            
            updateTimeDisplay() {{
                if (!this.showInfo) {{
                    this.timeDisplay.style.display = 'none';
                    return;
                }}
                
                const time = this.originalVideo.currentTime;
                const minutes = Math.floor(time / 60);
                const seconds = Math.floor(time % 60);
                this.timeDisplay.textContent = 'Time: ' + minutes.toString().padStart(2, '0') + ':' + seconds.toString().padStart(2, '0');
                this.timeDisplay.style.display = 'block';
            }}
            
            updateFPSDisplay() {{
                if (!this.showInfo) {{
                    this.fpsDisplay.style.display = 'none';
                    return;
                }}
                
                const fps = this.originalVideo.playbackRate;
                this.fpsDisplay.textContent = 'Speed: ' + fps.toFixed(1) + 'x';
                this.fpsDisplay.style.display = 'block';
            }}
            
            reset() {{
                this.originalVideoContainer.style.clipPath = 'inset(0 50% 0 0)';
                this.comparisonDivider.style.left = '50%';
            }}
            
            setSync(enabled) {{
                this.isSynced = enabled;
            }}
            
            togglePlayPause() {{
                if (this.isPlaying) {{
                    this.originalVideo.pause();
                    this.enhancedVideo.pause();
                }} else {{
                    this.originalVideo.play();
                    if (this.isSynced) {{
                        this.enhancedVideo.play();
                    }}
                }}
                this.isPlaying = !this.isPlaying;
            }}
            
            toggleLabels(show) {{
                this.showLabels = show;
                this.labels.forEach(label => {{
                    label.style.display = show ? 'block' : 'none';
                }});
            }}
            
            destroy() {{
                this.originalVideo.pause();
                this.enhancedVideo.pause();
                
                this.originalVideo.src = '';
                this.enhancedVideo.src = '';
                
                // Remove event listeners
                window.removeEventListener('resize', this.resizeVideos);
                document.removeEventListener('fullscreenchange', this.resizeVideos);
                document.removeEventListener('webkitfullscreenchange', this.resizeVideos);
                document.removeEventListener('msfullscreenchange', this.resizeVideos);
            }}
        }}
        
        // Initialize everything
        window.addEventListener('DOMContentLoaded', () => {{
            initSocket();
            loadModels();
            loadJobs();
            
            // Initialize GPU restriction
            updateGPUForBackend(document.getElementById('backend').value);
            
            // Initialize comparison settings from saved preferences
            const savedTheme = localStorage.getItem('theme') || 'dark';
            document.getElementById('theme').value = savedTheme;
            
            const savedMaxJobs = localStorage.getItem('maxJobs') || '2';
            document.getElementById('maxJobs').value = savedMaxJobs;
            
            const savedAutoSync = localStorage.getItem('autoSyncVideos') || 'true';
            document.getElementById('autoSyncVideos').checked = savedAutoSync === 'true';
            
            const savedShowInfo = localStorage.getItem('showComparisonInfo') || 'true';
            document.getElementById('showComparisonInfo').checked = savedShowInfo === 'true';
            
            const savedAutoHDR = localStorage.getItem('autoHDRMode') || 'true';
            document.getElementById('autoHDRMode').checked = savedAutoHDR === 'true';
            
            const savedTiling = localStorage.getItem('tilingEnabled') || 'false';
            document.getElementById('tilingEnabled').checked = savedTiling === 'true';
            
            // Save settings on change
            document.getElementById('theme').addEventListener('change', function() {{
                localStorage.setItem('theme', this.value);
            }});
            
            document.getElementById('maxJobs').addEventListener('change', function() {{
                localStorage.setItem('maxJobs', this.value);
            }});
            
            document.getElementById('autoSyncVideos').addEventListener('change', function() {{
                localStorage.setItem('autoSyncVideos', this.checked);
            }});
            
            document.getElementById('showComparisonInfo').addEventListener('change', function() {{
                localStorage.setItem('showComparisonInfo', this.checked);
            }});
            
            document.getElementById('autoHDRMode').addEventListener('change', function() {{
                localStorage.setItem('autoHDRMode', this.checked);
            }});
            
            document.getElementById('tilingEnabled').addEventListener('change', function() {{
                localStorage.setItem('tilingEnabled', this.checked);
            }});
        }});
    </script>
</body>
</html>'''
    
    with open(template_path, 'w') as f:
        f.write(html_content)
    
    print(f"Created HTML template at: {template_path}")
    return template_path

# Creează template-ul HTML
create_html_template()

# ============================================================================
# DATA MODELS CU GPU SUPPORT
# ============================================================================

@dataclass
class ProcessingSettings:
    """Settings for video processing"""
    upscale_enabled: bool = False
    upscale_model: str = ""
    upscale_factor: int = 2
    interpolate_enabled: bool = False
    interpolate_model: str = ""
    interpolate_factor: float = 2.0
    denoise_enabled: bool = False
    denoise_model: str = ""
    decompress_enabled: bool = False
    decompress_model: str = ""
    scene_detect_enabled: bool = True
    backend: str = "pytorch"
    gpu_id: int = 0  # Nou: ID-ul GPU-ului
    precision: str = "auto"
    tile_size: int = 0
    output_format: str = "mp4"
    output_codec: str = "libx264"
    crf: int = 18
    audio_quality: str = "copy"
    tiling_enabled: bool = False
    benchmark_mode: bool = False
    ensemble_mode: bool = False
    auto_hdr_mode: bool = True
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ProcessingSettings':
        """Create ProcessingSettings from dictionary"""
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})

@dataclass
class ProcessingJob:
    """A video processing job"""
    id: str
    input_path: str
    output_path: str
    display_name: str  # Numele afișat pentru job
    settings: ProcessingSettings
    status: str = "pending"
    progress: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    process_pid: Optional[int] = None
    log_file: Optional[str] = None
    use_real_backend: bool = True
    preview_path: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        result = asdict(self)
        result['settings'] = asdict(self.settings)
        if self.start_time:
            result['start_time'] = self.start_time.isoformat()
        if self.end_time:
            result['end_time'] = self.end_time.isoformat()
        
        # Add preview as base64 if exists
        if self.preview_path and os.path.exists(self.preview_path):
            try:
                with open(self.preview_path, 'rb') as f:
                    preview_data = base64.b64encode(f.read()).decode('utf-8')
                    result['preview'] = f"data:image/jpeg;base64,{preview_data}"
            except:
                pass
        
        # Add output URL for video playback
        if self.status == 'completed' and self.output_path and os.path.exists(self.output_path):
            filename = os.path.basename(self.output_path)
            result['output_url'] = f'/output/{filename}'
            result['output_filename'] = filename
        
        # Add original URL for comparison
        if self.input_path and os.path.exists(self.input_path):
            filename = os.path.basename(self.input_path)
            result['original_url'] = f'/uploads/{filename}'
        
        return result

# ============================================================================
# MODEL MANAGEMENT - ORGANIZED BY SUBFOLDERS (UPDATED VERSION)
# ============================================================================

class ModelManager:
    """Manages AI models for video enhancement organized by subfolders"""
    
    def __init__(self):
        self.categories = {
            'upscale': 'Upscaling Models',
            'interpolate': 'Frame Interpolation',
            'denoise': 'Denoising Models',
            'decompress': 'Compression Artifact Removal'
        }
    
    def get_available_models(self, category: str = None, backend: str = None) -> dict:
        """Get available models organized by subfolders - CORECT PENTRU FIECARE BACKEND"""
        all_models = {}
        
        models_dir = Path(config.MODELS_FOLDER)
        
        for cat in self.categories.keys():
            if category and cat != category:
                continue
                
            all_models[cat] = self._scan_models_for_backend(cat, backend, models_dir)
        
        return all_models if not category else {category: all_models.get(category, {})}
    
    def _scan_models_for_backend(self, category: str, backend: str, models_dir: Path) -> dict:
        """Scan models specifically for each backend"""
        models = {}
        
        # Extensii pentru fiecare backend
        extensions_map = {
            'pytorch': ['.pth', '.pt', '.safetensors', '.pkl'],
            'ncnn': ['.bin', '.param'],
            'tensorrt': ['.pth', '.pt', '.safetensors', '.pkl'],  # Folosește PyTorch pentru TensorRT fallback
        }
        
        extensions = extensions_map.get(backend, extensions_map['pytorch'])
        
        # Căutare recursivă
        for item in models_dir.rglob("*"):
            if item.is_file() and item.suffix.lower() in extensions:
                # Verifică categoria
                model_category = self._categorize_model(item.name.lower(), item.suffix.lower())
                if model_category != category:
                    continue
                
                # Pentru NCNN, asigură-te că avem ambele fișiere
                if backend == 'ncnn' and item.suffix.lower() == '.bin':
                    param_file = item.with_suffix('.param')
                    if not param_file.exists():
                        continue  # Skip .bin fără .param
                
                model_name = item.stem
                model_info = self._get_model_info(item, category, backend)
                
                models[model_name] = model_info
        
        return models
    
    def _get_model_info(self, item: Path, category: str, backend: str) -> dict:
        """Get model information"""
        model_name = item.stem
        
        # Clean name
        clean_name = model_name
        if item.suffix == '.engine':
            # Try to extract original PyTorch name
            if '_fp16_' in model_name:
                clean_name = model_name.split('_fp16_')[0]
        
        # Detect scale
        scale = self._detect_scale(clean_name)
        
        # Detect backend
        backends = []
        if item.suffix == '.engine':
            backends = ['tensorrt']
        elif item.suffix in ['.pth', '.pt', '.safetensors', '.pkl']:
            backends = ['pytorch', 'tensorrt']  # PyTorch works for TensorRT fallback
        elif item.suffix in ['.bin', '.param']:
            backends = ['ncnn']
        
        return {
            'scale': scale,
            'backend': backends,
            'filename': item.name,
            'path': str(item),
            'original_name': clean_name,
            'is_engine': item.suffix == '.engine',
            'category': category
        }
    
    def _categorize_model(self, filename: str, extension: str) -> str:
        """Categorize model based on filename and extension"""
        filename_lower = filename.lower()
        
        if 'rife' in filename_lower or 'gmfss' in filename_lower:
            return 'interpolate'
        elif 'drunet' in filename_lower or 'dncnn' in filename_lower or 'scunet' in filename_lower or 'denoise' in filename_lower:
            return 'denoise'
        elif 'deh264' in filename_lower or 'decompress' in filename_lower or ('span' in filename_lower and 'deh264' in filename_lower):
            return 'decompress'
        elif 'nomos' in filename_lower or 'realesr' in filename_lower or 'upscale' in filename_lower or 'anime' in filename_lower or '2x' in filename_lower or '4x' in filename_lower or 'span' in filename_lower or 'conservative' in filename_lower:
            return 'upscale'
        
        # Default pentru orice altceva
        return 'upscale'
    
    def _detect_scale(self, model_name: str) -> int:
        """Detect scale factor from model filename"""
        name_lower = model_name.lower()
        
        if 'x4' in name_lower or '4x' in name_lower or 'x4plus' in name_lower:
            return 4
        elif 'x3' in name_lower or '3x' in name_lower:
            return 3
        elif 'x2' in name_lower or '2x' in name_lower:
            return 2
        elif 'x1' in name_lower or '1x' in name_lower:
            return 1
        elif 'nomos8k' in name_lower:
            return 4
        elif 'up2x' in name_lower:
            return 2
        elif 'realesr-general-x4v3' in name_lower:
            return 4
        elif 'realesr-animevideov3-x2' in name_lower:
            return 2
        
        # Default
        return 2

# ============================================================================
# DOWNLOADABLE MODELS DEFINITION
# ============================================================================

DOWNLOADABLE_MODELS = [
    {
        'id': 'realesrgan-x4plus',
        'name': 'RealESRGAN x4plus',
        'description': 'General purpose x4 upscaler',
        'size': '64MB',
        'category': 'upscale',
        'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
    },
    {
        'id': 'realesr-animevideov3',
        'name': 'RealESR AnimeVideo v3',
        'description': 'Anime video upscaler',
        'size': '67MB',
        'category': 'upscale',
        'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth'
    },
    {
        'id': 'rife-v4.6',
        'name': 'RIFE v4.6',
        'description': 'Frame interpolation model',
        'size': '20MB',
        'category': 'interpolate',
        'url': 'https://github.com/hzwer/Practical-RIFE/releases/download/v1.0/rife-v4.6.pth'
    },
    {
        'id': 'drunet',
        'name': 'DRUNet',
        'description': 'Denoising model',
        'size': '55MB',
        'category': 'denoise',
        'url': 'https://github.com/cszn/KAIR/releases/download/v1.0/drunet.pth'
    },
    {
        'id': 'deh264',
        'name': 'DeH264',
        'description': 'H.264 compression artifact removal',
        'size': '42MB',
        'category': 'decompress',
        'url': 'https://github.com/cszn/KAIR/releases/download/v1.0/deh264.pth'
    }
]

# ============================================================================
# MODEL DOWNLOADER
# ============================================================================

class ModelDownloader:
    """Handles downloading of AI models"""
    
    def __init__(self):
        self.downloading_models = {}
        
    def get_downloadable_models(self):
        """Get list of downloadable models with status"""
        models_with_status = []
        model_manager = ModelManager()
        installed_models = model_manager.get_available_models()
        
        # Flatten installed models for easy checking
        installed_model_names = []
        for category_models in installed_models.values():
            for model_info in category_models.values():
                installed_model_names.append(model_info['original_name'].lower())
        
        for model in DOWNLOADABLE_MODELS:
            model_copy = model.copy()
            # Check if model is already installed
            is_installed = False
            for installed_name in installed_model_names:
                if model['id'] in installed_name.lower() or model['name'].lower() in installed_name.lower():
                    is_installed = True
                    break
            
            model_copy['status'] = 'downloaded' if is_installed else 'available'
            model_copy['is_downloading'] = model['id'] in self.downloading_models
            models_with_status.append(model_copy)
        
        return models_with_status
    
    def download_model(self, model_id):
        """Download a model"""
        # Find the model
        model = None
        for m in DOWNLOADABLE_MODELS:
            if m['id'] == model_id:
                model = m
                break
        
        if not model:
            return False, "Model not found"
        
        # Start download in background thread
        thread = threading.Thread(target=self._download_model_thread, args=(model,))
        thread.daemon = True
        thread.start()
        
        self.downloading_models[model_id] = True
        return True, "Download started"
    
    def _download_model_thread(self, model):
        """Download model in background thread"""
        try:
            import urllib.request
            import ssl
            import tempfile
            
            # Create temp file for download
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pth')
            temp_path = temp_file.name
            temp_file.close()
            
            # Download the file
            context = ssl._create_unverified_context()
            urllib.request.urlretrieve(
                model['url'],
                temp_path,
                reporthook=lambda count, block_size, total_size: self._download_progress(
                    model['id'], count * block_size, total_size
                )
            )
            
            # Determine destination directory based on category
            dest_dir = os.path.join(config.MODELS_FOLDER, model['category'])
            os.makedirs(dest_dir, exist_ok=True)
            
            # Save to models directory
            dest_path = os.path.join(dest_dir, os.path.basename(model['url']))
            shutil.move(temp_path, dest_path)
            
            # Emit download complete event
            socketio.emit('download_complete', {
                'model_id': model['id'],
                'model_name': model['name'],
                'path': dest_path
            })
            
        except Exception as e:
            print(f"Error downloading model {model['id']}: {e}")
            socketio.emit('download_error', {
                'model_id': model['id'],
                'error': str(e)
            })
        finally:
            # Remove from downloading list
            if model['id'] in self.downloading_models:
                del self.downloading_models[model['id']]
    
    def _download_progress(self, model_id, downloaded, total):
        """Report download progress"""
        if total > 0:
            progress = (downloaded / total) * 100
            socketio.emit('download_progress', {
                'model_id': model_id,
                'progress': progress,
                'downloaded': self._format_size(downloaded),
                'total': self._format_size(total)
            })
    
    def _format_size(self, size):
        """Format file size in human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:.1f}{unit}"
            size /= 1024.0
        return f"{size:.1f}TB"

# ============================================================================
# RVE BACKEND INTEGRATION CU GPU SUPPORT (ORIGINAL VERSION - CU FIX PENTRU GPU INDEXING)
# ============================================================================

class RVEBackendIntegration:
    """Integrates the REAL RVE backend with WebUI"""
    
    def __init__(self, socketio):
        self.socketio = socketio
        self.backend_available = RVE_BACKEND_AVAILABLE
        self.backend_info = {}
        
    def detect_backend(self):
        """Detect available backends and capabilities"""
        if not self.backend_available:
            return {
                'available': False,
                'error': f'RVE backend not found'
            }
        
        try:
            # Test simplu cu --version
            env = os.environ.copy()
            env['PYTHONPATH'] = f"{config.RVE_BACKEND_PATH}:{env.get('PYTHONPATH', '')}"
            
            result = subprocess.run(
                [config.RVE_PYTHON_PATH, RVE_BACKEND_FILE, '--version'],
                capture_output=True,
                text=True,
                cwd=config.RVE_BACKEND_PATH,
                env=env,
                timeout=3
            )
            
            if result.returncode == 0:
                version = result.stdout.strip()
                
                # Încercăm să obținem info despre GPU-uri
                gpu_info = []
                try:
                    # Run with --list_backends to get GPU info
                    result2 = subprocess.run(
                        [config.RVE_PYTHON_PATH, RVE_BACKEND_FILE, '--list_backends'],
                        capture_output=True,
                        text=True,
                        cwd=config.RVE_BACKEND_PATH,
                        env=env,
                        timeout=5
                    )
                    
                    if result2.returncode == 0:
                        output = result2.stdout
                        # Parse GPU info from output
                        for line in output.split('\n'):
                            if 'PyTorch GPU' in line or 'NCNN GPU' in line:
                                gpu_info.append(line.strip())
                except:
                    pass
                
                return {
                    'available': True,
                    'info': {
                        'version': version,
                        'backends': ['pytorch', 'ncnn', 'tensorrt'],
                        'gpus': gpu_info if gpu_info else AVAILABLE_GPUS,
                        'half_precision': True
                    }
                }
            else:
                print(f"Backend version check failed: {result.stderr}")
                return {
                    'available': False,
                    'error': 'Backend test failed'
                }
                
        except Exception as e:
            print(f"Backend detection error: {str(e)}")
            return {
                'available': False,
                'error': f'Backend error: {str(e)}'
            }
    
    def create_backend_arguments(self, job: ProcessingJob) -> list:
        """Create command line arguments for RVE backend with GPU support - VERSIUNEA COMPLET CORECTĂ"""
        args = [
            config.RVE_PYTHON_PATH,
            RVE_BACKEND_FILE,
            '-i', job.input_path,
            '-o', job.output_path,
            '--overwrite',
        ]
        
        settings = job.settings
        
        print(f"Job settings with GPU {settings.gpu_id}: {settings}")
        print(f"Backend: {settings.backend}")
        
        # IMPORTANT: Fix pentru GPU indexing - dacă GPU ID este 1, dar backend-ul are doar GPU 0
        # Trebuie să ajustăm pentru a evita IndexError
        effective_gpu_id = settings.gpu_id
        
        # Pentru fiecare backend, folosim argumente specifice
        if settings.backend == 'tensorrt':
            # Pentru TensorRT, trebuie să folosim argumentele corecte pentru fișiere .engine
            self._add_tensorrt_arguments(args, job, effective_gpu_id)
        elif settings.backend == 'ncnn':
            # Pentru NCNN, trebuie să găsim ambele fișiere .bin și .param
            self._add_ncnn_arguments(args, job, effective_gpu_id)
        else:  # pytorch
            self._add_pytorch_arguments(args, job, effective_gpu_id)
        
        # Argumente comune pentru toate backend-urile
        self._add_common_arguments(args, job)
        
        print(f"Backend arguments with GPU {effective_gpu_id}: {' '.join(args)}")
        
        return args
    
    def _add_tensorrt_arguments(self, args: list, job: ProcessingJob, gpu_id: int):
        """Add TensorRT specific arguments"""
        settings = job.settings
        
        # IMPORTANT: TensorRT nu funcționează în backend-ul RVE curent
        # Folosim PyTorch fallback dar menținem TensorRT pentru UI
        print(f"WARNING: TensorRT backend has compatibility issues in RVE")
        print(f"Using PyTorch fallback with GPU {gpu_id} for optimal performance")
        
        # Forțăm PyTorch dar menținem TensorRT în afișare pentru user
        args.extend(['--backend', 'pytorch'])
        args.extend(['--pytorch_gpu_id', str(gpu_id)])
        
        # Restul argumentelor pentru PyTorch
        self._add_pytorch_model_arguments(args, job)
    
    def _add_ncnn_arguments(self, args: list, job: ProcessingJob, gpu_id: int):
        """Add NCNN specific arguments"""
        settings = job.settings
        
        # Upscaling
        if settings.upscale_enabled and settings.upscale_model:
            model_path = self._find_ncnn_model_pair(settings.upscale_model, 'upscale')
            if model_path:
                args.extend(['--upscale_model', model_path])
                print(f"Found NCNN upscale model: {model_path}")
                
                if settings.upscale_factor > 1:
                    args.extend(['--override_upscale_scale', str(settings.upscale_factor)])
        
        # Interpolation
        if settings.interpolate_enabled and settings.interpolate_model:
            model_path = self._find_ncnn_model_pair(settings.interpolate_model, 'interpolate')
            if model_path:
                args.extend(['--interpolate_model', model_path])
                print(f"Found NCNN interpolate model: {model_path}")
                
                if settings.interpolate_factor > 1:
                    args.extend(['--interpolate_factor', str(settings.interpolate_factor)])
        
        # Denoising
        if settings.denoise_enabled and settings.denoise_model:
            model_path = self._find_ncnn_model_pair(settings.denoise_model, 'denoise')
            if model_path:
                args.extend(['--extra_restoration_models', model_path])
                print(f"Found NCNN denoise model: {model_path}")
        
        # Decompression
        if settings.decompress_enabled and settings.decompress_model:
            model_path = self._find_ncnn_model_pair(settings.decompress_model, 'decompress')
            if model_path:
                args.extend(['--extra_restoration_models', model_path])
                print(f"Found NCNN decompress model: {model_path}")
        
        # Backend și GPU settings pentru NCNN
        args.extend(['--backend', 'ncnn'])
        args.extend(['--ncnn_gpu_id', str(gpu_id)])
    
    def _add_pytorch_arguments(self, args: list, job: ProcessingJob, gpu_id: int):
        """Add PyTorch specific arguments"""
        settings = job.settings
        
        # CRITICAL FIX: Dacă GPU ID este 1 dar backend-ul RVE are doar GPU 0
        # Ajustăm la GPU 0 pentru a evita IndexError
        effective_gpu_id = gpu_id
        if gpu_id > 0:
            print(f"INFO: GPU ID {gpu_id} selected, but RVE backend might have GPU indexing issues")
            print(f"Using GPU 0 for compatibility. Performance will still use CUDA_VISIBLE_DEVICES")
            effective_gpu_id = 0
        
        # Upscaling
        if settings.upscale_enabled and settings.upscale_model:
            model_path = self._find_model_file_improved(
                settings.upscale_model, 'upscale', 'pytorch'
            )
            if model_path:
                args.extend(['--upscale_model', model_path])
                print(f"Found PyTorch upscale model: {model_path}")
                
                if settings.upscale_factor > 1:
                    args.extend(['--override_upscale_scale', str(settings.upscale_factor)])
            else:
                print(f"WARNING: PyTorch upscale model not found: {settings.upscale_model}")
        
        # Interpolation
        if settings.interpolate_enabled and settings.interpolate_model:
            model_path = self._find_model_file_improved(
                settings.interpolate_model, 'interpolate', 'pytorch'
            )
            if model_path:
                args.extend(['--interpolate_model', model_path])
                print(f"Found PyTorch interpolate model: {model_path}")
                
                if settings.interpolate_factor > 1:
                    args.extend(['--interpolate_factor', str(settings.interpolate_factor)])
        
        # Denoising - CORECT: folosește --extra_restoration_models pentru PyTorch
        if settings.denoise_enabled and settings.denoise_model:
            model_path = self._find_model_file_improved(
                settings.denoise_model, 'denoise', 'pytorch'
            )
            if model_path:
                args.extend(['--extra_restoration_models', model_path])
                print(f"Found PyTorch denoise model: {model_path}")
        
        # Decompression - CORECT: folosește --extra_restoration_models pentru PyTorch
        if settings.decompress_enabled and settings.decompress_model:
            model_path = self._find_model_file_improved(
                settings.decompress_model, 'decompress', 'pytorch'
            )
            if model_path:
                args.extend(['--extra_restoration_models', model_path])
                print(f"Found PyTorch decompress model: {model_path}")
        
        # Backend și GPU settings pentru PyTorch
        args.extend(['--backend', 'pytorch'])
        args.extend(['--pytorch_gpu_id', str(effective_gpu_id)])
    
    def _add_pytorch_model_arguments(self, args: list, job: ProcessingJob):
        """Add PyTorch model arguments (used by TensorRT fallback)"""
        settings = job.settings
        
        # Upscaling
        if settings.upscale_enabled and settings.upscale_model:
            model_path = self._find_model_file_improved(
                settings.upscale_model, 'upscale', 'pytorch'
            )
            if model_path:
                args.extend(['--upscale_model', model_path])
                print(f"Found PyTorch upscale model: {model_path}")
                
                if settings.upscale_factor > 1:
                    args.extend(['--override_upscale_scale', str(settings.upscale_factor)])
        
        # Interpolation
        if settings.interpolate_enabled and settings.interpolate_model:
            model_path = self._find_model_file_improved(
                settings.interpolate_model, 'interpolate', 'pytorch'
            )
            if model_path:
                args.extend(['--interpolate_model', model_path])
                print(f"Found PyTorch interpolate model: {model_path}")
                
                if settings.interpolate_factor > 1:
                    args.extend(['--interpolate_factor', str(settings.interpolate_factor)])
        
        # Denoising
        if settings.denoise_enabled and settings.denoise_model:
            model_path = self._find_model_file_improved(
                settings.denoise_model, 'denoise', 'pytorch'
            )
            if model_path:
                args.extend(['--extra_restoration_models', model_path])
                print(f"Found PyTorch denoise model: {model_path}")
        
        # Decompression
        if settings.decompress_enabled and settings.decompress_model:
            model_path = self._find_model_file_improved(
                settings.decompress_model, 'decompress', 'pytorch'
            )
            if model_path:
                args.extend(['--extra_restoration_models', model_path])
                print(f"Found PyTorch decompress model: {model_path}")
    
    def _add_common_arguments(self, args: list, job: ProcessingJob):
        """Add arguments common to all backends"""
        settings = job.settings
        
        # Precision
        if settings.precision != 'auto':
            args.extend(['--precision', settings.precision])
        
        # Tile size
        if settings.tiling_enabled:
            args.extend(['--tilesize', '512'])
        
        # Benchmark mode
        if settings.benchmark_mode:
            args.extend(['--benchmark'])
        
        # Ensemble mode
        if settings.ensemble_mode:
            args.extend(['--ensemble'])
        
        # HDR mode
        if settings.auto_hdr_mode:
            args.extend(['--hdr_mode'])
        
        # Output settings
        args.extend(['--crf', str(settings.crf)])
        
        # Audio
        if settings.audio_quality != 'copy':
            args.extend(['--audio_encoder_preset', 'aac'])
            args.extend(['--audio_bitrate', settings.audio_quality])
        else:
            args.extend(['--audio_encoder_preset', 'copy_audio'])
        
        # Scene detection
        if not settings.scene_detect_enabled:
            args.extend(['--scene_detect_method', 'none'])
    
    def _find_model_file_improved(self, model_name: str, category: str, backend: str) -> Optional[str]:
        """Improved model finding - handles complex directory structures"""
        models_dir = Path(config.MODELS_FOLDER)
        
        print(f"Searching for model '{model_name}' in category '{category}' for backend '{backend}'")
        
        # Define file extensions for each backend
        backend_extensions = {
            'tensorrt': ['.engine'],
            'ncnn': ['.bin', '.param'],
            'pytorch': ['.pth', '.pt', '.safetensors', '.pkl']
        }
        
        extensions = backend_extensions.get(backend, [])
        
        # Normalize search terms
        search_terms = model_name.lower().replace('_', ' ').replace('-', ' ').split()
        
        # Căutare recursivă în tot arborele de directoare
        for item in models_dir.rglob("*"):
            if item.is_file() and item.suffix.lower() in extensions:
                item_name_lower = item.stem.lower()
                
                # Verifică dacă modelul aparține categoriei corecte
                model_category = self._categorize_model(item_name_lower, item.suffix.lower())
                if model_category != category:
                    continue
                
                # Verifică potrivirea numelui
                match_score = 0
                for term in search_terms:
                    if term in item_name_lower:
                        match_score += 1
                
                # Dacă avem o potrivire bună, returnăm calea
                if match_score >= len(search_terms) * 0.7:  # 70% match
                    print(f"Found matching {backend} model: {item}")
                    return str(item)
        
        # Fallback: căutare mai laxa
        for item in models_dir.rglob("*"):
            if item.is_file() and item.suffix.lower() in extensions:
                item_name_lower = item.stem.lower()
                
                # Verifică dacă modelul aparține categoriei corecte
                model_category = self._categorize_model(item_name_lower, item.suffix.lower())
                if model_category != category:
                    continue
                
                # Căutare simplă: dacă numele modelului este conținut în numele fișierului
                if any(term in item_name_lower for term in search_terms):
                    print(f"Found similar {backend} model (fallback): {item}")
                    return str(item)
        
        print(f"Model '{model_name}' not found for backend '{backend}' in category '{category}'")
        return None
    
    def _find_ncnn_model_pair(self, model_name: str, category: str) -> Optional[str]:
        """Find NCNN model pair (.bin and .param files)"""
        models_dir = Path(config.MODELS_FOLDER)
        
        print(f"Searching for NCNN model pair '{model_name}' in category '{category}'")
        
        # Caută fișierul .bin
        bin_file = None
        for item in models_dir.rglob("*.bin"):
            item_name_lower = item.stem.lower()
            
            # Verifică dacă modelul aparține categoriei corecte
            model_category = self._categorize_model(item_name_lower, '.bin')
            if model_category != category:
                continue
            
            # Verifică potrivirea numelui
            if model_name.lower() in item_name_lower:
                bin_file = item
                break
        
        if not bin_file:
            print(f"NCNN .bin file not found for model '{model_name}'")
            return None
        
        # Caută fișierul .param corespunzător
        param_file = bin_file.with_suffix('.param')
        if not param_file.exists():
            # Încearcă să găsești orice fișier .param cu același nume de bază
            for item in models_dir.rglob("*.param"):
                if item.stem.lower() == bin_file.stem.lower():
                    param_file = item
                    break
        
        if not param_file.exists():
            print(f"NCNN .param file not found for model '{model_name}'")
            return None
        
        print(f"Found NCNN model pair: {bin_file} + {param_file}")
        # Pentru NCNN, backend-ul așteaptă calea către directorul care conține ambele fișiere
        return str(bin_file.parent)
    
    def _categorize_model(self, filename: str, extension: str) -> str:
        """Categorize model based on filename and extension"""
        filename_lower = filename.lower()
        
        if 'rife' in filename_lower or 'gmfss' in filename_lower:
            return 'interpolate'
        elif 'drunet' in filename_lower or 'dncnn' in filename_lower or 'scunet' in filename_lower or 'denoise' in filename_lower:
            return 'denoise'
        elif 'deh264' in filename_lower or 'decompress' in filename_lower or ('span' in filename_lower and 'deh264' in filename_lower):
            return 'decompress'
        elif 'nomos' in filename_lower or 'realesr' in filename_lower or 'upscale' in filename_lower or 'anime' in filename_lower or '2x' in filename_lower or '4x' in filename_lower or 'span' in filename_lower or 'conservative' in filename_lower:
            return 'upscale'
        
        # Default pentru orice altceva
        return 'upscale'
    
    def start_backend_process(self, job: ProcessingJob) -> subprocess.Popen:
        """Start RVE backend process for a job with GPU support"""
        args = self.create_backend_arguments(job)
        
        # Create log file
        log_file = os.path.join(config.TEMP_FOLDER, f"rve_job_{job.id}.log")
        job.log_file = log_file
        
        print(f"Starting RVE backend on GPU {job.settings.gpu_id}:")
        print(f"  Working dir: {config.RVE_BACKEND_PATH}")
        print(f"  Command: {' '.join(args)}")
        print(f"  Log file: {log_file}")
        
        # Setup environment with GPU support
        env = os.environ.copy()
        env['PYTHONPATH'] = f"{config.RVE_BACKEND_PATH}:{env.get('PYTHONPATH', '')}"
        
        # Set CUDA_VISIBLE_DEVICES for TensorRT și PyTorch
        # Folosim GPU-ul original selectat de user pentru CUDA_VISIBLE_DEVICES
        env['CUDA_VISIBLE_DEVICES'] = str(job.settings.gpu_id)
        print(f"Set CUDA_VISIBLE_DEVICES={job.settings.gpu_id}")
        
        # Start process
        try:
            with open(log_file, 'w') as log_f:
                log_f.write(f"Command: {' '.join(args)}\n")
                log_f.write(f"GPU ID: {job.settings.gpu_id}\n")
                log_f.write(f"Backend: {job.settings.backend}\n")
                log_f.write(f"Start time: {datetime.now()}\n")
                log_f.write("=" * 80 + "\n")
            
            process = subprocess.Popen(
                args,
                stdout=open(log_file, 'a'),
                stderr=subprocess.STDOUT,
                text=True,
                cwd=config.RVE_BACKEND_PATH,
                env=env
            )
            
            job.process_pid = process.pid
            print(f"Process started with PID: {process.pid} on GPU {job.settings.gpu_id}")
            return process
            
        except Exception as e:
            print(f"Error starting process on GPU {job.settings.gpu_id}: {e}")
            raise

# ============================================================================
# VIDEO PROCESSING ENGINE
# ============================================================================

class VideoProcessor:
    """Main video processing engine"""
    
    def __init__(self, socketio):
        self.socketio = socketio
        self.model_manager = ModelManager()
        self.rve_backend = RVEBackendIntegration(socketio)
        self.active_jobs: Dict[str, ProcessingJob] = {}
        self.job_queue = queue.Queue()
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.processing_thread.start()
    
    def create_job(self, input_path: str, settings_dict: dict, use_real_backend: bool = True) -> ProcessingJob:
        """Create a new processing job with unique output filename"""
        job_id = f"job_{int(time.time() * 1000)}"  # Folosim milisecunde pentru mai multă precizie
        original_filename = os.path.splitext(os.path.basename(input_path))[0]
        
        # Creează un nume unic pentru output
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_output_name = f"{original_filename}_enhanced_{timestamp}"
        
        # Folosește formatul specificat în setări sau MP4 implicit
        output_format = settings_dict.get('output_format', 'mp4')
        output_filename = f"{base_output_name}.{output_format}"
        output_path = os.path.join(config.OUTPUT_FOLDER, output_filename)
        
        # Asigură-te că fișierul output este unic
        counter = 1
        while os.path.exists(output_path):
            output_filename = f"{base_output_name}_{counter:03d}.{output_format}"
            output_path = os.path.join(config.OUTPUT_FOLDER, output_filename)
            counter += 1
        
        settings = ProcessingSettings.from_dict(settings_dict)
        
        # Force demo mode if backend not available
        if not self.rve_backend.backend_available:
            use_real_backend = False
            print("Backend not available, forcing demo mode")
        
        # Creează un nume de afișare pentru job
        display_name = f"{original_filename} (GPU {settings.gpu_id}, {settings.backend})"
        
        job = ProcessingJob(
            id=job_id,
            input_path=input_path,
            output_path=output_path,
            display_name=display_name,
            settings=settings,
            use_real_backend=use_real_backend
        )
        
        self.active_jobs[job_id] = job
        self.job_queue.put(job)
        
        print(f"Created job {job_id}")
        print(f"  Input: {original_filename}")
        print(f"  Output: {output_filename}")
        print(f"  GPU: {settings.gpu_id}")
        print(f"  Backend: {settings.backend}")
        print(f"  Real backend: {use_real_backend}")
        
        return job
    
    def _process_queue(self):
        """Process jobs from the queue"""
        while True:
            try:
                job = self.job_queue.get(timeout=1)
                self._process_job(job)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in processing queue: {e}")
                time.sleep(1)
    
    def _process_job(self, job: ProcessingJob):
        """Process a single job"""
        try:
            job.status = "processing"
            job.start_time = datetime.now()
            self._emit_job_update(job)
            
            print(f"Processing job {job.id} on GPU {job.settings.gpu_id}")
            
            if job.use_real_backend:
                self._process_with_rve_backend(job)
            else:
                self._process_simulated(job)
            
        except Exception as e:
            job.status = "failed"
            job.error_message = str(e)
            print(f"Job {job.id} failed: {e}")
            self._emit_job_update(job)
    
    def _process_with_rve_backend(self, job: ProcessingJob):
        """Process job using real RVE backend"""
        try:
            print(f"Starting RVE backend for job {job.id} on GPU {job.settings.gpu_id}")
            print(f"Backend: {job.settings.backend}")
            
            # Creează argumentele și le afișează pentru debugging
            args = self.rve_backend.create_backend_arguments(job)
            print("Final command arguments:")
            print(" ".join(args))
            
            process = self.rve_backend.start_backend_process(job)
            
            # Start a thread to monitor progress from log file
            progress_thread = threading.Thread(
                target=self._monitor_progress,
                args=(job, process),
                daemon=True
            )
            progress_thread.start()
            
            # Wait for process to complete
            print(f"Waiting for process {process.pid} to complete...")
            return_code = process.wait()
            
            print(f"Process completed with return code: {return_code}")
            
            # Check if output file was created
            output_exists = os.path.exists(job.output_path)
            print(f"Output file exists: {output_exists}")
            print(f"Output path: {job.output_path}")
            
            if output_exists:
                file_size = os.path.getsize(job.output_path)
                print(f"Output file size: {file_size} bytes")
            
            # Read log file for debugging
            if job.log_file and os.path.exists(job.log_file):
                with open(job.log_file, 'r') as f:
                    log_content = f.read()
                    print(f"Log file (last 500 chars):")
                    print(log_content[-500:] if len(log_content) > 500 else log_content)
            
            if return_code == 0 and output_exists:
                job.status = "completed"
                job.progress = 100.0
                print(f"Job {job.id} completed successfully on GPU {job.settings.gpu_id}")
                
                # Extract video preview frame
                self._create_video_preview(job)
            else:
                job.status = "failed"
                error_msg = f"Backend failed with exit code {return_code}"
                if not output_exists:
                    error_msg += " and output file was not created"
                job.error_message = error_msg
                
                # Get error details from log
                if job.log_file and os.path.exists(job.log_file):
                    try:
                        with open(job.log_file, 'r') as f:
                            lines = f.readlines()
                            last_lines = lines[-20:] if len(lines) > 20 else lines
                            error_details = "".join(last_lines)
                            job.error_message += f"\nLast log lines:\n{error_details}"
                    except:
                        pass
                
                print(f"Job {job.id} failed on GPU {job.settings.gpu_id}: {job.error_message}")
            
            job.end_time = datetime.now()
            
            # Force emit update
            self._emit_job_update(job)
            
            # Also print for debugging
            print(f"Job {job.id} final status: {job.status}")
            print(f"Job {job.id} output path: {job.output_path}")
            
        except Exception as e:
            job.status = "failed"
            job.error_message = f"Error in RVE backend: {str(e)}"
            job.end_time = datetime.now()
            print(f"Exception in RVE backend for job {job.id}: {e}")
            self._emit_job_update(job)

    def _monitor_progress(self, job: ProcessingJob, process):
        """Monitor progress by reading log file"""
        last_position = 0
        total_frames = 0
        
        # Try to get total frames from input video
        try:
            cap = cv2.VideoCapture(job.input_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            print(f"Total frames to process: {total_frames}")
        except:
            total_frames = 0
        
        while process.poll() is None:  # While process is running
            if job.log_file and os.path.exists(job.log_file):
                try:
                    with open(job.log_file, 'r') as f:
                        f.seek(last_position)
                        new_content = f.read()
                        last_position = f.tell()
                        
                        if new_content:
                            # Parse progress from log
                            lines = new_content.split('\n')
                            for line in lines:
                                # Look for progress indicators in log
                                if 'Current Frame:' in line:
                                    # Example: "Current Frame: 610 ETA: 0:00:02"
                                    frame_match = re.search(r'Current Frame:\s*(\d+)', line)
                                    if frame_match and total_frames > 0:
                                        current_frame = int(frame_match.group(1))
                                        progress = min(99.0, (current_frame / total_frames) * 100)
                                        if progress > job.progress:  # Only update if progress increased
                                            job.progress = progress
                                            self._emit_job_update(job)
                                            print(f"Progress update: {progress:.1f}% (frame {current_frame}/{total_frames})")
                
                except Exception as e:
                    print(f"Error reading progress log: {e}")
            
            time.sleep(1)  # Check every second
        
        # Final progress update
        job.progress = 99.9  # Almost done
        self._emit_job_update(job)
        
    def _create_video_preview(self, job: ProcessingJob):
        """Create a preview frame from processed video"""
        try:
            if not job.output_path or not os.path.exists(job.output_path):
                return
            
            # Create preview directory
            preview_dir = os.path.join(config.TEMP_FOLDER, "previews")
            os.makedirs(preview_dir, exist_ok=True)
            
            # Extract first frame
            preview_path = os.path.join(preview_dir, f"{job.id}_preview.jpg")
            
            cap = cv2.VideoCapture(job.output_path)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    # Resize for preview
                    max_size = (640, 360)
                    height, width = frame.shape[:2]
                    if width > max_size[0] or height > max_size[1]:
                        scale = min(max_size[0] / width, max_size[1] / height)
                        new_width = int(width * scale)
                        new_height = int(height * scale)
                        frame = cv2.resize(frame, (new_width, new_height))
                    
                    # Save preview
                    cv2.imwrite(preview_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    print(f"Created preview: {preview_path}")
                    
                    # Store preview path in job
                    job.preview_path = preview_path
                cap.release()
        except Exception as e:
            print(f"Error creating preview: {e}")
    
    def _process_simulated(self, job: ProcessingJob):
        """Simulate processing for demo purposes"""
        try:
            print(f"Simulating processing for job {job.id} (GPU {job.settings.gpu_id})")
            total_steps = 10
            
            for i in range(total_steps):
                time.sleep(1)
                job.progress = (i + 1) / total_steps * 100
                
                if i % 2 == 0:
                    self._emit_job_update(job)
                    print(f"  Progress: {job.progress:.1f}%")
            
            job.status = "completed"
            job.progress = 100.0
            job.end_time = datetime.now()
            
            # Create dummy output
            if os.path.exists(job.input_path):
                try:
                    shutil.copy(job.input_path, job.output_path)
                    print(f"Created demo output for job {job.id}")
                except Exception as e:
                    print(f"Error creating demo output: {e}")
            
            self._emit_job_update(job)
            print(f"Job {job.id} completed (simulated)")
            
        except Exception as e:
            job.status = "failed"
            job.error_message = str(e)
            job.end_time = datetime.now()
            print(f"Simulated job {job.id} failed: {e}")
            self._emit_job_update(job)
    
    def _emit_job_update(self, job: ProcessingJob):
        """Emit job update via SocketIO"""
        try:
            self.socketio.emit('job_update', job.to_dict())
        except Exception as e:
            print(f"Error emitting job update: {e}")
    
    def get_job(self, job_id: str) -> Optional[ProcessingJob]:
        """Get job by ID"""
        return self.active_jobs.get(job_id)
    
    def get_all_jobs(self) -> List[ProcessingJob]:
        """Get all jobs sorted by start_time (newest first)"""
        jobs = list(self.active_jobs.values())
        # Sortează după timpul de început (cel mai recent primul)
        jobs.sort(key=lambda x: x.start_time if x.start_time else datetime.min, reverse=True)
        return jobs
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job"""
        if job_id in self.active_jobs:
            job = self.active_jobs[job_id]
            job.status = "cancelled"
            self._emit_job_update(job)
            return True
        return False
    
    def delete_job(self, job_id: str) -> bool:
        """Delete a job"""
        if job_id in self.active_jobs:
            job = self.active_jobs[job_id]
            
            try:
                if os.path.exists(job.output_path):
                    os.remove(job.output_path)
                if job.log_file and os.path.exists(job.log_file):
                    os.remove(job.log_file)
                if job.preview_path and os.path.exists(job.preview_path):
                    os.remove(job.preview_path)
            except:
                pass
            
            del self.active_jobs[job_id]
            return True
        return False

# ============================================================================
# INITIALIZE COMPONENTS
# ============================================================================

video_processor = VideoProcessor(socketio)
model_manager = ModelManager()
rve_backend = RVEBackendIntegration(socketio)
model_downloader = ModelDownloader()

# ============================================================================
# FLASK ROUTES
# ============================================================================

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    """Upload a video file"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and '.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in config.ALLOWED_EXTENSIONS:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            video_info = VideoInfo.from_file(filepath)
            return jsonify({
                'success': True,
                'filename': filename,
                'video_info': asdict(video_info)
            })
        except Exception as e:
            print(f"Error analyzing video: {e}")
            return jsonify({'error': f'Failed to analyze video: {str(e)}'}), 500
    
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/backend/status')
def get_backend_status():
    """Get RVE backend status"""
    status = rve_backend.detect_backend()
    return jsonify(status)

@app.route('/models')
def get_models():
    """Get available models filtered by backend"""
    category = request.args.get('category')
    backend = request.args.get('backend', 'pytorch')  # Default to pytorch
    
    # Get all models
    all_models = model_manager.get_available_models(category, backend)
    
    # Filter models based on backend
    if backend:
        filtered_models = {}
        for cat, models in all_models.items():
            filtered_models[cat] = {}
            for model_name, model_info in models.items():
                # Check if model supports the selected backend
                model_backends = model_info.get('backend', [])
                supports_backend = False
                
                if backend == 'tensorrt':
                    supports_backend = model_info.get('is_engine', False) or any('tensorrt' in b.lower() for b in model_backends)
                elif backend == 'ncnn':
                    supports_backend = any('ncnn' in b.lower() for b in model_backends) or model_info['filename'].endswith(('.bin', '.param'))
                elif backend == 'pytorch':
                    supports_backend = any('pytorch' in b.lower() for b in model_backends) or model_info['filename'].endswith(('.pth', '.pt', '.safetensors', '.pkl'))
                
                if supports_backend:
                    filtered_models[cat][model_name] = model_info
        
        return jsonify(filtered_models)
    
    return jsonify(all_models)

@app.route('/downloadable_models')
def get_downloadable_models():
    """Get list of downloadable models"""
    models = model_downloader.get_downloadable_models()
    return jsonify(models)

@app.route('/download_model/<model_id>', methods=['POST'])
def download_model(model_id):
    """Download a model"""
    success, message = model_downloader.download_model(model_id)
    if success:
        return jsonify({'success': True, 'message': message})
    else:
        return jsonify({'success': False, 'error': message}), 400

@app.route('/delete_model/<model_id>', methods=['DELETE'])
def delete_model(model_id):
    """Delete a downloaded model"""
    # Find model file
    model = None
    for m in DOWNLOADABLE_MODELS:
        if m['id'] == model_id:
            model = m
            break
    
    if not model:
        return jsonify({'success': False, 'error': 'Model not found'}), 404
    
    # Try to find and delete the model file
    models_dir = Path(config.MODELS_FOLDER)
    model_filename = os.path.basename(model['url'])
    
    # Search in category folder
    category_dir = models_dir / model['category']
    if category_dir.exists():
        for item in category_dir.iterdir():
            if item.is_file() and model_filename in item.name:
                try:
                    os.remove(str(item))
                    return jsonify({'success': True, 'message': 'Model deleted'})
                except Exception as e:
                    return jsonify({'success': False, 'error': str(e)}), 500
    
    # Search in root
    for item in models_dir.iterdir():
        if item.is_file() and model_filename in item.name:
            try:
                os.remove(str(item))
                return jsonify({'success': True, 'message': 'Model deleted'})
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500
    
    return jsonify({'success': False, 'error': 'Model file not found'}), 404

@app.route('/import_model', methods=['POST'])
def import_model():
    """Import a custom model"""
    try:
        model_format = request.form.get('format')
        
        if model_format == 'pytorch':
            if 'file' not in request.files:
                return jsonify({'success': False, 'error': 'No file provided'}), 400
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({'success': False, 'error': 'No selected file'}), 400
            
            # Save to custom models directory
            filename = secure_filename(file.filename)
            filepath = os.path.join(config.CUSTOM_MODELS_PATH, filename)
            file.save(filepath)
            
            return jsonify({'success': True, 'message': 'Model imported successfully'})
        
        elif model_format == 'ncnn':
            if 'bin_file' not in request.files or 'param_file' not in request.files:
                return jsonify({'success': False, 'error': 'Both .bin and .param files are required'}), 400
            
            bin_file = request.files['bin_file']
            param_file = request.files['param_file']
            
            if bin_file.filename == '' or param_file.filename == '':
                return jsonify({'success': False, 'error': 'No selected files'}), 400
            
            # Create folder for the model
            model_name = os.path.splitext(bin_file.filename)[0]
            model_dir = os.path.join(config.CUSTOM_MODELS_PATH, model_name)
            os.makedirs(model_dir, exist_ok=True)
            
            # Save files
            bin_path = os.path.join(model_dir, secure_filename(bin_file.filename))
            param_path = os.path.join(model_dir, secure_filename(param_file.filename))
            
            bin_file.save(bin_path)
            param_file.save(param_path)
            
            return jsonify({'success': True, 'message': 'Model imported successfully'})
        
        else:
            return jsonify({'success': False, 'error': 'Invalid model format'}), 400
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/process', methods=['POST'])
def process_video():
    """Start video processing"""
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON data provided'}), 400
    
    filename = data.get('filename')
    settings = data.get('settings', {})
    use_real_backend = data.get('use_real_backend', True)
    
    if not filename:
        return jsonify({'error': 'Filename required'}), 400
    
    input_path = os.path.join(config.UPLOAD_FOLDER, filename)
    if not os.path.exists(input_path):
        return jsonify({'error': 'Video file not found'}), 404
    
    try:
        job = video_processor.create_job(input_path, settings, use_real_backend)
        
        return jsonify({
            'success': True,
            'job_id': job.id,
            'message': 'Processing started'
        })
    except Exception as e:
        return jsonify({'error': f'Failed to create job: {str(e)}'}), 500

@app.route('/jobs')
def get_jobs():
    """Get all processing jobs"""
    jobs = video_processor.get_all_jobs()
    return jsonify([job.to_dict() for job in jobs])

@app.route('/jobs/<job_id>/delete', methods=['DELETE'])
def delete_job(job_id):
    """Delete a job"""
    success = video_processor.delete_job(job_id)
    if success:
        return jsonify({'success': True, 'message': 'Job deleted'})
    else:
        return jsonify({'error': 'Job not found'}), 404
        
@app.route('/preview/<job_id>')
def get_preview(job_id):
    """Get preview image for a job"""
    job = video_processor.get_job(job_id)
    if not job or not job.preview_path or not os.path.exists(job.preview_path):
        return jsonify({'error': 'Preview not available'}), 404
    
    return send_file(job.preview_path, mimetype='image/jpeg')

@app.route('/output/<filename>')
def serve_output_video(filename):
    """Serve processed video files"""
    filepath = os.path.join(config.OUTPUT_FOLDER, filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'Video not found'}), 404
    
    # Verifică dacă browser-ul acceptă range requests pentru streaming
    range_header = request.headers.get('Range', None)
    file_size = os.path.getsize(filepath)
    
    if range_header:
        # Parse the range header
        byte1, byte2 = 0, None
        match = re.search(r'(\d+)-(\d*)', range_header)
        groups = match.groups()
        
        if groups[0]:
            byte1 = int(groups[0])
        if groups[1]:
            byte2 = int(groups[1])
        
        if byte2 is None:
            byte2 = file_size - 1
        
        length = byte2 - byte1 + 1
        
        # Create response with partial content
        with open(filepath, 'rb') as f:
            f.seek(byte1)
            data = f.read(length)
        
        rv = Response(data, 
                    206,  # Partial Content
                    mimetype='video/mp4',
                    direct_passthrough=True)
        rv.headers.add('Content-Range', f'bytes {byte1}-{byte2}/{file_size}')
        rv.headers.add('Accept-Ranges', 'bytes')
        rv.headers.add('Content-Length', str(length))
        return rv
    else:
        # Full file response
        return send_file(
            filepath,
            mimetype='video/mp4',
            as_attachment=False,
            conditional=True  # Enable conditional requests
        )

@app.route('/uploads/<filename>')
def serve_uploaded_video(filename):
    """Serve original uploaded video files"""
    filepath = os.path.join(config.UPLOAD_FOLDER, filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'Original video not found'}), 404
    
    range_header = request.headers.get('Range', None)
    file_size = os.path.getsize(filepath)
    
    if range_header:
        # Parse the range header
        byte1, byte2 = 0, None
        match = re.search(r'(\d+)-(\d*)', range_header)
        groups = match.groups()
        
        if groups[0]:
            byte1 = int(groups[0])
        if groups[1]:
            byte2 = int(groups[1])
        
        if byte2 is None:
            byte2 = file_size - 1
        
        length = byte2 - byte1 + 1
        
        # Create response with partial content
        with open(filepath, 'rb') as f:
            f.seek(byte1)
            data = f.read(length)
        
        rv = Response(data, 
                    206,
                    mimetype='video/mp4',
                    direct_passthrough=True)
        rv.headers.add('Content-Range', f'bytes {byte1}-{byte2}/{file_size}')
        rv.headers.add('Accept-Ranges', 'bytes')
        rv.headers.add('Content-Length', str(length))
        return rv
    else:
        # Full file response
        return send_file(
            filepath,
            mimetype='video/mp4',
            as_attachment=False,
            conditional=True
        )

@app.route('/download/<job_id>')
def download_result(job_id):
    """Download processed video"""
    job = video_processor.get_job(job_id)
    
    if not job:
        print(f"Download error: Job {job_id} not found in active_jobs")
        return jsonify({'error': 'Job not found'}), 404
    
    print(f"Download request for job {job_id}:")
    print(f"  Status: {job.status}")
    print(f"  Output path: {job.output_path}")
    print(f"  File exists: {os.path.exists(job.output_path) if job.output_path else 'No output path'}")
    
    if job.status != 'completed':
        print(f"  Error: Job status is {job.status}, not 'completed'")
        return jsonify({'error': f'Result not available (status: {job.status})'}), 404
    
    if not job.output_path or not os.path.exists(job.output_path):
        print(f"  Error: Output file not found at {job.output_path}")
        
        # Try to find the file with different extensions
        output_dir = os.path.dirname(job.output_path) if job.output_path else config.OUTPUT_FOLDER
        if os.path.exists(output_dir):
            files = os.listdir(output_dir)
            print(f"  Files in output directory: {files}")
            
            # Look for files starting with enhanced_ or containing job_id
            for file in files:
                if job_id in file or (job.input_path and os.path.basename(job.input_path).split('.')[0] in file):
                    found_path = os.path.join(output_dir, file)
                    print(f"  Found possible output: {found_path}")
                    if os.path.exists(found_path):
                        return send_file(found_path, as_attachment=True, download_name=f"enhanced_{job_id}.mp4")
        
        return jsonify({'error': 'Output file not found'}), 404
    
    try:
        # Get filename for download
        download_name = os.path.basename(job.output_path)
        
        print(f"  Downloading: {download_name}")
        return send_file(
            job.output_path, 
            as_attachment=True,
            download_name=download_name,
            mimetype='video/mp4'
        )
    except Exception as e:
        print(f"  Download error: {e}")
        return jsonify({'error': f'Download failed: {str(e)}'}), 500

# ============================================================================
# SOCKETIO EVENTS
# ============================================================================

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('Client connected')
    time.sleep(0.5)
    status = rve_backend.detect_backend()
    emit('backend_status', status)

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point for the application"""
    print("=" * 70)
    print("PRO AI Video UPSCALER - WebUI with GPU Selection and Video Comparison")
    print("=" * 70)
    print(f"Base directory: {config.BASE_DIR}")
    print(f"Models folder: {config.MODELS_FOLDER}")
    print(f"Backend path: {config.RVE_BACKEND_PATH}")
    print(f"Python path: {config.RVE_PYTHON_PATH}")
    
    # Display GPU information
    print(f"\nAvailable GPUs:")
    for i, gpu in enumerate(AVAILABLE_GPUS):
        print(f"  GPU {gpu['id']}: {gpu['name']} ({gpu['type']})")
    
    # Check backend availability
    if RVE_BACKEND_AVAILABLE:
        print("\nTesting backend connection...")
        
        try:
            env = os.environ.copy()
            env['PYTHONPATH'] = f"{config.RVE_BACKEND_PATH}:{env.get('PYTHONPATH', '')}"
            
            result = subprocess.run(
                [config.RVE_PYTHON_PATH, RVE_BACKEND_FILE, '--version'],
                capture_output=True,
                text=True,
                cwd=config.RVE_BACKEND_PATH,
                env=env,
                timeout=3
            )
            
            if result.returncode == 0:
                version = result.stdout.strip()
                print(f"  ✓ Backend version: {version}")
            else:
                print(f"  ✗ Backend version check failed")
                
        except Exception as e:
            print(f"  ✗ Backend test failed: {e}")
    else:
        print("\n✗ RVE backend file not found")
        print("  Running in demo mode only")
    
    print("\n✨ FEATURES:")
    print("  ✓ Model manager with organized categories")
    print("  ✓ GPU selection and backend configuration")
    print("  ✓ Video Comparison Mode with split-screen")
    print("  ✓ Fullscreen comparison mode (standard browser API)")
    print("  ✓ Downloadable models from internet")
    print("  ✓ Custom model import (PyTorch/NCNN)")
    print("  ✓ Denoise and Decompress models support (correct arguments)")
    print("  ✓ Models filtered by backend (PyTorch/NCNN/TensorRT)")
    print("  ✓ All original backend arguments preserved")
    
    print("\n🔧 IMPORTANT FIX APPLIED:")
    print("  ✓ GPU Indexing: Fixed IndexError for GPU ID 1")
    print("  ✓ TensorRT: Uses PyTorch fallback with GPU acceleration")
    print("  ✓ PyTorch: Works perfectly with GPU 0")
    print("  ✓ NCNN: Works perfectly")
    
    print("\n⚠️  GPU USAGE NOTE:")
    print("  For PyTorch/TensorRT: Use GPU 0 for best compatibility")
    print("  CUDA_VISIBLE_DEVICES is set to selected GPU for actual GPU usage")
    print("  RVE backend may show GPU 0 internally but uses correct GPU via CUDA_VISIBLE_DEVICES")
    
    print("\n📁 Model organization:")
    print(f"  {config.MODELS_FOLDER}")
    print("  ├── upscale/      # Upscaling models (.pth, .pt, .safetensors, .engine, .bin/.param)")
    print("  ├── interpolate/  # Frame interpolation models (.pkl, .bin/.param, .engine)")
    print("  ├── denoise/      # Denoising models (.pth, .engine)")
    print("  └── decompress/   # Compression artifact removal models (.pth, .safetensors, .bin/.param, .engine)")
    
    print("=" * 70)
    print(f"Server running at: http://{config.HOST}:{config.PORT}")
    print("=" * 70)
    
    socketio.run(
        app, 
        host=config.HOST, 
        port=config.PORT, 
        debug=config.DEBUG,
        allow_unsafe_werkzeug=True
    )

if __name__ == '__main__':
    main()
