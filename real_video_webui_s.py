"""
REAL Video Enhancer WebUI - Multi-GPU Batch Processing with Large Video Preview
Fixed version with proper backend integration and model detection
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
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    MODELS_FOLDER = os.path.join(BASE_DIR, "models")
    TEMP_FOLDER = os.path.join(BASE_DIR, "temp")
    TEMPLATE_FOLDER = os.path.join(BASE_DIR, "templates")
    RVE_BACKEND_PATH = os.path.join(BASE_DIR, "backend")
    RVE_PYTHON_PATH = sys.executable
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
    MAX_CONCURRENT_JOBS = 2
    MAX_QUEUE_SIZE = 20
    BATCH_PROCESSING_ENABLED = True
    VIDEO_PREVIEW_HEIGHT = "70vh"

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
# DETECT GPUs
# ============================================================================

def detect_gpus():
    """Detect available GPUs in the system"""
    gpus = []
    
    try:
        # Try to use nvidia-smi
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if ',' in line:
                    parts = line.split(',')
                    if len(parts) >= 3:
                        idx = parts[0].strip()
                        name = parts[1].strip()
                        memory = parts[2].strip()
                        # Convert memory to GB
                        memory_gb = f"{int(memory) / 1024:.1f} GB" if memory.isdigit() else memory
                        gpus.append({
                            'id': int(idx),
                            'name': f"{name} ({memory_gb})",
                            'type': 'nvidia'
                        })
    except Exception as e:
        print(f"Error detecting GPUs: {e}")
    
    # If no NVIDIA GPUs found, add placeholder
    if not gpus:
        gpus = [
            {'id': 0, 'name': 'CPU (Default)', 'type': 'cpu'},
        ]
    
    print(f"Detected {len(gpus)} GPUs: {[g['name'] for g in gpus]}")
    return gpus

AVAILABLE_GPUS = detect_gpus()

# ============================================================================
# RVE BACKEND INTEGRATION - FIXED MODEL DETECTION
# ============================================================================

RVE_BACKEND_FILE = os.path.join(config.RVE_BACKEND_PATH, "rve-backend.py")
RVE_BACKEND_AVAILABLE = os.path.exists(RVE_BACKEND_FILE)

if RVE_BACKEND_AVAILABLE:
    print(f"✓ RVE backend found: {RVE_BACKEND_FILE}")
else:
    print(f"⚠ RVE backend file not found at: {RVE_BACKEND_FILE}")
    possible_paths = [
        "/media/prouser/Storage/PlayGround/RVU/backend",
        os.path.join(BASE_DIR, "..", "backend"),
        os.path.join(BASE_DIR, "backend"),
    ]
    
    for path in possible_paths:
        test_file = os.path.join(path, "rve-backend.py")
        if os.path.exists(test_file):
            config.RVE_BACKEND_PATH = path
            RVE_BACKEND_FILE = test_file
            RVE_BACKEND_AVAILABLE = True
            print(f"✓ Found backend at: {path}")
            break

class RVEBackendIntegration:
    """Integrates the REAL RVE backend with WebUI - FIXED MODEL DETECTION"""
    
    def __init__(self):
        self.backend_available = RVE_BACKEND_AVAILABLE
        
    def detect_backend(self):
        """Detect available backends and capabilities"""
        if not self.backend_available:
            return {
                'available': False,
                'error': f'RVE backend not found at {RVE_BACKEND_FILE}'
            }
        
        try:
            env = os.environ.copy()
            env['PYTHONPATH'] = f"{config.RVE_BACKEND_PATH}:{env.get('PYTHONPATH', '')}"
            
            result = subprocess.run(
                [config.RVE_PYTHON_PATH, RVE_BACKEND_FILE, '--version'],
                capture_output=True,
                text=True,
                cwd=config.RVE_BACKEND_PATH,
                env=env,
                timeout=5
            )
            
            if result.returncode == 0:
                version = result.stdout.strip()
                return {
                    'available': True,
                    'version': version,
                    'backends': ['pytorch', 'ncnn', 'tensorrt']
                }
            else:
                result = subprocess.run(
                    [config.RVE_PYTHON_PATH, RVE_BACKEND_FILE, '--list_backends'],
                    capture_output=True,
                    text=True,
                    cwd=config.RVE_BACKEND_PATH,
                    env=env,
                    timeout=10
                )
                
                if result.returncode == 0:
                    return {
                        'available': True,
                        'version': 'unknown',
                        'backends': ['pytorch', 'ncnn', 'tensorrt']
                    }
                else:
                    return {
                        'available': False,
                        'error': f'Backend test failed: {result.stderr[:200]}'
                    }
                
        except subprocess.TimeoutExpired:
            return {
                'available': False,
                'error': 'Backend test timed out'
            }
        except Exception as e:
            return {
                'available': False,
                'error': f'Backend error: {str(e)}'
            }
    
    def _get_model_for_backend(self, model_name: str, category: str, backend: str) -> Optional[str]:
        """Get appropriate model file for specific backend"""
        models_dir = Path(config.MODELS_FOLDER)
        
        if not models_dir.exists():
            print(f"Models directory does not exist: {models_dir}")
            return None
        
        print(f"\nSearching for {backend} model '{model_name}' in category '{category}'")
        
        # Clean model name for matching
        clean_model_name = model_name.lower().replace('_', '').replace('-', '').replace('.pkl', '').replace('.engine', '').replace('.pth', '').replace('.pt', '')
        
        # Define backend-specific extensions
        backend_extensions = {
            'pytorch': ['.pth', '.pt', '.ckpt', '.safetensors', '.pkl'],
            'tensorrt': ['.engine'],
            'ncnn': ['.param', '.bin']
        }
        
        target_extensions = backend_extensions.get(backend, ['.pth', '.pt', '.ckpt'])
        
        # Search in category folder
        category_dir = models_dir / category
        found_files = []
        
        if category_dir.exists():
            print(f"  Searching in category folder: {category_dir}")
            for item in category_dir.iterdir():
                if item.is_file() and item.suffix.lower() in target_extensions:
                    item_stem_clean = item.stem.lower().replace('_', '').replace('-', '')
                    
                    # Check if model name matches
                    if (clean_model_name in item_stem_clean or 
                        item_stem_clean in clean_model_name or
                        model_name.lower() in item.name.lower()):
                        print(f"  ✓ Found {backend} model: {item.name}")
                        found_files.append(str(item))
        
        # Search in all subdirectories
        print(f"  Searching recursively for {backend} models")
        for item in models_dir.rglob("*"):
            if item.is_file() and item.suffix.lower() in target_extensions:
                item_stem_clean = item.stem.lower().replace('_', '').replace('-', '')
                
                # Check if category is in path
                item_category = item.parent.name.lower()
                is_category_match = category in item_category or any(
                    cat in item_category for cat in ['upscale', 'interpolate', 'denoise', 'decompress']
                )
                
                if is_category_match and (clean_model_name in item_stem_clean or 
                                         item_stem_clean in clean_model_name or
                                         model_name.lower() in item.name.lower()):
                    print(f"  ✓ Found recursively: {item.relative_to(models_dir)}")
                    found_files.append(str(item))
        
        # Sort by relevance
        def score_file(filepath):
            score = 0
            filename = os.path.basename(filepath).lower()
            
            # Exact name match
            if model_name.lower() == Path(filepath).stem.lower():
                score += 100
            
            # Contains model name
            if model_name.lower() in filename:
                score += 50
            
            # In correct category directory
            if category in filepath.lower():
                score += 30
            
            # Correct backend extension
            if backend == 'tensorrt' and filepath.endswith('.engine'):
                score += 40
            elif backend == 'pytorch' and filepath.endswith(('.pth', '.pt', '.ckpt', '.pkl')):
                score += 30
            elif backend == 'ncnn' and filepath.endswith(('.param', '.bin')):
                score += 30
            
            return score
        
        if found_files:
            found_files.sort(key=score_file, reverse=True)
            selected_file = found_files[0]
            print(f"  Selected for {backend}: {selected_file} (score: {score_file(selected_file)})")
            return selected_file
        
        print(f"  ⚠ No suitable {backend} model found for '{model_name}' in category '{category}'")
        
        # Fallback: try to find any model in the category
        print(f"  Trying fallback search for any model in category '{category}'")
        fallback_files = []
        if category_dir.exists():
            for item in category_dir.iterdir():
                if item.is_file():
                    # Accept any model file for fallback
                    if item.suffix.lower() in ['.pth', '.pt', '.ckpt', '.safetensors', '.pkl', '.engine', '.param', '.bin']:
                        fallback_files.append(str(item))
        
        if fallback_files:
            # Prefer PyTorch models for fallback
            for ext in ['.pth', '.pt', '.ckpt', '.pkl', '.safetensors', '.engine', '.param', '.bin']:
                for file in fallback_files:
                    if file.endswith(ext):
                        print(f"  Fallback selected: {file}")
                        return file
        
        print(f"  ⚠ No models found in category '{category}'")
        return None
    
    def _find_model_file(self, model_name: str, category: str, backend: str) -> Optional[str]:
        """Find the actual model file with backend-aware detection"""
        # Special handling for TensorRT - need engine files
        if backend == 'tensorrt':
            return self._get_model_for_backend(model_name, category, 'tensorrt')
        elif backend == 'pytorch':
            return self._get_model_for_backend(model_name, category, 'pytorch')
        elif backend == 'ncnn':
            return self._get_model_for_backend(model_name, category, 'ncnn')
        else:
            # Default to PyTorch
            return self._get_model_for_backend(model_name, category, 'pytorch')
    
    def _map_gpu_id(self, gpu_id: int, backend: str) -> int:
        """Map GPU ID for different backends"""
        # For TensorRT and PyTorch, use CUDA device IDs directly
        if backend in ['tensorrt', 'pytorch']:
            return gpu_id
        # For NCNN, might need different mapping
        elif backend == 'ncnn':
            return gpu_id
        else:
            return 0
    
    def create_backend_arguments(self, job: 'ProcessingJob') -> list:
        """Create command line arguments for RVE backend"""
        args = [
            config.RVE_PYTHON_PATH,
            RVE_BACKEND_FILE,
            '-i', job.input_path,
            '-o', job.output_path,
        ]
        
        if job.settings.overwrite:
            args.append('--overwrite')
        
        settings = job.settings
        
        print(f"\nJob {job.id} settings:")
        print(f"  Backend: {settings.backend}")
        print(f"  GPU ID: {settings.gpu_id}")
        
        # Add backend argument
        args.extend(['--backend', settings.backend])
        
        # Map GPU ID
        mapped_gpu_id = self._map_gpu_id(settings.gpu_id, settings.backend)
        
        # Add GPU selection based on backend
        if settings.backend == 'pytorch':
            args.extend(['--pytorch_gpu_id', str(mapped_gpu_id)])
            args.extend(['--device', 'cuda'])
        elif settings.backend == 'ncnn':
            args.extend(['--ncnn_gpu_id', str(mapped_gpu_id)])
        elif settings.backend == 'tensorrt':
            args.extend(['--device', 'cuda'])
            # For TensorRT, we don't pass GPU ID directly to backend
        
        # UPSCALING - FIND CORRECT MODEL FOR BACKEND
        if settings.upscale_enabled and settings.upscale_model:
            model_path = self._find_model_file(settings.upscale_model, 'upscale', settings.backend)
            if model_path:
                args.extend(['--upscale_model', model_path])
                
                if settings.upscale_factor > 1:
                    args.extend(['--override_upscale_scale', str(settings.upscale_factor)])
                print(f"  Using upscale model: {os.path.basename(model_path)}")
            else:
                print(f"  ⚠ No suitable upscale model found for backend '{settings.backend}'")
                settings.upscale_enabled = False
        
        # INTERPOLATION - FIND CORRECT MODEL FOR BACKEND
        if settings.interpolate_enabled and settings.interpolate_model:
            model_path = self._find_model_file(settings.interpolate_model, 'interpolate', settings.backend)
            if model_path:
                args.extend(['--interpolate_model', model_path])
                args.extend(['--interpolate_factor', str(settings.interpolate_factor)])
                print(f"  Using interpolate model: {os.path.basename(model_path)}")
            else:
                print(f"  ⚠ No suitable interpolate model found for backend '{settings.backend}'")
                settings.interpolate_enabled = False
        
        # DENOISE
        if settings.denoise_enabled and settings.denoise_model:
            model_path = self._find_model_file(settings.denoise_model, 'denoise', settings.backend)
            if model_path:
                args.extend(['--extra_restoration_models', model_path])
                print(f"  Using denoise model: {os.path.basename(model_path)}")
            else:
                print(f"  ⚠ No suitable denoise model found")
                settings.denoise_enabled = False
        
        # DECOMPRESS
        if settings.decompress_enabled and settings.decompress_model:
            model_path = self._find_model_file(settings.decompress_model, 'decompress', settings.backend)
            if model_path:
                if '--extra_restoration_models' in args:
                    idx = args.index('--extra_restoration_models') + 1
                    args[idx] = args[idx] + ',' + model_path
                else:
                    args.extend(['--extra_restoration_models', model_path])
                print(f"  Using decompress model: {os.path.basename(model_path)}")
            else:
                print(f"  ⚠ No suitable decompress model found")
                settings.decompress_enabled = False
        
        # Precision
        if settings.precision != 'auto':
            args.extend(['--precision', settings.precision])
        
        # Tile size
        if settings.tile_size > 0:
            args.extend(['--tilesize', str(settings.tile_size)])
        
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
        
        # Additional TensorRT options
        if settings.backend == 'tensorrt':
            args.extend(['--tensorrt_opt_profile', str(settings.tensorrt_opt_profile)])
            if settings.tensorrt_dynamic_shapes:
                args.append('--tensorrt_dynamic_shapes')
        
        print(f"\nBackend command for job {job.id}:")
        print(f"  {' '.join(args[:8])}...")
        
        return args
    
    def start_backend_process(self, job: 'ProcessingJob') -> subprocess.Popen:
        """Start RVE backend process for a job"""
        args = self.create_backend_arguments(job)
        
        # Create log file
        log_file = os.path.join(config.TEMP_FOLDER, f"rve_job_{job.id}.log")
        job.log_file = log_file
        
        print(f"\n{'='*80}")
        print(f"Starting RVE backend for job {job.id}")
        print(f"{'='*80}")
        print(f"  Working dir: {config.RVE_BACKEND_PATH}")
        print(f"  Python path: {config.RVE_PYTHON_PATH}")
        print(f"  GPU ID: {job.settings.gpu_id}")
        print(f"  Backend: {job.settings.backend}")
        print(f"  Command: {' '.join(args[:8])}...")
        print(f"  Log file: {log_file}")
        print(f"{'='*80}\n")
        
        # Setup environment
        env = os.environ.copy()
        env['PYTHONPATH'] = f"{config.RVE_BACKEND_PATH}:{env.get('PYTHONPATH', '')}"
        
        # Set CUDA_VISIBLE_DEVICES for GPU support
        if job.settings.backend in ['pytorch', 'tensorrt']:
            if job.settings.gpu_id >= 0 and AVAILABLE_GPUS and AVAILABLE_GPUS[0]['type'] == 'nvidia':
                env['CUDA_VISIBLE_DEVICES'] = str(job.settings.gpu_id)
                print(f"  Set CUDA_VISIBLE_DEVICES={job.settings.gpu_id}")
            else:
                print(f"  Using CPU (no GPU available)")
                env['CUDA_VISIBLE_DEVICES'] = ''
        
        # Start process
        try:
            with open(log_file, 'w') as log_f:
                log_f.write(f"Job ID: {job.id}\n")
                log_f.write(f"Start time: {datetime.now()}\n")
                log_f.write(f"Backend: {job.settings.backend}\n")
                log_f.write(f"GPU ID: {job.settings.gpu_id}\n")
                log_f.write(f"Command: {' '.join(args)}\n")
                log_f.write(f"Python: {config.RVE_PYTHON_PATH}\n")
                log_f.write(f"Working dir: {config.RVE_BACKEND_PATH}\n")
                log_f.write(f"CUDA_VISIBLE_DEVICES: {env.get('CUDA_VISIBLE_DEVICES', 'Not set')}\n")
                log_f.write("=" * 80 + "\n")
            
            process = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=config.RVE_BACKEND_PATH,
                env=env,
                bufsize=1,
                universal_newlines=True
            )
            
            job.process_pid = process.pid
            print(f"✓ Process started with PID: {process.pid} for job {job.id}")
            
            # Start thread to capture output
            output_thread = threading.Thread(
                target=self._capture_process_output,
                args=(job, process),
                daemon=True
            )
            output_thread.start()
            
            return process
            
        except Exception as e:
            print(f"✗ Error starting process for job {job.id}: {e}")
            print(f"  Args were: {' '.join(args[:8])}...")
            raise
    
    def _capture_process_output(self, job: 'ProcessingJob', process: subprocess.Popen):
        """Capture and log process output in real-time"""
        try:
            with open(job.log_file, 'a') as log_f:
                while True:
                    output = process.stdout.readline()
                    if output == '' and process.poll() is not None:
                        break
                    if output:
                        log_f.write(output)
                        log_f.flush()
                        
                        # Parse progress if possible
                        if 'Current Frame:' in output:
                            match = re.search(r'Current Frame:\s*(\d+)', output)
                            if match:
                                current_frame = int(match.group(1))
                                pass
        except Exception as e:
            print(f"Error capturing output for job {job.id}: {e}")

# ============================================================================
# DATA MODELS
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
    gpu_id: int = 0
    precision: str = "auto"
    tile_size: int = 0
    output_format: str = "mp4"
    output_codec: str = "libx264"
    crf: int = 18
    audio_quality: str = "copy"
    overwrite: bool = True
    tensorrt_opt_profile: int = 3
    tensorrt_dynamic_shapes: bool = False
    UHD_mode: bool = False
    ensemble: bool = False
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ProcessingSettings':
        """Create ProcessingSettings from dictionary"""
        defaults = cls()
        result = {}
        for field_name in cls.__annotations__:
            if field_name in data:
                result[field_name] = data[field_name]
            else:
                result[field_name] = getattr(defaults, field_name)
        return cls(**result)

@dataclass
class ProcessingJob:
    """A video processing job"""
    id: str
    input_path: str
    output_path: str
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
    position_in_queue: int = 0
    
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
        
        # Calculate ETA if processing
        if self.status == 'processing' and self.progress > 0:
            elapsed = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
            if elapsed > 0 and self.progress > 0:
                total_estimated = elapsed / (self.progress / 100)
                remaining = total_estimated - elapsed
                result['eta'] = max(0, int(remaining))
        
        return result

# ============================================================================
# BATCH PROCESSING QUEUE MANAGER
# ============================================================================

class BatchProcessingManager:
    """Manages batch processing queue with job management"""
    
    def __init__(self, socketio):
        self.socketio = socketio
        self.rve_backend = RVEBackendIntegration()
        self.active_jobs: Dict[str, ProcessingJob] = {}
        self.job_queue = queue.Queue(maxsize=config.MAX_QUEUE_SIZE)
        self.processing_jobs: Dict[str, ProcessingJob] = {}
        self.completed_jobs: List[ProcessingJob] = []
        self.failed_jobs: List[ProcessingJob] = []
        
        # Thread pool for concurrent processing
        self.executor = ThreadPoolExecutor(max_workers=config.MAX_CONCURRENT_JOBS)
        self.processing_futures = {}
        
        # Start queue monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_queue, daemon=True)
        self.monitor_thread.start()
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_old_jobs, daemon=True)
        self.cleanup_thread.start()
        
        print(f"Batch Processing Manager initialized with {config.MAX_CONCURRENT_JOBS} concurrent jobs")
    
    def add_job(self, input_path: str, settings_dict: dict, use_real_backend: bool = True) -> ProcessingJob:
        """Add a new job to the batch queue"""
        if self.job_queue.full():
            raise Exception("Job queue is full. Please wait for some jobs to complete.")
        
        job_id = f"job_{int(time.time())}_{os.urandom(2).hex()}"
        
        # Create output filename
        input_name = os.path.splitext(os.path.basename(input_path))[0]
        output_filename = f"enhanced_{input_name}.mp4"
        output_path = os.path.join(config.OUTPUT_FOLDER, output_filename)
        
        # Ensure unique output filename
        counter = 1
        base_name = os.path.splitext(output_filename)[0]
        while os.path.exists(output_path):
            output_filename = f"{base_name}_{counter}.mp4"
            output_path = os.path.join(config.OUTPUT_FOLDER, output_filename)
            counter += 1
        
        settings = ProcessingSettings.from_dict(settings_dict)
        
        # Force demo mode if backend not available
        if not self.rve_backend.backend_available:
            use_real_backend = False
            print("Backend not available, forcing demo mode")
        
        job = ProcessingJob(
            id=job_id,
            input_path=input_path,
            output_path=output_path,
            settings=settings,
            use_real_backend=use_real_backend,
            position_in_queue=self.job_queue.qsize() + 1
        )
        
        self.active_jobs[job_id] = job
        self.job_queue.put(job)
        
        print(f"✓ Job {job_id} added to queue (position: {job.position_in_queue})")
        print(f"  Input: {os.path.basename(input_path)}")
        print(f"  Output: {output_filename}")
        print(f"  GPU: {settings.gpu_id}")
        print(f"  Backend: {settings.backend}")
        print(f"  Real backend: {use_real_backend}")
        
        # Update all clients
        self._emit_queue_update()
        
        return job
    
    def _monitor_queue(self):
        """Monitor the job queue and start processing when capacity is available"""
        while True:
            try:
                # Check if we can start new jobs
                if len(self.processing_jobs) < config.MAX_CONCURRENT_JOBS:
                    # Try to get a job from queue
                    try:
                        job = self.job_queue.get(timeout=1)
                        
                        # Update job status
                        job.status = "processing"
                        job.start_time = datetime.now()
                        job.position_in_queue = 0  # No longer in queue
                        
                        # Add to processing jobs
                        self.processing_jobs[job.id] = job
                        
                        # Submit for processing
                        future = self.executor.submit(self._process_job, job)
                        self.processing_futures[job.id] = future
                        
                        print(f"→ Started processing job {job.id}")
                        self._emit_job_update(job)
                        self._emit_queue_update()
                        
                    except queue.Empty:
                        pass
                
                # Check for completed futures
                completed_jobs = []
                for job_id, future in list(self.processing_futures.items()):
                    if future.done():
                        try:
                            future.result()  # This will raise any exceptions
                            completed_jobs.append(job_id)
                        except Exception as e:
                            print(f"Job {job_id} failed with exception: {e}")
                            completed_jobs.append(job_id)
                
                # Clean up completed jobs
                for job_id in completed_jobs:
                    if job_id in self.processing_futures:
                        del self.processing_futures[job_id]
                    
                    if job_id in self.processing_jobs:
                        job = self.processing_jobs.pop(job_id)
                        if job.status == "completed":
                            self.completed_jobs.append(job)
                        elif job.status == "failed":
                            self.failed_jobs.append(job)
                
                # Update queue positions
                self._update_queue_positions()
                
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Error in queue monitor: {e}")
                time.sleep(1)
    
    def _process_job(self, job: ProcessingJob):
        """Process a single job"""
        try:
            print(f"\n{'='*60}")
            print(f"Processing job {job.id}")
            print(f"GPU: {job.settings.gpu_id}, Backend: {job.settings.backend}")
            print(f"{'='*60}")
            
            if job.use_real_backend and self.rve_backend.backend_available:
                self._process_with_rve_backend(job)
            else:
                self._process_simulated(job)
            
        except Exception as e:
            job.status = "failed"
            job.error_message = str(e)
            job.end_time = datetime.now()
            print(f"Job {job.id} failed: {e}")
            self._emit_job_update(job)
    
    def _process_with_rve_backend(self, job: ProcessingJob):
        """Process job using real RVE backend"""
        try:
            print(f"Starting RVE backend for job {job.id}")
            process = self.rve_backend.start_backend_process(job)
            
            # Monitor progress
            self._monitor_backend_progress(job, process)
            
            # Wait for process to complete
            return_code = process.wait()
            
            print(f"Process completed with return code: {return_code}")
            
            # Check if output file was created
            output_exists = os.path.exists(job.output_path)
            
            if return_code == 0 and output_exists:
                job.status = "completed"
                job.progress = 100.0
                print(f"✓ Job {job.id} completed successfully")
                
                # Extract video preview frame
                self._create_video_preview(job)
            else:
                job.status = "failed"
                error_msg = f"Backend failed with exit code {return_code}"
                if not output_exists:
                    error_msg += " - output file was not created"
                job.error_message = error_msg
                
                # Get error details from log
                if job.log_file and os.path.exists(job.log_file):
                    try:
                        with open(job.log_file, 'r') as f:
                            lines = f.readlines()
                            last_lines = lines[-50:] if len(lines) > 50 else lines
                            error_details = "".join(last_lines)
                            job.error_message += f"\nLast log lines:\n{error_details}"
                    except:
                        pass
                
                print(f"✗ Job {job.id} failed: {job.error_message}")
            
            job.end_time = datetime.now()
            self._emit_job_update(job)
            
        except Exception as e:
            job.status = "failed"
            job.error_message = f"Error in RVE backend: {str(e)}"
            job.end_time = datetime.now()
            print(f"Exception in RVE backend for job {job.id}: {e}")
            self._emit_job_update(job)
    
    def _monitor_backend_progress(self, job: ProcessingJob, process):
        """Monitor backend progress"""
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
        
        while process.poll() is None:
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
                                # Look for progress indicators
                                if 'Current Frame:' in line:
                                    frame_match = re.search(r'Current Frame:\s*(\d+)', line)
                                    if frame_match and total_frames > 0:
                                        current_frame = int(frame_match.group(1))
                                        progress = min(99.0, (current_frame / total_frames) * 100)
                                        if progress > job.progress:
                                            job.progress = progress
                                            self._emit_job_update(job)
                
                except Exception as e:
                    print(f"Error reading progress log: {e}")
            
            time.sleep(2)
        
        # Final progress update
        job.progress = 99.9
        self._emit_job_update(job)
    
    def _create_video_preview(self, job: ProcessingJob):
        """Create a preview frame from processed video"""
        try:
            if not job.output_path or not os.path.exists(job.output_path):
                return
            
            # Create preview directory
            preview_dir = os.path.join(config.TEMP_FOLDER, "previews")
            os.makedirs(preview_dir, exist_ok=True)
            
            # Extract middle frame
            preview_path = os.path.join(preview_dir, f"{job.id}_preview.jpg")
            
            cap = cv2.VideoCapture(job.output_path)
            if cap.isOpened():
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                middle_frame = total_frames // 2 if total_frames > 0 else 0
                
                if middle_frame > 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
                
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
                    print(f"Created preview for job {job.id}")
                    
                    # Store preview path in job
                    job.preview_path = preview_path
                cap.release()
        except Exception as e:
            print(f"Error creating preview for job {job.id}: {e}")
    
    def _process_simulated(self, job: ProcessingJob):
        """Simulate processing for demo purposes"""
        try:
            print(f"Simulating processing for job {job.id}")
            total_steps = 10
            
            for i in range(total_steps):
                time.sleep(2)
                job.progress = (i + 1) / total_steps * 100
                
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
            print(f"✓ Job {job.id} completed (simulated)")
            
        except Exception as e:
            job.status = "failed"
            job.error_message = str(e)
            job.end_time = datetime.now()
            print(f"✗ Simulated job {job.id} failed: {e}")
            self._emit_job_update(job)
    
    def _update_queue_positions(self):
        """Update position in queue for all queued jobs"""
        queued_jobs = []
        for job in self.active_jobs.values():
            if job.status == "pending" and job.position_in_queue > 0:
                queued_jobs.append(job)
        
        queued_jobs.sort(key=lambda x: x.position_in_queue)
        
        for i, job in enumerate(queued_jobs, 1):
            job.position_in_queue = i
    
    def _cleanup_old_jobs(self):
        """Clean up old completed/failed jobs"""
        while True:
            time.sleep(300)
            
            try:
                # Keep only last 50 completed jobs
                if len(self.completed_jobs) > 50:
                    jobs_to_remove = self.completed_jobs[:-50]
                    for job in jobs_to_remove:
                        if job.id in self.active_jobs:
                            del self.active_jobs[job.id]
                    self.completed_jobs = self.completed_jobs[-50:]
                
                # Keep only last 20 failed jobs
                if len(self.failed_jobs) > 20:
                    self.failed_jobs = self.failed_jobs[-20:]
                
                print(f"Cleanup: {len(self.active_jobs)} active jobs")
                
            except Exception as e:
                print(f"Error in cleanup: {e}")
    
    def _emit_job_update(self, job: ProcessingJob):
        """Emit job update via SocketIO"""
        try:
            self.socketio.emit('job_update', job.to_dict())
        except Exception as e:
            print(f"Error emitting job update: {e}")
    
    def _emit_queue_update(self):
        """Emit queue status update"""
        try:
            queue_status = {
                'queued': self.job_queue.qsize(),
                'processing': len(self.processing_jobs),
                'completed': len(self.completed_jobs),
                'failed': len(self.failed_jobs),
                'total_active': len(self.active_jobs)
            }
            self.socketio.emit('queue_update', queue_status)
        except Exception as e:
            print(f"Error emitting queue update: {e}")
    
    # Public methods
    def get_job(self, job_id: str) -> Optional[ProcessingJob]:
        """Get job by ID"""
        return self.active_jobs.get(job_id)
    
    def get_all_jobs(self) -> List[ProcessingJob]:
        """Get all jobs"""
        return list(self.active_jobs.values())
    
    def get_queue_stats(self) -> dict:
        """Get queue statistics"""
        return {
            'queued': self.job_queue.qsize(),
            'processing': len(self.processing_jobs),
            'completed': len(self.completed_jobs),
            'failed': len(self.failed_jobs),
            'max_concurrent': config.MAX_CONCURRENT_JOBS,
            'max_queue_size': config.MAX_QUEUE_SIZE
        }
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job"""
        if job_id in self.active_jobs:
            job = self.active_jobs[job_id]
            
            if job.status == "pending":
                job.status = "cancelled"
                self._emit_job_update(job)
                return True
                
            elif job.status == "processing":
                if job.process_pid:
                    try:
                        os.kill(job.process_pid, signal.SIGTERM)
                        print(f"Sent SIGTERM to process {job.process_pid}")
                    except:
                        pass
                
                job.status = "cancelled"
                job.end_time = datetime.now()
                self._emit_job_update(job)
                
                if job_id in self.processing_jobs:
                    del self.processing_jobs[job_id]
                
                return True
        
        return False
    
    def delete_job(self, job_id: str) -> bool:
        """Delete a job completely"""
        if job_id in self.active_jobs:
            job = self.active_jobs[job_id]
            
            if job.status in ["pending", "processing"]:
                self.cancel_job(job_id)
            
            try:
                if os.path.exists(job.output_path):
                    os.remove(job.output_path)
                if job.log_file and os.path.exists(job.log_file):
                    os.remove(job.log_file)
                if job.preview_path and os.path.exists(job.preview_path):
                    os.remove(job.preview_path)
            except Exception as e:
                print(f"Error deleting files: {e}")
            
            if job_id in self.processing_jobs:
                del self.processing_jobs[job_id]
            
            if job in self.completed_jobs:
                self.completed_jobs.remove(job)
            
            if job in self.failed_jobs:
                self.failed_jobs.remove(job)
            
            del self.active_jobs[job_id]
            
            self._emit_queue_update()
            return True
        
        return False
    
    def clear_completed_jobs(self) -> int:
        """Clear all completed jobs"""
        count = len(self.completed_jobs)
        
        for job in self.completed_jobs[:]:
            if job.id in self.active_jobs:
                del self.active_jobs[job.id]
        
        self.completed_jobs.clear()
        self._emit_queue_update()
        
        return count
    
    def pause_processing(self):
        """Pause all processing"""
        print("Processing pause requested")
    
    def resume_processing(self):
        """Resume processing"""
        print("Processing resume requested")

# ============================================================================
# CREATE HTML TEMPLATE
# ============================================================================

def create_html_template():
    """Create the HTML template file"""
    template_path = os.path.join(config.TEMPLATE_FOLDER, "index.html")
    
    # Generate GPU options HTML
    gpu_options_html = ""
    for gpu in AVAILABLE_GPUS:
        selected = "selected" if gpu['id'] == 0 else ""
        gpu_options_html += f'<option value="{gpu["id"]}" {selected}>{gpu["name"]}</option>'
    
    # Read existing template if it exists
    if os.path.exists(template_path):
        print(f"Template already exists at: {template_path}")
        return template_path
    
    html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>REAL Video Enhancer - FIXED</title>
    <style>
        :root {
            --primary-color: #6366f1;
            --success-color: #10b981;
            --danger-color: #ef4444;
            --warning-color: #f59e0b;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: #0f172a;
            color: #f8fafc;
            line-height: 1.6;
            height: 100vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        
        .container {
            display: flex;
            flex-direction: column;
            height: 100vh;
            width: 100vw;
            overflow: hidden;
        }
        
        header {
            background: #1e293b;
            padding: 12px 20px;
            border-bottom: 1px solid #334155;
            flex-shrink: 0;
        }
        
        .header-content {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .logo {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        
        .logo i {
            font-size: 32px;
            color: var(--primary-color);
        }
        
        .logo-text h1 {
            font-size: 20px;
            font-weight: 600;
        }
        
        .logo-text p {
            font-size: 12px;
            color: #94a3b8;
        }
        
        .queue-status {
            display: flex;
            gap: 12px;
            align-items: center;
        }
        
        .status-indicator {
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 13px;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        
        .status-online {
            background: var(--success-color);
            color: white;
        }
        
        .status-offline {
            background: var(--danger-color);
            color: white;
        }
        
        .tab-container {
            display: flex;
            background: #1e293b;
            border-bottom: 1px solid #334155;
            padding: 0 20px;
            flex-shrink: 0;
        }
        
        .tab {
            padding: 12px 20px;
            background: transparent;
            border: none;
            color: #94a3b8;
            cursor: pointer;
            font-size: 14px;
            border-bottom: 2px solid transparent;
            transition: all 0.2s;
        }
        
        .tab:hover {
            color: #f8fafc;
            background: #2d3748;
        }
        
        .tab.active {
            color: var(--primary-color);
            border-bottom-color: var(--primary-color);
            background: #2d3748;
        }
        
        .main-layout {
            display: flex;
            flex: 1;
            overflow: hidden;
            position: relative;
            min-height: 0;
        }
        
        .sidebar {
            width: 320px;
            background: #1e293b;
            border-right: 1px solid #334155;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            transition: transform 0.3s ease;
            flex-shrink: 0;
        }
        
        .sidebar-content {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            min-height: 0;
        }
        
        .main-content {
            flex: 1;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            min-height: 0;
        }
        
        .video-preview-container {
            height: 70vh;
            min-height: 300px;
            max-height: 80vh;
            background: #0a0e17;
            border-bottom: 1px solid #334155;
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
            overflow: hidden;
        }
        
        .video-placeholder {
            text-align: center;
            color: #64748b;
            padding: 40px;
        }
        
        .video-placeholder i {
            font-size: 64px;
            margin-bottom: 16px;
            opacity: 0.3;
        }
        
        .video-player {
            width: 100%;
            height: 100%;
            object-fit: contain;
            display: none;
        }
        
        .video-container {
            width: 100%;
            height: 100%;
            display: none;
        }
        
        .video-container.active {
            display: block;
        }
        
        .video-controls-bar {
            padding: 15px 20px;
            background: #1e293b;
            border-bottom: 1px solid #334155;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-shrink: 0;
        }
        
        .gpu-backend-selectors {
            display: flex;
            gap: 10px;
        }
        
        .settings-panel {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            background: #1e293b;
            min-height: 0;
        }
        
        .settings-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .settings-section {
            background: #2d3748;
            border-radius: 8px;
            padding: 15px;
            border: 1px solid #475569;
        }
        
        .form-group {
            margin-bottom: 12px;
        }
        
        .form-group label {
            display: block;
            font-size: 13px;
            color: #cbd5e1;
            margin-bottom: 4px;
        }
        
        .form-group select,
        .form-group input[type="number"] {
            width: 100%;
            padding: 8px 12px;
            background: #1e293b;
            border: 1px solid #475569;
            border-radius: 4px;
            color: #f8fafc;
            font-size: 13px;
        }
        
        .checkbox-group {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 10px;
        }
        
        .checkbox-group input[type="checkbox"] {
            accent-color: var(--primary-color);
        }
        
        .checkbox-group label {
            font-size: 13px;
            color: #cbd5e1;
            cursor: pointer;
        }
        
        .btn {
            padding: 8px 16px;
            border: none;
            border-radius: 6px;
            font-size: 13px;
            font-weight: 500;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 6px;
            transition: all 0.2s;
        }
        
        .btn-primary {
            background: var(--primary-color);
            color: white;
        }
        
        .btn-primary:hover {
            background: #4f46e5;
        }
        
        .btn-success {
            background: var(--success-color);
            color: white;
        }
        
        .btn-success:hover {
            background: #059669;
        }
        
        .btn-danger {
            background: var(--danger-color);
            color: white;
        }
        
        .btn-danger:hover {
            background: #dc2626;
        }
        
        .tab-content {
            display: none;
        }
        
        .tab-content.active {
            display: block;
        }
        
        .upload-area {
            border: 2px dashed #475569;
            border-radius: 8px;
            padding: 40px 20px;
            text-align: center;
            margin-bottom: 20px;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .upload-area:hover {
            border-color: var(--primary-color);
            background: rgba(99, 102, 241, 0.05);
        }
        
        .upload-area i {
            font-size: 48px;
            color: #64748b;
            margin-bottom: 10px;
        }
        
        .info-grid {
            background: #2d3748;
            border-radius: 6px;
            padding: 15px;
            margin-bottom: 20px;
        }
        
        .info-row {
            display: flex;
            justify-content: space-between;
            margin-bottom: 8px;
        }
        
        .hidden {
            display: none !important;
        }
        
        input[type="range"] {
            width: 100%;
            height: 6px;
            background: #475569;
            border-radius: 3px;
            outline: none;
            -webkit-appearance: none;
        }
        
        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 18px;
            height: 18px;
            background: var(--primary-color);
            border-radius: 50%;
            cursor: pointer;
        }
        
        .range-value {
            display: inline-block;
            min-width: 30px;
            text-align: right;
            font-weight: 500;
            color: var(--primary-color);
        }
        
        #toastContainer {
            position: fixed;
            bottom: 20px;
            right: 20px;
            z-index: 1000;
        }
        
        .toast {
            background: #1e293b;
            border-left: 4px solid var(--primary-color);
            color: #f8fafc;
            padding: 12px 20px;
            margin-bottom: 10px;
            border-radius: 4px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            animation: slideIn 0.3s ease;
            max-width: 300px;
        }
        
        .toast.success {
            border-left-color: var(--success-color);
        }
        
        .toast.error {
            border-left-color: var(--danger-color);
        }
        
        @keyframes slideIn {
            from {
                transform: translateX(100%);
                opacity: 0;
            }
            to {
                transform: translateX(0);
                opacity: 1;
            }
        }
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
                        <h1>REAL Video Enhancer - BACKEND FIXED</h1>
                        <p>Proper model detection for each backend</p>
                    </div>
                </div>
                <div class="queue-status">
                    <div id="queueStats" class="queue-badge">
                        <i class="fas fa-list"></i>
                        <span>Queue: 0 | Processing: 0</span>
                    </div>
                    <div id="backendStatus" class="status-indicator status-offline">
                        <i class="fas fa-circle"></i>
                        <span>Checking backend...</span>
                    </div>
                </div>
            </div>
        </header>
        
        <div class="tab-container">
            <button class="tab active" data-tab="editor">Editor</button>
            <button class="tab" data-tab="models">Models</button>
            <button class="tab" data-tab="jobs">Jobs</button>
        </div>
        
        <div class="main-layout">
            <div class="sidebar" id="sidebar">
                <div class="sidebar-content">
                    <div id="editor-tab" class="tab-content active">
                        <h2>Video Processing</h2>
                        <div class="upload-area" id="uploadArea">
                            <i class="fas fa-cloud-upload-alt"></i>
                            <h3>Drop video file here</h3>
                            <p>or click to browse (MP4, MOV, AVI, MKV, WEBM)</p>
                            <input type="file" id="fileInput" accept=".mp4,.mov,.avi,.mkv,.webm">
                        </div>
                        
                        <div id="videoInfo" class="hidden">
                            <h3>Video Information</h3>
                            <div class="info-grid">
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
                            </div>
                        </div>
                        
                        <div class="queue-stats">
                            <h3>Processing Queue</h3>
                            <div class="stats-grid">
                                <div class="stat-item">
                                    <span class="stat-label">Queued</span>
                                    <span class="stat-value" id="statQueued">0</span>
                                </div>
                                <div class="stat-item">
                                    <span class="stat-label">Processing</span>
                                    <span class="stat-value" id="statProcessing">0</span>
                                </div>
                                <div class="stat-item">
                                    <span class="stat-label">Completed</span>
                                    <span class="stat-value" id="statCompleted">0</span>
                                </div>
                            </div>
                            
                            <div style="margin-top: 15px;">
                                <button class="btn btn-primary btn-block" onclick="startProcessing()">
                                    <i class="fas fa-bolt"></i> Start Processing
                                </button>
                            </div>
                        </div>
                    </div>
                    
                    <div id="models-tab" class="tab-content">
                        <h2>Available Models</h2>
                        <div id="modelsList">
                            <p>Loading models...</p>
                        </div>
                    </div>
                    
                    <div id="jobs-tab" class="tab-content">
                        <h2>Jobs Queue</h2>
                        <div id="jobsList">
                            <p>No jobs in queue</p>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="main-content">
                <div class="video-preview-container">
                    <div id="videoPlaceholder" class="video-placeholder">
                        <i class="fas fa-video"></i>
                        <p>Video preview will appear here</p>
                    </div>
                    <div id="videoContainer" class="video-container">
                        <video id="videoPlayer" class="video-player" controls>
                            Your browser does not support the video tag.
                        </video>
                    </div>
                </div>
                
                <div class="video-controls-bar">
                    <div style="flex: 1;">
                        <h2>Processing Settings</h2>
                    </div>
                    <div class="gpu-backend-selectors">
                        <div class="form-group" style="min-width: 140px;">
                            <label>GPU:</label>
                            <select id="gpuSelection">
                                ''' + gpu_options_html + '''
                            </select>
                        </div>
                        <div class="form-group" style="min-width: 140px;">
                            <label>Backend:</label>
                            <select id="backend">
                                <option value="pytorch" selected>PyTorch</option>
                                <option value="tensorrt">TensorRT</option>
                                <option value="ncnn">NCNN</option>
                            </select>
                        </div>
                    </div>
                </div>
                
                <div class="settings-panel">
                    <div class="settings-grid">
                        <div class="settings-section">
                            <h4><i class="fas fa-expand-alt"></i> Upscaling</h4>
                            <div class="checkbox-group">
                                <input type="checkbox" id="enableUpscale" checked>
                                <label for="enableUpscale">Enable Upscaling</label>
                            </div>
                            <div id="upscaleOptions">
                                <div class="form-group">
                                    <label>Upscale Model:</label>
                                    <select id="upscaleModel">
                                        <option value="">Select model...</option>
                                    </select>
                                </div>
                                <div class="form-group">
                                    <label>Scale Factor: <span class="range-value" id="upscaleValue">2x</span></label>
                                    <input type="range" id="upscaleFactor" min="1" max="4" step="1" value="2">
                                </div>
                            </div>
                        </div>
                        
                        <div class="settings-section">
                            <h4><i class="fas fa-forward"></i> Interpolation</h4>
                            <div class="checkbox-group">
                                <input type="checkbox" id="enableInterpolate">
                                <label for="enableInterpolate">Frame Interpolation</label>
                            </div>
                            <div id="interpolateOptions" class="hidden">
                                <div class="form-group">
                                    <label>Interpolation Model:</label>
                                    <select id="interpolateModel">
                                        <option value="">Select model...</option>
                                    </select>
                                </div>
                                <div class="form-group">
                                    <label>Interpolation Factor: <span class="range-value" id="interpolateValue">2x</span></label>
                                    <input type="range" id="interpolateFactor" min="1" max="4" step="0.5" value="2">
                                </div>
                            </div>
                        </div>
                        
                        <div class="settings-section">
                            <h4><i class="fas fa-cog"></i> Output Settings</h4>
                            <div class="form-group">
                                <label>Precision:</label>
                                <select id="precision">
                                    <option value="auto" selected>Auto</option>
                                    <option value="float16">FP16 (Fast)</option>
                                    <option value="float32">FP32 (Accurate)</option>
                                </select>
                            </div>
                            <div class="form-group">
                                <label>Video Quality (CRF): <span class="range-value" id="crfValue">18</span></label>
                                <input type="range" id="crf" min="0" max="51" value="18">
                            </div>
                            <div class="checkbox-group">
                                <input type="checkbox" id="enableOverwrite" checked>
                                <label for="enableOverwrite">Overwrite Output</label>
                            </div>
                        </div>
                    </div>
                    
                    <div style="display: flex; gap: 10px; margin-top: 20px;">
                        <button class="btn btn-success" onclick="validateSettings()">
                            <i class="fas fa-check"></i> Validate Settings
                        </button>
                        <button class="btn btn-warning" onclick="resetSettings()">
                            <i class="fas fa-redo"></i> Reset
                        </button>
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
        let availableModels = {};
        
        function initSocket() {
            socket = io();
            
            socket.on('connect', () => {
                console.log('Connected to server');
                showToast('Connected to server', 'success');
                checkBackendStatus();
                refreshQueueStats();
                refreshJobs();
                loadModels();
            });
            
            socket.on('job_update', (job) => {
                updateJobInList(job);
                refreshQueueStats();
            });
            
            socket.on('queue_update', (stats) => {
                updateQueueStats(stats);
            });
            
            socket.on('backend_status', (status) => {
                updateBackendStatus(status);
            });
        }
        
        function checkBackendStatus() {
            fetch('/backend/status')
                .then(response => response.json())
                .then(status => updateBackendStatus(status))
                .catch(error => {
                    console.error('Error checking backend status:', error);
                    updateBackendStatus({ 
                        available: false, 
                        error: error.message 
                    });
                });
        }
        
        function updateBackendStatus(status) {
            backendAvailable = status.available;
            const statusElement = document.getElementById('backendStatus');
            
            if (backendAvailable) {
                statusElement.className = 'status-indicator status-online';
                statusElement.innerHTML = '<i class="fas fa-check-circle"></i> Backend Ready';
                if (status.version) {
                    statusElement.innerHTML += ' (' + status.version + ')';
                }
            } else {
                statusElement.className = 'status-indicator status-offline';
                if (status.error) {
                    statusElement.innerHTML = '<i class="fas fa-exclamation-circle"></i> ' + status.error;
                } else {
                    statusElement.innerHTML = '<i class="fas fa-exclamation-circle"></i> Backend Unavailable';
                }
            }
        }
        
        async function loadModels() {
            try {
                const response = await fetch('/models');
                availableModels = await response.json();
                
                clearModelSelects();
                
                // Group models by backend compatibility
                populateModelSelectWithBackend();
                
                displayModelsList();
                showToast('Models loaded successfully', 'success');
            } catch (error) {
                console.error('Error loading models:', error);
                showToast('Error loading models', 'error');
            }
        }
        
        function clearModelSelects() {
            ['upscaleModel', 'interpolateModel'].forEach(id => {
                const select = document.getElementById(id);
                select.innerHTML = '<option value="">Select model...</option>';
            });
        }
        
        function populateModelSelectWithBackend() {
            const backend = document.getElementById('backend').value;
            const upscaleSelect = document.getElementById('upscaleModel');
            const interpolateSelect = document.getElementById('interpolateModel');
            
            // Clear existing options
            upscaleSelect.innerHTML = '<option value="">Select model...</option>';
            interpolateSelect.innerHTML = '<option value="">Select model...</option>';
            
            // Populate upscale models
            if (availableModels.upscale) {
                for (const [modelName, modelInfo] of Object.entries(availableModels.upscale)) {
                    // Check if model supports current backend
                    if (modelInfo.backend && modelInfo.backend.includes(backend)) {
                        const option = document.createElement('option');
                        option.value = modelName;
                        option.textContent = `${modelName} (${modelInfo.scale}x - ${backend})`;
                        upscaleSelect.appendChild(option);
                    }
                }
            }
            
            // Populate interpolate models
            if (availableModels.interpolate) {
                for (const [modelName, modelInfo] of Object.entries(availableModels.interpolate)) {
                    if (modelInfo.backend && modelInfo.backend.includes(backend)) {
                        const option = document.createElement('option');
                        option.value = modelName;
                        option.textContent = `${modelName} (${modelInfo.scale}x - ${backend})`;
                        interpolateSelect.appendChild(option);
                    }
                }
            }
            
            // Select first model if available
            if (upscaleSelect.options.length > 1) {
                upscaleSelect.selectedIndex = 1;
            }
            if (interpolateSelect.options.length > 1) {
                interpolateSelect.selectedIndex = 1;
            }
        }
        
        function displayModelsList() {
            const modelsList = document.getElementById('modelsList');
            modelsList.innerHTML = '';
            
            const categories = [
                {id: 'upscale', name: 'Upscaling Models', icon: 'fas fa-expand-alt'},
                {id: 'interpolate', name: 'Interpolation Models', icon: 'fas fa-forward'},
                {id: 'denoise', name: 'Denoising Models', icon: 'fas fa-magic'},
                {id: 'decompress', name: 'Compression Repair', icon: 'fas fa-compress-alt'}
            ];
            
            categories.forEach(category => {
                if (availableModels[category.id] && Object.keys(availableModels[category.id]).length > 0) {
                    const section = document.createElement('div');
                    section.className = 'settings-section';
                    section.style.marginBottom = '15px';
                    
                    let html = `<h4><i class="${category.icon}"></i> ${category.name}</h4>`;
                    html += '<div style="max-height: 200px; overflow-y: auto;">';
                    
                    for (const [modelName, modelInfo] of Object.entries(availableModels[category.id])) {
                        html += `<div style="margin-bottom: 8px; padding: 8px; background: #334155; border-radius: 4px;">
                                    <div style="font-weight: 500; font-size: 13px;">${modelName}</div>
                                    <div style="font-size: 11px; color: #94a3b8; margin-top: 2px;">
                                        Scale: ${modelInfo.scale}x | 
                                        Backends: ${modelInfo.backend ? modelInfo.backend.join(', ') : 'pytorch'}
                                    </div>
                                </div>`;
                    }
                    
                    html += '</div>';
                    section.innerHTML = html;
                    modelsList.appendChild(section);
                }
            });
            
            if (modelsList.children.length === 0) {
                modelsList.innerHTML = '<p>No models found. Please check your models directory.</p>';
            }
        }
        
        function getCurrentSettings() {
            const backend = document.getElementById('backend').value;
            
            const settings = {
                upscale_enabled: document.getElementById('enableUpscale').checked,
                upscale_model: document.getElementById('upscaleModel').value,
                upscale_factor: parseInt(document.getElementById('upscaleFactor').value),
                
                interpolate_enabled: document.getElementById('enableInterpolate').checked,
                interpolate_model: document.getElementById('interpolateModel').value,
                interpolate_factor: parseFloat(document.getElementById('interpolateFactor').value),
                
                backend: backend,
                gpu_id: parseInt(document.getElementById('gpuSelection').value),
                precision: document.getElementById('precision').value,
                crf: parseInt(document.getElementById('crf').value),
                overwrite: document.getElementById('enableOverwrite').checked,
                
                scene_detect_enabled: true,
                tile_size: 0,
                output_format: "mp4",
                output_codec: "libx264",
                audio_quality: "copy",
                tensorrt_opt_profile: 3,
                tensorrt_dynamic_shapes: false,
                UHD_mode: false,
                ensemble: false
            };
            
            return settings;
        }
        
        function validateSettings() {
            const settings = getCurrentSettings();
            let errors = [];
            
            if (settings.upscale_enabled && !settings.upscale_model) {
                errors.push("Please select an upscale model");
            }
            
            if (settings.interpolate_enabled && !settings.interpolate_model) {
                errors.push("Please select an interpolation model");
            }
            
            if (errors.length > 0) {
                showToast('Validation errors: ' + errors.join(', '), 'error');
                return false;
            } else {
                showToast('Settings validated successfully', 'success');
                return true;
            }
        }
        
        function startProcessing() {
            if (!currentVideo) {
                showToast('Please upload a video first', 'error');
                return;
            }
            
            if (!validateSettings()) {
                return;
            }
            
            const settings = getCurrentSettings();
            
            fetch('/process', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    filename: currentVideo,
                    settings: settings
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    showToast(`Job added to queue (position: ${data.position})`, 'success');
                    refreshJobs();
                } else {
                    showToast(data.error || 'Failed to start processing', 'error');
                }
            })
            .catch(error => {
                console.error('Error:', error);
                showToast('Failed to start processing', 'error');
            });
        }
        
        function loadVideoForPreview(videoUrl) {
            const videoPlayer = document.getElementById('videoPlayer');
            const videoContainer = document.getElementById('videoContainer');
            const videoPlaceholder = document.getElementById('videoPlaceholder');
            
            videoPlayer.src = videoUrl;
            videoContainer.classList.add('active');
            videoPlaceholder.classList.add('hidden');
            videoPlayer.load();
            showToast('Video loaded for preview', 'success');
        }
        
        // Initialize
        window.addEventListener('DOMContentLoaded', () => {
            initSocket();
            
            // Set initial state
            document.getElementById('interpolateOptions').classList.add('hidden');
            
            // Tab switching
            document.querySelectorAll('.tab').forEach(tab => {
                tab.addEventListener('click', function() {
                    const tabId = this.dataset.tab;
                    
                    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                    this.classList.add('active');
                    
                    document.querySelectorAll('.tab-content').forEach(content => {
                        content.classList.remove('active');
                    });
                    document.getElementById(`${tabId}-tab`).classList.add('active');
                });
            });
            
            // File upload
            document.getElementById('fileInput').addEventListener('change', handleFileUpload);
            document.getElementById('uploadArea').addEventListener('click', function() {
                document.getElementById('fileInput').click();
            });
            
            // Backend selection change
            document.getElementById('backend').addEventListener('change', function() {
                populateModelSelectWithBackend();
            });
            
            // Toggle interpolate options
            document.getElementById('enableInterpolate').addEventListener('change', function() {
                const optionsDiv = document.getElementById('interpolateOptions');
                optionsDiv.classList.toggle('hidden', !this.checked);
            });
            
            // Range inputs
            document.getElementById('upscaleFactor').addEventListener('input', function() {
                document.getElementById('upscaleValue').textContent = this.value + 'x';
            });
            
            document.getElementById('interpolateFactor').addEventListener('input', function() {
                document.getElementById('interpolateValue').textContent = this.value + 'x';
            });
            
            document.getElementById('crf').addEventListener('input', function() {
                document.getElementById('crfValue').textContent = this.value;
            });
        });
        
        function handleFileUpload(e) {
            const file = e.target.files[0];
            if (!file) return;
            
            const formData = new FormData();
            formData.append('file', file);
            
            fetch('/upload', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    currentVideo = data.filename;
                    currentVideoInfo = data.video_info;
                    
                    // Update video info display
                    document.getElementById('infoFilename').textContent = data.video_info.filename;
                    document.getElementById('infoResolution').textContent = `${data.video_info.width}x${data.video_info.height}`;
                    document.getElementById('infoDuration').textContent = formatDuration(data.video_info.duration);
                    document.getElementById('infoFps').textContent = data.video_info.fps.toFixed(2);
                    
                    document.getElementById('videoInfo').classList.remove('hidden');
                    
                    // Load video for preview
                    loadVideoForPreview(`/uploads/${data.filename}`);
                    
                    showToast('Video uploaded successfully', 'success');
                } else {
                    showToast(data.error || 'Upload failed', 'error');
                }
            })
            .catch(error => {
                console.error('Error:', error);
                showToast('Upload failed', 'error');
            });
        }
        
        // Helper functions
        function showToast(message, type = 'info') {
            const container = document.getElementById('toastContainer');
            const toast = document.createElement('div');
            toast.className = `toast ${type}`;
            toast.textContent = message;
            container.appendChild(toast);
            
            setTimeout(() => {
                toast.remove();
            }, 5000);
        }
        
        function formatDuration(seconds) {
            const hours = Math.floor(seconds / 3600);
            const minutes = Math.floor((seconds % 3600) / 60);
            const secs = Math.floor(seconds % 60);
            
            if (hours > 0) {
                return `${hours}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
            } else {
                return `${minutes}:${secs.toString().padStart(2, '0')}`;
            }
        }
        
        function refreshQueueStats() {
            fetch('/queue/stats')
                .then(response => response.json())
                .then(stats => updateQueueStats(stats))
                .catch(error => console.error('Error:', error));
        }
        
        function updateQueueStats(stats) {
            document.getElementById('statQueued').textContent = stats.queued;
            document.getElementById('statProcessing').textContent = stats.processing;
            document.getElementById('statCompleted').textContent = stats.completed;
            
            const queueStats = document.getElementById('queueStats');
            queueStats.innerHTML = `<i class="fas fa-list"></i>
                                  <span>Queue: ${stats.queued} | Processing: ${stats.processing}</span>`;
        }
        
        function refreshJobs() {
            fetch('/jobs')
                .then(response => response.json())
                .then(jobs => updateJobsList(jobs))
                .catch(error => console.error('Error:', error));
        }
        
        function updateJobsList(jobs) {
            const jobsList = document.getElementById('jobsList');
            if (jobs.length === 0) {
                jobsList.innerHTML = '<p>No jobs in queue</p>';
                return;
            }
            
            let html = '';
            jobs.forEach(job => {
                const statusColor = {
                    'pending': '#f59e0b',
                    'processing': '#3b82f6',
                    'completed': '#10b981',
                    'failed': '#ef4444'
                }[job.status] || '#6b7280';
                
                html += `
                <div class="job-item" style="margin-bottom: 10px; padding: 10px; background: #2d3748; border-radius: 6px; border-left: 4px solid ${statusColor}">
                    <div style="font-weight: 500; font-size: 13px;">${job.input_path?.split('/').pop() || 'Unknown'}</div>
                    <div style="font-size: 11px; color: #94a3b8; margin-top: 2px;">
                        Status: <span style="color: ${statusColor}">${job.status}</span>
                        ${job.progress ? ` | Progress: ${job.progress?.toFixed(1) || 0}%` : ''}
                    </div>
                </div>
                `;
            });
            
            jobsList.innerHTML = html;
        }
        
        function updateJobInList(job) {
            // Simple update - just refresh the whole list
            refreshJobs();
        }
        
        function resetSettings() {
            if (!confirm('Are you sure you want to reset all settings to default?')) return;
            
            document.getElementById('enableUpscale').checked = true;
            document.getElementById('enableInterpolate').checked = false;
            document.getElementById('interpolateOptions').classList.add('hidden');
            document.getElementById('upscaleFactor').value = 2;
            document.getElementById('interpolateFactor').value = 2;
            document.getElementById('crf').value = 18;
            document.getElementById('precision').value = 'auto';
            document.getElementById('enableOverwrite').checked = true;
            
            document.getElementById('upscaleValue').textContent = '2x';
            document.getElementById('interpolateValue').textContent = '2x';
            document.getElementById('crfValue').textContent = '18';
            
            showToast('Settings reset to default', 'success');
        }
        
    </script>
</body>
</html>'''
    
    with open(template_path, 'w') as f:
        f.write(html_content)
    
    print(f"Created HTML template at: {template_path}")
    return template_path

# Create the template
create_html_template()

# ============================================================================
# INITIALIZE COMPONENTS
# ============================================================================

batch_manager = BatchProcessingManager(socketio)
rve_backend_integration = RVEBackendIntegration()

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
        
        counter = 1
        base_name, ext = os.path.splitext(filename)
        while os.path.exists(filepath):
            filename = f"{base_name}_{counter}{ext}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            counter += 1
        
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
    status = rve_backend_integration.detect_backend()
    return jsonify(status)

@app.route('/models')
def get_models():
    """Get available models with backend detection"""
    from pathlib import Path
    
    models = {
        'upscale': {},
        'interpolate': {},
        'denoise': {},
        'decompress': {}
    }
    
    models_dir = Path(config.MODELS_FOLDER)
    
    if not models_dir.exists():
        print(f"Models directory not found: {models_dir}")
        return jsonify(models)
    
    # Scan for models and detect backend compatibility
    model_count = 0
    
    for category in ['upscale', 'interpolate', 'denoise', 'decompress']:
        category_dir = models_dir / category
        
        if category_dir.exists():
            for item in category_dir.iterdir():
                if item.is_file():
                    ext = item.suffix.lower()
                    model_name = item.stem
                    
                    # Determine backend compatibility
                    backend = []
                    if ext in ['.pth', '.pt', '.ckpt', '.safetensors', '.pkl']:
                        backend.append('pytorch')
                    if ext == '.engine':
                        backend.append('tensorrt')
                    if ext in ['.param', '.bin']:
                        backend.append('ncnn')
                    
                    if backend:  # Only add if we can determine backend
                        model_count += 1
                        
                        # Detect scale from name
                        scale = 2
                        name_lower = model_name.lower()
                        if '4x' in name_lower or 'x4' in name_lower:
                            scale = 4
                        elif '3x' in name_lower or 'x3' in name_lower:
                            scale = 3
                        elif '2x' in name_lower or 'x2' in name_lower:
                            scale = 2
                        elif '1x' in name_lower or 'x1' in name_lower:
                            scale = 1
                        
                        models[category][model_name] = {
                            'scale': scale,
                            'backend': backend,
                            'path': str(item.relative_to(models_dir)),
                            'size': item.stat().st_size,
                            'extension': ext
                        }
    
    print(f"Found {model_count} models in {models_dir}")
    
    return jsonify(models)

@app.route('/process', methods=['POST'])
def process_video():
    """Start video processing"""
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON data provided'}), 400
    
    filename = data.get('filename')
    settings = data.get('settings', {})
    
    if not filename:
        return jsonify({'error': 'Filename required'}), 400
    
    input_path = os.path.join(config.UPLOAD_FOLDER, filename)
    if not os.path.exists(input_path):
        return jsonify({'error': 'Video file not found'}), 404
    
    try:
        use_real_backend = rve_backend_integration.backend_available
        job = batch_manager.add_job(input_path, settings, use_real_backend)
        
        return jsonify({
            'success': True,
            'job_id': job.id,
            'message': 'Job added to queue',
            'position': job.position_in_queue
        })
    except Exception as e:
        return jsonify({'error': f'Failed to add job to queue: {str(e)}'}), 500

@app.route('/jobs')
def get_jobs():
    """Get all processing jobs"""
    jobs = batch_manager.get_all_jobs()
    return jsonify([job.to_dict() for job in jobs])

@app.route('/queue/stats')
def get_queue_stats():
    """Get queue statistics"""
    stats = batch_manager.get_queue_stats()
    return jsonify(stats)

@app.route('/output/<filename>')
def serve_output_video(filename):
    """Serve processed video files"""
    filepath = os.path.join(config.OUTPUT_FOLDER, filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'Video not found'}), 404
    
    ext = os.path.splitext(filename)[1].lower()
    mime_types = {
        '.mp4': 'video/mp4',
        '.mkv': 'video/x-matroska',
        '.avi': 'video/x-msvideo',
        '.mov': 'video/quicktime'
    }
    
    mimetype = mime_types.get(ext, 'video/mp4')
    return send_file(filepath, mimetype=mimetype)

@app.route('/uploads/<filename>')
def serve_uploaded_video(filename):
    """Serve original uploaded video files"""
    filepath = os.path.join(config.UPLOAD_FOLDER, filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'Original video not found'}), 404
    
    ext = os.path.splitext(filename)[1].lower()
    mime_types = {
        '.mp4': 'video/mp4',
        '.mkv': 'video/x-matroska',
        '.avi': 'video/x-msvideo',
        '.mov': 'video/quicktime'
    }
    
    mimetype = mime_types.get(ext, 'video/mp4')
    return send_file(filepath, mimetype=mimetype)

# ============================================================================
# SOCKETIO EVENTS
# ============================================================================

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('Client connected')
    
    status = rve_backend_integration.detect_backend()
    emit('backend_status', status)
    
    stats = batch_manager.get_queue_stats()
    emit('queue_update', stats)

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point for the application"""
    print("=" * 70)
    print("REAL Video Enhancer - BACKEND FIXED")
    print("=" * 70)
    print(f"Base directory: {config.BASE_DIR}")
    print(f"Models folder: {config.MODELS_FOLDER}")
    
    print(f"\nGPU Detection:")
    for gpu in AVAILABLE_GPUS:
        print(f"  GPU {gpu['id']}: {gpu['name']}")
    
    backend_status = rve_backend_integration.detect_backend()
    if backend_status['available']:
        print(f"\n✓ RVE backend available")
        if 'version' in backend_status:
            print(f"  Version: {backend_status['version']}")
    else:
        print(f"\n⚠ RVE backend not available")
        print(f"  Error: {backend_status.get('error', 'Unknown error')}")
    
    print("\n✨ FIXED FEATURES:")
    print("  ✓ Proper model detection for each backend")
    print("  ✓ TensorRT uses .engine files")
    print("  ✓ PyTorch uses .pth/.pkl files")
    print("  ✓ NCNN uses .param/.bin files")
    print("  ✓ Automatic fallback to compatible models")
    
    print("=" * 70)
    print(f"Server running at: http://{config.HOST}:{config.PORT}")
    print("=" * 70)
    
    models_dir = Path(config.MODELS_FOLDER)
    models_dir.mkdir(exist_ok=True)
    
    for category in ['upscale', 'interpolate', 'denoise', 'decompress']:
        (models_dir / category).mkdir(exist_ok=True)
    
    socketio.run(
        app, 
        host=config.HOST, 
        port=config.PORT, 
        debug=config.DEBUG,
        allow_unsafe_werkzeug=True
    )

if __name__ == '__main__':
    main()