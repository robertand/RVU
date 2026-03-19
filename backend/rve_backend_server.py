#!/usr/bin/env python3
"""
RVE Backend Server - API pentru WebUI
Rulează ca server separat pentru a procesa cererile de upscale
"""

import os
import sys
import json
import subprocess
import threading
import time
import socket
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
import cgi
import torch

# Configurare
HOST = '0.0.0.0'  # Ascultă pe toate interfețele
PORT = 8765
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(os.path.dirname(BASE_DIR), 'uploads')
OUTPUT_DIR = os.path.join(os.path.dirname(BASE_DIR), 'outputs')
MODELS_DIR = os.path.join(os.path.dirname(BASE_DIR), 'models')
TEMP_DIR = os.path.join(os.path.dirname(BASE_DIR), 'temp')

# Asigură-te că directoarele există
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# Jobs în desfășurare
active_jobs = {}
job_history = []
job_lock = threading.Lock()

print(f"🚀 RVE Backend Server starting...")
print(f"📁 Uploads: {UPLOAD_DIR}")
print(f"📁 Outputs: {OUTPUT_DIR}")
print(f"📁 Models: {MODELS_DIR}")
print(f"🌐 Listening on http://{HOST}:{PORT}")


class RVEBackendHandler(BaseHTTPRequestHandler):

    def _send_response(self, data, status=200):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _send_error(self, message, status=500):
        self._send_response({'success': False, 'error': message}, status)

    def log_message(self, format, *args):
        # Suprimă logurile prea verbose
        if 'GET /jobs' not in args[0] and 'GET /status' not in args[0]:
            print(f"[{self.log_date_time_string()}] {args[0] % args[1:]}")

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        # Status backend
        if path == '/status' or path == '/backend/status':
            status_data = {
                'available': True,
                'version': '2.0.0',
                'cuda': torch.cuda.is_available(),
                'device': 'cuda' if torch.cuda.is_available() else 'cpu'
            }

            if torch.cuda.is_available():
                status_data['gpus'] = []
                for i in range(torch.cuda.device_count()):
                    try:
                        status_data['gpus'].append({
                            'id': i,
                            'name': torch.cuda.get_device_name(i),
                            'memory': torch.cuda.get_device_properties(i).total_memory // (1024**3)
                        })
                    except:
                        status_data['gpus'].append({'id': i, 'name': f'GPU {i}'})

            self._send_response(status_data)

        # Listă jobs
        elif path == '/jobs':
            with job_lock:
                # Merge active jobs and history
                jobs_to_return = []
                # Add active jobs first
                for jid, job in active_jobs.items():
                    jobs_to_return.append(job.copy())
                # Add history (last 50)
                for job in job_history[-50:]:
                    # Don't add if already in active_jobs
                    if not any(j['id'] == job['id'] for j in jobs_to_return):
                        jobs_to_return.append(job.copy())
            self._send_response(jobs_to_return)

        # Models disponibile
        elif path == '/models':
            models = self._scan_models()
            self._send_response(models)

        # Verifică dacă e request pentru fișiere
        elif path.startswith('/uploads/'):
            filename = path.replace('/uploads/', '')
            filepath = os.path.join(UPLOAD_DIR, filename)
            if os.path.exists(filepath):
                self._serve_file(filepath)
            else:
                self._send_error('File not found', 404)

        elif path.startswith('/output/'):
            filename = path.replace('/output/', '')
            filepath = os.path.join(OUTPUT_DIR, filename)
            if os.path.exists(filepath):
                self._serve_file(filepath)
            else:
                self._send_error('File not found', 404)

        else:
            self._send_error('Not found', 404)

    def _serve_file(self, filepath):
        """Servește un fișier"""
        try:
            with open(filepath, 'rb') as f:
                self.send_response(200)
                self.send_header('Content-Type', 'video/mp4')
                self.send_header('Content-Length', str(os.path.getsize(filepath)))
                self.send_header('Accept-Ranges', 'bytes')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(f.read())
        except Exception as e:
            self._send_error(f'Error serving file: {e}', 500)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        # Process video
        if path == '/process' or path == '/backend/process':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data)

            job_id = f"job_{int(time.time() * 1000)}"

            print(f"\n📥 New job {job_id} received")
            print(f"   File: {data.get('filename')}")

            # Pornește procesarea în thread separat
            thread = threading.Thread(
                target=self._process_video,
                args=(job_id, data)
            )
            thread.daemon = True
            thread.start()

            self._send_response({
                'success': True,
                'job_id': job_id,
                'message': 'Processing started'
            })

        # Stop processing
        elif path.startswith('/stop_processing/'):
            job_id = path.split('/')[-1]
            with job_lock:
                if job_id in active_jobs:
                    # TODO: Implement stop logic
                    active_jobs[job_id]['status'] = 'stopping'
                    self._send_response({'success': True})
                else:
                    self._send_response({'success': False, 'error': 'Job not found'})

        else:
            self._send_error('Not found', 404)

    def _scan_models(self):
        """Scanează modelele disponibile"""
        categories = ['upscale', 'interpolate', 'denoise', 'decompress']
        result = {}

        for category in categories:
            cat_dir = os.path.join(MODELS_DIR, category)
            if os.path.exists(cat_dir):
                models = {}
                for f in os.listdir(cat_dir):
                    if f.endswith(('.pth', '.pt', '.safetensors', '.onnx')):
                        models[f] = {
                            'filename': f,
                            'path': os.path.join(cat_dir, f),
                            'scale': self._guess_scale(f)
                        }
                result[category] = models

        return result

    def _guess_scale(self, filename):
        """Ghicește scala modelului din nume"""
        filename = filename.lower()
        if '2x' in filename or 'x2' in filename:
            return 2
        elif '3x' in filename or 'x3' in filename:
            return 3
        elif '4x' in filename or 'x4' in filename:
            return 4
        elif '8x' in filename or 'x8' in filename:
            return 8
        return None

    def _process_video(self, job_id, data):
        """Procesează video cu modelul AI"""
        try:
            with job_lock:
                active_jobs[job_id] = {
                    'id': job_id,
                    'status': 'processing',
                    'progress': 0,
                    'start_time': time.time()
                }

            input_file = data.get('filename')
            settings = data.get('settings', {})

            # Construiește calea completă
            input_path = os.path.join(UPLOAD_DIR, input_file)
            if not os.path.exists(input_path):
                raise Exception(f"Input file not found: {input_path}")

            # Generează nume output
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            base_name = os.path.splitext(input_file)[0]
            output_filename = f"{base_name}_enhanced_{timestamp}.mp4"
            output_path = os.path.join(OUTPUT_DIR, output_filename)

            # Găsește modelul
            model_path = None
            if settings.get('upscale_enabled') and settings.get('upscale_model'):
                model_name = settings['upscale_model']
                print(f"🔍 Searching for model: {model_name}")

                # Caută modelul în directorul upscale
                upscale_dir = os.path.join(MODELS_DIR, 'upscale')
                if os.path.exists(upscale_dir):
                    for f in os.listdir(upscale_dir):
                        if model_name in f or f.startswith(model_name) or model_name in os.path.splitext(f)[0]:
                            model_path = os.path.join(upscale_dir, f)
                            print(f"✅ Found model: {model_path}")
                            break

            if not model_path and settings.get('upscale_enabled'):
                # Fallback: caută în toate directoarele
                print(f"⚠️ Model not found in upscale dir, searching all subdirectories...")
                for root, dirs, files in os.walk(MODELS_DIR):
                    for f in files:
                        if f.endswith(('.pth', '.pt', '.safetensors')):
                            if model_name in f or f.startswith(model_name):
                                model_path = os.path.join(root, f)
                                print(f"✅ Found model in {root}: {f}")
                                break
                    if model_path:
                        break

            # Construiește comanda pentru noul backend
            backend_script = os.path.join(BASE_DIR, 'simple_backend.py')

            cmd = [
                sys.executable,
                backend_script,
                '-i', input_path,
                '-o', output_path,
                '--overwrite'
            ]

            if model_path:
                cmd.extend(['--upscale_model', model_path])

            # Adaugă settings
            if settings.get('gpu_id') is not None:
                cmd.extend(['--pytorch_gpu_id', str(settings['gpu_id'])])

            if settings.get('crf'):
                cmd.extend(['--crf', str(settings['crf'])])

            if settings.get('auto_hdr_mode'):
                cmd.append('--hdr_mode')

            if settings.get('tile_size'):
                cmd.extend(['--tilesize', str(settings['tile_size'])])

            if settings.get('overlap'):
                cmd.extend(['--overlap', str(settings['overlap'])])

            if settings.get('precision'):
                cmd.extend(['--precision', str(settings['precision'])])

            if settings.get('tta_enabled'):
                cmd.append('--tta')

            if settings.get('scene_detect_cuda'):
                cmd.extend(['--scene_detect_method', 'cuda'])
            elif settings.get('scene_detect_enabled'):
                cmd.extend(['--scene_detect_method', 'cpu'])

            if settings.get('deinterlace_method') and settings['deinterlace_method'] != 'none':
                cmd.extend(['--deinterlace_method', settings['deinterlace_method']])

            print(f"\n🚀 Starting job {job_id}")
            print(f"   Command: {' '.join(cmd)}")

            # Rulează procesul
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )

            # Get video total frames for progress calculation
            try:
                cap = cv2.VideoCapture(input_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                with job_lock:
                    if job_id in active_jobs:
                        active_jobs[job_id]['total_frames'] = total_frames
            except:
                total_frames = 0

            # Monitorizează output-ul
            for line in process.stdout:
                line = line.strip()
                if line:
                    # print(f"[{job_id}] {line}") # Don't spam server console

                    # Parsează frame count
                    if 'Current Frame:' in line:
                        try:
                            frame_str = line.split('Current Frame:')[1].strip()
                            frame = int(frame_str)
                            with job_lock:
                                if job_id in active_jobs:
                                    total = active_jobs[job_id].get('total_frames', 0)
                                    progress = (frame / total) * 100 if total > 0 else 0
                                    active_jobs[job_id]['progress'] = progress
                        except:
                            pass

            process.wait()

            if process.returncode == 0 and os.path.exists(output_path):
                with job_lock:
                    if job_id in active_jobs:
                        active_jobs[job_id]['status'] = 'completed'
                        active_jobs[job_id]['progress'] = 100
                        active_jobs[job_id]['output'] = output_filename

                    # Adaugă în history
                    job_history.append({
                        'id': job_id,
                        'input': input_file,
                        'output': output_filename,
                        'status': 'completed',
                        'timestamp': time.time(),
                        'settings': settings
                    })

                print(f"✅ Job {job_id} completed successfully")
                print(f"   Output: {output_path}")
            else:
                raise Exception(f"Process failed with code {process.returncode}")

        except Exception as e:
            print(f"❌ Job {job_id} failed: {e}")
            import traceback
            # traceback.print_exc()

            with job_lock:
                if job_id in active_jobs:
                    active_jobs[job_id]['status'] = 'failed'
                    active_jobs[job_id]['error'] = str(e)

                job_history.append({
                    'id': job_id,
                    'input': input_file if 'input_file' in locals() else 'unknown',
                    'status': 'failed',
                    'error': str(e),
                    'timestamp': time.time(),
                    'settings': settings if 'settings' in locals() else {}
                })


def run_server():
    """Pornește serverul"""
    server = HTTPServer((HOST, PORT), RVEBackendHandler)
    print(f"\n{'='*50}")
    print(f"🚀 RVE Backend Server running on http://{HOST}:{PORT}")
    print(f"{'='*50}")
    print(f"📁 Uploads: {UPLOAD_DIR}")
    print(f"📁 Outputs: {OUTPUT_DIR}")
    print(f"📁 Models: {MODELS_DIR}")
    print(f"\nPress Ctrl+C to stop\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")
    finally:
        server.server_close()


if __name__ == '__main__':
    # Make sure cv2 is available for frame count
    try:
        import cv2
    except ImportError:
        print("Installing opencv-python-headless for frame counting...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python-headless"])
        import cv2

    run_server()
