#!/usr/bin/env python3
"""
Script pentru debugging RVE backend și modele
"""

import os
import sys
import subprocess
from pathlib import Path

def check_backend():
    """Verifică backend-ul RVE"""
    backend_path = "/media/prouser/Storage/PlayGround/RVU/backend/rve-backend.py"
    python_path = "/media/prouser/Storage/PlayGround/RVU/python/python/bin/python3"
    
    if not os.path.exists(backend_path):
        print(f"❌ Backend not found: {backend_path}")
        return False
    
    if not os.path.exists(python_path):
        print(f"❌ Python not found: {python_path}")
        return False
    
    # Test backend
    print("🔧 Testing backend...")
    try:
        env = os.environ.copy()
        env['PYTHONPATH'] = f"/media/prouser/Storage/PlayGround/RVU/backend:{env.get('PYTHONPATH', '')}"
        
        result = subprocess.run(
            [python_path, backend_path, '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            print(f"✅ Backend version: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ Backend test failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Backend error: {e}")
        return False

def check_models():
    """Verifică structura modelelor"""
    models_dir = Path("/media/prouser/Storage/PlayGround/RVU/models")
    
    if not models_dir.exists():
        print(f"❌ Models directory not found: {models_dir}")
        return False
    
    print(f"📁 Models directory: {models_dir}")
    
    # Listează toate modelele
    models = []
    categories = {}
    
    for item in models_dir.rglob("*"):
        if item.is_file() and item.suffix in ['.pth', '.pt', '.pkl', '.safetensors', '.engine']:
            models.append(item)
            category = item.parent.name if item.parent != models_dir else "root"
            categories.setdefault(category, []).append(item.name)
    
    print(f"📊 Found {len(models)} model files")
    
    # Afișează structura
    for category, files in categories.items():
        print(f"\n📂 {category}:")
        for file in sorted(files):
            print(f"  - {file}")
    
    return True

def test_specific_model(model_name):
    """Testează un model specific"""
    models_dir = Path("/media/prouser/Storage/PlayGround/RVU/models")
    
    print(f"\n🔍 Searching for model: {model_name}")
    
    found = []
    for item in models_dir.rglob("*"):
        if item.is_file() and model_name.lower() in item.name.lower():
            found.append(item)
            print(f"✅ Found: {item}")
    
    if not found:
        print(f"❌ Model '{model_name}' not found")
        
        # Sugestii
        print("\n💡 Suggestions:")
        print("1. Check if model exists in the models folder")
        print("2. Download model using REAL Video Enhancer original app")
        print("3. Place model in correct subfolder (upscale/, interpolate/, etc.)")
    
    return found

def main():
    print("🔬 REAL Video Enhancer - Debug Tool")
    print("=" * 50)
    
    # Check backend
    backend_ok = check_backend()
    
    # Check models
    models_ok = check_models()
    
    if backend_ok and models_ok:
        print("\n✅ All checks passed!")
        
        # Test specific models
        test_models = ["RealESRGAN", "rife", "4x"]
        for model in test_models:
            test_specific_model(model)
    else:
        print("\n❌ Some checks failed. Please fix the issues above.")
    
    print("\n🛠️ Recommended fixes:")
    print("1. Run REAL Video Enhancer original app at least once")
    print("2. Download models through the original app")
    print("3. Copy models folder structure from original app")
    print("4. Check CUDA and PyTorch installation")

if __name__ == "__main__":
    main()