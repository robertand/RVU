#!/bin/bash
# Script simplu pentru a porni WebUI-ul REAL Video Enhancer

echo "=============================================="
echo "REAL Video Enhancer WebUI"
echo "=============================================="

REAL_ROOT="/media/prouser/Storage/PlayGround/RVU"

# Verifică existența backend-ului
if [ ! -f "$REAL_ROOT/backend/rve-backend.py" ]; then
    echo "❌ EROARE: Backend-ul REAL nu a fost găsit!"
    echo "   Calea: $REAL_ROOT/backend/rve-backend.py"
    exit 1
fi

# Activează environment-ul (dacă este necesar)
if [ -f "$REAL_ROOT/activate_real_env.sh" ]; then
    source "$REAL_ROOT/activate_real_env.sh"
fi

# Pornește WebUI-ul
echo "✓ Pornire WebUI..."
echo "📱 Interfața va fi accesibilă la: http://localhost:7860"
echo "🔄 Încărcare..."

cd "$REAL_ROOT"
python real_video_enhancer_webui_final.py
