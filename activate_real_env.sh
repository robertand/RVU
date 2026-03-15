#!/bin/bash
# Activează mediul Python self-contained al REAL-Video-Enhancer

export REAL_ROOT="/media/prouser/Storage/PlayGround/RVU"
export PYTHONHOME="$REAL_ROOT/bin/python/python"
export PYTHONPATH="$REAL_ROOT:$REAL_ROOT/src:$REAL_ROOT/bin/python/python/lib/python3.12/site-packages:$PYTHONPATH"

# Adaugă bin la PATH
export PATH="$REAL_ROOT/bin:$REAL_ROOT/bin/python/python/bin:$PATH"

# Activează venv-ul intern (dacă există activate)
if [ -f "$REAL_ROOT/bin/python/python/bin/activate" ]; then
    source "$REAL_ROOT/bin/python/python/bin/activate"
fi

echo "✓ REAL-Video-Enhancer environment activated"
echo "  Python: $(which python)"
echo "  Python version: $(python --version)"
echo "  Root: $REAL_ROOT"
