#!/bin/bash
# HAL Agent-Debug Setup Script
# ============================
# Run this script to install all dependencies in one go:
#   bash setup.sh
#
# Or for a specific benchmark:
#   bash setup.sh colbench
#   bash setup.sh corebench
#   bash setup.sh scicode

set -e

echo "================================================"
echo "HAL Agent-Debug Environment Setup"
echo "================================================"

# Install base requirements
echo ""
echo "[1/4] Installing base requirements..."
pip install -r requirements.txt

# Install HAL harness (editable)
echo ""
echo "[2/4] Installing HAL harness..."
pip install -e hal-harness/

# Install docent SDK (editable)
echo ""
echo "[3/4] Installing docent SDK..."
pip install -e docent/docent/
pip install -e docent/

# Install benchmark-specific dependencies if requested
BENCHMARK="${1:-}"
if [ -n "$BENCHMARK" ]; then
    echo ""
    echo "[4/4] Installing $BENCHMARK dependencies..."
    pip install -e "hal-harness/[$BENCHMARK]"
else
    echo ""
    echo "[4/4] No specific benchmark requested."
    echo "     To install benchmark dependencies later, run:"
    echo "       pip install -e hal-harness/[colbench]"
    echo "       pip install -e hal-harness/[corebench]"
    echo "       pip install -e hal-harness/[scicode]"
    echo "       pip install -e hal-harness/[swebench]"
    echo "       pip install -e hal-harness/[usaco]"
    echo "       pip install -e hal-harness/[assistantbench]"
fi

echo ""
echo "================================================"
echo "Setup complete!"
echo "================================================"
echo ""
echo "Quick verification:"
python -c "import hal; print(f'  HAL harness: OK')" 2>/dev/null || echo "  HAL harness: Failed to import"
python -c "import docent; print(f'  docent SDK: OK')" 2>/dev/null || echo "  docent SDK: Failed to import"
python -c "import litellm; print(f'  litellm: OK')" 2>/dev/null || echo "  litellm: Failed to import"
python -c "import openai; print(f'  openai: OK')" 2>/dev/null || echo "  openai: Failed to import"
echo ""
