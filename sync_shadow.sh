#!/bin/bash
# Script para mantener AiphaShadow sincronizado

while true; do
    echo "Sincronizando AiphaShadow..."
    python aipha_shadow.py --sync ./aipha_1
    sleep 60  # Sincroniza cada minuto
done