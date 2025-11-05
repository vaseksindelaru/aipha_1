#!/bin/bash
# shadow/periodic_sync.sh

# Script para ejecutar sincronización periódica
# Útil como backup si los hooks fallan

REPO_PATH="/home/vaclav/Aipha_0.0.1"
cd "$REPO_PATH"

# Verificar si hay cambios recientes
MINUTES_SINCE_LAST_COMMIT=$(( ( $(date +%s) - $(git log -1 --format=%ct) ) / 60 ))

if [ $MINUTES_SINCE_LAST_COMMIT -lt 60 ]; then
    echo "[$(date)] Cambios recientes detectados, sincronizando..."
    python3 shadow/gemini_sync.py
else
    echo "[$(date)] Sin cambios recientes, omitiendo sincronización"
fi