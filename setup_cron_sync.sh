#!/bin/bash
# setup_cron_sync.sh

# Configura sincronización periódica con cron
# Ejecuta cada hora si hay cambios recientes

echo "Configurando sincronización periódica con cron..."
echo ""

# Verificar si crontab existe
if crontab -l > /dev/null 2>&1; then
    echo "Crontab existente encontrado. Agregando nueva tarea..."
else
    echo "No hay crontab existente. Creando nueva..."
fi

# Crear entrada de cron (cada hora)
CRON_JOB="0 * * * * /home/vaclav/aipha_1/shadow/periodic_sync.sh >> /home/vaclav/aipha_1/shadow/cron_sync.log 2>&1"

# Agregar al crontab si no existe
(crontab -l 2>/dev/null | grep -v "periodic_sync.sh"; echo "$CRON_JOB") | crontab -

echo "✓ Tarea de cron configurada:"
echo "  $CRON_JOB"
echo ""
echo "La sincronización se ejecutará cada hora si hay cambios en los últimos 60 minutos."
echo ""
echo "Para verificar:"
echo "  crontab -l"
echo ""
echo "Para ver logs:"
echo "  tail -f /home/vaclav/aipha_1/shadow/cron_sync.log"