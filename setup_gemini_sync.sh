#!/bin/bash
# setup_gemini_sync.sh

echo "=========================================="
echo "CONFIGURACIÓN DE SINCRONIZACIÓN GEMINI"
echo "=========================================="
echo ""

cd /home/vaclav/Aipha_0.0.1

# 1. Crear directorio shadow si no existe
mkdir -p shadow

# 2. Copiar scripts
echo "1. Instalando scripts..."
# (Aquí irían los scripts que creamos arriba)

# 3. Hacer ejecutables
chmod +x generate_full_context.sh
chmod +x shadow/gemini_sync.py
chmod +x shadow/periodic_sync.sh

# 4. Configurar Git hook
echo "2. Configurando Git hook..."
cp .git/hooks/post-commit .git/hooks/post-commit.backup 2>/dev/null
cat > .git/hooks/post-commit << 'HOOK'
#!/bin/bash
# .git/hooks/post-commit-with-gemini

# Hook mejorado que:
# 1. Registra en Shadow (como antes)
# 2. Sincroniza con Gemini automáticamente

REPO_ROOT=$(git rev-parse --show-toplevel)

# 1. Registro en Shadow (funcionalidad existente)
COMMIT_HASH=$(git rev-parse HEAD)
COMMIT_MSG=$(git log -1 --pretty=%B)
CHANGED_FILES=$(git diff-tree --no-commit-id --name-only -r HEAD | tr '\n' ',')

if [ -f "$REPO_ROOT/shadow/log_git_event.py" ]; then
    python3 "$REPO_ROOT/shadow/log_git_event.py" \
        --event "commit" \
        --commit-hash "$COMMIT_HASH" \
        --message "$COMMIT_MSG" \
        --files "$CHANGED_FILES"

    echo "✓ Evento registrado en Shadow"
fi

# 2. Sincronización con Gemini (NUEVA FUNCIONALIDAD)
if [ -f "$REPO_ROOT/shadow/gemini_sync.py" ]; then
    # Solo sincronizar si la API key está configurada
    if [ -n "$GEMINI_API_KEY" ]; then
        echo "🔄 Sincronizando con Gemini..."
        python3 "$REPO_ROOT/shadow/gemini_sync.py"
        if [ $? -eq 0 ]; then
            echo "✓ Contexto sincronizado con shadowAipha_1.0"
        else
            echo "⚠ Advertencia: Sincronización con Gemini falló"
            echo "  El commit fue registrado en Shadow local"
        fi
    else
        echo "ℹ Sincronización con Gemini deshabilitada (GEMINI_API_KEY no configurada)"
    fi
fi

exit 0
HOOK
chmod +x .git/hooks/post-commit

# 5. Solicitar API key
echo ""
echo "3. Configuración de API Key de Gemini:"
echo ""
echo "Obtén tu API key en: https://makersuite.google.com/app/apikey"
echo ""
read -p "Ingresa tu GEMINI_API_KEY: " api_key

if [ -n "$api_key" ]; then
    # Guardar en .bashrc
    if ! grep -q "GEMINI_API_KEY" ~/.bashrc; then
        echo "export GEMINI_API_KEY=\"$api_key\"" >> ~/.bashrc
        echo "✓ API key guardada en ~/.bashrc"
    fi

    # Exportar para sesión actual
    export GEMINI_API_KEY="$api_key"

    # Probar conexión
    echo ""
    echo "4. Probando conexión con Gemini..."
    python3 shadow/gemini_sync.py --test

    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ Configuración exitosa"

        # Hacer sincronización inicial
        echo ""
        echo "5. Ejecutando sincronización inicial..."
        python3 shadow/gemini_sync.py

        echo ""
        echo "=========================================="
        echo "✓✓✓ INSTALACIÓN COMPLETADA ✓✓✓"
        echo "=========================================="
        echo ""
        echo "Ahora Gemini se sincronizará automáticamente después de cada commit."
        echo ""
        echo "Comandos útiles:"
        echo "  - Sincronización manual: python3 shadow/gemini_sync.py"
        echo "  - Ver logs: tail -f shadow/gemini_sync.log"
        echo "  - Test de API: python3 shadow/gemini_sync.py --test"
    fi
else
    echo "⚠ API key no configurada. La sincronización automática no funcionará."
    echo "  Configura manualmente: export GEMINI_API_KEY=\"tu-api-key\""
fi

# 6. Configurar cron (opcional)
echo ""
read -p "¿Configurar sincronización periódica cada hora? (y/n): " setup_cron

if [ "$setup_cron" = "y" ]; then
    (crontab -l 2>/dev/null; echo "0 * * * * /home/vaclav/Aipha_0.0.1/shadow/periodic_sync.sh >> /home/vaclav/Aipha_0.0.1/shadow/cron.log 2>&1") | crontab -
    echo "✓ Cron configurado (sincronización cada hora)"
fi