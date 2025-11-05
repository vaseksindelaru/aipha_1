#!/bin/bash
# shadow_aiphalab_launcher.sh - Lanzador fácil del sistema Shadow-AiphaLab

echo "🔗 SHADOW-AIPHALAB INTEGRATION SYSTEM"
echo "====================================="
echo ""

# Configuración
MEMORY_PATH="./aipha_memory_storage/action_history"
PORT=8080

# Verificar que existe la memoria de Shadow
if [ ! -d "$MEMORY_PATH" ]; then
    echo "❌ Error: No se encuentra la memoria de Shadow en $MEMORY_PATH"
    echo "   Asegúrate de que el sistema esté inicializado."
    exit 1
fi

# Mostrar opciones disponibles
echo "🚀 MODOS DISPONIBLES:"
echo ""
echo "1. 📋 Modo CLI - Generar contexto para AiphaLab (RECOMENDADO)"
echo "2. 🌐 Modo Web - Interfaz web completa"
echo "3. 🔌 Modo API - API REST para consultas"
echo "4. 🧪 Demo - Ejecutar ejemplo completo"
echo "5. ❓ Ayuda - Mostrar opciones detalladas"
echo ""

read -p "Selecciona una opción (1-5): " choice

case $choice in
    1)
        echo ""
        read -p "🔍 Ingresa tu consulta (opcional, presiona Enter para estado general): " query
        echo ""
        echo "📋 Generando contexto para AiphaLab..."
        python3 shadow/shadow_aiphalab_integration.py --mode cli --memory-path "$MEMORY_PATH" --query "$query"
        ;;
    2)
        echo ""
        echo "🌐 Iniciando interfaz web en puerto $PORT..."
        echo "📊 Accede a: http://localhost:$PORT"
        echo "⏹️  Presiona Ctrl+C para detener"
        python3 shadow/shadow_aiphalab_integration.py --mode web --memory-path "$MEMORY_PATH" --port $PORT
        ;;
    3)
        echo ""
        echo "🔌 Iniciando API REST en puerto $PORT..."
        echo "📊 Endpoints disponibles:"
        echo "   - http://localhost:$PORT/api/status"
        echo "   - http://localhost:$PORT/api/files"  
        echo "   - http://localhost:$PORT/api/context"
        echo "   - http://localhost:$PORT/api/integrity"
        echo "⏹️  Presiona Ctrl+C para detener"
        python3 shadow/shadow_aiphalab_integration.py --mode api --memory-path "$MEMORY_PATH" --port $PORT
        ;;
    4)
        echo ""
        echo "🧪 Ejecutando demo completo..."
        echo ""
        echo "1. Estado del repositorio:"
        python3 shadow/shadow_aiphalab_integration.py --mode cli --memory-path "$MEMORY_PATH" --query "estado del repositorio" --force-refresh
        echo ""
        echo "2. Análisis de integridad:"
        python3 shadow/integrity_analyzer.py
        echo ""
        echo "✅ Demo completado"
        ;;
    5)
        echo ""
        echo "❓ AYUDA DETALLADA"
        echo "=================="
        echo ""
        echo "📋 MODO CLI:"
        echo "   Genera contexto formateado para AiphaLab"
        echo "   Uso: python3 shadow/shadow_aiphalab_integration.py --mode cli --query 'consulta'"
        echo ""
        echo "🌐 MODO WEB:"
        echo "   Interfaz web completa con dashboard"
        echo "   Uso: python3 shadow/shadow_aiphalab_integration.py --mode web --port 8080"
        echo ""
        echo "🔌 MODO API:"
        echo "   API REST para integración con otras aplicaciones"
        echo "   Uso: python3 shadow/shadow_aiphalab_integration.py --mode api --port 8080"
        echo ""
        echo "📊 CARACTERÍSTICAS:"
        echo "   ✅ Análisis de integridad profunda"
        echo "   ✅ Cache inteligente para optimización"
        echo "   ✅ Acceso en tiempo real al repositorio"
        echo "   ✅ Generación automática de contexto"
        echo "   ✅ API REST para integraciones"
        echo ""
        echo "🔗 INTEGRACIÓN CON AIPHALAB:"
        echo "   1. Ejecuta el modo CLI"
        echo "   2. Copia el contexto generado"
        echo "   3. Pega en AiphaLab como contexto"
        echo "   4. AiphaLab tendrá acceso completo al estado del proyecto"
        echo ""
        ;;
    *)
        echo "❌ Opción inválida. Ejecuta el script nuevamente."
        exit 1
        ;;
esac

echo ""
echo "✅ Sistema completado. ¿Deseas hacer otra consulta? (y/n)"
read -p "Respuesta: " again

if [[ $again =~ ^[Yy]$ ]]; then
    exec "$0"  # Reiniciar el script
fi

echo "👋 ¡Gracias por usar Shadow-AiphaLab Integration!"