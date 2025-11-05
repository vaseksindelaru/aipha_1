#!/bin/bash
# quick_test.sh - Test rápido del sistema Shadow completo

set -e

echo "🚀 TEST RÁPIDO DEL SISTEMA SHADOW"
echo "================================="

# 1. Verificar que estamos en el directorio correcto
if [ ! -d "shadow" ]; then
    echo "❌ Error: Ejecutar desde el directorio raíz del proyecto Shadow"
    exit 1
fi

echo "✅ Ubicación correcta"

# 2. Verificar archivos críticos
files=("shadow/aiphalab_bridge.py" "shadow/log_git_event.py" "aipha_memory_storage/action_history/current_history.json")
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file existe"
    else
        echo "❌ $file no encontrado"
        exit 1
    fi
done

# 3. Test básico de importación
echo ""
echo "🐍 Probando imports..."
python3 -c "
try:
    from shadow.aiphalab_bridge import AiphaLabBridge
    print('✅ AiphaLabBridge importado correctamente')
except ImportError as e:
    print(f'❌ Error importando AiphaLabBridge: {e}')
    exit(1)

try:
    from shadow.log_git_event import GitEventLogger
    print('✅ GitEventLogger importado correctamente')
except ImportError as e:
    print(f'❌ Error importando GitEventLogger: {e}')
    exit(1)
"

# 4. Test de memoria
echo ""
echo "💾 Probando memoria Shadow..."
python3 -c "
from shadow.aiphalab_bridge import AiphaLabBridge

bridge = AiphaLabBridge('./aipha_memory_storage/action_history')
result = bridge.query_shadow_memory({'limit': 1})

if result['status'] == 'success':
    print(f'✅ Memoria operativa: {result[\"total_entries\"]} entradas totales')
else:
    print(f'❌ Error en memoria: {result[\"message\"]}')
    exit(1)
"

# 5. Test de registro de evento
echo ""
echo "📝 Probando registro de evento..."
python3 -c "
from shadow.log_git_event import GitEventLogger

logger = GitEventLogger('./aipha_memory_storage/action_history')
result = logger.log_git_event({
    'event_type': 'test',
    'commit_hash': 'quicktest123',
    'commit_message': 'Quick test event',
    'files_changed': ['test.py']
})

if result:
    print('✅ Evento registrado correctamente')
else:
    print('❌ Error registrando evento')
    exit(1)
"

# 6. Test de export de contexto
echo ""
echo "📤 Probando export de contexto..."
python3 shadow/aiphalab_bridge.py --aiphalab-context --time-range 1h --output /tmp/quick_test_context.md >/dev/null 2>&1

if [ -f "/tmp/quick_test_context.md" ]; then
    size=$(stat -c%s "/tmp/quick_test_context.md")
    if [ $size -gt 100 ]; then
        echo "✅ Contexto exportado correctamente ($size bytes)"
        rm "/tmp/quick_test_context.md"
    else
        echo "❌ Archivo de contexto muy pequeño"
        exit 1
    fi
else
    echo "❌ Error exportando contexto"
    exit 1
fi

# 7. Verificar Git hooks (si existe repo)
if [ -d ".git" ]; then
    echo ""
    echo "🔗 Verificando Git hooks..."
    if [ -f ".git/hooks/post-commit" ] && [ -x ".git/hooks/post-commit" ]; then
        echo "✅ post-commit hook instalado y ejecutable"
    else
        echo "⚠️  post-commit hook no encontrado o no ejecutable"
    fi

    if [ -f ".git/hooks/post-push" ] && [ -x ".git/hooks/post-push" ]; then
        echo "✅ post-push hook instalado y ejecutable"
    else
        echo "⚠️  post-push hook no encontrado o no ejecutable"
    fi
fi

echo ""
echo "🎉 TEST COMPLETADO EXITOSAMENTE"
echo "================================"
echo ""
echo "📊 Resumen:"
echo "✅ Imports funcionando"
echo "✅ Memoria Shadow operativa"
echo "✅ Registro de eventos funcionando"
echo "✅ Export de contexto funcionando"
echo "✅ Sistema listo para uso"
echo ""
echo "🚀 El sistema Shadow está completamente operativo!"