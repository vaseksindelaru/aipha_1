#!/usr/bin/env python3
"""
test_aiphalab_query.py - Test de consulta a Gemini a través del puente enhanced
"""
import sys
import os

# Añadir el directorio shadow al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'shadow'))

from aiphalab_enhanced_bridge import AiphaLabEnhancedBridge
import time

def main():
    """Simula la consulta a Gemini que el usuario realizaría"""
    
    print("🤖 PRUEBA DE CONSULTA A GEMINI - PUENTE SHADOW ENHANCED")
    print("=" * 60)
    
    # Crear el puente enhanced con la ruta correcta del proyecto
    bridge = AiphaLabEnhancedBridge(
        shadow_memory_path='./aipha_memory_storage/action_history',
        local_repo_path='../Aipha_0.0.1'
    )
    
    # Generar el contexto actualizado para la consulta
    query = "¿Cuáles archivos contiene el repositorio Aipha_0.0.1 actual?"
    
    print(f"📝 Consulta: {query}")
    print()
    
    # Obtener contexto actualizado del puente enhanced
    context = bridge.get_context_for_aiphalab(query)
    
    print("🔍 CONTEXTO GENERADO POR EL PUENTE ENHANCED:")
    print("-" * 50)
    print(context)
    print("-" * 50)
    
    # Simular lo que respondería Gemini con esta información
    print()
    print("💬 RESPUESTA SIMULADA DE GEMINI:")
    print("-" * 50)
    
    # Obtener lista de archivos del repositorio real
    repo_files = bridge.get_repository_files()
    files_list = repo_files['files']
    
    # Filtrar solo archivos principales (sin directorios de sistema)
    main_files = []
    for f in files_list:
        if f.endswith(('.py', '.json', '.md', '.txt')) and not f.startswith('.') and 'pycache' not in f:
            # Obtener solo el nombre del archivo
            filename = os.path.basename(f)
            main_files.append(filename)
    
    # Mostrar resultado en formato similar al esperado
    main_files.sort()
    
    print("vaclav@vaclav:~/Aipha_0.0.1$ ls")
    for f in main_files:
        print(f"{f}")
    
    print()
    print("✅ Verificación del Puente Enhanced:")
    print(f"📁 Total archivos detectados: {len(main_files)}")
    print(f"🎯 Información actualizada: {'✅' if repo_files['source'] == 'shadow_monitor' else '❌'}")
    
    return main_files

if __name__ == "__main__":
    files = main()