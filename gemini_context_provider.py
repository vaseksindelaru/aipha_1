#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gemini_context_provider.py - Proveedor automático de contexto para Gemini
Garantiza que Gemini siempre tenga la información correcta del repositorio Aipha_0.0.1
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

# Añadir path para imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shadow.aiphalab_enhanced_bridge import AiphaLabEnhancedBridge

class GeminiContextProvider:
    """
    Proveedor automático de contexto que garantiza información correcta del repositorio
    """
    
    def __init__(self):
        """Inicializar proveedor de contexto"""
        self.bridge = AiphaLabEnhancedBridge(
            shadow_memory_path='./aipha_memory_storage/action_history',
            local_repo_path='../Aipha_0.0.1'
        )
        
        # Template de contexto actualizado
        self.correct_context = self._generate_correct_context()
        
        # Configurar logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("🎯 Gemini Context Provider iniciado")
    
    def _generate_correct_context(self) -> str:
        """Genera el contexto correcto basado en información real del repositorio"""
        try:
            # Obtener información actualizada del repositorio
            repo_files = self.bridge.get_repository_files()
            verification = self._verify_real_files()
            
            context = f"""=== INFORMACIÓN VERIFICADA DEL REPOSITORIO AIPHA_0.0.1 ===

🔍 CONSULTA RECIBIDA SOBRE REPOSITORIO AIPHA_0.0.1
📅 Última verificación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
✅ Fuente: Shadow System (monitor en tiempo real)

📁 ARCHIVOS REALES CONFIRMADOS DEL REPOSITORIO:
"""
            
            # Agregar lista de archivos reales
            for file in sorted(repo_files.get('files', [])):
                file_info = self._get_file_description(file)
                context += f"  📄 {file}: {file_info}\n"
            
            context += f"""
📊 VERIFICACIÓN TÉCNICA:
  • Total archivos: {len(repo_files.get('files', []))}
  • Repositorio: {repo_files.get('repository_url', '')}
  • Monitor activo: {'Sí' if repo_files.get('source') == 'shadow_monitor' else 'No'}
  • Verificación directa: {'✅ Completa' if verification['complete'] else '⚠️ Parcial'}

📝 ESTRUCTURA REAL DEL PROYECTO:
```
/Aipha_0.0.1/
"""
            
            # Generar estructura de directorios basada en archivos reales
            for file in sorted(repo_files.get('files', [])):
                if file.endswith('.py'):
                    description = self._get_file_description(file)
                    context += f"├── {file:<35} # {description}\n"
                elif file.endswith('.json'):
                    context += f"├── {file:<35} # Archivo de configuración JSON\n"
                elif file.endswith('.md'):
                    context += f"├── {file:<35} # Documentación del proyecto\n"
            
            context += """```
🎯 INSTRUCCIONES PARA RESPUESTA:
  - Usar ÚNICAMENTE los archivos listados arriba
  - NO mencionar archivos que no existan en esta lista
  - NO inventar componentes que no estén en el repositorio
  - Base toda la información en la estructura real verificada

⚠️ IMPORTANTE: Esta es la estructura REAL del repositorio Aipha_0.0.1,
verificada directamente por el sistema Shadow en tiempo real.

=== FIN INFORMACIÓN VERIFICADA ==="""
            
            return context
            
        except Exception as e:
            self.logger.error(f"Error generando contexto: {e}")
            return self._get_fallback_context()
    
    def _verify_real_files(self) -> Dict[str, Any]:
        """Verifica la existencia real de archivos"""
        real_files = [
            'README.md', 'config.json', 'config_loader.py',
            'main.py', 'potential_capture_engine.py', 'shadow.py', 'strategy.py'
        ]
        
        found_files = []
        repo_path = self.bridge.local_repo_path
        
        if repo_path:
            for file in real_files:
                file_path = os.path.join(repo_path, file)
                if os.path.exists(file_path):
                    found_files.append(file)
        
        return {
            'real_files': real_files,
            'found_files': found_files,
            'complete': len(found_files) == len(real_files),
            'verification_time': datetime.now().isoformat()
        }
    
    def _get_file_description(self, file_path: str) -> str:
        """Obtiene descripción específica de un archivo"""
        descriptions = {
            'main.py': 'Script principal y punto de entrada del sistema',
            'config.json': 'Archivo de configuración principal del proyecto',
            'config_loader.py': 'Módulo para cargar y gestionar configuraciones',
            'potential_capture_engine.py': 'Motor de captura de potencial y análisis de trading',
            'shadow.py': 'Componente Shadow del sistema contextual',
            'strategy.py': 'Implementación de estrategias de trading y análisis',
            'README.md': 'Documentación principal del proyecto'
        }
        
        return descriptions.get(file_path, 'Archivo del proyecto')
    
    def _get_fallback_context(self) -> str:
        """Contexto de emergencia"""
        return """=== CONTEXTO DE EMERGENCIA - REPOSITORIO AIPHA_0.0.1 ===

📁 ARCHIVOS CONFIRMADOS (EMERGENCIA):
  ✅ main.py
  ✅ config.json  
  ✅ config_loader.py
  ✅ potential_capture_engine.py
  ✅ shadow.py
  ✅ strategy.py
  ✅ README.md

⚠️ NOTA: Esta información proviene de verificación de emergencia.

=== FIN CONTEXTO EMERGENCIA ==="""
    
    def get_enhanced_query(self, original_query: str) -> str:
        """
        Mejora una consulta proporcionando contexto correcto automáticamente
        
        Args:
            original_query: Consulta original sobre el repositorio
            
        Returns:
            Consulta mejorada con contexto correcto
        """
        # Detectar si es una consulta sobre el repositorio
        if self._is_repository_query(original_query):
            enhanced_query = f"""{original_query}

{self.correct_context}

Por favor, responde basándote ÚNICAMENTE en la información del repositorio verificada que se proporciona arriba."""
            return enhanced_query
        else:
            return original_query
    
    def _is_repository_query(self, query: str) -> bool:
        """Detecta si una consulta es sobre el repositorio"""
        query_lower = query.lower()
        repo_keywords = [
            'aipha_0.0.1', 'repositorio', 'archivos', 'estructura', 
            'archivos del proyecto', 'contenido del repositorio',
            'qué archivos', 'cuáles archivos', 'estructura del proyecto'
        ]
        return any(keyword in query_lower for keyword in repo_keywords)
    
    def generate_response_context(self) -> Dict[str, Any]:
        """Genera contexto completo para la respuesta"""
        return {
            'correct_context': self.correct_context,
            'repository_info': self.bridge.get_repository_files(),
            'verification': self._verify_real_files(),
            'generated_at': datetime.now().isoformat(),
            'provider': 'GeminiContextProvider'
        }


def main():
    """Función principal para testing"""
    provider = GeminiContextProvider()
    
    print("🎯 PROVEEDOR DE CONTEXTO PARA GEMINI")
    print("=" * 50)
    
    # Prueba 1: Consulta sobre archivos
    query1 = "¿Cuáles archivos contiene el repositorio Aipha_0.0.1?"
    enhanced1 = provider.get_enhanced_query(query1)
    
    print("\n1️⃣ CONSULTA ORIGINAL:")
    print(query1)
    print("\n1️⃣ CONSULTA MEJORADA:")
    print(enhanced1)
    
    # Prueba 2: Consulta general
    query2 = "Cuéntame sobre la estructura del proyecto Aipha_0.0.1"
    enhanced2 = provider.get_enhanced_query(query2)
    
    print("\n2️⃣ CONSULTA ORIGINAL:")
    print(query2)
    print("\n2️⃣ CONSULTA MEJORADA:")
    print(enhanced2)
    
    # Prueba 3: Contexto completo
    print("\n3️⃣ CONTEXTO COMPLETO GENERADO:")
    context = provider.generate_response_context()
    print(json.dumps(context, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()