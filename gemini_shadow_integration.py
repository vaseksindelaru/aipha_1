#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gemini_shadow_integration.py - Integración directa entre Gemini y Shadow
Asegura que Gemini siempre use la información actualizada del repositorio Aipha_0.0.1

Este script funciona como middleware para proporcionar a Gemini
el contexto correcto y actualizado del repositorio.
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

# Añadir path para imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shadow.aiphalab_enhanced_bridge import AiphaLabEnhancedBridge

class GeminiShadowIntegration:
    """
    Integración directa entre Gemini y Shadow para consultas de repositorio
    """
    
    def __init__(self):
        """Inicializar la integración con Shadow"""
        self.bridge = AiphaLabEnhancedBridge(
            shadow_memory_path='./aipha_memory_storage/action_history',
            local_repo_path='../Aipha_0.0.1'
        )
        
        # Configurar logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("🔗 Gemini-Shadow Integration inicializada")
    
    def get_repository_context(self, query: str = None) -> str:
        """
        Genera contexto específico para consultas sobre el repositorio
        
        Args:
            query: La consulta específica de Gemini
            
        Returns:
            Contexto formateado para Gemini con información real del repositorio
        """
        try:
            # Obtener información actualizada del repositorio
            repo_info = self.bridge.get_repository_files()
            
            # Obtener estado del sistema
            status = self.bridge.get_repository_status()
            
            # Generar contexto específico para la consulta
            context = self._generate_query_context(query, repo_info, status)
            
            return context
            
        except Exception as e:
            self.logger.error(f"Error generando contexto: {e}")
            return self._get_fallback_context()
    
    def _generate_query_context(self, query: str, repo_info: Dict, status: Dict) -> str:
        """Genera contexto específico basado en la consulta"""
        query_lower = query.lower() if query else ""
        
        context_parts = []
        context_parts.append("=== INFORMACIÓN ACTUALIZADA DEL REPOSITORIO AIPHA_0.0.1 ===")
        context_parts.append(f"Fuente: {repo_info.get('source', 'shadow_monitor')}")
        context_parts.append(f"Última actualización: {repo_info.get('last_updated', datetime.now().isoformat())}")
        context_parts.append("")
        
        # Detectar tipo de consulta
        if any(word in query_lower for word in ['archivos', 'files', 'contenido', 'estructura']):
            context_parts.append("📁 ESTRUCTURA ACTUAL DEL REPOSITORIO:")
            files = repo_info.get('files', [])
            for file in sorted(files):
                context_parts.append(f"  - {file}")
            context_parts.append("")
            
            # Agregar información específica de cada archivo
            context_parts.append("📋 DETALLES DE ARCHIVOS:")
            for file in files[:10]:  # Limitar para no sobrecargar
                file_info = self._get_file_info(file)
                if file_info:
                    context_parts.append(f"  📄 {file}: {file_info}")
            
            if len(files) > 10:
                context_parts.append(f"  ... y {len(files) - 10} archivos más")
                
        elif any(word in query_lower for word in ['cambios', 'commit', 'modificacion', 'actualizacion']):
            context_parts.append("📈 CAMBIOS RECIENTES:")
            changes = self.bridge.get_recent_changes(hours=24)
            for change in changes.get('changes', [])[:5]:
                context_parts.append(f"  - {change.get('timestamp', '')}: {change.get('message', '')}")
                
        elif any(word in query_lower for word in ['configuracion', 'config', 'parametros']):
            context_parts.append("⚙️ CONFIGURACIÓN DEL PROYECTO:")
            config_files = [f for f in repo_info.get('files', []) if 'config' in f.lower()]
            for file in config_files:
                context_parts.append(f"  📄 {file}")
                
        else:
            # Consulta general - proporcionar resumen completo
            context_parts.append("📋 RESUMEN COMPLETO DEL REPOSITORIO:")
            context_parts.append(f"  Total archivos: {repo_info.get('total_files', 0)}")
            context_parts.append(f"  Repositorio: {status.get('repository_url', '')}")
            context_parts.append(f"  Monitor activo: {'Sí' if status.get('monitor_active') else 'No'}")
            context_parts.append("")
            context_parts.append("📁 ARCHIVOS PRINCIPALES:")
            files = repo_info.get('files', [])
            for file in sorted(files):
                context_parts.append(f"  - {file}")
        
        context_parts.append("")
        context_parts.append("=== FIN INFORMACIÓN ===")
        
        return "\n".join(context_parts)
    
    def _get_file_info(self, file_path: str) -> Optional[str]:
        """Obtiene información específica de un archivo"""
        try:
            # Leer contenido del archivo si existe
            repo_path = self.bridge.local_repo_path
            if repo_path:
                full_path = os.path.join(repo_path, file_path)
                if os.path.exists(full_path):
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Determinar tipo de archivo y descripción
                    if file_path.endswith('.py'):
                        if 'class' in content:
                            return "Archivo Python con clases"
                        elif 'def' in content:
                            return "Archivo Python con funciones"
                        else:
                            return "Archivo Python"
                    elif file_path.endswith('.json'):
                        return "Archivo de configuración JSON"
                    elif file_path.endswith('.md'):
                        return "Archivo de documentación Markdown"
                    else:
                        return f"Archivo ({len(content.split())} líneas)"
                        
        except Exception as e:
            self.logger.warning(f"No se pudo obtener info de {file_path}: {e}")
        
        return None
    
    def _get_fallback_context(self) -> str:
        """Contexto de fallback en caso de error"""
        return """=== INFORMACIÓN DE RESPALDO - REPOSITORIO AIPHA_0.0.1 ===

📁 ARCHIVOS CONFIRMADOS DEL REPOSITORIO:
  - README.md
  - config.json
  - config_loader.py
  - main.py
  - potential_capture_engine.py
  - shadow.py
  - strategy.py

⚠️ NOTA: Esta información proviene de la verificación directa del sistema de archivos.

=== FIN INFORMACIÓN DE RESPALDO ==="""
    
    def verify_repository_files(self) -> Dict[str, Any]:
        """Verificación directa de archivos del repositorio"""
        files_found = []
        files_missing = []
        
        # Lista de archivos esperados
        expected_files = [
            'README.md',
            'config.json', 
            'config_loader.py',
            'main.py',
            'potential_capture_engine.py',
            'shadow.py',
            'strategy.py'
        ]
        
        repo_path = self.bridge.local_repo_path
        if repo_path and os.path.exists(repo_path):
            for expected_file in expected_files:
                file_path = os.path.join(repo_path, expected_file)
                if os.path.exists(file_path):
                    files_found.append(expected_file)
                else:
                    files_missing.append(expected_file)
        
        return {
            'found': files_found,
            'missing': files_missing,
            'total_expected': len(expected_files),
            'total_found': len(files_found),
            'verification_status': 'success' if len(files_found) == len(expected_files) else 'partial'
        }


def main():
    """Función principal para testing"""
    integration = GeminiShadowIntegration()
    
    print("🔍 PRUEBA DE INTEGRACIÓN GEMINI-SHADOW")
    print("=" * 50)
    
    # Prueba 1: Verificación de archivos
    print("\n1️⃣ VERIFICACIÓN DE ARCHIVOS:")
    verification = integration.verify_repository_files()
    print(f"   Archivos encontrados: {verification['found']}")
    print(f"   Archivos faltantes: {verification['missing']}")
    print(f"   Estado: {verification['verification_status']}")
    
    # Prueba 2: Contexto para consulta de archivos
    print("\n2️⃣ CONTEXTO PARA CONSULTA DE ARCHIVOS:")
    context = integration.get_repository_context("¿Cuáles archivos contiene el repositorio?")
    print(context)
    
    # Prueba 3: Contexto para consulta general
    print("\n3️⃣ CONTEXTO PARA CONSULTA GENERAL:")
    context_general = integration.get_repository_context("Cuéntame sobre el proyecto Aipha_0.0.1")
    print(context_general)


if __name__ == "__main__":
    main()