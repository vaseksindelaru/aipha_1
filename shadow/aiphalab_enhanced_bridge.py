#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aiphalab_enhanced_bridge.py - Puente mejorado entre Shadow y AiphaLab

Este módulo proporciona una interfaz mejorada para que AiphaLab pueda acceder
a información actualizada del repositorio a través del sistema Shadow.

Características:
- Acceso directo a información del repositorio en tiempo real
- Contexto actualizado de cambios recientes
- Información de archivos y estructura del proyecto
- Integración con memoria Shadow para contexto histórico

Autor: Shadow System
Versión: 2.0.0
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

# Añadir directorio padre al path para imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shadow.aiphalab_bridge import AiphaLabBridge
from shadow.enhanced_github_monitor import EnhancedGitHubRepositoryMonitor

class AiphaLabEnhancedBridge:
    """
    Puente mejorado entre Shadow Memory y AiphaLab con acceso directo al repositorio
    """

    def __init__(self, shadow_memory_path: str, repo_url: str = None, local_repo_path: str = None):
        """
        Inicializa el puente mejorado

        Args:
            shadow_memory_path: Ruta a la memoria de Shadow
            repo_url: URL del repositorio (opcional, se detecta automáticamente)
            local_repo_path: Ruta local del repositorio (opcional, se detecta automáticamente)
        """
        self.shadow_memory_path = shadow_memory_path
        self.basic_bridge = AiphaLabBridge(shadow_memory_path)

        # Detectar configuración del repositorio
        self.repo_url = repo_url or self._detect_repo_url()
        self.local_repo_path = local_repo_path or self._detect_local_repo_path()

        # Inicializar monitor si es posible
        self.monitor = None
        if self.local_repo_path and os.path.exists(self.local_repo_path):
            try:
                self.monitor = EnhancedGitHubRepositoryMonitor(
                    repo_url=self.repo_url,
                    local_path=self.local_repo_path,
                    shadow_memory_path=shadow_memory_path
                )
            except Exception as e:
                logging.warning(f"No se pudo inicializar monitor: {e}")

        # Configurar logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _detect_repo_url(self) -> str:
        """Detecta la URL del repositorio desde la memoria de Shadow"""
        try:
            # Buscar en la memoria de Shadow información del repositorio
            result = self.basic_bridge.query_shadow_memory({
                'category': 'code_understanding',
                'limit': 10
            })

            for entry in result.get('data', []):
                details = entry.get('details', {})
                if isinstance(details, dict) and 'repository' in details:
                    return f"https://github.com/vaseksindelaru/{details['repository']}.git"

        except Exception as e:
            self.logger.warning(f"Error detectando repo URL: {e}")

        # Fallback
        return "https://github.com/vaseksindelaru/aipha_0.0.1.git"

    def _detect_local_repo_path(self) -> str:
        """Detecta la ruta local del repositorio"""
        # Buscar directorios con .git
        current_dir = Path.cwd()
        for parent in [current_dir] + list(current_dir.parents):
            if (parent / '.git').exists():
                return str(parent)

        # Fallback: buscar en directorios comunes
        common_paths = [
            "../Aipha_0.0.1",
            "./monitored_repos/aipha_0.0.1",
            "/home/vaclav/Aipha_0.0.1"
        ]

        for path in common_paths:
            if os.path.exists(path) and os.path.exists(os.path.join(path, '.git')):
                return path

        return None

    def get_repository_files(self) -> Dict[str, Any]:
        """
        Obtiene la lista actualizada de archivos del repositorio

        Returns:
            Dict con información de archivos y estructura
        """
        result = {
            'repository_url': self.repo_url,
            'last_updated': datetime.now().isoformat(),
            'files': [],
            'structure': {},
            'total_files': 0,
            'source': 'unknown'
        }

        try:
            # Método 1: Usar monitor si está disponible
            if self.monitor:
                status = self.monitor.get_repository_status()
                if status.get('codebase_analyzed'):
                    # Obtener información del análisis de código
                    code_analysis = self._get_codebase_analysis()
                    if code_analysis:
                        # Usar all_files si está disponible, sino fallback a codebase_summary
                        all_files = code_analysis.get('all_files', [])
                        if not all_files:
                            all_files = list(code_analysis.get('codebase_summary', {}).keys())
                        
                        result.update({
                            'files': all_files,
                            'structure': code_analysis.get('codebase_summary', {}),
                            'total_files': len(all_files),
                            'source': 'shadow_monitor'
                        })
                        return result

            # Método 2: Leer directamente del repositorio local
            if self.local_repo_path and os.path.exists(self.local_repo_path):
                files = self._scan_repository_files(self.local_repo_path)
                result.update({
                    'files': files,
                    'total_files': len(files),
                    'source': 'direct_scan'
                })
                return result

            # Método 3: Buscar en memoria de Shadow
            shadow_files = self._get_files_from_shadow_memory()
            if shadow_files:
                result.update({
                    'files': shadow_files,
                    'total_files': len(shadow_files),
                    'source': 'shadow_memory'
                })
                return result

        except Exception as e:
            self.logger.error(f"Error obteniendo archivos del repositorio: {e}")
            result['error'] = str(e)

        return result

    def _scan_repository_files(self, repo_path: str) -> List[str]:
        """Escanea archivos del repositorio local"""
        files = []
        exclude_patterns = ['.git', '__pycache__', '.pytest_cache', '*.pyc']

        for root, dirs, files_in_dir in os.walk(repo_path):
            # Filtrar directorios excluidos
            dirs[:] = [d for d in dirs if not any(pattern in d for pattern in exclude_patterns)]

            for file in files_in_dir:
                # Filtrar archivos excluidos
                if not any(pattern in file for pattern in exclude_patterns):
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, repo_path)
                    files.append(rel_path)

        return sorted(files)

    def _get_codebase_analysis(self) -> Optional[Dict[str, Any]]:
        """Obtiene el análisis de código más reciente de Shadow"""
        try:
            result = self.basic_bridge.query_shadow_memory({
                'component': 'code_understanding',
                'limit': 1
            })

            if result.get('data'):
                entry = result['data'][0]
                return entry.get('details', {})
        except Exception as e:
            self.logger.error(f"Error obteniendo análisis de código: {e}")

        return None

    def _get_files_from_shadow_memory(self) -> List[str]:
        """Obtiene lista de archivos desde memoria de Shadow"""
        try:
            # Buscar entradas de análisis de código
            result = self.basic_bridge.query_shadow_memory({
                'component': 'code_understanding',
                'limit': 5
            })

            all_files = set()
            for entry in result.get('data', []):
                details = entry.get('details', {})
                if isinstance(details, dict):
                    summary = details.get('codebase_summary', {})
                    all_files.update(summary.keys())

            return sorted(list(all_files))

        except Exception as e:
            self.logger.error(f"Error obteniendo archivos de memoria: {e}")
            return []

    def get_recent_changes(self, hours: int = 24) -> Dict[str, Any]:
        """
        Obtiene cambios recientes con información detallada

        Args:
            hours: Número de horas hacia atrás para buscar cambios

        Returns:
            Dict con cambios recientes y metadatos
        """
        result = {
            'time_range': f'{hours}h',
            'changes': [],
            'total_changes': 0,
            'last_updated': datetime.now().isoformat(),
            'source': 'shadow_memory'
        }

        try:
            # Obtener cambios de Shadow memory
            changes_result = self.basic_bridge.query_shadow_memory({
                'category': 'GIT_EVENT',
                'time_range': f'{hours}h',
                'limit': 50
            })

            changes = []
            for entry in changes_result.get('data', []):
                change_info = {
                    'timestamp': entry.get('timestamp'),
                    'type': entry.get('data_payload', {}).get('event_type'),
                    'commit_hash': entry.get('data_payload', {}).get('commit_hash', '')[:8],
                    'message': entry.get('data_payload', {}).get('commit_message', ''),
                    'files': entry.get('data_payload', {}).get('files_changed', [])
                }
                changes.append(change_info)

            result.update({
                'changes': changes,
                'total_changes': len(changes)
            })

        except Exception as e:
            self.logger.error(f"Error obteniendo cambios recientes: {e}")
            result['error'] = str(e)

        return result

    def get_file_content(self, file_path: str) -> Dict[str, Any]:
        """
        Obtiene el contenido de un archivo específico

        Args:
            file_path: Ruta relativa del archivo

        Returns:
            Dict con contenido del archivo y metadatos
        """
        result = {
            'file_path': file_path,
            'found': False,
            'content': None,
            'size': 0,
            'last_modified': None,
            'source': 'unknown'
        }

        try:
            # Método 1: Leer del repositorio local
            if self.local_repo_path:
                full_path = os.path.join(self.local_repo_path, file_path)
                if os.path.exists(full_path):
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    stat = os.stat(full_path)
                    result.update({
                        'found': True,
                        'content': content,
                        'size': stat.st_size,
                        'last_modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        'source': 'local_repository'
                    })
                    return result

            # Método 2: Buscar en memoria de Shadow (limitado)
            shadow_content = self._get_file_content_from_shadow(file_path)
            if shadow_content:
                result.update({
                    'found': True,
                    'content': shadow_content,
                    'source': 'shadow_memory'
                })
                return result

        except Exception as e:
            self.logger.error(f"Error obteniendo contenido de archivo {file_path}: {e}")
            result['error'] = str(e)

        return result

    def _get_file_content_from_shadow(self, file_path: str) -> Optional[str]:
        """Busca contenido de archivo en memoria de Shadow (funcionalidad limitada)"""
        # Esta es una implementación básica - en producción se necesitaría
        # almacenar snapshots de archivos en la memoria
        return None

    def get_repository_status(self) -> Dict[str, Any]:
        """
        Obtiene el estado general del repositorio

        Returns:
            Dict con estado del repositorio y metadatos
        """
        status = {
            'repository_url': self.repo_url,
            'local_path': self.local_repo_path,
            'monitor_active': self.monitor is not None,
            'last_check': datetime.now().isoformat(),
            'capabilities': []
        }

        # Verificar capacidades
        if self.monitor:
            status['capabilities'].append('real_time_monitoring')
            status['capabilities'].append('code_analysis')

        if self.local_repo_path and os.path.exists(self.local_repo_path):
            status['capabilities'].append('direct_file_access')

        if self.basic_bridge:
            status['capabilities'].append('shadow_memory_access')

        # Obtener información adicional del monitor si está disponible
        if self.monitor:
            try:
                monitor_status = self.monitor.get_repository_status()
                status.update({
                    'tracked_files': monitor_status.get('tracked_files', 0),
                    'last_commit': monitor_status.get('last_commit_hash', '')[:8],
                    'recent_commits': len(monitor_status.get('recent_commits', []))
                })
            except Exception as e:
                self.logger.warning(f"Error obteniendo status del monitor: {e}")

        return status

    def get_context_for_aiphalab(self, query: str = None) -> str:
        """
        Genera contexto completo para AiphaLab con información actualizada

        Args:
            query: Consulta específica (opcional)

        Returns:
            String con contexto formateado para AiphaLab
        """
        context_parts = []

        # Header
        context_parts.append("# 🔍 CONTEXTO SHADOW MEMORY - ENHANCED")
        context_parts.append(f"**Consulta:** {query or 'Información actualizada del repositorio'}")
        context_parts.append(f"**Timestamp:** {datetime.now().isoformat()}")
        context_parts.append("")

        # Información del repositorio
        repo_info = self.get_repository_files()
        context_parts.append("## 📁 INFORMACIÓN DEL REPOSITORIO")
        context_parts.append(f"- **URL:** {repo_info['repository_url']}")
        context_parts.append(f"- **Archivos totales:** {repo_info['total_files']}")
        context_parts.append(f"- **Fuente:** {repo_info['source']}")
        context_parts.append(f"- **Última actualización:** {repo_info['last_updated']}")
        context_parts.append("")

        # Lista de archivos
        if repo_info['files']:
            context_parts.append("### Archivos del repositorio:")
            for file in repo_info['files'][:20]:  # Limitar a 20 archivos
                context_parts.append(f"- `{file}`")
            if len(repo_info['files']) > 20:
                context_parts.append(f"- ... y {len(repo_info['files']) - 20} archivos más")
            context_parts.append("")

        # Cambios recientes
        recent_changes = self.get_recent_changes(hours=24)
        if recent_changes['changes']:
            context_parts.append("## 📈 CAMBIOS RECIENTES (24h)")
            context_parts.append(f"**Total de cambios:** {recent_changes['total_changes']}")
            context_parts.append("")

            for change in recent_changes['changes'][:10]:  # Últimos 10 cambios
                context_parts.append(f"### {change['timestamp'][:19]}")
                context_parts.append(f"- **Tipo:** {change['type']}")
                context_parts.append(f"- **Commit:** {change['commit_hash']}")
                context_parts.append(f"- **Mensaje:** {change['message']}")
                if change['files']:
                    context_parts.append(f"- **Archivos:** {', '.join(change['files'])}")
                context_parts.append("")

        # Estado del sistema
        repo_status = self.get_repository_status()
        context_parts.append("## ⚙️ ESTADO DEL SISTEMA")
        context_parts.append(f"- **Monitor activo:** {'✅' if repo_status['monitor_active'] else '❌'}")
        context_parts.append(f"- **Acceso directo:** {'✅' if 'direct_file_access' in repo_status['capabilities'] else '❌'}")
        context_parts.append(f"- **Capacidades:** {', '.join(repo_status['capabilities'])}")
        context_parts.append("")

        # Footer con instrucciones
        context_parts.append("---")
        context_parts.append("💡 **Información actualizada proporcionada por Shadow System**")
        context_parts.append("🔄 **Datos obtenidos en tiempo real del repositorio**")

        return "\n".join(context_parts)


def main():
    """Función principal para testing"""
    import argparse

    parser = argparse.ArgumentParser(description='AiphaLab Enhanced Bridge')
    parser.add_argument('--memory-path', default='./aipha_memory_storage/action_history',
                       help='Ruta a memoria Shadow')
    parser.add_argument('--repo-url', help='URL del repositorio')
    parser.add_argument('--local-repo', help='Ruta local del repositorio')
    parser.add_argument('--query', choices=['files', 'changes', 'status', 'context'],
                       default='context', help='Tipo de consulta')

    args = parser.parse_args()

    # Crear puente
    bridge = AiphaLabEnhancedBridge(
        shadow_memory_path=args.memory_path,
        repo_url=args.repo_url,
        local_repo_path=args.local_repo
    )

    # Ejecutar consulta
    if args.query == 'files':
        result = bridge.get_repository_files()
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif args.query == 'changes':
        result = bridge.get_recent_changes()
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif args.query == 'status':
        result = bridge.get_repository_status()
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif args.query == 'context':
        context = bridge.get_context_for_aiphalab()
        print(context)


if __name__ == "__main__":
    main()