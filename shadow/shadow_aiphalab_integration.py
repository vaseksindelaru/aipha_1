#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
shadow_aiphalab_integration.py - Sistema completo de integración Shadow-AiphaLab

Este módulo proporciona una solución integral para que AiphaLab pueda acceder
a información actualizada del repositorio Aipha_0.0.1 a través del sistema Shadow.

Características:
- API REST para consultas en tiempo real
- Integración con analizador de integridad profunda
- Cache inteligente para optimizar rendimiento
- Detección automática de repositorio
- Interface web opcional
- Generación automática de contexto

Autor: Shadow System
Versión: 3.0.0
"""

import os
import sys
import json
import logging
import hashlib
import sqlite3
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from flask import Flask, jsonify, request, render_template_string
import argparse

# Añadir directorio padre al path para imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shadow.aiphalab_enhanced_bridge import AiphaLabEnhancedBridge
from shadow.integrity_analyzer import IntegrityAnalyzer

class ShadowAiphaLabIntegration:
    """
    Sistema completo de integración entre Shadow y AiphaLab
    """

    def __init__(self, shadow_memory_path: str, cache_db_path: str = None):
        """
        Inicializa el sistema de integración
        
        Args:
            shadow_memory_path: Ruta a la memoria de Shadow
            cache_db_path: Ruta a la base de datos de cache (opcional)
        """
        self.shadow_memory_path = shadow_memory_path
        self.cache_db_path = cache_db_path or os.path.join(shadow_memory_path, 'shadow_aiphalab_cache.db')
        
        # Inicializar componentes
        self.enhanced_bridge = AiphaLabEnhancedBridge(shadow_memory_path)
        self.integrity_analyzer = IntegrityAnalyzer(
            repo_path="../Aipha_0.0.1",
            shadow_memory_path=shadow_memory_path
        )
        
        # Configurar cache
        self.cache_enabled = True
        self.cache_timeout = 300  # 5 minutos
        self._init_cache()
        
        # Configurar logging
        self._setup_logging()
        
        # Variables de estado
        self.last_analysis = None
        self.repository_status = None
        self._lock = threading.Lock()
        
        self.logger.info("🔗 Shadow-AiphaLab Integration inicializado")

    def _setup_logging(self):
        """Configura el sistema de logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def _init_cache(self):
        """Inicializa la base de datos de cache"""
        try:
            os.makedirs(os.path.dirname(self.cache_db_path), exist_ok=True)
            
            conn = sqlite3.connect(self.cache_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS cache (
                    cache_key TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    expires_at REAL NOT NULL
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS repository_files (
                    file_path TEXT PRIMARY KEY,
                    content_hash TEXT NOT NULL,
                    content TEXT NOT NULL,
                    size INTEGER,
                    last_modified REAL,
                    timestamp REAL NOT NULL
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.warning(f"Error inicializando cache: {e}")
            self.cache_enabled = False

    def get_current_repository_state(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Obtiene el estado actual del repositorio con cache inteligente
        
        Args:
            force_refresh: Forzar actualización del cache
            
        Returns:
            Dict con estado completo del repositorio
        """
        cache_key = "repository_state"
        
        # Intentar obtener del cache
        if not force_refresh:
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                return cached_result
        
        # Obtener estado fresco
        with self._lock:
            state = {
                'timestamp': datetime.now().isoformat(),
                'repository_url': self.enhanced_bridge.repo_url,
                'files': [],
                'integrity_score': 0,
                'recent_changes': [],
                'analysis_summary': {},
                'status': 'unknown'
            }
            
            try:
                # 1. Obtener archivos del repositorio
                repo_files = self.enhanced_bridge.get_repository_files()
                state['files'] = repo_files.get('files', [])
                state['status'] = repo_files.get('source', 'unknown')
                
                # 2. Realizar análisis de integridad si es necesario
                if force_refresh or not self.last_analysis or \
                   (datetime.now() - self.last_analysis).total_seconds() > 3600:  # 1 hora
                    self.logger.info("🔍 Realizando análisis de integridad...")
                    integrity_result = self.integrity_analyzer.perform_deep_integrity_analysis()
                    self.last_analysis = datetime.now()
                    
                    state['integrity_score'] = integrity_result.get('integrity_score', 0)
                    state['analysis_summary'] = {
                        'score': integrity_result.get('integrity_score', 0),
                        'files_analyzed': len(integrity_result.get('file_checksums', {})),
                        'issues_found': len(integrity_result.get('issues_found', [])),
                        'structure_score': integrity_result.get('structure_validation', {}).get('structure_score', 0)
                    }
                else:
                    # Usar análisis previo si está disponible
                    if hasattr(self, '_last_integrity_result'):
                        integrity_result = self._last_integrity_result
                        state['integrity_score'] = integrity_result.get('integrity_score', 0)
                        state['analysis_summary'] = {
                            'score': integrity_result.get('integrity_score', 0),
                            'files_analyzed': len(integrity_result.get('file_checksums', {})),
                            'issues_found': len(integrity_result.get('issues_found', [])),
                            'structure_score': integrity_result.get('structure_validation', {}).get('structure_score', 0)
                        }
                        self._last_integrity_result = integrity_result
                
                # 3. Obtener cambios recientes
                recent_changes = self.enhanced_bridge.get_recent_changes(hours=24)
                state['recent_changes'] = recent_changes.get('changes', [])
                
                # 4. Guardar en cache
                if self.cache_enabled:
                    self._save_to_cache(cache_key, state, timeout=self.cache_timeout)
                
            except Exception as e:
                self.logger.error(f"Error obteniendo estado del repositorio: {e}")
                state['error'] = str(e)
        
        return state

    def get_file_content_with_analysis(self, file_path: str) -> Dict[str, Any]:
        """
        Obtiene el contenido de un archivo con análisis
        
        Args:
            file_path: Ruta relativa del archivo
            
        Returns:
            Dict con contenido y análisis del archivo
        """
        cache_key = f"file_content_{hashlib.md5(file_path.encode()).hexdigest()}"
        
        # Intentar obtener del cache
        if self.cache_enabled:
            cached_result = self._get_from_cache(cache_key)
            if cached_result and cached_result.get('content'):
                return cached_result
        
        # Obtener contenido fresco
        result = {
            'file_path': file_path,
            'found': False,
            'content': None,
            'analysis': {},
            'size': 0,
            'last_modified': None,
            'error': None
        }
        
        try:
            # Obtener contenido del repositorio
            file_content = self.enhanced_bridge.get_file_content(file_path)
            
            if file_content.get('found'):
                result.update({
                    'found': True,
                    'content': file_content.get('content', ''),
                    'size': file_content.get('size', 0),
                    'last_modified': file_content.get('last_modified'),
                    'analysis': self._analyze_file_content(file_path, file_content.get('content', ''))
                })
                
                # Guardar en cache
                if self.cache_enabled:
                    self._save_to_cache(cache_key, result, timeout=600)  # 10 minutos
            else:
                result['error'] = 'Archivo no encontrado'
                
        except Exception as e:
            self.logger.error(f"Error obteniendo contenido de {file_path}: {e}")
            result['error'] = str(e)
        
        return result

    def _analyze_file_content(self, file_path: str, content: str) -> Dict[str, Any]:
        """Analiza el contenido de un archivo"""
        analysis = {
            'language': 'unknown',
            'size_bytes': len(content.encode('utf-8')),
            'lines': len(content.split('\n')),
            'complexity': 'simple'
        }
        
        try:
            if file_path.endswith('.py'):
                analysis['language'] = 'python'
                # Análisis básico de Python
                functions = content.count('def ')
                classes = content.count('class ')
                imports = content.count('import ') + content.count('from ')
                
                analysis.update({
                    'functions': functions,
                    'classes': classes,
                    'imports': imports,
                    'complexity': 'high' if functions > 10 or classes > 5 else 'medium' if functions > 5 else 'simple'
                })
                
            elif file_path.endswith('.json'):
                analysis['language'] = 'json'
                try:
                    json.loads(content)
                    analysis['valid_json'] = True
                    analysis['structure'] = 'valid'
                except:
                    analysis['valid_json'] = False
                    analysis['structure'] = 'invalid'
                    
            elif file_path.endswith('.md'):
                analysis['language'] = 'markdown'
                headers = content.count('#')
                code_blocks = content.count('```')
                analysis.update({
                    'headers': headers,
                    'code_blocks': code_blocks
                })
        
        except Exception as e:
            self.logger.warning(f"Error analizando archivo {file_path}: {e}")
            analysis['analysis_error'] = str(e)
        
        return analysis

    def generate_complete_context(self, query: str = "") -> str:
        """
        Genera contexto completo para AiphaLab
        
        Args:
            query: Consulta específica (opcional)
            
        Returns:
            String con contexto formateado para AiphaLab
        """
        # Obtener estado actual
        repo_state = self.get_current_repository_state()
        
        context_parts = []
        
        # Header principal
        context_parts.append("# 🔍 CONTEXTO COMPLETO - REPOSITORIO AIPHA_0.0.1")
        context_parts.append(f"**Generado:** {datetime.now().isoformat()}")
        context_parts.append(f"**Consulta:** {query or 'Estado general del proyecto'}")
        context_parts.append("")
        
        # Información general del repositorio
        context_parts.append("## 📊 ESTADO DEL REPOSITORIO")
        context_parts.append(f"- **URL:** {repo_state['repository_url']}")
        context_parts.append(f"- **Archivos totales:** {len(repo_state['files'])}")
        context_parts.append(f"- **Score de integridad:** {repo_state['integrity_score']}/100")
        context_parts.append(f"- **Fuente de datos:** {repo_state['status']}")
        context_parts.append("")
        
        # Análisis de integridad
        if repo_state.get('analysis_summary'):
            analysis = repo_state['analysis_summary']
            context_parts.append("## 🔍 ANÁLISIS DE INTEGRIDAD")
            context_parts.append(f"- **Score:** {analysis['score']}/100")
            context_parts.append(f"- **Archivos analizados:** {analysis['files_analyzed']}")
            context_parts.append(f"- **Issues encontrados:** {analysis['issues_found']}")
            context_parts.append(f"- **Score de estructura:** {analysis['structure_score']}/100")
            context_parts.append("")
        
        # Lista de archivos principales
        if repo_state['files']:
            context_parts.append("## 📁 ARCHIVOS DEL PROYECTO")
            python_files = [f for f in repo_state['files'] if f.endswith('.py')]
            config_files = [f for f in repo_state['files'] if f.endswith(('.json', '.yaml'))]
            doc_files = [f for f in repo_state['files'] if f.endswith('.md')]
            
            if python_files:
                context_parts.append("### Archivos Python:")
                for file in python_files[:10]:
                    context_parts.append(f"- `{file}`")
                if len(python_files) > 10:
                    context_parts.append(f"- ... y {len(python_files) - 10} archivos Python más")
                context_parts.append("")
            
            if config_files:
                context_parts.append("### Archivos de Configuración:")
                for file in config_files:
                    context_parts.append(f"- `{file}`")
                context_parts.append("")
            
            if doc_files:
                context_parts.append("### Documentación:")
                for file in doc_files:
                    context_parts.append(f"- `{file}`")
                context_parts.append("")
        
        # Cambios recientes
        if repo_state['recent_changes']:
            context_parts.append("## 📈 CAMBIOS RECIENTES")
            context_parts.append(f"**Total de cambios:** {len(repo_state['recent_changes'])}")
            context_parts.append("")
            
            for change in repo_state['recent_changes'][:5]:  # Últimos 5
                context_parts.append(f"### {change.get('timestamp', '')[:19]}")
                context_parts.append(f"- **Tipo:** {change.get('type', 'unknown')}")
                context_parts.append(f"- **Commit:** {change.get('commit_hash', 'N/A')}")
                context_parts.append(f"- **Mensaje:** {change.get('message', 'Sin mensaje')}")
                files = change.get('files', [])
                if files:
                    context_parts.append(f"- **Archivos:** {', '.join(files[:3])}{'...' if len(files) > 3 else ''}")
                context_parts.append("")
        
        # Instrucciones para consultas específicas
        context_parts.append("## 💡 CONSULTAS ESPECÍFICAS DISPONIBLES")
        context_parts.append("Puedes solicitar información más específica sobre:")
        context_parts.append("- `¿Cuál es la estructura de main.py?`")
        context_parts.append("- `Muéstrame el contenido de config.json`")
        context_parts.append("- `¿Qué hace la función X en archivo Y?`")
        context_parts.append("- `¿Cuáles son los últimos commits?`")
        context_parts.append("- `¿Hay algún error de integridad?`")
        context_parts.append("")
        
        # Footer
        context_parts.append("---")
        context_parts.append("🔄 **Información actualizada automáticamente por Shadow System**")
        context_parts.append("📊 **Datos verificados con análisis de integridad profunda**")
        context_parts.append(f"⏰ **Última actualización:** {repo_state['timestamp']}")
        
        return "\n".join(context_parts)

    def create_web_interface(self, port: int = 8080) -> Flask:
        """
        Crea una interfaz web para el sistema de integración
        
        Args:
            port: Puerto para el servidor web
            
        Returns:
            Aplicación Flask configurada
        """
        app = Flask(__name__)
        
        @app.route('/')
        def index():
            """Página principal con estado del repositorio"""
            state = self.get_current_repository_state()
            context = self.generate_complete_context()
            
            html_template = '''
            <!DOCTYPE html>
            <html>
            <head>
                <title>Shadow-AiphaLab Integration</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; }
                    .header { background: #f0f0f0; padding: 20px; border-radius: 5px; }
                    .status { background: #e8f5e8; padding: 15px; margin: 10px 0; border-radius: 5px; }
                    .error { background: #ffe8e8; padding: 15px; margin: 10px 0; border-radius: 5px; }
                    pre { background: #f5f5f5; padding: 15px; border-radius: 5px; overflow-x: auto; }
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>🔗 Shadow-AiphaLab Integration</h1>
                    <p>Sistema de integración completo para acceso a datos del repositorio Aipha_0.0.1</p>
                </div>
                
                <div class="status">
                    <h2>📊 Estado Actual del Repositorio</h2>
                    <p><strong>URL:</strong> {{ state.repository_url }}</p>
                    <p><strong>Archivos:</strong> {{ state.files|length }}</p>
                    <p><strong>Score de Integridad:</strong> {{ state.integrity_score }}/100</p>
                    <p><strong>Última actualización:</strong> {{ state.timestamp }}</p>
                </div>
                
                <h2>📋 Contexto Completo para AiphaLab</h2>
                <pre>{{ context }}</pre>
                
                <h2>🔗 Endpoints Disponibles</h2>
                <ul>
                    <li><a href="/api/status">/api/status</a> - Estado JSON del repositorio</li>
                    <li><a href="/api/files">/api/files</a> - Lista de archivos JSON</li>
                    <li><a href="/api/context">/api/context</a> - Contexto completo JSON</li>
                    <li><a href="/api/integrity">/api/integrity</a> - Análisis de integridad JSON</li>
                </ul>
            </body>
            </html>
            '''
            
            return render_template_string(html_template, state=state, context=context)
        
        @app.route('/api/status')
        def api_status():
            """API endpoint para estado del repositorio"""
            state = self.get_current_repository_state()
            return jsonify(state)
        
        @app.route('/api/files')
        def api_files():
            """API endpoint para lista de archivos"""
            state = self.get_current_repository_state()
            return jsonify({
                'files': state['files'],
                'total': len(state['files']),
                'timestamp': state['timestamp']
            })
        
        @app.route('/api/context')
        def api_context():
            """API endpoint para contexto completo"""
            context = self.generate_complete_context()
            return jsonify({
                'context': context,
                'timestamp': datetime.now().isoformat()
            })
        
        @app.route('/api/integrity')
        def api_integrity():
            """API endpoint para análisis de integridad"""
            try:
                result = self.integrity_analyzer.perform_deep_integrity_analysis()
                return jsonify(result)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @app.route('/api/file/<path:file_path>')
        def api_file_content(file_path):
            """API endpoint para contenido de archivo"""
            result = self.get_file_content_with_analysis(file_path)
            if result.get('found'):
                return jsonify(result)
            else:
                return jsonify(result), 404
        
        return app

    # Métodos de cache
    def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Obtiene datos del cache"""
        if not self.cache_enabled:
            return None
        
        try:
            conn = sqlite3.connect(self.cache_db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                'SELECT data, expires_at FROM cache WHERE cache_key = ?',
                (cache_key,)
            )
            
            row = cursor.fetchone()
            conn.close()
            
            if row and row[1] > datetime.now().timestamp():
                return json.loads(row[0])
            
        except Exception as e:
            self.logger.warning(f"Error obteniendo del cache: {e}")
        
        return None

    def _save_to_cache(self, cache_key: str, data: Dict[str, Any], timeout: int = 300):
        """Guarda datos en el cache"""
        if not self.cache_enabled:
            return
        
        try:
            conn = sqlite3.connect(self.cache_db_path)
            cursor = conn.cursor()
            
            expires_at = datetime.now().timestamp() + timeout
            
            cursor.execute(
                'INSERT OR REPLACE INTO cache (cache_key, data, timestamp, expires_at) VALUES (?, ?, ?, ?)',
                (cache_key, json.dumps(data), datetime.now().timestamp(), expires_at)
            )
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.warning(f"Error guardando en cache: {e}")

    def cleanup_cache(self, max_age_hours: int = 24):
        """Limpia entradas de cache expiradas"""
        if not self.cache_enabled:
            return
        
        try:
            conn = sqlite3.connect(self.cache_db_path)
            cursor = conn.cursor()
            
            cutoff_time = (datetime.now() - timedelta(hours=max_age_hours)).timestamp()
            cursor.execute('DELETE FROM cache WHERE timestamp < ?', (cutoff_time,))
            
            deleted_count = cursor.rowcount
            conn.commit()
            conn.close()
            
            self.logger.info(f"Cache cleanup: {deleted_count} entradas eliminadas")
            
        except Exception as e:
            self.logger.error(f"Error en limpieza de cache: {e}")


def main():
    """Función principal para testing y demo"""
    parser = argparse.ArgumentParser(description='Shadow-AiphaLab Integration System')
    parser.add_argument('--memory-path', default='./aipha_memory_storage/action_history',
                       help='Ruta a memoria Shadow')
    parser.add_argument('--cache-db', help='Ruta a base de datos de cache')
    parser.add_argument('--mode', choices=['api', 'web', 'cli'], default='cli',
                       help='Modo de operación')
    parser.add_argument('--port', type=int, default=8080, help='Puerto para servidor web')
    parser.add_argument('--query', help='Consulta para modo CLI')
    parser.add_argument('--force-refresh', action='store_true',
                       help='Forzar actualización de cache')
    
    args = parser.parse_args()
    
    # Crear sistema de integración
    integration = ShadowAiphaLabIntegration(
        shadow_memory_path=args.memory_path,
        cache_db_path=args.cache_db
    )
    
    if args.mode == 'api':
        # Modo API REST
        app = integration.create_web_interface(port=args.port)
        print(f"🚀 Iniciando servidor API en puerto {args.port}...")
        print(f"📊 Acceso: http://localhost:{args.port}")
        app.run(host='0.0.0.0', port=args.port, debug=False)
        
    elif args.mode == 'web':
        # Modo interfaz web
        app = integration.create_web_interface(port=args.port)
        print(f"🌐 Iniciando interfaz web en puerto {args.port}...")
        print(f"📊 Acceso: http://localhost:{args.port}")
        app.run(host='0.0.0.0', port=args.port, debug=True)
        
    else:
        # Modo CLI
        print("🔗 SHADOW-AIPHALAB INTEGRATION SYSTEM")
        print("=" * 50)
        
        if args.query:
            print(f"📋 Generando contexto para consulta: '{args.query}'")
            context = integration.generate_complete_context(args.query)
            print("\n" + context)
        else:
            print("📊 Obteniendo estado del repositorio...")
            state = integration.get_current_repository_state(force_refresh=args.force_refresh)
            
            print(f"\n📈 ESTADO DEL REPOSITORIO:")
            print(f"   URL: {state['repository_url']}")
            print(f"   Archivos: {len(state['files'])}")
            print(f"   Score Integridad: {state['integrity_score']}/100")
            print(f"   Cambios Recientes: {len(state['recent_changes'])}")
            print(f"   Estado: {state['status']}")
            
            print(f"\n📋 CONTEXTO COMPLETO:")
            context = integration.generate_complete_context()
            print("\n" + context)
        
        # Limpiar cache al final
        integration.cleanup_cache()


if __name__ == "__main__":
    main()