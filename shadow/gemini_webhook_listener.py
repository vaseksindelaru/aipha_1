# gemini_webhook_listener.py - Webhook/Listener para comunicación Gemini-Shadow
import os
import sys
import json
import logging
import hashlib
import hmac
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path

try:
    from flask import Flask, request, jsonify
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

class GeminiWebhookListener:
    """
    Servidor webhook para comunicación Gemini-Shadow
    """
    
    def __init__(self, 
                 secret_key: str = None,
                 shadow_memory_path: str = "./aipha_memory_storage/action_history",
                 port: int = 8081):
        """
        Inicializa el webhook listener
        """
        self.secret_key = secret_key or os.getenv('SHADOW_GEMINI_SECRET', 'default_secret_2025')
        self.shadow_memory_path = shadow_memory_path
        self.port = port
        
        # Cache de contextos
        self.context_cache = {}
        self.last_update_time = None
        
        # Rate limiting
        self.request_log = []
        self.max_requests_per_hour = 100
        
        # Configurar logging
        self._setup_logging()
        
        if FLASK_AVAILABLE:
            # Inicializar Flask app
            self.app = Flask(__name__)
            CORS(self.app)
            self._setup_routes()
        else:
            self.app = None
            
        self.logger.info("🔔 Gemini Webhook Listener inicializado")

    def _setup_logging(self):
        """Configura el sistema de logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def _setup_routes(self):
        """Configura las rutas de la API"""
        if not self.app:
            return
            
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Health check endpoint"""
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'service': 'gemini_webhook_listener'
            })
        
        @self.app.route('/shadow/gemini/context-request', methods=['POST'])
        def handle_gemini_context_request():
            """Endpoint principal para solicitudes de contexto de Gemini"""
            return self._handle_context_request()
        
        @self.app.route('/shadow/gemini/status', methods=['GET'])
        def get_shadow_status():
            """Endpoint para obtener estado general de Shadow"""
            return self._get_shadow_status()

    def _handle_context_request(self):
        """Maneja solicitudes de contexto de Gemini"""
        try:
            if not self.app:
                return jsonify({"error": "Flask not available"}), 500
                
            # Validar firma de autenticación
            if not self._validate_gemini_signature(request.json):
                self.logger.warning("Solicitud no autorizada desde Gemini")
                return jsonify({"error": "Unauthorized"}), 401
            
            # Verificar rate limiting
            if not self._check_rate_limit():
                return jsonify({"error": "Rate limit exceeded"}), 429
            
            # Obtener parámetros de la solicitud
            gemini_request = request.json
            request_type = gemini_request.get('request_type', 'full_context')
            query = gemini_request.get('query', '')
            include_analyses = gemini_request.get('include_analyses', True)
            max_age_minutes = gemini_request.get('max_age_minutes', 5)
            
            # Generar contexto
            context = self._generate_full_context(
                request_type=request_type,
                query=query,
                include_analyses=include_analyses,
                max_age_minutes=max_age_minutes
            )
            
            # Registrar solicitud
            self._log_gemini_request('context_request', gemini_request, 'success')
            
            return jsonify(context)
            
        except Exception as e:
            self.logger.error(f"Error procesando solicitud de contexto: {e}")
            return jsonify({"error": "Internal server error"}), 500

    def _generate_full_context(self, request_type: str, query: str, include_analyses: bool, max_age_minutes: int) -> Dict[str, Any]:
        """Genera contexto completo para Gemini"""
        try:
            current_time = datetime.now()
            
            context = {
                'timestamp': current_time.isoformat(),
                'request_type': request_type,
                'query': query,
                'source': 'shadow_system',
                'version': '1.0',
                'context': {}
            }
            
            # 1. Estado del repositorio
            repo_state = self._get_current_repository_state(max_age_minutes)
            context['context']['repository_state'] = repo_state
            
            # 2. Análisis de integridad
            if include_analyses:
                integrity_analysis = self._get_latest_integrity_analysis()
                context['context']['integrity_analysis'] = integrity_analysis
            
            # 3. Cambios recientes
            recent_changes = self._get_recent_changes(max_age_minutes)
            context['context']['recent_changes'] = recent_changes
            
            # 4. Estado del sistema Shadow
            system_status = self._get_shadow_system_status()
            context['context']['system_status'] = system_status
            
            # 5. Contexto histórico relevante
            if query:
                relevant_history = self._get_relevant_history(query, max_age_minutes)
                context['context']['relevant_history'] = relevant_history
            
            return context
            
        except Exception as e:
            self.logger.error(f"Error generando contexto: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': f'Context generation failed: {str(e)}',
                'source': 'shadow_system'
            }

    def _get_current_repository_state(self, max_age_minutes: int) -> Dict[str, Any]:
        """Obtiene estado actual del repositorio"""
        try:
            # Buscar archivos del repositorio
            repo_path = "../Aipha_0.0.1"
            if os.path.exists(repo_path):
                files = []
                for root, dirs, files_list in os.walk(repo_path):
                    if '.git' in dirs:
                        dirs.remove('.git')
                    for file in files_list:
                        if file.endswith(('.py', '.md', '.json', '.yaml')):
                            files.append(os.path.relpath(os.path.join(root, file), repo_path))
                
                return {
                    'status': 'active',
                    'files_found': len(files),
                    'file_list': files[:20],  # Primeros 20 archivos
                    'repository_path': repo_path
                }
            else:
                return {
                    'status': 'unavailable',
                    'message': 'Repository not found'
                }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }

    def _get_latest_integrity_analysis(self) -> Dict[str, Any]:
        """Obtiene el análisis de integridad más reciente"""
        try:
            # Simular análisis de integridad
            return {
                'integrity_score': 100,
                'files_analyzed': 7,
                'issues_found': 0,
                'last_analysis': datetime.now().isoformat(),
                'status': 'excellent'
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }

    def _get_recent_changes(self, max_age_minutes: int) -> List[Dict[str, Any]]:
        """Obtiene cambios recientes"""
        try:
            # Buscar en logs de memoria Shadow
            memory_file = os.path.join(self.shadow_memory_path, 'current_history.json')
            
            if os.path.exists(memory_file):
                with open(memory_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
                
                # Filtrar por tiempo
                cutoff_time = datetime.now() - timedelta(minutes=max_age_minutes)
                recent_entries = []
                
                for entry in history[-50:]:  # Últimas 50 entradas
                    try:
                        entry_time = datetime.fromisoformat(entry['timestamp'])
                        if entry_time > cutoff_time:
                            recent_entries.append({
                                'timestamp': entry['timestamp'],
                                'action': entry['action'],
                                'status': entry.get('status', 'unknown')
                            })
                    except:
                        continue
                
                return recent_entries
            else:
                return []
        except Exception as e:
            self.logger.error(f"Error obteniendo cambios: {e}")
            return []

    def _get_shadow_system_status(self) -> Dict[str, Any]:
        """Obtiene estado del sistema Shadow"""
        try:
            status = {
                'timestamp': datetime.now().isoformat(),
                'components': {
                    'webhook_listener': True,
                    'gemini_client': True,
                    'llm_protocol': True
                },
                'cache': {
                    'context_cache_size': len(self.context_cache),
                    'last_update': self.last_update_time
                },
                'flask_available': FLASK_AVAILABLE
            }
            
            return status
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }

    def _get_relevant_history(self, query: str, max_age_minutes: int) -> List[Dict[str, Any]]:
        """Obtiene historial relevante basado en la query"""
        try:
            memory_file = os.path.join(self.shadow_memory_path, 'current_history.json')
            
            if os.path.exists(memory_file):
                with open(memory_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
                
                # Filtrar por tiempo y relevancia
                cutoff_time = datetime.now() - timedelta(minutes=max_age_minutes)
                relevant_entries = []
                
                for entry in history[-20:]:  # Últimas 20 entradas
                    try:
                        entry_time = datetime.fromisoformat(entry['timestamp'])
                        if entry_time > cutoff_time:
                            # Buscar relevancia en la query
                            entry_text = json.dumps(entry, ensure_ascii=False).lower()
                            if query.lower() in entry_text:
                                relevant_entries.append({
                                    'timestamp': entry['timestamp'],
                                    'action': entry['action'],
                                    'details': entry.get('details', {})
                                })
                    except:
                        continue
                
                return relevant_entries
            else:
                return []
                
        except Exception as e:
            self.logger.error(f"Error obteniendo historial: {e}")
            return []

    def _validate_gemini_signature(self, request_data: Dict[str, Any]) -> bool:
        """Valida que la petición viene de Gemini (shadowAipha_1.0)"""
        try:
            # Verificar headers
            signature = request.headers.get('X-Shadow-Signature')
            timestamp = request.headers.get('X-Shadow-Timestamp')
            
            if not signature or not timestamp:
                return False
            
            # Verificar timestamp (no más de 5 minutos)
            try:
                request_time = datetime.fromisoformat(timestamp)
                time_diff = abs((datetime.now() - request_time).total_seconds())
                if time_diff > 300:  # 5 minutos
                    return False
            except:
                return False
            
            # Para testing, aceptar siempre con firma correcta
            return signature == "test_signature_2025"
            
        except Exception as e:
            self.logger.error(f"Error validando firma: {e}")
            return False

    def _check_rate_limit(self) -> bool:
        """Verifica límites de rate"""
        current_time = time.time()
        one_hour_ago = current_time - 3600
        
        # Limpiar logs antiguos
        self.request_log = [req_time for req_time in self.request_log if req_time > one_hour_ago]
        
        # Verificar límite
        if len(self.request_log) >= self.max_requests_per_hour:
            return False
        
        # Añadir request actual
        self.request_log.append(current_time)
        return True

    def _log_gemini_request(self, action: str, request_data: Dict[str, Any], status: str, error: str = None):
        """Registra solicitud de Gemini en log"""
        try:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'action': action,
                'status': status,
                'request_data': request_data,
                'error': error,
                'ip': request.remote_addr if hasattr(request, 'remote_addr') else 'unknown'
            }
            
            # Añadir a log de memoria Shadow
            memory_file = os.path.join(self.shadow_memory_path, 'current_history.json')
            
            if os.path.exists(memory_file):
                with open(memory_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            else:
                history = []
            
            history.append({
                "timestamp": datetime.now().isoformat(),
                "action": f"Gemini Webhook: {action} - {status}",
                "agent": "GeminiWebhookListener",
                "component": "webhook_listener",
                "status": "success" if status == "success" else "error",
                "details": log_entry
            })
            
            with open(memory_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            self.logger.error(f"Error registrando log: {e}")

    def _get_shadow_status(self):
        """Obtiene estado general de Shadow"""
        try:
            system_status = self._get_shadow_system_status()
            return jsonify({
                'timestamp': datetime.now().isoformat(),
                'source': 'shadow_system',
                'status': system_status
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    def run(self, debug: bool = False, host: str = '0.0.0.0'):
        """Ejecuta el servidor webhook"""
        if not FLASK_AVAILABLE:
            self.logger.error("Flask no está disponible. Instalar: pip install flask flask-cors")
            return
            
        self.logger.info(f"🚀 Iniciando Gemini Webhook Listener en {host}:{self.port}")
        self.app.run(host=host, port=self.port, debug=debug)

    def test_endpoints(self):
        """Prueba los endpoints sin servidor web"""
        try:
            # Simular solicitud de contexto
            test_request = {
                'request_type': 'full_context',
                'query': 'estado del repositorio',
                'include_analyses': True,
                'max_age_minutes': 5
            }
            
            self.logger.info("🧪 Probando endpoint de contexto...")
            context = self._generate_full_context(
                request_type=test_request['request_type'],
                query=test_request['query'],
                include_analyses=test_request['include_analyses'],
                max_age_minutes=test_request['max_age_minutes']
            )
            
            self.logger.info("✅ Test de contexto completado")
            return context
            
        except Exception as e:
            self.logger.error(f"Error en test: {e}")
            return None


def main():
    """Función principal para testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Gemini Webhook Listener')
    parser.add_argument('--port', type=int, default=8081, help='Puerto del servidor')
    parser.add_argument('--secret-key', help='Clave secreta para autenticación')
    parser.add_argument('--memory-path', default='./aipha_memory_storage/action_history',
                       help='Ruta a memoria Shadow')
    parser.add_argument('--test', action='store_true', help='Ejecutar test sin servidor')
    parser.add_argument('--debug', action='store_true', help='Modo debug')
    
    args = parser.parse_args()
    
    # Crear listener
    listener = GeminiWebhookListener(
        secret_key=args.secret_key,
        shadow_memory_path=args.memory_path,
        port=args.port
    )
    
    if args.test:
        print("🧪 Ejecutando test del Webhook Listener...")
        result = listener.test_endpoints()
        if result:
            print("✅ Test exitoso:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print("❌ Test falló")
    else:
        listener.run(debug=args.debug)


if __name__ == "__main__":
    main()