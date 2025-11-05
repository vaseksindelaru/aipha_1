#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
shadow_proactive.py - Shadow Proactivo: Context Sentinel con Monitoreo Persistente

Este módulo implementa Shadow como un servicio proactivo de monitoreo continuo
del sistema de archivos del proyecto Aipha. Utiliza watchdog para detectar cambios
en tiempo real y mantener un estado interno actualizado.

Características:
- Monitoreo persistente del directorio del proyecto
- Detección automática de cambios (creación, modificación, eliminación)
- Estado interno como "fuente de verdad" del sistema de archivos
- Logging detallado de todos los eventos
- Interfaz de consulta para otros componentes

Autor: Shadow System
Versión: 2.0.0
"""

import os
import sys
import time
import json
import logging
import signal
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from collections import defaultdict

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    print("⚠️  watchdog no está instalado. Instala con: pip install watchdog")
    sys.exit(1)


class ProjectFileState:
    """
    Representa el estado interno del sistema de archivos del proyecto
    """

    def __init__(self, project_path: str):
        self.project_path = Path(project_path).resolve()
        self.files: Dict[str, Dict[str, Any]] = {}
        self.directories: Set[str] = set()
        self.last_scan = None
        self.logger = logging.getLogger(__name__)

    def scan_initial_state(self):
        """Escanea el estado inicial del proyecto"""
        self.logger.info(f"Escaneando estado inicial del proyecto: {self.project_path}")

        self.files.clear()
        self.directories.clear()

        for root, dirs, files in os.walk(self.project_path):
            # Filtrar directorios que no queremos monitorear
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules']]

            # Registrar directorios
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                rel_path = os.path.relpath(dir_path, self.project_path)
                self.directories.add(rel_path)

            # Registrar archivos
            for file_name in files:
                file_path = os.path.join(root, file_name)
                rel_path = os.path.relpath(file_path, self.project_path)

                # Obtener información del archivo
                try:
                    stat = os.stat(file_path)
                    file_info = {
                        'path': rel_path,
                        'size': stat.st_size,
                        'modified': stat.st_mtime,
                        'created': stat.st_ctime,
                        'is_file': True,
                        'extension': Path(file_path).suffix,
                        'last_seen': datetime.now().isoformat()
                    }
                    self.files[rel_path] = file_info
                except OSError as e:
                    self.logger.warning(f"Error obteniendo info de {file_path}: {e}")

        self.last_scan = datetime.now()
        self.logger.info(f"Estado inicial escaneado: {len(self.files)} archivos, {len(self.directories)} directorios")

    def update_file(self, file_path: str, event_type: str):
        """Actualiza el estado de un archivo"""
        rel_path = os.path.relpath(file_path, self.project_path)

        if event_type in ['created', 'modified']:
            try:
                stat = os.stat(file_path)
                file_info = {
                    'path': rel_path,
                    'size': stat.st_size,
                    'modified': stat.st_mtime,
                    'created': stat.st_ctime,
                    'is_file': True,
                    'extension': Path(file_path).suffix,
                    'last_seen': datetime.now().isoformat(),
                    'last_event': event_type
                }
                self.files[rel_path] = file_info
                self.logger.info(f"Archivo {event_type}: {rel_path}")
            except OSError as e:
                self.logger.error(f"Error actualizando {file_path}: {e}")

        elif event_type == 'deleted':
            if rel_path in self.files:
                del self.files[rel_path]
                self.logger.info(f"Archivo eliminado: {rel_path}")
            else:
                self.logger.warning(f"Archivo no encontrado para eliminar: {rel_path}")

    def update_directory(self, dir_path: str, event_type: str):
        """Actualiza el estado de un directorio"""
        rel_path = os.path.relpath(dir_path, self.project_path)

        if event_type in ['created', 'modified']:
            self.directories.add(rel_path)
            self.logger.info(f"Directorio {event_type}: {rel_path}")
        elif event_type == 'deleted':
            self.directories.discard(rel_path)
            self.logger.info(f"Directorio eliminado: {rel_path}")

    def get_state_summary(self) -> Dict[str, Any]:
        """Obtiene un resumen del estado actual"""
        # Contar por extensiones
        extensions = defaultdict(int)
        for file_info in self.files.values():
            ext = file_info.get('extension', 'no_ext')
            extensions[ext] += 1

        return {
            'total_files': len(self.files),
            'total_directories': len(self.directories),
            'extensions': dict(extensions),
            'last_scan': self.last_scan.isoformat() if self.last_scan else None,
            'project_path': str(self.project_path)
        }

    def get_file_info(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Obtiene información de un archivo específico"""
        rel_path = os.path.relpath(file_path, self.project_path)
        return self.files.get(rel_path)

    def list_files(self, extension: str = None) -> List[str]:
        """Lista archivos, opcionalmente filtrados por extensión"""
        if extension:
            return [f for f in self.files.keys() if f.endswith(extension)]
        return list(self.files.keys())

    def save_state(self, state_file: str):
        """Guarda el estado actual a un archivo"""
        state_data = {
            'timestamp': datetime.now().isoformat(),
            'project_path': str(self.project_path),
            'files': self.files,
            'directories': list(self.directories),
            'summary': self.get_state_summary()
        }

        try:
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(state_data, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Estado guardado en: {state_file}")
        except Exception as e:
            self.logger.error(f"Error guardando estado: {e}")

    def load_state(self, state_file: str) -> bool:
        """Carga el estado desde un archivo"""
        try:
            with open(state_file, 'r', encoding='utf-8') as f:
                state_data = json.load(f)

            self.files = state_data.get('files', {})
            self.directories = set(state_data.get('directories', []))
            self.last_scan = datetime.fromisoformat(state_data.get('timestamp'))

            self.logger.info(f"Estado cargado desde: {state_file}")
            return True
        except Exception as e:
            self.logger.error(f"Error cargando estado: {e}")
            return False


class ShadowFileSystemHandler(FileSystemEventHandler):
    """
    Manejador de eventos del sistema de archivos para Shadow
    """

    def __init__(self, file_state: ProjectFileState, logger: logging.Logger):
        self.file_state = file_state
        self.logger = logger

    def on_created(self, event):
        """Maneja evento de creación"""
        path = event.src_path
        if os.path.isfile(path):
            self.file_state.update_file(path, 'created')
        else:
            self.file_state.update_directory(path, 'created')

    def on_modified(self, event):
        """Maneja evento de modificación"""
        path = event.src_path
        if os.path.isfile(path):
            self.file_state.update_file(path, 'modified')
        else:
            self.file_state.update_directory(path, 'modified')

    def on_deleted(self, event):
        """Maneja evento de eliminación"""
        path = event.src_path
        if os.path.isfile(path):
            self.file_state.update_file(path, 'deleted')
        else:
            self.file_state.update_directory(path, 'deleted')

    def on_moved(self, event):
        """Maneja evento de movimiento/renombrado"""
        # Tratar como eliminación del origen y creación del destino
        self.on_deleted(event)
        # Crear un evento simulado para el destino
        event.src_path = event.dest_path
        self.on_created(event)


class ShadowProactive:
    """
    Shadow Proactivo: Servicio de monitoreo continuo del proyecto Aipha
    """

    def __init__(self, project_path: str = None, state_file: str = None, log_level: str = 'INFO'):
        # Configurar logging
        self.setup_logging(log_level)

        self.logger = logging.getLogger(__name__)
        self.logger.info("Inicializando Shadow Proactivo v2.0.0")

        # Determinar rutas
        self.project_path = project_path or self._find_project_path()
        self.state_file = state_file or os.path.join(self.project_path, 'shadow_state.json')

        # Estado del proyecto
        self.file_state = ProjectFileState(self.project_path)

        # Componentes de monitoreo
        self.observer = None
        self.handler = None
        self.running = False
        self.thread = None

        # Estado de consultas
        self.query_interface = None

        self.logger.info(f"Proyecto a monitorear: {self.project_path}")
        self.logger.info(f"Archivo de estado: {self.state_file}")

    def setup_logging(self, log_level: str):
        """Configura el sistema de logging"""
        numeric_level = getattr(logging, log_level.upper(), logging.INFO)

        # Configurar logging a archivo y consola
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=numeric_level,
            format=log_format,
            handlers=[
                logging.FileHandler('shadow_proactive.log'),
                logging.StreamHandler()
            ]
        )

    def _find_project_path(self) -> str:
        """Encuentra automáticamente el directorio del proyecto Aipha"""
        # Buscar hacia arriba desde el directorio actual
        current = Path.cwd()
        for parent in [current] + list(current.parents):
            if (parent / 'shadow.py').exists() or (parent / 'main.py').exists():
                return str(parent)

        # Fallback al directorio actual
        return str(Path.cwd())

    def start_monitoring(self):
        """Inicia el monitoreo del sistema de archivos"""
        if self.running:
            self.logger.warning("El monitoreo ya está activo")
            return

        self.logger.info("Iniciando monitoreo del sistema de archivos...")

        # Cargar estado anterior si existe
        if os.path.exists(self.state_file):
            self.file_state.load_state(self.state_file)

        # Escanear estado inicial
        self.file_state.scan_initial_state()

        # Configurar el observador
        self.handler = ShadowFileSystemHandler(self.file_state, self.logger)
        self.observer = Observer()
        self.observer.schedule(self.handler, self.project_path, recursive=True)

        # Iniciar el observador
        self.observer.start()
        self.running = True

        # Guardar estado inicial
        self.file_state.save_state(self.state_file)

        self.logger.info("✅ Monitoreo iniciado exitosamente")
        self.logger.info(f"📁 Monitoreando: {self.project_path}")
        self.logger.info(f"💾 Estado guardado en: {self.state_file}")

    def stop_monitoring(self):
        """Detiene el monitoreo del sistema de archivos"""
        if not self.running:
            return

        self.logger.info("Deteniendo monitoreo...")

        if self.observer:
            self.observer.stop()
            self.observer.join(timeout=5)

        self.running = False

        # Guardar estado final
        self.file_state.save_state(self.state_file)

        self.logger.info("✅ Monitoreo detenido")

    def get_status(self) -> Dict[str, Any]:
        """Obtiene el estado actual del sistema Shadow"""
        return {
            'running': self.running,
            'project_path': str(self.project_path),
            'state_file': self.state_file,
            'file_state': self.file_state.get_state_summary(),
            'uptime': 'N/A',  # Podría implementarse con timestamp de inicio
            'last_activity': datetime.now().isoformat()
        }

    def query_files(self, extension: str = None) -> List[str]:
        """Consulta archivos del proyecto"""
        return self.file_state.list_files(extension)

    def query_file_info(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Consulta información de un archivo específico"""
        return self.file_state.get_file_info(file_path)

    def save_current_state(self):
        """Guarda el estado actual"""
        self.file_state.save_state(self.state_file)
        self.logger.info("Estado actual guardado")

    def run_interactive(self):
        """Ejecuta en modo interactivo para testing"""
        print("🚀 Shadow Proactivo - Modo Interactivo")
        print("=" * 50)

        self.start_monitoring()

        try:
            while True:
                print("\nComandos disponibles:")
                print("  status  - Mostrar estado actual")
                print("  files   - Listar archivos")
                print("  save    - Guardar estado")
                print("  stop    - Detener monitoreo")
                print("  quit    - Salir")

                cmd = input("\nComando: ").strip().lower()

                if cmd == 'status':
                    status = self.get_status()
                    print(f"Estado: {'Activo' if status['running'] else 'Inactivo'}")
                    print(f"Archivos: {status['file_state']['total_files']}")
                    print(f"Directorios: {status['file_state']['total_directories']}")

                elif cmd == 'files':
                    files = self.query_files()
                    print(f"Total archivos: {len(files)}")
                    for f in files[:10]:  # Mostrar primeros 10
                        print(f"  {f}")
                    if len(files) > 10:
                        print(f"  ... y {len(files) - 10} más")

                elif cmd == 'save':
                    self.save_current_state()
                    print("✅ Estado guardado")

                elif cmd == 'stop':
                    self.stop_monitoring()
                    break

                elif cmd == 'quit':
                    break

                else:
                    print("Comando no reconocido")

        except KeyboardInterrupt:
            print("\nInterrupción detectada...")
        finally:
            self.stop_monitoring()

    def run_daemon(self):
        """Ejecuta como daemon en segundo plano"""
        self.logger.info("Iniciando Shadow como daemon...")

        self.start_monitoring()

        # Configurar señales para shutdown graceful
        def signal_handler(signum, frame):
            self.logger.info(f"Señal {signum} recibida, deteniendo...")
            self.stop_monitoring()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Mantener vivo
        try:
            while self.running:
                time.sleep(1)
                # Aquí podría implementarse lógica adicional
                # como guardado periódico del estado, etc.
        except Exception as e:
            self.logger.error(f"Error en ejecución daemon: {e}")
        finally:
            self.stop_monitoring()


def main():
    """Función principal"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Shadow Proactivo - Context Sentinel con Monitoreo Persistente',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

# Modo interactivo (para testing)
python3 shadow_proactive.py --interactive

# Modo daemon (producción)
python3 shadow_proactive.py --daemon

# Con rutas específicas
python3 shadow_proactive.py --project-path /path/to/aipha --state-file /path/to/state.json --daemon

# Logging detallado
python3 shadow_proactive.py --log-level DEBUG --interactive
        """
    )

    parser.add_argument(
        '--project-path',
        help='Ruta al directorio del proyecto Aipha'
    )

    parser.add_argument(
        '--state-file',
        help='Archivo para guardar el estado del proyecto'
    )

    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Ejecutar en modo interactivo'
    )

    parser.add_argument(
        '--daemon', '-d',
        action='store_true',
        help='Ejecutar como daemon en segundo plano'
    )

    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Nivel de logging (default: INFO)'
    )

    args = parser.parse_args()

    # Validar argumentos
    if not args.interactive and not args.daemon:
        print("Debe especificar --interactive o --daemon")
        sys.exit(1)

    if args.interactive and args.daemon:
        print("No puede usar --interactive y --daemon juntos")
        sys.exit(1)

    # Crear instancia de Shadow
    shadow = ShadowProactive(
        project_path=args.project_path,
        state_file=args.state_file,
        log_level=args.log_level
    )

    # Ejecutar según el modo
    if args.interactive:
        shadow.run_interactive()
    elif args.daemon:
        shadow.run_daemon()


if __name__ == "__main__":
    main()