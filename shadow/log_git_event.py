#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
log_git_event.py - Script para registrar eventos de Git en Shadow Memory

Este script registra automáticamente eventos de Git (commits y pushes) en el sistema
de memoria de Shadow, manteniendo un registro inmutable de la evolución del código.

Uso:
    python3 shadow/log_git_event.py --event commit --commit-hash abc123 --message "Fix bug" --files "file1.py,file2.py"
    python3 shadow/log_git_event.py --event push --commit-hash abc123 --github-url "https://github.com/user/repo/commit/abc123"

Autor: Shadow System
Versión: 1.0.0
"""

import os
import sys
import json
import hashlib
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional
import uuid

class GitEventLogger:
    """
    Logger para eventos de Git en el sistema de memoria Shadow
    """

    def __init__(self, shadow_memory_path: str):
        self.shadow_memory_path = shadow_memory_path
        self.memory_file = os.path.join(shadow_memory_path, 'current_history.json')
        self.logger = logging.getLogger(__name__)

        # Configurar logging básico
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    def log_git_event(self, event_data: Dict[str, Any]) -> bool:
        """
        Registra un evento de Git en la memoria de Shadow

        Args:
            event_data: Diccionario con datos del evento Git

        Returns:
            bool: True si el registro fue exitoso, False en caso contrario
        """
        try:
            # Generar ID único para la entrada
            entry_id = str(uuid.uuid4())

            # Obtener timestamp en formato ISO 8601
            timestamp = datetime.now().isoformat()

            # Preparar la estructura de la entrada
            shadow_entry = {
                "entry_id": entry_id,
                "timestamp": timestamp,
                "version_id": "Aipha_0.0.1",
                "source_component": "Git_Hook",
                "entry_category": "GIT_EVENT",
                "data_payload": event_data,
                "previous_entry_hash": self._get_previous_entry_hash(),
                "entry_hash": ""
            }

            # Calcular hash SHA-256 de la entrada para integridad
            entry_content = json.dumps(shadow_entry, sort_keys=True, ensure_ascii=False)
            shadow_entry["entry_hash"] = hashlib.sha256(entry_content.encode('utf-8')).hexdigest()

            # Agregar entrada a la memoria de Shadow
            success = self._add_to_shadow_memory(shadow_entry)

            if success:
                self.logger.info(f"Git event logged successfully: {event_data.get('event_type', 'unknown')} - {entry_id}")
                return True
            else:
                self.logger.error("Failed to add entry to Shadow memory")
                return False

        except Exception as e:
            self.logger.error(f"Error logging Git event: {e}")
            return False

    def _get_previous_entry_hash(self) -> str:
        """
        Obtiene el hash de la entrada anterior para mantener la cadena de integridad

        Returns:
            str: Hash de la entrada anterior o cadena vacía si no existe
        """
        try:
            if not os.path.exists(self.memory_file):
                return ""

            with open(self.memory_file, 'r', encoding='utf-8') as f:
                history = json.load(f)

            if not history:
                return ""

            # Obtener la última entrada
            last_entry = history[-1]
            return last_entry.get("entry_hash", "")

        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            self.logger.warning(f"Could not get previous entry hash: {e}")
            return ""

    def _add_to_shadow_memory(self, entry: Dict[str, Any]) -> bool:
        """
        Agrega una entrada al archivo de memoria de Shadow

        Args:
            entry: Entrada a agregar

        Returns:
            bool: True si fue exitosa, False en caso contrario
        """
        try:
            # Leer historia existente
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            else:
                history = []

            # Agregar nueva entrada
            history.append(entry)

            # Escribir de vuelta al archivo
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)

            return True

        except Exception as e:
            self.logger.error(f"Error writing to Shadow memory: {e}")
            return False


def parse_arguments() -> argparse.Namespace:
    """Parsea los argumentos de línea de comandos"""
    parser = argparse.ArgumentParser(
        description='Registra eventos de Git en Shadow Memory',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

# Registrar commit:
python3 shadow/log_git_event.py --event commit --commit-hash abc123 --message "Fix bug" --files "file1.py,file2.py"

# Registrar push:
python3 shadow/log_git_event.py --event push --commit-hash abc123 --github-url "https://github.com/user/repo/commit/abc123"
        """
    )

    parser.add_argument(
        '--event',
        required=True,
        choices=['commit', 'push'],
        help='Tipo de evento Git (commit o push)'
    )

    parser.add_argument(
        '--commit-hash',
        required=True,
        help='Hash del commit'
    )

    parser.add_argument(
        '--message',
        help='Mensaje del commit (requerido para commits)'
    )

    parser.add_argument(
        '--files',
        help='Archivos modificados separados por coma (requerido para commits)'
    )

    parser.add_argument(
        '--github-url',
        help='URL completa del commit en GitHub (requerido para pushes)'
    )

    parser.add_argument(
        '--memory-path',
        default='./aipha_memory_storage/action_history',
        help='Ruta al directorio de memoria Shadow (default: ./aipha_memory_storage/action_history)'
    )

    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> bool:
    """Valida que los argumentos sean correctos"""
    if args.event == 'commit':
        if not args.message:
            print("ERROR: --message es requerido para eventos de commit")
            return False
        if not args.files:
            print("ERROR: --files es requerido para eventos de commit")
            return False
    elif args.event == 'push':
        if not args.github_url:
            print("ERROR: --github-url es requerido para eventos de push")
            return False

    return True


def main():
    """Función principal"""
    try:
        # Parsear argumentos
        args = parse_arguments()

        # Validar argumentos
        if not validate_arguments(args):
            sys.exit(1)

        # Crear logger
        logger = GitEventLogger(args.memory_path)

        # Preparar datos del evento
        event_data = {
            "event_type": args.event,
            "commit_hash": args.commit_hash,
        }

        if args.event == 'commit':
            event_data.update({
                "commit_message": args.message,
                "files_changed": args.files.split(',') if args.files else []
            })
        elif args.event == 'push':
            event_data.update({
                "github_url": args.github_url
            })

        # Registrar evento
        success = logger.log_git_event(event_data)

        if success:
            print(f"✅ Evento Git registrado exitosamente: {args.event} - {args.commit_hash}")
            sys.exit(0)
        else:
            print("❌ Error al registrar evento Git")
            sys.exit(1)

    except Exception as e:
        print(f"❌ Error fatal: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()