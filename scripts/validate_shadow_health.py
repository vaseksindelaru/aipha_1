#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/validate_shadow_health.py - Healthcheck del sistema Shadow

Valida el estado completo del sistema Shadow incluyendo:
- Integridad de memoria
- Instalación de Git hooks
- Dependencias Python
- Permisos de archivos
- Actividad reciente

Autor: Shadow System
Versión: 1.0.0
"""

import os
import sys
import json
import hashlib
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

# Añadir directorio shadow al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shadow'))

try:
    from aiphalab_bridge import AiphaLabBridge
    AIPHALAB_AVAILABLE = True
except ImportError:
    AIPHALAB_AVAILABLE = False


class ShadowHealthChecker:
    """Validador de salud del sistema Shadow"""

    def __init__(self, shadow_project_path=None):
        self.shadow_project_path = shadow_project_path or self._find_shadow_project()
        self.memory_path = os.path.join(self.shadow_project_path, 'aipha_memory_storage', 'action_history')
        self.memory_file = os.path.join(self.memory_path, 'current_history.json')

        # Resultados de validación
        self.checks = {}
        self.overall_score = 0
        self.total_checks = 0

    def _find_shadow_project(self):
        """Encuentra el directorio del proyecto Shadow"""
        current_path = Path(__file__).resolve()

        # Buscar hacia arriba hasta encontrar el directorio con shadow/
        for parent in current_path.parents:
            if (parent / 'shadow').exists():
                return str(parent)

        # Fallback al directorio actual
        return str(Path(__file__).parent.parent)

    def run_full_health_check(self):
        """Ejecuta verificación completa de salud"""
        print("🔍 Verificando Shadow Memory System...\n")

        self.check_shadow_memory_integrity()
        self.check_git_hooks_installation()
        self.check_dependencies()
        self.check_memory_storage()
        self.check_recent_activity()

        self._calculate_overall_score()
        self._print_final_report()

    def check_shadow_memory_integrity(self):
        """Verifica integridad de la memoria de Shadow"""
        print("📋 Verificando integridad de memoria...")

        check_name = "Shadow Memory Integrity"
        self.total_checks += 1

        try:
            if not os.path.exists(self.memory_file):
                self.checks[check_name] = {
                    'status': 'FAIL',
                    'message': 'Archivo de memoria no encontrado',
                    'details': f'Expected: {self.memory_file}'
                }
                return

            # Cargar datos de memoria
            with open(self.memory_file, 'r', encoding='utf-8') as f:
                memory_data = json.load(f)

            if not isinstance(memory_data, list):
                self.checks[check_name] = {
                    'status': 'FAIL',
                    'message': 'Formato de memoria inválido',
                    'details': 'Se esperaba una lista JSON'
                }
                return

            # Verificar cadena de hashes
            integrity_issues = self._validate_hash_chain(memory_data)

            if integrity_issues:
                self.checks[check_name] = {
                    'status': 'WARN',
                    'message': 'Cadena de hashes comprometida',
                    'details': f'{len(integrity_issues)} problemas encontrados'
                }
            else:
                self.checks[check_name] = {
                    'status': 'PASS',
                    'message': 'Integridad de memoria OK',
                    'details': f'{len(memory_data)} entradas válidas'
                }
                self.overall_score += 1

        except Exception as e:
            self.checks[check_name] = {
                'status': 'FAIL',
                'message': 'Error verificando integridad',
                'details': str(e)
            }

    def _validate_hash_chain(self, memory_data):
        """Valida la cadena de hashes de integridad"""
        issues = []

        for i, entry in enumerate(memory_data):
            entry_hash = entry.get('entry_hash', '')
            stored_prev_hash = entry.get('previous_entry_hash', '')

            # Calcular hash de la entrada
            entry_content = json.dumps({
                k: v for k, v in entry.items()
                if k != 'entry_hash'
            }, sort_keys=True, ensure_ascii=False)

            calculated_hash = hashlib.sha256(entry_content.encode('utf-8')).hexdigest()

            # Verificar hash de entrada actual
            if calculated_hash != entry_hash:
                issues.append(f"Entry {i}: hash mismatch")

            # Verificar cadena con entrada anterior
            if i > 0:
                prev_entry_hash = memory_data[i-1].get('entry_hash', '')
                if stored_prev_hash != prev_entry_hash:
                    issues.append(f"Entry {i}: chain break")

        return issues

    def check_git_hooks_installation(self):
        """Verifica instalación de Git hooks"""
        print("🔗 Verificando instalación de Git hooks...")

        check_name = "Git Hooks Installation"
        self.total_checks += 1

        # Buscar repositorios con hooks
        hook_repos = self._find_repositories_with_hooks()

        if not hook_repos:
            self.checks[check_name] = {
                'status': 'WARN',
                'message': 'No se encontraron repositorios con hooks',
                'details': 'Hooks no instalados en ningún repositorio'
            }
            return

        # Verificar hooks en cada repositorio
        hook_status = {}
        all_good = True

        for repo_path in hook_repos:
            repo_name = os.path.basename(repo_path)
            hooks_status = self._check_repo_hooks(repo_path)
            hook_status[repo_name] = hooks_status

            if not all(hooks_status.values()):
                all_good = False

        if all_good:
            self.checks[check_name] = {
                'status': 'PASS',
                'message': 'Git hooks instalados correctamente',
                'details': f'{len(hook_repos)} repositorio(s) con hooks activos'
            }
            self.overall_score += 1
        else:
            self.checks[check_name] = {
                'status': 'WARN',
                'message': 'Algunos hooks faltantes o no ejecutables',
                'details': f'Hooks verificados en {len(hook_repos)} repositorio(s)'
            }

    def _find_repositories_with_hooks(self):
        """Encuentra repositorios Git con hooks instalados"""
        repos = []

        # Buscar en el directorio actual y subdirectorios
        for root, dirs, files in os.walk('.'):
            if '.git' in dirs:
                git_dir = os.path.join(root, '.git')
                hooks_dir = os.path.join(git_dir, 'hooks')

                if os.path.exists(hooks_dir):
                    # Verificar si hay hooks de Shadow
                    post_commit = os.path.join(hooks_dir, 'post-commit')
                    post_push = os.path.join(hooks_dir, 'post-push')

                    if os.path.exists(post_commit) or os.path.exists(post_push):
                        repos.append(root)

        return repos

    def _check_repo_hooks(self, repo_path):
        """Verifica hooks en un repositorio específico"""
        hooks_dir = os.path.join(repo_path, '.git', 'hooks')
        hooks_status = {}

        # Verificar post-commit
        post_commit = os.path.join(hooks_dir, 'post-commit')
        if os.path.exists(post_commit):
            hooks_status['post-commit'] = os.access(post_commit, os.X_OK)
        else:
            hooks_status['post-commit'] = False

        # Verificar post-push
        post_push = os.path.join(hooks_dir, 'post-push')
        if os.path.exists(post_push):
            hooks_status['post-push'] = os.access(post_push, os.X_OK)
        else:
            hooks_status['post-push'] = False

        return hooks_status

    def check_dependencies(self):
        """Verifica dependencias Python"""
        print("🐍 Verificando dependencias Python...")

        check_name = "Dependencies"
        self.total_checks += 1

        required_modules = ['json', 'os', 'sys', 'hashlib', 'pathlib']
        missing_modules = []

        for module in required_modules:
            try:
                __import__(module)
            except ImportError:
                missing_modules.append(module)

        # Verificar módulos opcionales
        optional_modules = ['ast', 'logging']
        optional_missing = []

        for module in optional_modules:
            try:
                __import__(module)
            except ImportError:
                optional_missing.append(module)

        if missing_modules:
            self.checks[check_name] = {
                'status': 'FAIL',
                'message': 'Dependencias críticas faltantes',
                'details': f'Módulos faltantes: {", ".join(missing_modules)}'
            }
        elif optional_missing:
            self.checks[check_name] = {
                'status': 'WARN',
                'message': 'Algunas dependencias opcionales faltantes',
                'details': f'Módulos opcionales faltantes: {", ".join(optional_missing)}'
            }
        else:
            self.checks[check_name] = {
                'status': 'PASS',
                'message': 'Todas las dependencias disponibles',
                'details': f'Python {sys.version.split()[0]}'
            }
            self.overall_score += 1

    def check_memory_storage(self):
        """Verifica almacenamiento de memoria"""
        print("💾 Verificando almacenamiento de memoria...")

        check_name = "Memory Storage"
        self.total_checks += 1

        try:
            # Verificar que el directorio existe
            if not os.path.exists(self.memory_path):
                self.checks[check_name] = {
                    'status': 'FAIL',
                    'message': 'Directorio de memoria no encontrado',
                    'details': f'Expected: {self.memory_path}'
                }
                return

            # Verificar permisos de escritura
            if not os.access(self.memory_path, os.W_OK):
                self.checks[check_name] = {
                    'status': 'FAIL',
                    'message': 'Sin permisos de escritura en directorio de memoria',
                    'details': f'Directory: {self.memory_path}'
                }
                return

            # Verificar archivo de memoria
            if os.path.exists(self.memory_file):
                # Verificar permisos de lectura/escritura
                readable = os.access(self.memory_file, os.R_OK)
                writable = os.access(self.memory_file, os.W_OK)

                if readable and writable:
                    # Obtener tamaño del archivo
                    size = os.path.getsize(self.memory_file)
                    self.checks[check_name] = {
                        'status': 'PASS',
                        'message': 'Almacenamiento de memoria operativo',
                        'details': f'Archivo accesible ({size} bytes)'
                    }
                    self.overall_score += 1
                else:
                    self.checks[check_name] = {
                        'status': 'FAIL',
                        'message': 'Permisos insuficientes en archivo de memoria',
                        'details': f'Readable: {readable}, Writable: {writable}'
                    }
            else:
                # Archivo no existe pero directorio sí - esto es OK para nueva instalación
                self.checks[check_name] = {
                    'status': 'PASS',
                    'message': 'Directorio de memoria listo',
                    'details': 'Archivo se creará en primer uso'
                }
                self.overall_score += 1

        except Exception as e:
            self.checks[check_name] = {
                'status': 'FAIL',
                'message': 'Error verificando almacenamiento',
                'details': str(e)
            }

    def check_recent_activity(self):
        """Verifica actividad reciente del sistema"""
        print("📈 Verificando actividad reciente...")

        check_name = "Recent Activity"
        self.total_checks += 1

        try:
            if not os.path.exists(self.memory_file):
                self.checks[check_name] = {
                    'status': 'WARN',
                    'message': 'Sin actividad (archivo de memoria no existe)',
                    'details': 'Sistema no ha registrado eventos aún'
                }
                return

            with open(self.memory_file, 'r', encoding='utf-8') as f:
                memory_data = json.load(f)

            if not memory_data:
                self.checks[check_name] = {
                    'status': 'WARN',
                    'message': 'Sin actividad registrada',
                    'details': 'Archivo de memoria vacío'
                }
                return

            # Encontrar entrada más reciente
            most_recent = max(memory_data, key=lambda x: x.get('timestamp', ''))

            timestamp_str = most_recent.get('timestamp', '')
            if timestamp_str:
                try:
                    # Parsear timestamp
                    if timestamp_str.endswith('Z'):
                        timestamp_str = timestamp_str[:-1] + '+00:00'

                    entry_time = datetime.fromisoformat(timestamp_str)
                    now = datetime.now(entry_time.tzinfo)
                    time_diff = now - entry_time

                    # Verificar si es reciente (últimas 24 horas)
                    if time_diff < timedelta(hours=24):
                        self.checks[check_name] = {
                            'status': 'PASS',
                            'message': 'Sistema activo recientemente',
                            'details': f'Última actividad: {time_diff.seconds // 3600} horas atrás'
                        }
                        self.overall_score += 1
                    else:
                        self.checks[check_name] = {
                            'status': 'WARN',
                            'message': 'Actividad antigua detectada',
                            'details': f'Última actividad: {time_diff.days} días atrás'
                        }
                except ValueError:
                    self.checks[check_name] = {
                        'status': 'WARN',
                        'message': 'Timestamp inválido en entrada reciente',
                        'details': 'Formato de timestamp problemático'
                    }
            else:
                self.checks[check_name] = {
                    'status': 'WARN',
                    'message': 'Entrada sin timestamp',
                    'details': 'Formato de entrada incompleto'
                }

        except Exception as e:
            self.checks[check_name] = {
                'status': 'FAIL',
                'message': 'Error verificando actividad',
                'details': str(e)
            }

    def _calculate_overall_score(self):
        """Calcula puntuación general"""
        if self.total_checks > 0:
            percentage = (self.overall_score / self.total_checks) * 100
            self.overall_percentage = round(percentage, 1)
        else:
            self.overall_percentage = 0

    def _print_final_report(self):
        """Imprime reporte final"""
        print("\n" + "="*50)
        print("REPORTE DE SALUD SHADOW SYSTEM")
        print("="*50)

        for check_name, result in self.checks.items():
            status = result['status']
            message = result['message']
            details = result.get('details', '')

            if status == 'PASS':
                print(f"✅ {check_name}: {message}")
            elif status == 'WARN':
                print(f"⚠️  {check_name}: {message}")
            else:  # FAIL
                print(f"❌ {check_name}: {message}")

            if details:
                print(f"   └─ {details}")

        print(f"\nOverall Health Score: {self.overall_score}/{self.total_checks} ({self.overall_percentage}%)")

        if self.overall_percentage >= 80:
            print("🎉 Estado: EXCELENTE")
        elif self.overall_percentage >= 60:
            print("👍 Estado: BUENO")
        elif self.overall_percentage >= 40:
            print("⚠️  Estado: REQUIERE ATENCIÓN")
        else:
            print("❌ Estado: CRÍTICO")


def main():
    """Función principal"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Healthcheck del sistema Shadow',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

# Healthcheck completo
python3 scripts/validate_shadow_health.py

# Healthcheck con path específico
python3 scripts/validate_shadow_health.py --path /ruta/al/proyecto/shadow
        """
    )

    parser.add_argument(
        '--path',
        help='Ruta al proyecto Shadow (detecta automáticamente si no se especifica)'
    )

    args = parser.parse_args()

    # Crear y ejecutar healthcheck
    checker = ShadowHealthChecker(args.path)
    checker.run_full_health_check()


if __name__ == "__main__":
    main()