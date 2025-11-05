#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
llm_communication_protocol.py - Protocolo de Comunicación LLM-to-LLM para Shadow

Este módulo implementa el protocolo de comunicación entre diferentes versiones
de Shadow (Shadow_1.0 ↔ Shadow_2.0), permitiendo la transferencia de contexto,
validación de integridad y coordinación durante transiciones del sistema.

Autor: Shadow System
Versión: 1.0.0
"""

import os
import sys
import json
import logging
import hashlib
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum

# Añadir directorio padre al path para imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class ProtocolState(Enum):
    """Estados del protocolo de comunicación"""
    IDLE = "idle"
    HANDSHAKE = "handshake"
    CONTEXT_TRANSFER = "context_transfer"
    VALIDATION = "validation"
    COMPLETE = "complete"
    ERROR = "error"

class ShadowVersion(Enum):
    """Versiones de Shadow compatibles"""
    SHADOW_1_0 = "shadow_1.0"
    SHADOW_2_0 = "shadow_2.0"

@dataclass
class ContextPacket:
    """Paquete de contexto para transferencia"""
    version: str
    timestamp: str
    sequence_number: int
    total_packets: int
    checksum: str
    data: Dict[str, Any]
    metadata: Dict[str, str]

@dataclass
class CommunicationLog:
    """Registro de comunicación para auditoría"""
    timestamp: str
    from_version: str
    to_version: str
    action: str
    status: str
    details: Dict[str, Any]

class LLMCommunicationProtocol:
    """
    Protocolo de comunicación LLM-to-LLM para sistema Shadow
    """
    
    def __init__(self, 
                 shadow_version: str = ShadowVersion.SHADOW_1_0.value,
                 shadow_memory_path: str = "./aipha_memory_storage/action_history",
                 communication_log_path: str = None):
        """
        Inicializa el protocolo de comunicación
        
        Args:
            shadow_version: Versión actual de Shadow
            shadow_memory_path: Ruta a memoria Shadow
            communication_log_path: Ruta para logs de comunicación
        """
        self.shadow_version = shadow_version
        self.shadow_memory_path = shadow_memory_path
        self.communication_log_path = communication_log_path or os.path.join(shadow_memory_path, 'communication_log.json')
        
        # Estado del protocolo
        self.current_state = ProtocolState.IDLE
        self.peer_version = None
        self.context_transfer_id = None
        self.sequence_counter = 0
        
        # Cache de contextos
        self.context_cache = {}
        self.transfer_queue = []
        
        # Configurar logging
        self._setup_logging()
        
        # Inicializar estructura de directorios
        self._initialize_directories()
        
        self.logger.info(f"🔗 LLM Communication Protocol inicializado como {shadow_version}")

    def _setup_logging(self):
        """Configura el sistema de logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def _initialize_directories(self):
        """Inicializa directorios necesarios"""
        try:
            os.makedirs(os.path.dirname(self.communication_log_path), exist_ok=True)
            
            # Crear archivo de log si no existe
            if not os.path.exists(self.communication_log_path):
                with open(self.communication_log_path, 'w', encoding='utf-8') as f:
                    json.dump([], f, indent=2, ensure_ascii=False)
                    
        except Exception as e:
            self.logger.error(f"Error inicializando directorios: {e}")

    async def register_shadow_2_0_startup(self, 
                                        shadow_2_0_endpoint: str = None,
                                        verification_token: str = None) -> Dict[str, Any]:
        """
        Notifica a shadowAipha_1.0 que Shadow_2.0 arrancó
        
        Args:
            shadow_2_0_endpoint: Endpoint de Shadow_2.0
            verification_token: Token de verificación
            
        Returns:
            Dict con resultado del registro
        """
        result = {
            'success': False,
            'timestamp': datetime.now().isoformat(),
            'action': 'shadow_2_0_startup_registration',
            'peer_version': ShadowVersion.SHADOW_2_0.value,
            'status': 'unknown',
            'error': None
        }
        
        try:
            self.logger.info("🔔 Registrando startup de Shadow_2.0...")
            
            # Cambiar estado a handshake
            self.current_state = ProtocolState.HANDSHAKE
            
            # Preparar mensaje de registro
            startup_message = {
                'type': 'startup_notification',
                'source_version': self.shadow_version,
                'target_version': ShadowVersion.SHADOW_2_0.value,
                'timestamp': datetime.now().isoformat(),
                'endpoint': shadow_2_0_endpoint,
                'verification_token': verification_token or self._generate_verification_token(),
                'capabilities': self._get_capabilities()
            }
            
            # Enviar notificación (simulado)
            response = await self._send_notification(startup_message)
            
            if response.get('success'):
                result.update({
                    'success': True,
                    'status': 'registered',
                    'peer_endpoint': response.get('endpoint'),
                    'response_data': response
                })
                
                # Registrar en memoria Shadow
                await self._register_communication_event(
                    from_version=self.shadow_version,
                    to_version=ShadowVersion.SHADOW_2_0.value,
                    action='startup_registration',
                    status='success',
                    details=response
                )
                
                # Cambiar estado a idle
                self.current_state = ProtocolState.IDLE
                
                self.logger.info("✅ Startup de Shadow_2.0 registrado exitosamente")
            else:
                result['error'] = response.get('error', 'Unknown error')
                self.current_state = ProtocolState.ERROR
                
        except Exception as e:
            self.logger.error(f"Error registrando startup de Shadow_2.0: {e}")
            result['error'] = str(e)
            self.current_state = ProtocolState.ERROR
        
        return result

    async def request_full_context_transfer(self, 
                                          target_version: str = ShadowVersion.SHADOW_2_0.value,
                                          transfer_mode: str = "incremental") -> Dict[str, Any]:
        """
        Pide transferencia completa de contexto
        
        Args:
            target_version: Versión objetivo
            transfer_mode: Modo de transferencia (full, incremental, differential)
            
        Returns:
            Dict con resultado de la transferencia
        """
        result = {
            'success': False,
            'timestamp': datetime.now().isoformat(),
            'action': 'context_transfer_request',
            'target_version': target_version,
            'transfer_id': None,
            'status': 'unknown',
            'error': None
        }
        
        try:
            self.logger.info(f"📦 Solicitando transferencia de contexto a {target_version}...")
            
            # Validar estado
            if self.current_state not in [ProtocolState.IDLE, ProtocolState.HANDSHAKE]:
                raise Exception(f"Protocolo en estado inválido: {self.current_state}")
            
            # Cambiar estado
            self.current_state = ProtocolState.CONTEXT_TRANSFER
            
            # Generar ID de transferencia
            transfer_id = hashlib.md5(f"{datetime.now().isoformat()}{self.sequence_counter}".encode()).hexdigest()[:8]
            self.context_transfer_id = transfer_id
            result['transfer_id'] = transfer_id
            
            # Obtener contexto desde memoria Shadow
            context_data = await self._gather_full_context()
            
            # Preparar solicitud de transferencia
            transfer_request = {
                'type': 'context_transfer_request',
                'transfer_id': transfer_id,
                'source_version': self.shadow_version,
                'target_version': target_version,
                'transfer_mode': transfer_mode,
                'timestamp': datetime.now().isoformat(),
                'context_size': len(str(context_data)),
                'sequence_count': len(context_data.get('entries', []))
            }
            
            # Enviar solicitud
            response = await self._send_transfer_request(transfer_request, context_data)
            
            if response.get('success'):
                result.update({
                    'success': True,
                    'status': 'transfer_initiated',
                    'response_data': response
                })
                
                # Registrar evento
                await self._register_communication_event(
                    from_version=self.shadow_version,
                    to_version=target_version,
                    action='context_transfer_request',
                    status='success',
                    details={'transfer_id': transfer_id, 'mode': transfer_mode}
                )
                
                self.logger.info(f"✅ Transferencia de contexto iniciada: {transfer_id}")
            else:
                result['error'] = response.get('error', 'Transfer request failed')
                self.current_state = ProtocolState.ERROR
                
        except Exception as e:
            self.logger.error(f"Error solicitando transferencia de contexto: {e}")
            result['error'] = str(e)
            self.current_state = ProtocolState.ERROR
        
        return result

    async def validate_context_integrity(self, 
                                       received_context: Dict[str, Any],
                                       expected_checksum: str = None) -> Dict[str, Any]:
        """
        Valida que el contexto recibido está íntegro
        
        Args:
            received_context: Contexto recibido
            expected_checksum: Checksum esperado
            
        Returns:
            Dict con resultado de validación
        """
        result = {
            'valid': False,
            'timestamp': datetime.now().isoformat(),
            'action': 'context_integrity_validation',
            'checksum_match': False,
            'data_integrity': False,
            'completeness': False,
            'issues': [],
            'warnings': []
        }
        
        try:
            self.logger.info("🔍 Validando integridad del contexto recibido...")
            
            # Cambiar estado
            self.current_state = ProtocolState.VALIDATION
            
            # Validación 1: Checksum
            calculated_checksum = self._calculate_context_checksum(received_context)
            checksum_match = calculated_checksum == expected_checksum if expected_checksum else True
            result['checksum_match'] = checksum_match
            
            if not checksum_match:
                result['issues'].append(f"Checksum mismatch: expected {expected_checksum}, calculated {calculated_checksum}")
            
            # Validación 2: Integridad de datos
            data_integrity = self._validate_data_integrity(received_context)
            result['data_integrity'] = data_integrity
            
            if not data_integrity['valid']:
                result['issues'].extend(data_integrity['issues'])
            
            # Validación 3: Completitud
            completeness = self._validate_completeness(received_context)
            result['completeness'] = completeness['valid']
            
            if not completeness['valid']:
                result['warnings'].extend(completeness['warnings'])
            
            # Resultado final
            result['valid'] = checksum_match and data_integrity['valid'] and completeness['valid']
            
            # Cambiar estado según resultado
            if result['valid']:
                self.current_state = ProtocolState.COMPLETE
                self.logger.info("✅ Validación de integridad exitosa")
            else:
                self.current_state = ProtocolState.ERROR
                self.logger.warning(f"⚠️  Validación falló: {result['issues']}")
            
            # Registrar evento
            await self._register_communication_event(
                from_version=received_context.get('source_version', 'unknown'),
                to_version=self.shadow_version,
                action='context_integrity_validation',
                status='success' if result['valid'] else 'failed',
                details={
                    'checksum_match': checksum_match,
                    'data_integrity': data_integrity['valid'],
                    'completeness': completeness['valid'],
                    'issues_count': len(result['issues']),
                    'warnings_count': len(result['warnings'])
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error validando integridad: {e}")
            result['issues'].append(f"Validation error: {str(e)}")
            self.current_state = ProtocolState.ERROR
        
        return result

    # Métodos auxiliares privados
    
    def _generate_verification_token(self) -> str:
        """Genera token de verificación"""
        return hashlib.sha256(f"{self.shadow_version}{datetime.now().isoformat()}".encode()).hexdigest()[:16]
    
    def _get_capabilities(self) -> Dict[str, Any]:
        """Obtiene capacidades del sistema actual"""
        return {
            'context_transfer': True,
            'real_time_monitoring': True,
            'integrity_validation': True,
            'bidirectional_communication': True,
            'version_compatibility': [ShadowVersion.SHADOW_1_0.value, ShadowVersion.SHADOW_2_0.value]
        }
    
    async def _send_notification(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Envía notificación (simulado para testing)"""
        # Simular envío exitoso
        await asyncio.sleep(0.1)  # Simular delay de red
        
        return {
            'success': True,
            'endpoint': message.get('endpoint'),
            'response': 'acknowledged',
            'timestamp': datetime.now().isoformat()
        }
    
    async def _gather_full_context(self) -> Dict[str, Any]:
        """Recopila contexto completo desde memoria Shadow"""
        try:
            # Leer memoria Shadow
            memory_file = os.path.join(self.shadow_memory_path, 'current_history.json')
            
            if os.path.exists(memory_file):
                with open(memory_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            else:
                history = []
            
            context_data = {
                'source_version': self.shadow_version,
                'timestamp': datetime.now().isoformat(),
                'total_entries': len(history),
                'entries': history[-1000:],  # Últimas 1000 entradas
                'metadata': {
                    'generation_time': datetime.now().isoformat(),
                    'data_size': len(str(history)),
                    'integrity_hash': self._calculate_context_checksum({'entries': history})
                }
            }
            
            return context_data
            
        except Exception as e:
            self.logger.error(f"Error recopilando contexto: {e}")
            return {
                'source_version': self.shadow_version,
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
    
    async def _send_transfer_request(self, request: Dict[str, Any], context_data: Dict[str, Any]) -> Dict[str, Any]:
        """Envía solicitud de transferencia"""
        # Simular envío exitoso
        await asyncio.sleep(0.1)
        
        return {
            'success': True,
            'transfer_accepted': True,
            'response_timestamp': datetime.now().isoformat(),
            'estimated_completion': datetime.now().isoformat()
        }
    
    def _calculate_context_checksum(self, context: Dict[str, Any]) -> str:
        """Calcula checksum del contexto"""
        context_str = json.dumps(context, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(context_str.encode()).hexdigest()
    
    def _validate_data_integrity(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Valida integridad de datos"""
        issues = []
        
        # Validar estructura básica
        required_fields = ['source_version', 'timestamp', 'entries']
        for field in required_fields:
            if field not in context:
                issues.append(f"Campo requerido faltante: {field}")
        
        # Validar entradas
        if 'entries' in context:
            for i, entry in enumerate(context['entries']):
                if not isinstance(entry, dict):
                    issues.append(f"Entrada {i} no es un diccionario")
                elif 'timestamp' not in entry:
                    issues.append(f"Entrada {i} sin timestamp")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues
        }
    
    def _validate_completeness(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Valida completitud del contexto"""
        warnings = []
        
        # Verificar completitud de entradas
        if 'entries' in context:
            if len(context['entries']) < 10:
                warnings.append(f"Contexto muy pequeño: {len(context['entries'])} entradas")
            
            # Verificar que hay entradas recientes
            recent_threshold = datetime.now() - timedelta(days=1)
            recent_entries = 0
            
            for entry in context['entries']:
                try:
                    entry_time = datetime.fromisoformat(entry.get('timestamp', ''))
                    if entry_time > recent_threshold:
                        recent_entries += 1
                except:
                    pass
            
            if recent_entries == 0:
                warnings.append("No hay entradas recientes en el contexto")
        
        return {
            'valid': len(warnings) == 0,
            'warnings': warnings
        }
    
    async def _register_communication_event(self, 
                                           from_version: str,
                                           to_version: str,
                                           action: str,
                                           status: str,
                                           details: Dict[str, Any]):
        """Registra evento de comunicación"""
        try:
            # Crear registro
            log_entry = CommunicationLog(
                timestamp=datetime.now().isoformat(),
                from_version=from_version,
                to_version=to_version,
                action=action,
                status=status,
                details=details
            )
            
            # Leer logs existentes
            if os.path.exists(self.communication_log_path):
                with open(self.communication_log_path, 'r', encoding='utf-8') as f:
                    logs = json.load(f)
            else:
                logs = []
            
            # Añadir nuevo log
            logs.append(asdict(log_entry))
            
            # Mantener solo últimos 1000 logs
            if len(logs) > 1000:
                logs = logs[-1000:]
            
            # Escribir de vuelta
            with open(self.communication_log_path, 'w', encoding='utf-8') as f:
                json.dump(logs, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            self.logger.error(f"Error registrando evento: {e}")
    
    def get_protocol_status(self) -> Dict[str, Any]:
        """Obtiene estado actual del protocolo"""
        return {
            'timestamp': datetime.now().isoformat(),
            'current_state': self.current_state.value,
            'shadow_version': self.shadow_version,
            'peer_version': self.peer_version,
            'context_transfer_id': self.context_transfer_id,
            'sequence_counter': self.sequence_counter,
            'cache_size': len(self.context_cache),
            'queue_size': len(self.transfer_queue)
        }

    async def simulate_full_communication(self) -> Dict[str, Any]:
        """Simula comunicación completa para testing"""
        results = {
            'simulation_id': hashlib.md5(datetime.now().isoformat().encode()).hexdigest()[:8],
            'start_time': datetime.now().isoformat(),
            'steps': []
        }
        
        try:
            # Paso 1: Registro startup Shadow_2.0
            step1 = await self.register_shadow_2_0_startup(
                shadow_2_0_endpoint="shadow_2.0.endpoint.local:8080"
            )
            results['steps'].append({
                'step': 1,
                'action': 'register_shadow_2_0_startup',
                'result': step1
            })
            
            # Paso 2: Solicitar transferencia de contexto
            step2 = await self.request_full_context_transfer(
                target_version=ShadowVersion.SHADOW_2_0.value,
                transfer_mode="full"
            )
            results['steps'].append({
                'step': 2,
                'action': 'request_context_transfer',
                'result': step2
            })
            
            # Paso 3: Validar contexto recibido (simulado)
            sample_context = await self._gather_full_context()
            step3 = await self.validate_context_integrity(
                received_context=sample_context
            )
            results['steps'].append({
                'step': 3,
                'action': 'validate_context_integrity',
                'result': step3
            })
            
            # Resultado final
            results['success'] = all(
                step['result'].get('success', False) for step in results['steps']
            )
            results['end_time'] = datetime.now().isoformat()
            
        except Exception as e:
            results['error'] = str(e)
            results['success'] = False
        
        return results


def main():
    """Función principal para testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='LLM Communication Protocol')
    parser.add_argument('--shadow-version', default=ShadowVersion.SHADOW_1_0.value,
                       choices=[v.value for v in ShadowVersion],
                       help='Versión de Shadow')
    parser.add_argument('--memory-path', default='./aipha_memory_storage/action_history',
                       help='Ruta a memoria Shadow')
    parser.add_argument('--mode', choices=['status', 'simulate', 'register', 'transfer', 'validate'], 
                       default='status', help='Modo de operación')
    
    args = parser.parse_args()
    
    # Crear protocolo
    protocol = LLMCommunicationProtocol(
        shadow_version=args.shadow_version,
        shadow_memory_path=args.memory_path
    )
    
    if args.mode == 'status':
        print("🔍 Estado del Protocolo de Comunicación:")
        status = protocol.get_protocol_status()
        print(json.dumps(status, indent=2, ensure_ascii=False))
        
    elif args.mode == 'simulate':
        print("🧪 Simulando comunicación completa...")
        asyncio.run(protocol.simulate_full_communication()).then(lambda result: 
            print(json.dumps(result, indent=2, ensure_ascii=False))
        )
        
    elif args.mode == 'register':
        print("🔔 Registrando startup Shadow_2.0...")
        result = asyncio.run(protocol.register_shadow_2_0_startup())
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
    elif args.mode == 'transfer':
        print("📦 Solicitando transferencia de contexto...")
        result = asyncio.run(protocol.request_full_context_transfer())
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
    elif args.mode == 'validate':
        print("🔍 Validando integridad de contexto...")
        sample_context = {
            'source_version': ShadowVersion.SHADOW_2_0.value,
            'timestamp': datetime.now().isoformat(),
            'entries': [{'timestamp': datetime.now().isoformat(), 'action': 'test'}]
        }
        result = asyncio.run(protocol.validate_context_integrity(sample_context))
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()