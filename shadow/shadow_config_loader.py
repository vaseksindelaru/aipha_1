#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
shadow_config_loader.py - Cargador de configuración para el sistema Shadow

Este módulo carga y gestiona la configuración centralizada del sistema Shadow
incluyendo todos los componentes críticos implementados.

Autor: Shadow System
Versión: 1.0.0
"""

import os
import sys
import yaml
import logging
from typing import Dict, Any, Optional
from pathlib import Path

class ShadowConfigLoader:
    """
    Cargador de configuración para el sistema Shadow-AiphaLab
    """
    
    def __init__(self, config_path: str = None):
        """
        Inicializa el cargador de configuración
        
        Args:
            config_path: Ruta al archivo de configuración YAML
        """
        self.config_path = config_path or self._find_config_file()
        self.config = {}
        self.logger = logging.getLogger(__name__)
        
        # Cargar configuración
        self.load_config()
    
    def _find_config_file(self) -> str:
        """Busca el archivo de configuración en ubicaciones estándar"""
        possible_paths = [
            os.path.join(os.path.dirname(__file__), 'config_shadow.yaml'),
            './config_shadow.yaml',
            './shadow/config_shadow.yaml',
            './aipha_0.0.1/config_shadow.yaml'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        raise FileNotFoundError("No se encontró el archivo config_shadow.yaml")
    
    def load_config(self):
        """Carga la configuración desde el archivo YAML"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            
            # Procesar variables de entorno
            self._process_environment_variables()
            
            # Validar configuración
            self._validate_config()
            
            self.logger.info(f"✅ Configuración cargada desde: {self.config_path}")
            
        except Exception as e:
            self.logger.error(f"Error cargando configuración: {e}")
            raise
    
    def _process_environment_variables(self):
        """Procesa variables de entorno en la configuración"""
        import re
        
        def replace_env_vars(obj):
            if isinstance(obj, dict):
                return {k: replace_env_vars(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [replace_env_vars(item) for item in obj]
            elif isinstance(obj, str):
                # Buscar patrones ${VAR_NAME}
                env_var_pattern = r'\$\{([^}]+)\}'
                def replacer(match):
                    env_var = match.group(1)
                    return os.getenv(env_var, match.group(0))  # Retornar original si no existe
                
                return re.sub(env_var_pattern, replacer, obj)
            else:
                return obj
        
        self.config = replace_env_vars(self.config)
    
    def _validate_config(self):
        """Valida la configuración cargada"""
        required_sections = [
            'gemini_integration',
            'webhook_listener', 
            'llm_protocol',
            'repository_monitoring',
            'integrity_analysis',
            'shadow_memory'
        ]
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Sección requerida faltante: {section}")
        
        # Validar gemini_integration
        gemini = self.config.get('gemini_integration', {})
        if not gemini.get('api_key'):
            self.logger.warning("API key de Gemini no configurada")
        
        # Validar webhook_listener
        webhook = self.config.get('webhook_listener', {})
        if webhook.get('enabled') and not webhook.get('port'):
            raise ValueError("Webhook listener habilitado pero sin puerto configurado")
        
        self.logger.info("✅ Configuración validada correctamente")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Obtiene un valor de configuración usando notación de puntos
        
        Args:
            key: Clave en formato 'section.subsection.field'
            default: Valor por defecto si no se encuentra
            
        Returns:
            Valor de configuración
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def is_enabled(self, component: str) -> bool:
        """
        Verifica si un componente está habilitado
        
        Args:
            component: Nombre del componente
            
        Returns:
            True si está habilitado
        """
        return self.get(f'{component}.enabled', False)
    
    def get_gemini_config(self) -> Dict[str, Any]:
        """Obtiene configuración específica de Gemini"""
        return {
            'api_endpoint': self.get('gemini_integration.api_endpoint'),
            'api_key': self.get('gemini_integration.api_key'),
            'communication_mode': self.get('gemini_integration.communication_mode', 'hybrid'),
            'sync_interval': self.get('gemini_integration.sync_interval', 300),
            'enabled': self.is_enabled('gemini_integration')
        }
    
    def get_webhook_config(self) -> Dict[str, Any]:
        """Obtiene configuración específica del webhook"""
        return {
            'host': self.get('webhook_listener.host', '0.0.0.0'),
            'port': self.get('webhook_listener.port', 8081),
            'debug': self.get('webhook_listener.debug', False),
            'endpoints': self.get('webhook_listener.endpoints', {}),
            'enabled': self.is_enabled('webhook_listener')
        }
    
    def get_protocol_config(self) -> Dict[str, Any]:
        """Obtiene configuración del protocolo LLM-to-LLM"""
        return {
            'current_version': self.get('llm_protocol.current_version', 'shadow_1.0'),
            'target_versions': self.get('llm_protocol.target_versions', []),
            'context_transfer': self.get('llm_protocol.context_transfer', {}),
            'enabled': self.is_enabled('llm_protocol')
        }
    
    def get_repository_config(self) -> Dict[str, Any]:
        """Obtiene configuración de monitoreo de repositorio"""
        return {
            'url': self.get('repository_monitoring.target_repository.url'),
            'local_path': self.get('repository_monitoring.target_repository.local_path'),
            'check_interval': self.get('repository_monitoring.monitoring.check_interval_seconds', 30),
            'enabled': self.is_enabled('repository_monitoring')
        }
    
    def get_memory_config(self) -> Dict[str, Any]:
        """Obtiene configuración de memoria Shadow"""
        return {
            'base_path': self.get('shadow_memory.base_path'),
            'max_entries': self.get('shadow_memory.max_history_entries', 10000),
            'enabled': self.is_enabled('shadow_memory')
        }
    
    def get_all_config(self) -> Dict[str, Any]:
        """Obtiene toda la configuración"""
        return self.config
    
    def get_summary(self) -> Dict[str, Any]:
        """Obtiene resumen de la configuración"""
        summary = {
            'config_file': self.config_path,
            'components_enabled': {},
            'gemini_integration': self.get_gemini_config(),
            'webhook_listener': self.get_webhook_config(),
            'llm_protocol': self.get_protocol_config(),
            'repository_monitoring': self.get_repository_config(),
            'integrity_analysis': {
                'enabled': self.is_enabled('integrity_analysis'),
                'interval_minutes': self.get('integrity_analysis.analysis_interval_minutes', 15)
            },
            'shadow_memory': self.get_memory_config()
        }
        
        # Componentes habilitados
        components = [
            'gemini_integration', 'webhook_listener', 'llm_protocol', 
            'repository_monitoring', 'integrity_analysis', 'shadow_memory'
        ]
        
        for component in components:
            summary['components_enabled'][component] = self.is_enabled(component)
        
        return summary


def test_config_loader():
    """Función de prueba para el cargador de configuración"""
    try:
        print("🧪 Probando Shadow Config Loader...")
        
        # Crear loader
        loader = ShadowConfigLoader()
        
        # Obtener resumen
        summary = loader.get_summary()
        print("✅ Resumen de configuración obtenido")
        
        # Probar acceso a valores
        print(f"📊 Componentes habilitados: {summary['components_enabled']}")
        print(f"🔗 Webhook puerto: {summary['webhook_listener']['port']}")
        print(f"📁 Memoria base: {summary['shadow_memory']['base_path']}")
        
        # Probar configuración de Gemini
        gemini_config = loader.get_gemini_config()
        print(f"🤖 Gemini endpoint: {gemini_config['api_endpoint']}")
        print(f"🔑 Gemini key configurada: {'Sí' if gemini_config['api_key'] else 'No'}")
        
        print("✅ Test de configuración exitoso")
        return True
        
    except Exception as e:
        print(f"❌ Error en test de configuración: {e}")
        return False


if __name__ == "__main__":
    # Configurar logging básico
    logging.basicConfig(level=logging.INFO)
    
    # Ejecutar test
    if test_config_loader():
        print("\n🎉 Sistema de configuración operativo")
    else:
        print("\n❌ Problemas con la configuración")