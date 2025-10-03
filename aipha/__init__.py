# aipha/__init__.py

"""
Aipha_1.1 - Sistema de Auto-Construcción de Trading Algorítmico
Paquete principal que contiene todos los componentes del sistema.
"""

__version__ = "1.1.0"
__author__ = "Vaclav Sindelar"

# Imports públicos del paquete
from .core.redesign_helper import RedesignHelper

__all__ = ['RedesignHelper', '__version__']