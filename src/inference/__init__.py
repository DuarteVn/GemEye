"""
Módulo de inferência.
Contém integrações com modelos multimodais e de detecção de objetos.
"""

from .multimodal import generate_multimodal_answer
from .object_detection import detect_objects, DetectedObject

__all__ = ["generate_multimodal_answer", "detect_objects", "DetectedObject"]

