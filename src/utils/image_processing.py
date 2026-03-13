"""
Módulo de processamento de imagens.

Fornece funções utilitárias para pré-processamento, redimensionamento
e conversão de imagens.
"""

import numpy as np
from typing import Tuple, Optional, Union
from PIL import Image
import cv2


def convert_to_rgb(image: np.ndarray) -> np.ndarray:
    """
    Converte uma imagem para formato RGB.
    
    Detecta automaticamente o formato de entrada (BGR, grayscale, RGBA)
    e converte para RGB.
    
    Args:
        image: Imagem como array NumPy.
    
    Returns:
        Imagem em formato RGB.
    
    Example:
        >>> bgr_image = cv2.imread("photo.jpg")  # OpenCV lê em BGR
        >>> rgb_image = convert_to_rgb(bgr_image)
    """
    if image is None:
        raise ValueError("Imagem não pode ser None")
    
    # Se já tem 3 canais, assume BGR (OpenCV) e converte
    if len(image.shape) == 3:
        if image.shape[2] == 3:
            # Assume BGR -> RGB
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif image.shape[2] == 4:
            # RGBA/BGRA -> RGB
            return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    
    # Grayscale -> RGB
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    return image


def resize_image(
    image: np.ndarray,
    target_size: Tuple[int, int],
    keep_aspect_ratio: bool = True,
    interpolation: int = cv2.INTER_LINEAR
) -> np.ndarray:
    """
    Redimensiona uma imagem para o tamanho alvo.
    
    Args:
        image: Imagem como array NumPy.
        target_size: Tupla (largura, altura) desejada.
        keep_aspect_ratio: Se True, mantém proporção e adiciona padding.
        interpolation: Método de interpolação do OpenCV.
    
    Returns:
        Imagem redimensionada.
    
    Example:
        >>> image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        >>> resized = resize_image(image, (640, 480))
        >>> print(resized.shape)  # (480, 640, 3) ou com padding
    """
    if image is None:
        raise ValueError("Imagem não pode ser None")
    
    target_width, target_height = target_size
    
    if not keep_aspect_ratio:
        return cv2.resize(image, (target_width, target_height), interpolation=interpolation)
    
    # Mantém aspect ratio
    h, w = image.shape[:2]
    scale = min(target_width / w, target_height / h)
    
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    
    # Adiciona padding para atingir o tamanho alvo
    pad_w = target_width - new_w
    pad_h = target_height - new_h
    
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    
    padded = cv2.copyMakeBorder(
        resized,
        top, bottom, left, right,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0)
    )
    
    return padded


def preprocess_image(
    image: np.ndarray,
    target_size: Optional[Tuple[int, int]] = None,
    normalize: bool = False,
    to_float: bool = False
) -> np.ndarray:
    """
    Pré-processa uma imagem para inferência em modelos.
    
    Aplica uma série de transformações comuns:
    - Conversão para RGB
    - Redimensionamento (opcional)
    - Normalização (opcional)
    - Conversão para float (opcional)
    
    Args:
        image: Imagem como array NumPy.
        target_size: Tupla (largura, altura) para redimensionar.
        normalize: Se True, normaliza valores para [0, 1].
        to_float: Se True, converte para float32.
    
    Returns:
        Imagem pré-processada.
    
    Example:
        >>> raw_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        >>> processed = preprocess_image(
        ...     raw_image, 
        ...     target_size=(224, 224),
        ...     normalize=True,
        ...     to_float=True
        ... )
    """
    if image is None:
        raise ValueError("Imagem não pode ser None")
    
    result = image.copy()
    
    # Garante RGB
    if len(result.shape) == 2 or result.shape[2] != 3:
        result = convert_to_rgb(result)
    
    # Redimensiona se necessário
    if target_size is not None:
        result = resize_image(result, target_size, keep_aspect_ratio=True)
    
    # Converte para float
    if to_float:
        result = result.astype(np.float32)
    
    # Normaliza
    if normalize:
        if not to_float:
            result = result.astype(np.float32)
        result = result / 255.0
    
    return result


def numpy_to_pil(image: np.ndarray) -> Image.Image:
    """
    Converte array NumPy para imagem PIL.
    
    Args:
        image: Imagem como array NumPy (RGB, 0-255).
    
    Returns:
        Imagem PIL.
    """
    if image.dtype == np.float32 or image.dtype == np.float64:
        # Assume normalizado [0, 1]
        image = (image * 255).astype(np.uint8)
    return Image.fromarray(image)


def pil_to_numpy(image: Image.Image) -> np.ndarray:
    """
    Converte imagem PIL para array NumPy.
    
    Args:
        image: Imagem PIL.
    
    Returns:
        Array NumPy em formato RGB.
    """
    return np.array(image.convert("RGB"))


def add_text_overlay(
    image: np.ndarray,
    text: str,
    position: Tuple[int, int] = (10, 30),
    font_scale: float = 1.0,
    color: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
    background: bool = True
) -> np.ndarray:
    """
    Adiciona texto sobreposto à imagem.
    
    Args:
        image: Imagem original (será copiada).
        text: Texto a adicionar.
        position: Posição (x, y) do texto.
        font_scale: Escala da fonte.
        color: Cor do texto (RGB).
        thickness: Espessura do texto.
        background: Se True, adiciona fundo semi-transparente.
    
    Returns:
        Nova imagem com o texto.
    """
    output = image.copy()
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Obtém tamanho do texto
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )
    
    x, y = position
    
    # Adiciona fundo
    if background:
        padding = 5
        cv2.rectangle(
            output,
            (x - padding, y - text_height - padding),
            (x + text_width + padding, y + baseline + padding),
            (0, 0, 0),
            -1  # Preenchido
        )
    
    # Adiciona texto
    cv2.putText(
        output,
        text,
        position,
        font,
        font_scale,
        color,
        thickness
    )
    
    return output


def create_grid(
    images: list,
    cols: int = 2,
    cell_size: Tuple[int, int] = (320, 240)
) -> np.ndarray:
    """
    Cria uma grade de imagens.
    
    Args:
        images: Lista de imagens NumPy.
        cols: Número de colunas na grade.
        cell_size: Tamanho (largura, altura) de cada célula.
    
    Returns:
        Imagem combinada em grade.
    """
    if not images:
        raise ValueError("Lista de imagens vazia")
    
    n = len(images)
    rows = (n + cols - 1) // cols
    
    cell_w, cell_h = cell_size
    grid = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)
    
    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        
        # Redimensiona para caber na célula
        resized = resize_image(img, cell_size, keep_aspect_ratio=True)
        
        y_start = row * cell_h
        x_start = col * cell_w
        
        grid[y_start:y_start + cell_h, x_start:x_start + cell_w] = resized
    
    return grid


if __name__ == "__main__":
    # Teste das funções
    print("Testando módulo de processamento de imagens...")
    
    # Cria imagem de teste
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Testa pré-processamento
    processed = preprocess_image(test_image, target_size=(224, 224), normalize=True)
    print(f"Original: {test_image.shape}, dtype={test_image.dtype}")
    print(f"Processado: {processed.shape}, dtype={processed.dtype}")
    print(f"Range: [{processed.min():.2f}, {processed.max():.2f}]")
    
    # Testa overlay de texto
    with_text = add_text_overlay(test_image, "VisionFNT Live", position=(50, 50))
    print(f"Com texto: {with_text.shape}")
    
    print("\nTestes concluídos!")

