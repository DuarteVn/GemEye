"""
Módulo de inferência multimodal.

Fornece integração com modelos de visão-linguagem (VLM) para
responder perguntas sobre imagens em linguagem natural.

Utiliza o Google Gemini como backend principal.
"""

import numpy as np
from typing import Optional, Dict, Any
from PIL import Image
import io
import os
import time
from dotenv import load_dotenv

# Importa Google Generative AI
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠️ google-generativeai não instalado. Usando modo stub.")

from ..utils.logger import log_interaction, get_logger


# Carrega variáveis do arquivo .env (para ambiente local)
load_dotenv()

# Configuração da API Key (via variável de ambiente ou .env)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Validação da API Key
if not GOOGLE_API_KEY:
    print("⚠️ GOOGLE_API_KEY não configurada. Configure via variável de ambiente ou Secrets do HF Space.")


class GeminiVisionModel:
    """
    Classe para integração com Google Gemini Vision.
    
    Utiliza o modelo Gemini para análise de imagens e resposta
    a perguntas em linguagem natural.
    
    Attributes:
        model_name: Nome do modelo Gemini a utilizar.
        is_loaded: Indica se o modelo está pronto.
        model: Instância do modelo Gemini.
    """
    
    def __init__(self, model_name: str = GEMINI_MODEL) -> None:
        """
        Inicializa o modelo Gemini.
        
        Args:
            model_name: Nome do modelo (ex: gemini-1.5-flash, gemini-1.5-pro).
        """
        self.model_name = model_name
        self.is_loaded = False
        self.model = None
        self.logger = get_logger()
    
    def load(self) -> bool:
        """
        Configura e carrega o modelo Gemini.
        
        Returns:
            True se carregado com sucesso, False caso contrário.
        """
        if not GEMINI_AVAILABLE:
            self.logger.warning("Gemini não disponível, usando modo stub")
            return False
        
        if not GOOGLE_API_KEY:
            self.logger.warning("GOOGLE_API_KEY não configurada, usando modo stub")
            return False
        
        try:
            # Configura a API
            genai.configure(api_key=GOOGLE_API_KEY)
            
            # Cria o modelo
            self.model = genai.GenerativeModel(self.model_name)
            
            self.is_loaded = True
            self.logger.info(f"Modelo Gemini '{self.model_name}' carregado com sucesso")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao carregar Gemini: {e}")
            return False
    
    def generate(self, image: np.ndarray, question: str) -> str:
        """
        Gera uma resposta para a pergunta sobre a imagem usando Gemini.
        
        Args:
            image: Imagem como array NumPy (RGB).
            question: Pergunta em linguagem natural.
        
        Returns:
            Resposta gerada pelo modelo.
        """
        if not self.is_loaded:
            if not self.load():
                return self._stub_response(image, question)
        
        try:
            # Converte NumPy array para PIL Image
            pil_image = Image.fromarray(image)
            
            # Prepara o prompt
            prompt = f"""Você é um assistente de visão computacional inteligente e prestativo.
Analise a imagem fornecida e responda à pergunta do usuário de forma clara e detalhada.

Pergunta do usuário: {question}

Responda em português brasileiro de forma natural e informativa."""
            
            # Gera a resposta
            start_time = time.time()
            response = self.model.generate_content([prompt, pil_image])
            elapsed_ms = (time.time() - start_time) * 1000
            
            self.logger.info(f"Resposta gerada em {elapsed_ms:.1f}ms")
            
            # Extrai o texto da resposta
            if response and response.text:
                return response.text
            else:
                return "Não foi possível gerar uma resposta para esta imagem."
                
        except Exception as e:
            self.logger.error(f"Erro na geração: {e}")
            return f"❌ Erro ao analisar imagem: {str(e)}"
    
    def _stub_response(self, image: np.ndarray, question: str) -> str:
        """
        Gera uma resposta stub quando o modelo não está disponível.
        """
        height, width = image.shape[:2]
        channels = image.shape[2] if len(image.shape) > 2 else 1
        mean_brightness = np.mean(image)
        
        api_key_status = "não configurada" if not GOOGLE_API_KEY else "configurada"
        
        return (
            f"[MODO DEMO - Gemini não disponível]\n\n"
            f"📷 Imagem recebida: {width}x{height}, {channels} canais\n"
            f"💡 Brilho médio: {mean_brightness:.1f}/255\n\n"
            f"❓ Pergunta: \"{question}\"\n\n"
            f"🔑 API Key: {api_key_status}\n\n"
            f"**Para ativar o modelo:**\n"
            f"1. Obtenha uma API key em: https://aistudio.google.com/apikey\n"
            f"2. Configure a variável de ambiente GOOGLE_API_KEY\n"
            f"   - Local: crie um arquivo .env com GOOGLE_API_KEY=sua-key\n"
            f"   - HF Space: adicione em Settings → Secrets"
        )


class MultimodalModel:
    """
    Classe wrapper para compatibilidade com código existente.
    Redireciona para GeminiVisionModel.
    """
    
    def __init__(self, model_name: str = "gemini") -> None:
        self.model_name = model_name
        self._gemini = GeminiVisionModel()
        self.is_loaded = False
    
    def load(self) -> bool:
        self.is_loaded = self._gemini.load()
        return self.is_loaded
    
    def generate(self, image: np.ndarray, question: str) -> str:
        return self._gemini.generate(image, question)


# Instância global do modelo
_model_instance: Optional[GeminiVisionModel] = None


def get_model() -> GeminiVisionModel:
    """
    Obtém a instância global do modelo Gemini.
    
    Returns:
        Instância do GeminiVisionModel.
    """
    global _model_instance
    if _model_instance is None:
        _model_instance = GeminiVisionModel()
    return _model_instance


def generate_multimodal_answer(
    image: np.ndarray, 
    question: str,
    model_config: Optional[Dict[str, Any]] = None
) -> str:
    """
    Função principal para gerar respostas sobre imagens.
    
    Esta é a função de entrada para o módulo multimodal.
    Utiliza o Google Gemini para análise de imagens.
    
    Args:
        image: Imagem como array NumPy em formato RGB.
               Shape esperado: (H, W, 3) com valores 0-255.
        question: Pergunta em linguagem natural sobre a imagem.
        model_config: Configurações opcionais do modelo.
    
    Returns:
        Resposta gerada pelo modelo Gemini.
    
    Raises:
        ValueError: Se a imagem ou pergunta forem inválidas.
    
    Example:
        >>> import numpy as np
        >>> image = np.zeros((480, 640, 3), dtype=np.uint8)
        >>> answer = generate_multimodal_answer(
        ...     image, 
        ...     "O que você vê nesta imagem?"
        ... )
    """
    # Validação de entrada
    if image is None:
        raise ValueError("Imagem não pode ser None")
    
    if not isinstance(image, np.ndarray):
        raise ValueError(f"Imagem deve ser np.ndarray, recebido: {type(image)}")
    
    if len(image.shape) < 2:
        raise ValueError(f"Imagem deve ter pelo menos 2 dimensões, recebido: {image.shape}")
    
    if not question or not question.strip():
        raise ValueError("Pergunta não pode ser vazia")
    
    # Obtém o modelo e gera resposta
    model = get_model()
    answer = model.generate(image, question.strip())
    
    # Log da interação
    log_interaction(
        question=question, 
        answer=answer, 
        image_shape=image.shape,
        model_name=model.model_name
    )
    
    return answer


def image_to_base64(image: np.ndarray, format: str = "PNG") -> str:
    """
    Converte uma imagem NumPy para string base64.
    
    Args:
        image: Imagem como array NumPy (RGB).
        format: Formato de saída (PNG, JPEG, etc.).
    
    Returns:
        String base64 da imagem.
    """
    import base64
    pil_image = Image.fromarray(image)
    buffer = io.BytesIO()
    pil_image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


if __name__ == "__main__":
    # Teste do módulo
    print("Testando módulo multimodal com Gemini...")
    print(f"API Key configurada: {'Sim' if GOOGLE_API_KEY else 'Não'}")
    print(f"Modelo: {GEMINI_MODEL}")
    
    # Cria imagem de teste (um gradiente simples)
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    test_image[:, :, 0] = np.linspace(0, 255, 640).astype(np.uint8)  # Gradiente vermelho
    test_image[:, :, 2] = np.linspace(255, 0, 640).astype(np.uint8)  # Gradiente azul
    
    test_question = "Descreva o que você vê nesta imagem."
    
    print(f"\nPergunta: {test_question}")
    print("\nGerando resposta...")
    
    try:
        response = generate_multimodal_answer(test_image, test_question)
        print(f"\nResposta:\n{response}")
    except Exception as e:
        print(f"\nErro: {e}")
