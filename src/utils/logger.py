"""
Módulo de logging.

Fornece funcionalidades para registro de interações, eventos e erros
do sistema GemEye.
"""

import logging
import sys
from datetime import datetime
from typing import Optional, Any, Tuple
from pathlib import Path
import json


# Configuração do logger principal
_logger: Optional[logging.Logger] = None
_log_file: Optional[Path] = None


def setup_logger(
    name: str = "gemeye",
    level: int = logging.INFO,
    log_to_file: bool = False,
    log_dir: str = "logs"
) -> logging.Logger:
    """
    Configura e retorna o logger principal da aplicação.
    
    Args:
        name: Nome do logger.
        level: Nível de logging (DEBUG, INFO, WARNING, ERROR).
        log_to_file: Se True, também salva logs em arquivo.
        log_dir: Diretório para arquivos de log.
    
    Returns:
        Logger configurado.
    
    Example:
        >>> logger = setup_logger(level=logging.DEBUG)
        >>> logger.info("Aplicação iniciada")
    """
    global _logger, _log_file
    
    if _logger is not None:
        return _logger
    
    _logger = logging.getLogger(name)
    _logger.setLevel(level)
    
    # Evita duplicação de handlers
    if _logger.handlers:
        return _logger
    
    # Formato do log
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Handler para console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    _logger.addHandler(console_handler)
    
    # Handler para arquivo (opcional)
    if log_to_file:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        _log_file = log_path / f"gemeye_{timestamp}.log"
        
        file_handler = logging.FileHandler(_log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        _logger.addHandler(file_handler)
    
    return _logger


def get_logger() -> logging.Logger:
    """
    Obtém o logger global da aplicação.
    
    Se o logger não foi configurado, cria um com configurações padrão.
    
    Returns:
        Logger da aplicação.
    """
    global _logger
    if _logger is None:
        _logger = setup_logger()
    return _logger


def log_interaction(
    question: str,
    answer: str,
    image_shape: Optional[Tuple[int, ...]] = None,
    model_name: str = "unknown",
    latency_ms: Optional[float] = None
) -> None:
    """
    Registra uma interação pergunta-resposta.
    
    Args:
        question: Pergunta feita pelo usuário.
        answer: Resposta gerada pelo modelo.
        image_shape: Shape da imagem processada.
        model_name: Nome do modelo utilizado.
        latency_ms: Tempo de resposta em milissegundos.
    
    Example:
        >>> log_interaction(
        ...     question="O que você vê?",
        ...     answer="Vejo uma pessoa sentada.",
        ...     image_shape=(480, 640, 3),
        ...     latency_ms=150.5
        ... )
    """
    logger = get_logger()
    
    # Trunca resposta longa para log
    answer_preview = answer[:100] + "..." if len(answer) > 100 else answer
    answer_preview = answer_preview.replace("\n", " ")
    
    image_info = f"{image_shape}" if image_shape else "N/A"
    latency_info = f"{latency_ms:.1f}ms" if latency_ms else "N/A"
    
    logger.info(
        f"INTERACTION | "
        f"Q: \"{question[:50]}...\" | "
        f"A: \"{answer_preview}\" | "
        f"Image: {image_info} | "
        f"Model: {model_name} | "
        f"Latency: {latency_info}"
    )


def log_error(
    error: Exception,
    context: str = "",
    extra_info: Optional[dict] = None
) -> None:
    """
    Registra um erro com contexto adicional.
    
    Args:
        error: Exceção capturada.
        context: Descrição do contexto onde o erro ocorreu.
        extra_info: Informações adicionais para debug.
    """
    logger = get_logger()
    
    error_info = {
        "type": type(error).__name__,
        "message": str(error),
        "context": context
    }
    
    if extra_info:
        error_info["extra"] = extra_info
    
    logger.error(f"ERROR | {json.dumps(error_info, ensure_ascii=False)}")


def log_model_load(
    model_name: str,
    load_time_ms: float,
    success: bool = True
) -> None:
    """
    Registra o carregamento de um modelo.
    
    Args:
        model_name: Nome do modelo carregado.
        load_time_ms: Tempo de carregamento em milissegundos.
        success: Se o carregamento foi bem-sucedido.
    """
    logger = get_logger()
    status = "SUCCESS" if success else "FAILED"
    logger.info(f"MODEL_LOAD | {model_name} | {status} | {load_time_ms:.1f}ms")


def log_webcam_event(
    event: str,
    details: Optional[str] = None
) -> None:
    """
    Registra eventos relacionados à webcam.
    
    Args:
        event: Tipo de evento (start, stop, error, frame_drop, etc.).
        details: Detalhes adicionais do evento.
    """
    logger = get_logger()
    msg = f"WEBCAM | {event}"
    if details:
        msg += f" | {details}"
    logger.info(msg)


class InteractionLogger:
    """
    Classe para logging estruturado de interações.
    
    Mantém histórico de interações em memória e permite
    exportação para diferentes formatos.
    
    Attributes:
        history: Lista de interações registradas.
        max_history: Número máximo de interações mantidas em memória.
    """
    
    def __init__(self, max_history: int = 100) -> None:
        """
        Inicializa o logger de interações.
        
        Args:
            max_history: Número máximo de interações a manter.
        """
        self.history: list = []
        self.max_history = max_history
    
    def log(
        self,
        question: str,
        answer: str,
        image_shape: Optional[Tuple[int, ...]] = None,
        metadata: Optional[dict] = None
    ) -> None:
        """
        Registra uma interação.
        
        Args:
            question: Pergunta do usuário.
            answer: Resposta do modelo.
            image_shape: Shape da imagem.
            metadata: Metadados adicionais.
        """
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer,
            "image_shape": image_shape,
            "metadata": metadata or {}
        }
        
        self.history.append(interaction)
        
        # Limita o tamanho do histórico
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
        
        # Também loga no logger padrão
        log_interaction(question, answer, image_shape)
    
    def get_history(self, limit: Optional[int] = None) -> list:
        """
        Retorna o histórico de interações.
        
        Args:
            limit: Número máximo de interações a retornar.
        
        Returns:
            Lista de interações.
        """
        if limit:
            return self.history[-limit:]
        return self.history.copy()
    
    def export_json(self, filepath: str) -> None:
        """
        Exporta o histórico para arquivo JSON.
        
        Args:
            filepath: Caminho do arquivo de saída.
        """
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)
    
    def clear(self) -> None:
        """Limpa o histórico de interações."""
        self.history.clear()


# Instância global do logger de interações
interaction_logger = InteractionLogger()


if __name__ == "__main__":
    # Teste do módulo
    print("Testando módulo de logging...")
    
    # Configura logger
    logger = setup_logger(level=logging.DEBUG)
    
    # Testa diferentes tipos de log
    logger.info("Teste de log INFO")
    logger.debug("Teste de log DEBUG")
    logger.warning("Teste de log WARNING")
    
    # Testa log de interação
    log_interaction(
        question="O que você vê na imagem?",
        answer="Vejo uma pessoa sentada em frente a um computador.",
        image_shape=(480, 640, 3),
        model_name="stub",
        latency_ms=125.3
    )
    
    # Testa InteractionLogger
    interaction_logger.log(
        question="Teste",
        answer="Resposta de teste",
        image_shape=(100, 100, 3)
    )
    
    print(f"\nHistórico: {interaction_logger.get_history()}")
    print("\nTestes de logging concluídos!")

