"""
Módulo de captura de webcam.

Fornece funcionalidades para capturar frames da webcam em tempo real.
Utiliza OpenCV para acesso à câmera.
"""

import cv2
import numpy as np
from typing import Optional
import threading
import time


class WebcamCapture:
    """
    Classe para gerenciar a captura de frames da webcam.
    
    Suporta captura contínua em thread separada para melhor performance,
    ou captura sob demanda para casos mais simples.
    
    Attributes:
        camera_index: Índice da câmera a ser utilizada (padrão: 0).
        cap: Objeto VideoCapture do OpenCV.
        _frame: Último frame capturado.
        _running: Flag indicando se a captura está ativa.
        _thread: Thread de captura contínua.
        _lock: Lock para acesso thread-safe ao frame.
    
    Example:
        >>> webcam = WebcamCapture()
        >>> webcam.start()
        >>> frame = webcam.get_frame()
        >>> webcam.stop()
    """
    
    def __init__(self, camera_index: int = 0) -> None:
        """
        Inicializa o capturador de webcam.
        
        Args:
            camera_index: Índice da câmera (0 para câmera padrão).
        """
        self.camera_index = camera_index
        self.cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._running: bool = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
    
    def start(self) -> bool:
        """
        Inicia a captura da webcam.
        
        Abre a conexão com a câmera e inicia a thread de captura contínua.
        
        Returns:
            True se a câmera foi aberta com sucesso, False caso contrário.
        """
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            return False
        
        # Configurações opcionais para melhor qualidade
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        
        return True
    
    def _capture_loop(self) -> None:
        """
        Loop interno de captura contínua.
        
        Executa em thread separada, capturando frames continuamente
        e armazenando o mais recente.
        """
        while self._running:
            if self.cap is not None:
                ret, frame = self.cap.read()
                if ret:
                    # Converte BGR (OpenCV) para RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    with self._lock:
                        self._frame = frame_rgb
            time.sleep(0.01)  # Pequeno delay para não sobrecarregar CPU
    
    def get_frame(self) -> Optional[np.ndarray]:
        """
        Retorna o frame mais recente capturado.
        
        Returns:
            Frame como array NumPy em formato RGB, ou None se não houver frame.
        """
        with self._lock:
            if self._frame is not None:
                return self._frame.copy()
            return None
    
    def capture_single_frame(self) -> Optional[np.ndarray]:
        """
        Captura um único frame diretamente (sem thread).
        
        Útil para capturas pontuais sem necessidade de streaming contínuo.
        
        Returns:
            Frame como array NumPy em formato RGB, ou None em caso de erro.
        """
        if self.cap is None:
            self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            return None
        
        ret, frame = self.cap.read()
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None
    
    def stop(self) -> None:
        """
        Para a captura e libera os recursos.
        """
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def is_running(self) -> bool:
        """
        Verifica se a captura está ativa.
        
        Returns:
            True se a captura está rodando, False caso contrário.
        """
        return self._running
    
    def __enter__(self) -> "WebcamCapture":
        """Context manager: inicia a captura."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager: para a captura."""
        self.stop()


# Instância global para uso simplificado
_global_webcam: Optional[WebcamCapture] = None


def get_frame(camera_index: int = 0) -> Optional[np.ndarray]:
    """
    Função de conveniência para capturar um único frame.
    
    Cria/reutiliza uma instância global de WebcamCapture para
    facilitar o uso em scripts simples.
    
    Args:
        camera_index: Índice da câmera a utilizar.
    
    Returns:
        Frame como array NumPy em formato RGB, ou None em caso de erro.
    
    Example:
        >>> frame = get_frame()
        >>> if frame is not None:
        ...     print(f"Frame capturado: {frame.shape}")
    """
    global _global_webcam
    
    if _global_webcam is None:
        _global_webcam = WebcamCapture(camera_index)
    
    return _global_webcam.capture_single_frame()


def release_global_webcam() -> None:
    """Libera a instância global da webcam."""
    global _global_webcam
    if _global_webcam is not None:
        _global_webcam.stop()
        _global_webcam = None


if __name__ == "__main__":
    # Teste básico da captura
    print("Testando captura de webcam...")
    print("Pressione 'q' para sair.")
    
    with WebcamCapture() as webcam:
        while True:
            frame = webcam.get_frame()
            if frame is not None:
                # Converte de volta para BGR para exibição no OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imshow("Webcam Test", frame_bgr)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cv2.destroyAllWindows()
    print("Teste finalizado.")

