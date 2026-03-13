"""
GemEye - Hugging Face Spaces Entry Point.

Este arquivo é o ponto de entrada para o Hugging Face Spaces.
"""

import os
import sys
from pathlib import Path

# Adiciona o diretório raiz ao path para imports
sys.path.insert(0, str(Path(__file__).parent))

from src.interface.interface import create_interface

# Cria e lança a aplicação
app = create_interface()

if __name__ == "__main__":
    # HF Spaces expõe a porta via env var PORT
    port = int(os.getenv("PORT", "7860"))
    app.launch(server_name="0.0.0.0", server_port=port)
