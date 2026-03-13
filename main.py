#!/usr/bin/env python3
"""
GemEye - Ponto de entrada principal.

Este script inicia a aplicação GemEye, que permite
interagir com um modelo multimodal usando imagens da webcam.

Usage:
    python main.py [--share] [--port PORT]

Examples:
    # Execução local
    python main.py
    
    # Com link público (para compartilhar)
    python main.py --share
    
    # Em porta específica
    python main.py --port 8080
"""

import argparse
import sys
from pathlib import Path

# Adiciona o diretório src ao path
sys.path.insert(0, str(Path(__file__).parent))

from src.interface.interface import create_interface, launch_app
from src.utils.logger import setup_logger


def parse_args() -> argparse.Namespace:
    """
    Parse argumentos da linha de comando.
    
    Returns:
        Namespace com os argumentos parseados.
    """
    parser = argparse.ArgumentParser(
        description="GemEye - Assistente de Visão Computacional",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python main.py                    # Inicia na porta 7860
  python main.py --share            # Cria link público
  python main.py --port 8080        # Usa porta 8080
  python main.py --debug            # Modo debug com logs detalhados
        """
    )
    
    parser.add_argument(
        "--share",
        action="store_true",
        help="Cria um link público via Gradio (útil para compartilhar)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Porta do servidor (padrão: 7860)"
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host do servidor (padrão: 127.0.0.1, use 0.0.0.0 para acesso externo)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Ativa modo debug com logs detalhados"
    )
    
    parser.add_argument(
        "--log-file",
        action="store_true",
        help="Salva logs em arquivo"
    )
    
    return parser.parse_args()


def main() -> None:
    """
    Função principal que inicializa e executa a aplicação.
    """
    args = parse_args()
    
    # Configura logging
    import logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = setup_logger(
        level=log_level,
        log_to_file=args.log_file
    )
    
    # Banner
    print("""
    ===========================================================
    |                                                         |
    |   GemEye                                                    |
    |   Assistente de Visao Computacional em Tempo Real       |
    |                                                         |
    ===========================================================
    """)
    
    logger.info("Iniciando GemEye...")
    logger.info(f"Configurações: port={args.port}, host={args.host}, share={args.share}")
    
    try:
        # Cria e lança a aplicação
        app = create_interface()
        
        logger.info(f"Servidor iniciando em http://{args.host}:{args.port}")
        if args.share:
            logger.info("Link público será gerado...")
        
        app.launch(
            server_name=args.host,
            server_port=args.port,
            share=args.share,
            show_error=True
        )
        
    except KeyboardInterrupt:
        logger.info("Aplicação encerrada pelo usuário")
    except Exception as e:
        logger.error(f"Erro ao iniciar aplicação: {e}")
        raise


if __name__ == "__main__":
    main()

