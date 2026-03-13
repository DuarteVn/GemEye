"""
Módulo de interface Gradio.

Fornece a interface web para interação com o sistema GemEye.
Permite captura de imagens via webcam ou upload, e chat com o modelo
multimodal.
"""

import gradio as gr
import numpy as np
from typing import List, Dict, Tuple, Optional
import time

from ..inference.multimodal import generate_multimodal_answer
from ..utils.logger import get_logger


# Tipo para histórico de chat do Gradio 6.0 (lista de dicts com role/content)
ChatMessage = Dict[str, str]
ChatHistory = List[ChatMessage]


def process_image_and_question(
    image: Optional[np.ndarray],
    question: str,
    chat_history: Optional[ChatHistory],
) -> Tuple[ChatHistory, str]:
    """
    Processa uma imagem e pergunta, gerando uma resposta.
    
    Esta é a função principal que conecta a interface ao backend.
    
    Args:
        image: Imagem capturada (RGB, np.ndarray).
        question: Pergunta do usuário.
        chat_history: Histórico de conversas anterior.
    
    Returns:
        Tupla com:
        - Histórico de chat atualizado
        - Mensagem de status
    """
    logger = get_logger()
    
    # Garante que chat_history é uma lista
    if chat_history is None:
        chat_history = []
    
    # Validação
    if image is None:
        new_history = chat_history + [
            {"role": "user", "content": question if question else ""},
            {"role": "assistant", "content": "⚠️ Por favor, capture ou envie uma imagem primeiro."}
        ]
        return new_history, "Nenhuma imagem fornecida"
    
    if not question or not question.strip():
        new_history = chat_history + [
            {"role": "user", "content": ""},
            {"role": "assistant", "content": "⚠️ Por favor, digite uma pergunta."}
        ]
        return new_history, "Nenhuma pergunta fornecida"
    
    logger.info(f"Processando pergunta: {question[:50]}...")
    start_time = time.time()
    
    try:
        # Gera resposta do modelo multimodal
        answer = generate_multimodal_answer(image, question.strip())
        
        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(f"Resposta gerada em {elapsed_ms:.1f}ms")
        
        # Atualiza histórico no formato Gradio 6.0
        new_history = chat_history + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ]
        
        return new_history, f"✅ Processado em {elapsed_ms:.0f}ms"
        
    except Exception as e:
        logger.error(f"Erro ao processar: {e}")
        error_msg = f"❌ Erro ao processar: {str(e)}"
        new_history = chat_history + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": error_msg}
        ]
        return new_history, error_msg


def clear_chat() -> Tuple[ChatHistory, str]:
    """
    Limpa o histórico de chat.
    
    Returns:
        Histórico vazio e mensagem de status.
    """
    return [], "Chat limpo"


def create_interface() -> gr.Blocks:
    """
    Cria e retorna a interface Gradio completa.
    
    A interface inclui:
    - Webcam para captura de imagens
    - Upload de imagens alternativo
    - Chat para perguntas e respostas
    - Opções de detecção de objetos
    - Histórico de conversas
    
    Returns:
        Aplicação Gradio configurada.
    
    Example:
        >>> app = create_interface()
        >>> app.launch()
    """
    
    with gr.Blocks(
        title="GemEye",
    ) as app:

        # question_input definido antes dos exemplos para poder ser referenciado
        question_input = gr.Textbox(
            label="Sua pergunta",
            placeholder="Ex: O que você vê nesta imagem? Quantas pessoas há? Descreva a cena...",
            lines=2,
            visible=False  # será substituído pelo render dentro da coluna
        )

        # Header
        gr.Markdown(
            """
            # GemEye
            **Assistente de Visão Computacional em Tempo Real** — Capture uma imagem pela webcam ou faça upload, depois faça perguntas sobre o que você vê!
            """
        )

        # Exemplos de perguntas (topo, sempre visível)
        gr.Markdown("### 💡 Exemplos de Perguntas")
        gr.Examples(
            examples=[
                ["O que você vê nesta imagem?"],
                ["Descreva a cena em detalhes."],
                ["Quantas pessoas você consegue identificar?"],
                ["Quais objetos estão presentes na imagem?"],
                ["Qual é a cor predominante?"],
                ["O que está acontecendo na cena?"],
            ],
            inputs=[question_input],
            label="Clique para usar"
        )

        with gr.Row():
            # Coluna da esquerda: Webcam
            with gr.Column(scale=1):
                gr.Markdown("### 📷 Captura de Imagem")
                
                # Webcam para captura
                webcam = gr.Image(
                    label="Webcam",
                    sources=["webcam", "upload"],
                    type="numpy",
                    height=450
                )

                # Status (abaixo da webcam)
                status_text = gr.Textbox(
                    label="Status",
                    value="Pronto para começar!",
                    interactive=False,
                    max_lines=1
                )
            
            # Coluna da direita: Chat
            with gr.Column(scale=1):
                gr.Markdown("### 💬 Chat")
                
                # Área de chat
                chatbot = gr.Chatbot(
                    label="Conversa",
                    height=450
                )
                
                # Input de pergunta (visível aqui)
                question_input_visible = gr.Textbox(
                    label="Sua pergunta",
                    placeholder="Ex: O que você vê nesta imagem? Quantas pessoas há? Descreva a cena...",
                    lines=2
                )
                
                # Botões
                with gr.Row():
                    submit_btn = gr.Button(
                        "🚀 Enviar",
                        variant="primary",
                        scale=2
                    )
                    clear_btn = gr.Button(
                        "🗑️ Limpar Chat",
                        variant="secondary",
                        scale=1
                    )

        # Sincroniza o exemplo clicado para o input visível
        question_input.change(
            fn=lambda x: x,
            inputs=[question_input],
            outputs=[question_input_visible]
        )


        # Event handlers
        submit_btn.click(
            fn=process_image_and_question,
            inputs=[webcam, question_input_visible, chatbot],
            outputs=[chatbot, status_text]
        ).then(
            fn=lambda: "",
            outputs=[question_input_visible]
        )
        
        # Submit também com Enter
        question_input_visible.submit(
            fn=process_image_and_question,
            inputs=[webcam, question_input_visible, chatbot],
            outputs=[chatbot, status_text]
        ).then(
            fn=lambda: "",
            outputs=[question_input_visible]
        )
        
        clear_btn.click(
            fn=clear_chat,
            outputs=[chatbot, status_text]
        )

    
    return app


def launch_app(
    share: bool = False,
    server_port: int = 7860,
    server_name: str = "0.0.0.0"
) -> None:
    """
    Inicia a aplicação Gradio.
    
    Args:
        share: Se True, cria link público via Gradio.
        server_port: Porta do servidor.
        server_name: Nome/IP do servidor.
    """
    logger = get_logger()
    logger.info(f"Iniciando GemEye na porta {server_port}...")
    
    app = create_interface()
    app.launch(
        share=share,
        server_port=server_port,
        server_name=server_name
    )


if __name__ == "__main__":
    # Execução direta do módulo
    launch_app()

