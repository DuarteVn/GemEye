# VisionFNT Live

**Assistente de Visão Computacional em Tempo Real**

Um sistema que usa webcam + Google Gemini para responder perguntas sobre o que você vê.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Gradio](https://img.shields.io/badge/Gradio-6.0+-orange.svg)
![Gemini](https://img.shields.io/badge/Gemini-2.5--flash-green.svg)

---

## 🚀 Início Rápido

### 1. Clonar e entrar no projeto

```bash
git clone <seu-repositorio>
cd webcam-test
```

### 2. Criar ambiente virtual (recomendado)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
.\venv\Scripts\activate
```

### 3. Instalar dependências

```bash
pip install -r requirements.txt
```

### 4. Iniciar a aplicação

```bash
python app.py
```

### 5. Acessar no navegador

Abra: **http://127.0.0.1:7860**

---

## ⚙️ Opções de Execução

```bash
# Execução padrão (porta 7860)
python app.py

# Com link público (para compartilhar)
python app.py --share

# Em porta específica
python app.py --port 8080

# Modo debug (logs detalhados)
python app.py --debug

# Acessível na rede local
python app.py --host 0.0.0.0
```

---

## 🔑 Configuração da API Key

A API key do Google Gemini **não deve** ficar no código nem no repositório. Para usar sua key localmente:

1. Crie um arquivo `.env` na raiz do projeto:

```env
GOOGLE_API_KEY=sua-api-key-aqui
GEMINI_MODEL=gemini-2.5-flash
```

2. Obtenha sua key em: https://aistudio.google.com/apikey

---

## 📁 Estrutura do Projeto

```
webcam-test/
├── src/
│   ├── capture/
│   │   └── webcam.py           # Captura de webcam
│   ├── inference/
│   │   ├── multimodal.py       # Integração Gemini
│   │   └── object_detection.py # Detecção de objetos
│   ├── utils/
│   │   ├── image_processing.py # Processamento de imagem
│   │   └── logger.py           # Sistema de logs
│   └── interface/
│       └── interface.py        # Interface Gradio
├── app.py                      # Ponto de entrada (local e HF Spaces)
├── requirements.txt            # Dependências
└── README.md
```

---

## 🎮 Como Usar

1. **Capture uma imagem** - Use a webcam ou faça upload
2. **Faça uma pergunta** - Ex: "O que você vê?", "Descreva a cena"
3. **Receba a resposta** - O Gemini analisa e responde

### Exemplos de perguntas:
- "O que você vê nesta imagem?"
- "Quantas pessoas tem na foto?"
- "Descreva os objetos na cena"
- "Qual é a cor predominante?"
- "O que está acontecendo?"

---

## 🧪 Testar Módulos Individualmente

```bash
# Testar webcam
python -m src.capture.webcam

# Testar modelo Gemini
python -m src.inference.multimodal

# Testar processamento de imagem
python -m src.utils.image_processing
```

---

## 📦 Dependências Principais

| Pacote | Versão | Uso |
|--------|--------|-----|
| gradio | >=4.0.0 | Interface web |
| opencv-python | >=4.8.0 | Captura de vídeo |
| google-generativeai | >=0.8.0 | API Gemini |
| numpy | >=1.24.0 | Processamento |
| pillow | >=10.0.0 | Imagens |

---

## 🛠️ Solução de Problemas

### Erro de quota do Gemini
```
429 You exceeded your current quota
```
**Solução:** Aguarde alguns minutos ou use outra API key.

### Webcam não funciona
```
Não foi possível abrir a câmera
```
**Solução:** Verifique se a webcam está conectada e não está sendo usada por outro programa.

### Erro de encoding no Windows
```
UnicodeEncodeError
```
**Solução:** Execute com `chcp 65001` antes de `python app.py`.

---

## 🚀 Deploy no Hugging Face Spaces

1. Crie um Space em huggingface.co/spaces
2. Faça upload dos arquivos
3. Configure os secrets com sua API key
4. O Gradio será detectado automaticamente

---

## 📄 Licença

MIT License - Use livremente!

---

**Desenvolvido como projeto de portfólio** | Visão Computacional + IA Multimodal
