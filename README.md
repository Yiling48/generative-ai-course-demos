# Generative AI Applications & Prototyping

Selected demos from the master's course **Generative AI Applications and Implementation**.

This repository presents cleaned, portfolio-ready versions of course experiments covering local LLMs, agent workflows, RAG, multi-provider APIs, and Stable Diffusion interfaces.

## Demos

### 1. Local LLM AI Interviewer
**Ollama · Gemma 3 4B · OpenAI-compatible API · Gradio**

A role-based interview simulator that:
- runs a local model through Ollama,
- keeps multi-turn conversation history,
- asks position-specific questions,
- provides feedback and follow-up questions,
- exposes the workflow through a Gradio UI.

Code: [llm-chatbots/ai_interviewer.py](llm-chatbots/ai_interviewer.py)

### 2. Reflection Story Writer
**AISuite · Groq · Gradio**

A three-stage reflection workflow:

1. Writer creates a first draft.
2. Reviewer provides revision suggestions.
3. Writer rewrites the story using the feedback.

Code: [agents/reflection_story_writer.py](agents/reflection_story_writer.py)

### 3. Two-Stage Planning & Generation
**AISuite · Groq · Gradio**

Separates planning from final generation:
- stage 1 generates candidate interpretations,
- stage 2 selects and converts one into the final response.

Code: [agents/two_stage_planning.py](agents/two_stage_planning.py)

### 4. RAG Vector Database
**LangChain · multilingual-e5-small · FAISS**

Builds a vector database from TXT / PDF / DOCX files using:
- document loading,
- chunking with overlap,
- E5 `passage:` / `query:` prefixes,
- FAISS storage.

Code: [rag/build_faiss_db.py](rag/build_faiss_db.py)

### 5. Taiwan Travel RAG Assistant
**FAISS · E5 Embeddings · Groq-compatible API · Gradio**

Retrieves relevant Taiwan travel content and injects the retrieved context into an LLM prompt.

Code: [rag/taiwan_travel_assistant.py](rag/taiwan_travel_assistant.py)

### 6. Stable Diffusion WebUI
**diffusers · PyTorch · Gradio**

An interactive image-generation interface supporting:
- prompt enhancement,
- negative prompts,
- custom seeds,
- configurable image dimensions,
- inference-step control,
- multiple generated images.

Code: [image-generation/stable_diffusion_webui.py](image-generation/stable_diffusion_webui.py)

## Repository Structure

```text
generative-ai-course-demos/
├── README.md
├── llm-chatbots/
│   └── ai_interviewer.py
├── agents/
│   ├── reflection_story_writer.py
│   └── two_stage_planning.py
├── rag/
│   ├── build_faiss_db.py
│   └── taiwan_travel_assistant.py
├── image-generation/
│   └── stable_diffusion_webui.py
├── requirements.txt
└── .gitignore
```

## Setup

Create a virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

For demos using Groq, set:

```bash
export GROQ_API_KEY="your_key_here"
```

For the local interviewer, install Ollama and pull the model:

```bash
ollama pull gemma3:4b
ollama serve
```

## Security Note

The original course notebooks included local / Colab credential-loading experiments. This public portfolio repository contains **cleaned scripts only** and does not publish API tokens or access credentials.

Never commit API keys. Use environment variables, Colab Secrets, or another secret manager.

## Portfolio

[Data Analytics & Science Portfolio](https://app.notion.com/p/Data-Analytics-Science-Portfolio-160bcf9053b6800198faddd8f6e6a8ab)

---
Portfolio repository maintained by **Yi-Ling Dai**.
