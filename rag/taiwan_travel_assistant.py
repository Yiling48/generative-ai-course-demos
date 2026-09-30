import os

import gradio as gr
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from openai import OpenAI


class CustomE5Embedding(HuggingFaceEmbeddings):
    def embed_documents(self, texts):
        return super().embed_documents([f"passage: {text}" for text in texts])

    def embed_query(self, text):
        return super().embed_query(f"query: {text}")


SYSTEM_PROMPT = (
    "請使用繁體中文回答。你是台灣旅遊景點推薦助理，"
    "請優先根據檢索到的資料回答，並提供具體建議。"
)

PROMPT_TEMPLATE = """
根據下列資料回答問題：
{retrieved_chunks}

使用者的問題是：{question}

請根據資料內容使用繁體中文回覆。
如果資料不足，請明確說明資料不足，不要捏造。
"""


def load_retriever(db_path="faiss_db"):
    embeddings = CustomE5Embedding(
        model_name="intfloat/multilingual-e5-small"
    )
    db = FAISS.load_local(
        db_path,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return db.as_retriever()


def build_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("Please set GROQ_API_KEY before running this demo.")

    return OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )


retriever = load_retriever()
client = build_client()
MODEL = "llama3-70b-8192"


def chat_with_rag(user_input):
    docs = retriever.invoke(user_input)
    retrieved_chunks = "\n\n".join(doc.page_content for doc in docs)

    final_prompt = PROMPT_TEMPLATE.format(
        retrieved_chunks=retrieved_chunks,
        question=user_input,
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": final_prompt},
        ],
    )
    return response.choices[0].message.content


with gr.Blocks() as demo:
    gr.Markdown("# AI 台灣旅遊景點諮詢師")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="請輸入你的問題...")

    def respond(message, history):
        answer = chat_with_rag(message)
        history = history + [(message, answer)]
        return "", history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])


if __name__ == "__main__":
    demo.launch()
