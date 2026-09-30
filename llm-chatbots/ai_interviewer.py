import gradio as gr
from openai import OpenAI

MODEL = "gemma3:4b"
SYSTEM_PROMPT = (
    "請用繁體中文回答。你是一位專業的面試官，說話有條理、直白但不失禮貌，"
    "習慣用邏輯引導面試者反思問題。請針對職位提出合理但具有挑戰性的問題，"
    "並根據回答給予明確的建議與回饋，並提出其他問題。語氣請保持專業、穩重。"
)

DESCRIPTION = (
    "歡迎使用 AI 面試模擬系統。本平台透過本地語言模型模擬專業面試官，"
    "協助你進行職位模擬問答，並依回答提供回饋與後續問題。"
)

client = OpenAI(
    api_key="ollama",
    base_url="http://localhost:11434/v1",
)


def start_interview(position):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "assistant", "content": DESCRIPTION},
        {
            "role": "user",
            "content": f"我想應徵「{position}」，請開始第一題面試問題。",
        },
    ]
    response = client.chat.completions.create(model=MODEL, messages=messages)
    reply = response.choices[0].message.content
    messages.append({"role": "assistant", "content": reply})
    return messages, messages


def continue_interview(prompt, messages):
    messages = list(messages)
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(model=MODEL, messages=messages)
    reply = response.choices[0].message.content
    messages.append({"role": "assistant", "content": reply})
    return "", messages, messages


with gr.Blocks(title="專業 AI 面試官") as demo:
    gr.Markdown(f"## 🧑‍💼 專業 AI 面試官\n{DESCRIPTION}")

    position_input = gr.Textbox(
        label="輸入你要應徵的職位",
        placeholder="例如：資料分析師",
    )
    start_btn = gr.Button("▶️ 開始模擬面試")

    chatbot = gr.Chatbot(type="messages")
    state = gr.State([])
    msg = gr.Textbox(label="輸入回答")

    start_btn.click(
        fn=start_interview,
        inputs=position_input,
        outputs=[chatbot, state],
    )
    msg.submit(
        fn=continue_interview,
        inputs=[msg, state],
        outputs=[msg, chatbot, state],
    )


if __name__ == "__main__":
    demo.launch()
