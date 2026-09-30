import os

import aisuite as ai
import gradio as gr

PROVIDER_PLANNER = "groq"
MODEL_PLANNER = "llama3-70b-8192"
PROVIDER_WRITER = "groq"
MODEL_WRITER = "llama3-70b-8192"

SYSTEM_PLANNER = (
    "請用台灣習慣的中文回應。你是一位正向思考導師，"
    "請針對使用者的事件提出五種不同的正向重新詮釋。"
)

SYSTEM_WRITER = (
    "請用台灣習慣的中文回應。你是一位活潑的社群貼文寫手，"
    "請從候選理由中挑一個有趣的角度，改寫成自然的第一人稱貼文。"
)


def reply(system, prompt, provider, model):
    client = ai.Client()
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    response = client.chat.completions.create(
        model=f"{provider}:{model}",
        messages=messages,
    )
    return response.choices[0].message.content


def two_stage_generate(prompt):
    planning_prompt = (
        f"使用者說：{prompt}\n"
        "請幫我提出五種不同的正向重新詮釋。"
    )
    reasons = reply(
        SYSTEM_PLANNER,
        planning_prompt,
        PROVIDER_PLANNER,
        MODEL_PLANNER,
    )

    generation_prompt = (
        f"候選理由如下：\n{reasons}\n\n"
        "請選出一個最有趣的理由並寫成社群貼文。"
    )
    post = reply(
        SYSTEM_WRITER,
        generation_prompt,
        PROVIDER_WRITER,
        MODEL_WRITER,
    )
    return reasons, post


with gr.Blocks() as demo:
    gr.Markdown("### 🧠 Two-Stage Planning & Generation")
    user_input = gr.Textbox(label="輸入一件想重新詮釋的事情")
    btn = gr.Button("開始生成")

    with gr.Row():
        out1 = gr.Textbox(label="Planning")
        out2 = gr.Textbox(label="Final Generation")

    btn.click(two_stage_generate, inputs=user_input, outputs=[out1, out2])


if __name__ == "__main__":
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("Please set GROQ_API_KEY before running this demo.")
    demo.launch()
