import os

import aisuite as ai
import gradio as gr

PROVIDER_WRITER = "groq"
MODEL_WRITER = "llama3-70b-8192"
PROVIDER_REVIEWER = "groq"
MODEL_REVIEWER = "llama3-70b-8192"

SYSTEM_WRITER = (
    "你是一位富有創意的小說作家，只會用繁體中文來創作故事，"
    "擅長撰寫充滿情感起伏與細膩描寫的短篇故事。"
)

SYSTEM_REVIEWER = (
    "你是一位文案潤稿專家，只會用繁體中文協助修改，"
    "請針對文章給出具體、可執行的修改建議。"
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


def reflect_post(prompt):
    first_version = reply(
        SYSTEM_WRITER,
        prompt,
        PROVIDER_WRITER,
        MODEL_WRITER,
    )

    suggestion = reply(
        SYSTEM_REVIEWER,
        first_version,
        PROVIDER_REVIEWER,
        MODEL_REVIEWER,
    )

    second_prompt = (
        "這是我剛剛創作的短篇小說：\n"
        f"{first_version}\n\n"
        "這是修改建議：\n"
        f"{suggestion}\n\n"
        "請根據建議改寫，並只輸出修改後的文章。"
    )

    second_version = reply(
        SYSTEM_WRITER,
        second_prompt,
        PROVIDER_WRITER,
        MODEL_WRITER,
    )

    return first_version, suggestion, second_version


with gr.Blocks() as demo:
    gr.Markdown("### 🤖 短小說生成幫手（Reflection Agent）")
    user_input = gr.Textbox(label="請輸入想創作的小說要素")
    btn = gr.Button("生成文章 & 修正建議")

    with gr.Row():
        out1 = gr.Textbox(label="🌟 第一版小說")
        out2 = gr.Textbox(label="🧐 修改建議")
        out3 = gr.Textbox(label="✨ 第二版小說")

    btn.click(
        reflect_post,
        inputs=[user_input],
        outputs=[out1, out2, out3],
    )


if __name__ == "__main__":
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("Please set GROQ_API_KEY before running this demo.")
    demo.launch()
