import gc
import random

import gradio as gr
import torch
from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler

MODEL_NAME = "stablediffusionapi/anything-v5"

pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
).to("cuda")

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)


def generate_images(
    prompt,
    use_enhance,
    enhance_text,
    use_negative,
    negative_text,
    use_custom_seed,
    custom_seed,
    height,
    width,
    steps,
    num_images,
):
    height = int(height)
    width = int(width)
    steps = int(steps)
    num_images = int(num_images)

    if height % 8 != 0 or width % 8 != 0:
        raise ValueError("Height and width must be multiples of 8.")

    base_seed = int(custom_seed) if use_custom_seed else random.randint(0, 2**32 - 1)
    seeds = [base_seed + i for i in range(num_images)]

    final_prompt = prompt
    if use_enhance and enhance_text:
        final_prompt = f"{prompt}, {enhance_text}"

    final_negative = negative_text if use_negative else None

    gc.collect()
    torch.cuda.empty_cache()

    images = []
    for seed in seeds:
        generator = torch.Generator("cuda").manual_seed(seed)
        with torch.no_grad():
            image = pipe(
                prompt=final_prompt,
                negative_prompt=final_negative,
                height=height,
                width=width,
                num_inference_steps=steps,
                guidance_scale=7.5,
                generator=generator,
            ).images[0]
        images.append(image)

    return images, f"Seeds: {seeds}"


DEFAULT_ENHANCE = (
    "masterpiece, ultra high quality, intricate details, cinematic lighting"
)
DEFAULT_NEGATIVE = (
    "bad anatomy, blurry, disfigured, poorly drawn hands, extra fingers, "
    "mutated hands, low quality, worst quality"
)

with gr.Blocks() as demo:
    gr.Markdown("# 🎨 Interactive Stable Diffusion Generator")

    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt", lines=3)
            use_enhance = gr.Checkbox(label="Enhance prompt", value=True)
            enhance_text = gr.Textbox(label="Enhancement", value=DEFAULT_ENHANCE)
            use_negative = gr.Checkbox(label="Use negative prompt", value=True)
            negative_text = gr.Textbox(label="Negative prompt", value=DEFAULT_NEGATIVE)
            use_custom_seed = gr.Checkbox(label="Use custom seed", value=False)
            custom_seed = gr.Number(label="Seed", value=42)
            height = gr.Dropdown(["512", "768", "1024"], label="Height", value="512")
            width = gr.Dropdown(["512", "768", "1024"], label="Width", value="512")
            steps = gr.Slider(10, 50, value=20, step=5, label="Steps")
            num_images = gr.Slider(1, 4, value=1, step=1, label="Number of images")
            generate_btn = gr.Button("Generate")

        with gr.Column():
            gallery = gr.Gallery(label="Generated images", columns=2)
            seed_info = gr.Label(label="Seeds")

    generate_btn.click(
        fn=generate_images,
        inputs=[
            prompt,
            use_enhance,
            enhance_text,
            use_negative,
            negative_text,
            use_custom_seed,
            custom_seed,
            height,
            width,
            steps,
            num_images,
        ],
        outputs=[gallery, seed_info],
    )


if __name__ == "__main__":
    demo.launch()
