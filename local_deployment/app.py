"""
MAGE - Masked and Generative Editor
Run locally with:  python app.py
Then open:         http://127.0.0.1:7860
"""

import os
import re
import multiprocessing
from io import BytesIO

# Disable tokenizer parallelism warning (HuggingFace thing)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Core ML + image libraries
import torch
import numpy as np
import cv2
from PIL import Image

# UI
import gradio as gr

# Visualization
import matplotlib.pyplot as plt

# Models
from diffusers import StableDiffusionInpaintPipeline  # image editing
from ultralytics import YOLO                          # object detection
from transformers import (                            # NLP + scoring
    CLIPProcessor, CLIPModel,
    T5Tokenizer, T5ForConditionalGeneration,
)

# Metric for "how much image changed"
from skimage.metrics import structural_similarity as ssim_metric

# Style-words list
STYLE_WORDS = {
    "cartoon", "anime", "manga", "comic", "comics",
    "illustration", "illustrated", "drawing", "drawn",
    "sketch", "sketched", "doodle",
    "painting", "painted", "painterly",
    "watercolor", "oil painting", "acrylic", "gouache",
    "pencil", "charcoal", "ink", "pastel",
    "pixel art", "8-bit", "16-bit", "voxel",
    "3d render", "cgi", "cel-shaded", "cel shaded",
    "low poly", "low-poly",
    "ghibli", "pixar", "disney", "simpsons",
    "van gogh", "picasso", "monet",
    "stylized", "stylised", "artistic", "abstract",
    "surreal", "vector", "flat design",
}

STYLE_PRESETS = {
    "Auto (detect from prompt)": None,
    "Photorealistic": {
        "suffix": "highly detailed, photorealistic, sharp focus, natural lighting, high quality",
        "negative_add": "cartoon, anime, illustration, drawing, painting, 3d render",
        "guidance": 7.5, "steps_min": 30, "dilate": 10,
    },
    "Cartoon": {
        "suffix": "cartoon illustration, bold outlines, flat vibrant colors, cel-shaded, animated style",
        "negative_add": "photorealistic, photograph, realistic, 3d render, grainy",
        "guidance": 11.0, "steps_min": 45, "dilate": 22,
    },
    "Anime": {
        "suffix": "anime style, cel-shaded, clean lineart, expressive eyes, vibrant colors",
        "negative_add": "photorealistic, photograph, realistic, western cartoon, 3d render",
        "guidance": 11.0, "steps_min": 45, "dilate": 22,
    },
    "Oil Painting": {
        "suffix": "oil painting, visible brushstrokes, rich textured pigment, classical art, painterly",
        "negative_add": "photorealistic, photograph, digital art, 3d render, sharp edges",
        "guidance": 10.0, "steps_min": 45, "dilate": 20,
    },
    "Watercolor": {
        "suffix": "watercolor painting, soft edges, flowing pigment, paper texture, delicate washes",
        "negative_add": "photorealistic, sharp edges, 3d render, harsh lines",
        "guidance": 10.0, "steps_min": 40, "dilate": 20,
    },
    "Pencil Sketch": {
        "suffix": "pencil sketch, graphite drawing, cross-hatching, sketchy lineart, paper background",
        "negative_add": "color, photorealistic, 3d render, painting",
        "guidance": 10.0, "steps_min": 40, "dilate": 20,
    },
    "3D Render": {
        "suffix": "3d rendered, pixar style, smooth surfaces, subsurface scattering, soft studio lighting",
        "negative_add": "2d, flat, photograph, sketch, painting",
        "guidance": 10.0, "steps_min": 45, "dilate": 18,
    },
}

STYLE_CHOICES = list(STYLE_PRESETS.keys())
DEFAULT_STYLE = STYLE_CHOICES[0]


def apply_style(prompt, negative, style_name, user_steps, user_guidance):
    preset = STYLE_PRESETS.get(style_name)

    if preset is None:
        # Auto mode: fall back to keyword detection
        if is_stylized(prompt):
            return user_steps, user_guidance, negative, 22, ", clean lines, vibrant colors, high quality"
        return user_steps, user_guidance, negative, 10, ", highly detailed, photorealistic, high quality"

    merged_neg = negative.strip()
    if preset["negative_add"] not in merged_neg:
        merged_neg = f"{merged_neg}, {preset['negative_add']}" if merged_neg else preset["negative_add"]

    eff_steps = max(user_steps, preset["steps_min"])
    eff_guidance = max(user_guidance, preset["guidance"])
    return eff_steps, eff_guidance, merged_neg, preset["dilate"], f", {preset['suffix']}"


# Image Preprocessing to standardize input image
def preprocessImage(image, size):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image = image.convert("RGB")
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    image = image.crop((left, top, left + side, top + side))
    image = image.resize((size, size), Image.LANCZOS)
    return image


# Expands mask region
def dilate_mask(mask, dilate):
    if dilate > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate))
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def is_stylized(prompt):
    if not prompt:
        return False
    text = prompt.lower()
    for phrase in STYLE_WORDS:
        if (" " in phrase or "-" in phrase) and phrase in text:
            return True
    words = re.findall(r"\b[\w-]+\b", text)
    if any(w in STYLE_WORDS for w in words):
        return True
    if re.search(r"\bstyle of\b|\bstyled? as\b|\blike a (drawing|painting|cartoon|sketch)", text):
        return True
    return False


# Uses YOLO to detect objects
def detectObject(image):
    image_np = np.array(image)
    results = yoloModel(image_np)[0]
    labels_idx = results.boxes.cls.cpu().numpy().astype(int).tolist()
    labels = [yoloModel.names[i] for i in labels_idx]
    boxes = results.boxes.xyxy.cpu().numpy().tolist()
    scores = results.boxes.conf.cpu().numpy().tolist()
    masks = results.masks.data.cpu().numpy() if results.masks is not None else None
    return {"labels": labels, "boxes": boxes, "scores": scores, "masks": masks}


# Creates rectangular mask from bounding box when YOLO fails to detect objects
def maskBox(image, box, dilate=15):
    width, height = image.size
    x1, y1, x2, y2 = map(int, box)
    x1, x2 = max(0, min(x1, width)), max(0, min(x2, width))
    y1, y2 = max(0, min(y1, height)), max(0, min(y2, height))
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    mask = dilate_mask(mask, dilate)
    return Image.fromarray(mask)


# Makes YOLO detection into mask
def maskFromSegmentation(image, seg_mask, dilate=10):
    width, height = image.size
    mask = (seg_mask > 0.5).astype(np.uint8) * 255
    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    mask = dilate_mask(mask, dilate)
    return Image.fromarray(mask)


# Measures how output image is to prompt
def clipScore(image, prompt):
    inputs = clipProcessor(
        text=[prompt], images=image,
        return_tensors="pt", padding=True, truncation=True,
    )
    with torch.no_grad():
        logits = clipModel(
            pixel_values=inputs["pixel_values"].to(device),
            input_ids=inputs["input_ids"].to(device),
            attention_mask=inputs["attention_mask"].to(device),
            return_dict=True,
        )
        img_emb = logits.image_embeds / logits.image_embeds.norm(dim=1, keepdim=True)
        txt_emb = logits.text_embeds / logits.text_embeds.norm(dim=1, keepdim=True)
    return round((img_emb * txt_emb).sum().item(), 4)


# Measures how much area outside of mask was changed
def preservationScore(original, edited, mask):
    orig_np = np.array(original.convert("RGB"))
    edit_np = np.array(edited.convert("RGB").resize(original.size))
    mask_np = np.array(mask.convert("L").resize(original.size))
    outside = mask_np < 128
    if outside.sum() == 0:
        return 1.0
    og = cv2.cvtColor(orig_np, cv2.COLOR_RGB2GRAY)
    eg = cv2.cvtColor(edit_np, cv2.COLOR_RGB2GRAY)
    _, ssim_map = ssim_metric(og, eg, full=True)
    return round(float(ssim_map[outside].mean()), 4)


# Runs Stable Diffusion In-Painting
def mainProcess(image, maskImage, prompt, negativePrompt,
                steps=30, guidanceScale=7.5, seed=1024, score_prompt=None):
    generator = torch.Generator(device=device).manual_seed(int(seed))
    result = stableDiffusion(
        prompt=prompt,
        negative_prompt=negativePrompt,
        image=image,
        mask_image=maskImage,
        num_inference_steps=steps,
        guidance_scale=guidanceScale,
        generator=generator,
    ).images[0]
    score = clipScore(result, score_prompt if score_prompt else prompt)
    return result, score


# Uses T5 to add more detail to prompt
def enhancePrompt(user_prompt, target_object=None, style=DEFAULT_STYLE):
    if not user_prompt or user_prompt.strip() == "":
        return user_prompt
    cleaned = user_prompt.strip()
    for verb in ("replace ", "remove ", "change ", "add ", "modify "):
        if cleaned.lower().startswith(verb):
            cleaned = cleaned[len(verb):]
            for sep in (" to a ", " to an ", " to ", " with a ", " with an ", " with "):
                if sep in cleaned.lower():
                    cleaned = cleaned.lower().split(sep, 1)[1]
                    break
            break
    _, _, _, _, suffix = apply_style(user_prompt, "", style, 0, 0)
    instruction = (
        f"Expand this phrase into a detailed, descriptive image caption. "
        f"Focus on visual details like colors, lighting, and style. "
        f"Phrase: {cleaned}. Descriptive caption:"
    )
    inputs = t5Tokenizer(instruction, return_tensors="pt", truncation=True, max_length=256).to(device)
    with torch.no_grad():
        output_ids = t5Model.generate(**inputs, max_new_tokens=80, num_beams=4,
                                      do_sample=False, early_stopping=True)
    enhanced = t5Tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    for prefix in ("replace ", "remove ", "change ", "add ", "modify "):
        if enhanced.lower().startswith(prefix):
            enhanced = enhanced[len(prefix):]
            break
    if len(enhanced) < len(cleaned) or not enhanced.strip():
        return f"{cleaned}{suffix}"
    return f"{enhanced}{suffix}"


# Classifies action of prompt
def classifyIntent(user_prompt):
    instruction = (
        f"Classify this image edit instruction into exactly one category: "
        f"replace, remove, add, or modify. "
        f"Instruction: '{user_prompt}'. Category:"
    )
    inputs = t5Tokenizer(instruction, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        output_ids = t5Model.generate(**inputs, max_new_tokens=5, do_sample=False)
    label = t5Tokenizer.decode(output_ids[0], skip_special_tokens=True).lower().strip()
    for category in ["replace", "remove", "add", "modify"]:
        if category in label:
            return category
    return "replace"


def styleAware(prompt, user_steps, user_guidance, user_negative):
    if not is_stylized(prompt):
        return user_steps, user_guidance, user_negative, 10
    eff_steps = max(user_steps, 45)
    eff_guidance = max(user_guidance, 10)
    bad_defaults = {
        "blurry, low quality, distorted, deformed, ugly",
        "blurry, low quality, distorted",
    }
    negative = "blurry, low quality, photorealistic, photograph" \
        if user_negative.strip() in bad_defaults else user_negative
    return eff_steps, eff_guidance, negative, 22


# Auto-Mask Pipeline using YOLO and Stable Diffusion
def auto_mask(image, target, prompt, negativePrompt, style=DEFAULT_STYLE, use_t5=True, steps=30, guidanceScale=7.5, seed=1024):
    if image is None:
        return None, None, "Upload an image first."
    
    image = preprocessImage(image, 512)
    detections = detectObject(image)
    labels, boxes, scores, masks = (
        detections["labels"], detections["boxes"],
        detections["scores"], detections["masks"],
    )
    if not labels:
        return None, None, "No objects detected by YOLO."
    
    target_low = target.lower().strip()
    target_idx = next(
        (i for i, l in enumerate(labels) if l.lower() == target_low), None
    )
    if target_idx is None:
        found = ", ".join(sorted(set(labels)))
        return None, None, f"No '{target}' detected. YOLO found: {found}"

    eff_steps, eff_guidance, eff_negative, eff_dilate, _ = apply_style(prompt, negativePrompt, style, steps, guidanceScale)

    if masks is not None:
        mask = maskFromSegmentation(image, masks[target_idx], dilate=eff_dilate)
    else:
        mask = maskBox(image, boxes[target_idx], dilate=eff_dilate)

    final_prompt = enhancePrompt(prompt, target_object=target, style=style) if use_t5 else prompt
    intent = classifyIntent(prompt) if use_t5 else "n/a"
    result, score = mainProcess(image, 
                                mask, 
                                final_prompt, 
                                eff_negative,
                                steps=eff_steps, 
                                guidanceScale=eff_guidance,
                                seed=seed, 
                                score_prompt=prompt)
    ssim = preservationScore(image, result, mask)
    stats = (
        f"Style: {style}\n"
        f"Detected: {labels[target_idx]} (conf {scores[target_idx]:.2f})\n"
        f"Intent (T5): {intent}\n"
        f"Enhanced prompt: {final_prompt[:150]}{'...' if len(final_prompt) > 150 else ''}\n"
        f"CLIP Score: {score:.4f} (higher = better prompt match)\n"
        f"SSIM Preservation: {ssim:.4f} (higher = less unwanted change)"
    )
    return result, mask, stats


# Manual Mask Pipeline
def manual_mask(image, prompt, negativePrompt, style=DEFAULT_STYLE, use_t5=True, steps=30, guidanceScale=7.5, seed=1024):
    if image is None:
        return None, "Upload an image first."
    
    background = image.get("background")
    layers = image.get("layers")
    if background is None:
        return None, "No image."
    
    if not layers or all(layer is None for layer in layers):
        return None, "Draw a mask on the image first."
    
    background = preprocessImage(background, 512)
    width, height = background.size
    combined_array = np.zeros((height, width), dtype=np.uint8)
    for layer in layers:
        if layer is None:
            continue

        layer = layer.resize((width, height))
        layer_array = np.array(layer)
        if layer_array.ndim == 3 and layer_array.shape[2] == 4:
            alpha = layer_array[:, :, 3]

        elif layer_array.ndim == 3:
            alpha = np.max(layer_array, axis=2)

        else:
            alpha = layer_array

        combined_array = np.maximum(combined_array, alpha)
    binary_mask = np.where(combined_array > 10, 255, 0).astype(np.uint8)
    if binary_mask.sum() == 0:
        return None, "Mask is empty — draw over the area you want to edit."
    
    mask = Image.fromarray(binary_mask, mode="L")
    eff_steps, eff_guidance, eff_negative, _, _ = apply_style(prompt, negativePrompt, style, steps, guidanceScale)
    final_prompt = enhancePrompt(prompt, style=style) if use_t5 else prompt
    result, score = mainProcess(background, 
                                mask, 
                                final_prompt, 
                                eff_negative,
                                steps=eff_steps, 
                                guidanceScale=eff_guidance,
                                seed=seed, 
                                score_prompt=prompt)
    ssim = preservationScore(background, result, mask)
    return result, (
        f"Style: {style}\n"
        f"Enhanced prompt: {final_prompt[:150]}{'...' if len(final_prompt) > 150 else ''}\n"
        f"CLIP Score: {score:.4f}\n"
        f"SSIM Preservation: {ssim:.4f}"
    )


# Generates variations of output using different noise/seeds
def variationGrid(image, target, prompt, negativePrompt, num_variants=4):
    if image is None:
        return None, "Upload an image first."
    
    image = preprocessImage(image, 512)
    detections = detectObject(image)
    labels = detections["labels"]
    target_low = target.lower().strip()
    target_idxs = [i for i, l in enumerate(labels) if l.lower() == target_low]
    if not target_idxs and target_low.endswith("s"):
        target_idxs = [i for i, l in enumerate(labels) if l.lower() == target_low[:-1]]

    if not target_idxs:
        found = ", ".join(sorted(set(labels))) if labels else "nothing"
        return None, f"No '{target}' detected. YOLO found: {found}"
    
    eff_steps, eff_guidance, eff_negative, eff_dilate = styleAware(prompt, 25, 7.5, negativePrompt)
    if detections["masks"] is not None:
        combined = np.zeros_like(detections["masks"][target_idxs[0]])

        for i in target_idxs:
            combined = np.maximum(combined, detections["masks"][i])
        mask = maskFromSegmentation(image, combined, dilate=eff_dilate)
    else:
        best = max(target_idxs, key=lambda i: detections["scores"][i])
        mask = maskBox(image, detections["boxes"][best], dilate=eff_dilate)

    final_prompt = enhancePrompt(prompt, target_object=target)
    variations, scores = [], []
    for s in range(num_variants):
        result, score = mainProcess(
            image, mask, final_prompt, negativePrompt,
            steps=25, guidanceScale=7.5,
            seed=s * 1000 + 42, score_prompt=prompt,
        )
        variations.append(result)
        scores.append(score)
    w, h = variations[0].size
    grid = Image.new("RGB", (w * 2, h * 2))

    for i, v in enumerate(variations):
        grid.paste(v, ((i % 2) * w, (i // 2) * h))
    info = ", ".join([f"Seed {i}: CLIP {sc:.3f}" for i, sc in enumerate(scores)])
    return grid, info


# Shows latent representation of image
def visualizeLatent(image):
    if image is None:
        return None
    image = preprocessImage(image, 512)
    img_tensor = stableDiffusion.image_processor.preprocess(image).to(device, dtype=dtype)
    with torch.no_grad():
        latent = stableDiffusion.vae.encode(img_tensor).latent_dist.sample()
        latent = latent * stableDiffusion.vae.config.scaling_factor
    latent_np = latent[0].cpu().float().numpy()
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for i in range(4):
        ch = latent_np[i]
        ch = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)
        axes[i].imshow(ch, cmap="viridis")
        axes[i].set_title(f"Latent Channel {i + 1}")
        axes[i].axis("off")
    plt.suptitle("VAE Latent Representation (64x64x4)", fontsize=14)
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=80, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def build_ui():
    with gr.Blocks(title="M.A.G.E", theme=gr.themes.Soft()) as demo:
        gr.Markdown("## M.A.G.E 🤖\nMasked and Generative Editor - AI based Stable Diffusion Image Editor")

        with gr.Tabs():

            # Auto Masking
            with gr.TabItem("Auto-Masking (YOLO)"):
                with gr.Row():
                    with gr.Column():
                        auto_input = gr.Image(type="pil", label="Upload Image")
                        auto_mask_img = gr.Image(label="Generated Mask", interactive=False)
                        auto_target = gr.Textbox(label="Target object (e.g. 'car', 'dog')")
                        auto_prompt = gr.Textbox(label="Prompt")
                        auto_style = gr.Dropdown(choices=STYLE_CHOICES, value=DEFAULT_STYLE, label="Style")
                        auto_neg = gr.Textbox(
                            label="Negative Prompt",
                            value="blurry, low quality, distorted, deformed, ugly",
                        )
                        with gr.Accordion("Advanced settings", open=False):
                            auto_t5 = gr.Checkbox(label="Use T5 prompt enhancement", value=True)
                            auto_steps = gr.Slider(10, 100, value=30, step=1, label="Denoising steps")
                            auto_guide = gr.Slider(1.0, 15.0, value=7.5, step=0.5, label="Guidance scale")
                            auto_seed = gr.Number(value=1024, label="Seed", precision=0)
                        auto_btn = gr.Button("Edit Image", variant="primary")
                    with gr.Column():
                        auto_output = gr.Image(label="Output", interactive=False)
                        auto_info = gr.Textbox(label="Info", lines=6)
                auto_btn.click(
                    auto_mask,
                    inputs=[auto_input, auto_target, auto_prompt, auto_neg,
                            auto_style, auto_t5, auto_steps, auto_guide, auto_seed],
                    outputs=[auto_output, auto_mask_img, auto_info],
                )

            # Manual Masking
            with gr.TabItem("Manual Masking"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("Draw over the area you want to edit. NOTE: don't outline, completely white-out the area")
                        man_drawing = gr.ImageEditor(
                            label="Upload image then draw mask",
                            brush=gr.Brush(colors=["#FFFFFF"]),
                            type="pil",
                            height=700,
                        )
                        man_prompt = gr.Textbox(label="Prompt")
                        man_style = gr.Dropdown(choices=STYLE_CHOICES, value=DEFAULT_STYLE, label="Style")
                        man_neg = gr.Textbox(
                            label="Negative Prompt",
                            value="blurry, low quality, distorted",
                        )
                        with gr.Accordion("Advanced settings", open=False):
                            man_t5 = gr.Checkbox(label="Use T5 prompt enhancement", value=True)
                            man_steps = gr.Slider(10, 100, value=30, step=1, label="Denoising steps")
                            man_guide = gr.Slider(1.0, 15.0, value=7.5, step=0.5, label="Guidance scale")
                            man_seed = gr.Number(value=1024, label="Seed", precision=0)
                        man_btn = gr.Button("Edit Image", variant="primary")
                    with gr.Column():
                        man_output = gr.Image(label="Output", interactive=False)
                        man_info = gr.Textbox(label="Info", lines=4)
                man_btn.click(
                    manual_mask,
                    inputs=[man_drawing, man_prompt, man_neg,
                            man_style, man_t5, man_steps, man_guide, man_seed],
                    outputs=[man_output, man_info],
                )

            # Variations
            with gr.TabItem("Variation using DDPM"):
                gr.Markdown(
                    "Generate 4 outputs with different DDPM sampling seeds"
                    "\nDifferent noise = different valid outputs "
                )
                with gr.Row():
                    with gr.Column():
                        var_img = gr.Image(type="pil", label="Upload Image")
                        var_target = gr.Textbox(label="Target object")
                        var_prompt = gr.Textbox(label="Prompt")
                        var_neg = gr.Textbox(
                            label="Negative Prompt",
                            value="blurry, low quality, distorted",
                        )
                        var_btn = gr.Button("Generate 4 Variations", variant="primary")
                    with gr.Column():
                        var_out = gr.Image(label="4 Variations (2x2 grid)")
                        var_info = gr.Textbox(label="CLIP scores per variant", lines=2)
                var_btn.click(
                    variationGrid,
                    inputs=[var_img, var_target, var_prompt, var_neg],
                    outputs=[var_out, var_info],
                )

            # VAE Visualizer
            with gr.TabItem("VAE Visualizer"):
                gr.Markdown(
                    "Visualize latent representation of image layer, "
                    "this shows the compressed space that diffusion runs on "
                )
                with gr.Row():
                    with gr.Column():
                        vae_img = gr.Image(type="pil", label="Upload Image")
                        vae_btn = gr.Button("Encode to Latent Space")
                    with gr.Column():
                        vae_out = gr.Image(label="Latent channels")
                vae_btn.click(visualizeLatent, inputs=[vae_img], outputs=[vae_out])

    return demo


# Entry Point
if __name__ == "__main__":
    multiprocessing.freeze_support()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Using device: {device}")

    # Stable Diffusion
    print("Loading Stable Diffusion inpainting pipeline...")
    stableDiffusion = StableDiffusionInpaintPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-inpainting",
        torch_dtype=dtype,
        safety_checker=None,
    ).to(device=device)
    stableDiffusion.enable_attention_slicing()

    # YOLO
    print("Loading YOLO model...")
    yoloModel = YOLO("yolo11s-seg.pt")

    # T5
    print("Loading T5 model...")
    t5Tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
    t5Model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base").to(device)

    # CLIP
    print("Loading CLIP model...")
    clipModel = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clipProcessor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    print("All models loaded.\n")

    # Build and launch UI
    demo = build_ui()
    demo.launch(server_name="127.0.0.1", server_port=7860, debug=True)
