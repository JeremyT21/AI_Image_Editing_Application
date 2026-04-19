# G.R.I.D. - Generative Retargeting & Editing Robot

Presentation Link: [Presentation HERE](https://docs.google.com/presentation/d/1mV8537iCb7vSeOqNhG83GdzGO182wE9M3IOhMXgK8NA/edit?slide=id.g3d72a841141_3_51#slide=id.g3d72a841141_3_51) -- change to YouTube video when uploaded


| Name | Student Number |
| ----------- | ----------- |
| Adam Kolodziejczak | 100874535 |
| Jeremy Thummel | 100 ... |
| Tyson Grant | 100875284


## The Problem
Traditional image editing requires the user to select regions of an image using a magic wand or lasso tool, then hand-craft layers or images to replace or distort these images. Traditional techniques do not have a grasp on what objects are, or what they are being replaced with.

Using AI-guided image editing with stable diffusion, it allows inpatining to detect and mask a region to then replace this area using a text-guided prompt. This allows for more efficient image editing and gives a AI creative-touch to the final output, saving the user time from having to manually create an image to overlay.

## The Neural Network Components
1. Stable Diffusion In-Painting
2. YOLO (Auto-Masking)
3. CLIP Score


## End-to-End Application Pipeline

| Step | Component | Detail |
|---|---|---|
| 1 | **Input** | PIL image + text prompt |
| 2 | **preprocessImage()** | Square crop, resize to 512×512 |
| 3a | **Auto mask** | YOLO detects object → bounding box → binary mask |
| 3b | **Manual mask** | User brush strokes → merged layers → binary mask |
| 4 | **Stable Diffusion inpainting** | VAE encodes → U-Net denoises 30 steps |
| 5 | **CLIP scoring** | Computes Cosine Similarity to give the float score between output and prompt |
| 6 | **Gradio output** | Resulting image |


## Deployment



