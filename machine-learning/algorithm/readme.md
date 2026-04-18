# ML Algorithm Reference

A structured reference of algorithms by domain. Each section covers the algorithm's core idea, architecture, strengths, weaknesses, and when to use it.

---

## Contents

| Area | File |
|------|------|
| Computer Vision | [computer-vision.md](computer-vision.md) |
| NLP & Language Models | [nlp.md](nlp.md) |
| Generative AI | [generative-ai.md](generative-ai.md) |
| Graph Neural Networks | [graph-networks.md](graph-networks.md) |
| Reinforcement Learning from Human Feedback | [rlhf-dpo.md](rlhf-dpo.md) |

---

## Quick Selection Guide

```
Tabular data, classification/regression   → Gradient Boosting (XGBoost, LightGBM)
Image classification, high accuracy       → EfficientNet / ConvNeXt / ViT
Image classification, edge / mobile       → MobileNet
Object detection, real-time               → YOLOv8 / YOLOv10
Object detection, open-vocabulary         → YOLO-World / Grounding DINO
Object detection, highest accuracy        → DINO-DETR / EfficientDet-D7
Image segmentation, any object zero-shot  → SAM / SAM 2
Image segmentation, medical               → U-Net
Image segmentation, instance              → Mask R-CNN
Video classification, real-time           → MoViNet
Video classification, offline             → VideoMAE / Video Swin
Multimodal (image + text)                 → CLIP / LLaVA / BLIP-2
Text understanding / classification       → BERT / RoBERTa / DeBERTa fine-tune
Text generation / chatbot                 → GPT-4o / Claude 3.5 / Gemini 2.0
Text generation, open-weight              → LLaMA 3 / Mistral / Qwen 2.5
Reasoning tasks (maths, coding)           → o3 / DeepSeek-R1 / QwQ
Preference alignment, limited compute     → DPO / SimPO
Preference alignment, full control        → RLHF (PPO)
Preference alignment, reasoning           → GRPO (DeepSeek-R1 approach)
Parameter-efficient fine-tuning           → LoRA / QLoRA
Image generation, open-weight             → FLUX.1 / Stable Diffusion XL
Image generation, API                     → DALL-E 3 / Midjourney v6
Graph-structured data                     → GCN / GraphSAGE
Recommendation / link prediction          → Node2Vec / Matrix Factorisation
Retrieval-augmented generation            → RAG (vector DB + LLM)
LLM agents / tool use                     → ReAct / Function calling / MCP
```
