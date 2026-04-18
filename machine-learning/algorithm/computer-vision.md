# Computer Vision Algorithms

---

## Image Classification

### VGG (VGG16 / VGG19)

**Core idea:** very deep networks using only 3×3 convolution filters stacked sequentially. Depth is the primary driver of performance.

**Architecture:**
```
Input → [Conv3×3 → Conv3×3 → MaxPool] × 5 blocks → FC(4096) → FC(4096) → Softmax
```

| Pros | Cons |
|------|------|
| Simple, uniform architecture — easy to understand | Very large — 138M parameters (VGG16) |
| Strong feature extractor for transfer learning | Slow inference, high memory |
| Well-studied, many pretrained weights available | Outperformed by ResNet and EfficientNet |

**When to use:** transfer learning baseline when simplicity matters. Rarely used as a primary model today.

---

### ResNet (Residual Network)

**Core idea:** residual (skip) connections allow gradients to flow directly through the network, solving the vanishing gradient problem and enabling very deep networks (50, 101, 152 layers).

**Architecture:**
```
Input → Conv → [Residual Block × N] → GlobalAvgPool → FC → Softmax

Residual Block:
x → Conv → BN → ReLU → Conv → BN → (+x) → ReLU
            ↑___________________________|  (skip connection)
```

The skip connection adds the input directly to the output of the block. If the block learns nothing useful, the gradient still flows through the identity path.

| Pros | Cons |
|------|------|
| Solves vanishing gradient — enables 100+ layer networks | Larger than MobileNet for edge deployment |
| Strong baseline for most vision tasks | Not as parameter-efficient as EfficientNet |
| Widely supported, many pretrained variants | |

**When to use:** general-purpose image classification and as a backbone for detection/segmentation. ResNet-50 is a reliable default.

---

### EfficientNet

**Core idea:** compound scaling — simultaneously scale network depth, width, and input resolution using a fixed ratio derived by neural architecture search. EfficientNet-B0 through B7 trade off speed vs accuracy.

**Architecture:** MBConv blocks (mobile inverted bottleneck) with squeeze-and-excitation, scaled by compound coefficient φ:
```
depth    = α^φ
width    = β^φ
resolution = γ^φ
```

| Variant | Params | Top-1 ImageNet | Use case |
|---------|--------|---------------|----------|
| B0 | 5.3M | 77.1% | Mobile / edge |
| B4 | 19M | 82.9% | Balanced |
| B7 | 66M | 84.3% | Highest accuracy |

| Pros | Cons |
|------|------|
| Best accuracy/parameter trade-off on ImageNet | Slower than MobileNet on edge hardware |
| Compound scaling is principled — not ad-hoc | B7 is large and slow |
| Strong transfer learning backbone | |

**When to use:** default choice for image classification when accuracy matters. B4 is a strong balanced option.

---

### MobileNet (V1 / V2 / V3)

**Core idea:** depthwise separable convolutions — split a standard convolution into a depthwise convolution (per-channel spatial filter) and a pointwise convolution (1×1 cross-channel mix). This reduces computation by ~8–9x with minimal accuracy loss.

**Architecture (MobileNetV2):**
```
Input → Conv → [Inverted Residual Block × N] → Conv → GlobalAvgPool → FC
```
Inverted residual: expand channels → depthwise conv → project back to narrow channels.

| Pros | Cons |
|------|------|
| Very fast — designed for mobile and edge | Lower accuracy than EfficientNet / ResNet |
| Small model size — fits on device | Less suitable for complex scenes |
| Hardware-optimised (TFLite, CoreML) | |

**When to use:** real-time inference on mobile, edge devices, or when latency is the primary constraint.

---

### Vision Transformer (ViT)

**Core idea:** apply the Transformer architecture (originally designed for NLP) directly to images by splitting the image into fixed-size patches, treating each patch as a token, and processing the sequence with self-attention.

**Architecture:**
```
Image (H×W×C)
  → Split into N patches of size P×P
  → Flatten each patch → Linear projection → Patch embeddings (N × D)
  → Prepend [CLS] token
  → Add positional embeddings
  → Transformer Encoder × L layers
      (Multi-Head Self-Attention → LayerNorm → MLP → LayerNorm)
  → [CLS] token output → MLP Head → Class probabilities
```

**Patch embedding:** an image of 224×224 with patch size 16×16 produces (224/16)² = 196 patch tokens. Each patch is flattened to a vector of size 16×16×3 = 768 and projected to dimension D.

**Self-attention across patches:** unlike CNNs which have local receptive fields, every patch attends to every other patch from the first layer. This gives ViT a global receptive field by design — it can model long-range spatial dependencies that CNNs only capture in deep layers.

**Positional embeddings:** since Transformers have no inherent notion of position, learnable 1-D positional embeddings are added to patch embeddings to encode spatial location.

**[CLS] token:** a learnable token prepended to the sequence. Its output representation after the final Transformer layer is used as the image-level feature for classification.

| Variant | Patches | Layers | Heads | Params |
|---------|---------|--------|-------|--------|
| ViT-B/16 | 16×16 | 12 | 12 | 86M |
| ViT-L/16 | 16×16 | 24 | 16 | 307M |
| ViT-H/14 | 14×14 | 32 | 16 | 632M |

| Pros | Cons |
|------|------|
| Global receptive field from layer 1 — captures long-range dependencies | Requires large datasets or strong pretraining to outperform CNNs |
| Scales well with data and compute | Quadratic attention cost with sequence length — large images are expensive |
| Unified architecture with NLP — same Transformer for vision and text | Less inductive bias than CNNs — needs more data to learn spatial structure |
| Strong transfer learning with MAE / DINO pretraining | Patch size is a hard hyperparameter — smaller patches = more tokens = more compute |

**When to use:** large-scale image classification with abundant data or strong pretrained weights (ImageNet-21k, CLIP, MAE). For small datasets, EfficientNet or ResNet with transfer learning is more practical.

**ViT variants worth knowing:**

| Model | Key idea |
|-------|---------|
| DeiT | Data-efficient ViT — distillation token enables training on ImageNet without extra data |
| Swin Transformer | Hierarchical ViT with shifted windows — local attention, scales to dense prediction tasks |
| CLIP | ViT trained with contrastive image-text pairs — powerful zero-shot classifier |
| MAE | Masked Autoencoder — self-supervised ViT pretraining by reconstructing masked patches |
| DINOv2 | Self-supervised ViT trained with self-distillation — strong general-purpose visual features without labels |
| SAM (Segment Anything) | ViT-based promptable segmentation — accepts point, box, or mask prompts to segment any object |

---

### ConvNeXt

**Core idea:** a pure CNN redesigned to match ViT performance by incorporating design choices from Transformers — larger kernels (7×7), inverted bottleneck, fewer activation functions, and LayerNorm instead of BatchNorm. Demonstrates that CNNs can match ViTs when modernised.

| Pros | Cons |
|------|------|
| Strong accuracy with CNN simplicity — no attention overhead | Does not capture global context as naturally as ViT |
| Faster than ViT on standard hardware | Less flexible for non-image modalities |
| Drop-in replacement for ResNet in most pipelines | |

**When to use:** when you want ViT-level accuracy with CNN-level inference speed and deployment simplicity.

---

## Object Detection

### YOLOv5

**Core idea:** single-stage detector — predict bounding boxes and class probabilities directly from the full image in one forward pass. No separate region proposal step.

**Architecture:**
```
Input → CSPDarknet backbone → PANet neck (multi-scale feature fusion) → Detection head
Detection head outputs: [x, y, w, h, objectness, class_probs] per anchor per grid cell
```

| Pros | Cons |
|------|------|
| Very fast — real-time at 30–140 FPS | Lower accuracy than two-stage detectors on small objects |
| Small model variants (YOLOv5n, s, m, l, x) | Anchor-based — requires anchor tuning per dataset |
| Easy to train and deploy | |

**When to use:** real-time detection, edge deployment, video streams.

---

### Faster R-CNN

**Core idea:** two-stage detector. Stage 1: Region Proposal Network (RPN) proposes candidate object regions. Stage 2: RoI pooling extracts features per region, then classifies and refines bounding boxes.

**Architecture:**
```
Input → Backbone (ResNet/VGG) → Feature Map
  → RPN → Region proposals (anchors scored by objectness)
  → RoI Pooling → Per-region FC → Class + Box regression
```

| Pros | Cons |
|------|------|
| High accuracy, especially on small objects | Slow — two forward passes per image |
| Flexible backbone | Not suitable for real-time |
| Strong on dense, overlapping objects | |

**When to use:** offline detection where accuracy matters more than speed.

---

### EfficientDet-D7

**Core idea:** EfficientNet backbone + BiFPN (bidirectional feature pyramid network) neck + compound scaling applied to detection. D0–D7 scale accuracy vs speed.

| Pros | Cons |
|------|------|
| State-of-the-art accuracy on COCO | D7 is slow — not real-time |
| Principled scaling like EfficientNet | Complex architecture |

**When to use:** highest accuracy offline detection.

---

### DETR (Detection Transformer)

**Core idea:** reformulate object detection as a set prediction problem using a Transformer encoder-decoder. Eliminates anchors, NMS (non-maximum suppression), and hand-crafted components entirely.

**Architecture:**
```
Image → CNN backbone → flatten → Transformer encoder
  → Transformer decoder (N learned object queries)
  → FFN per query → (class, bounding box) × N
```

Object queries are learned embeddings that attend to the encoder output. Each query specialises in detecting objects at different positions and scales. The Hungarian algorithm matches predictions to ground truth during training — no NMS needed.

| Pros | Cons |
|------|------|
| No anchors or NMS — cleaner pipeline | Slow to train — requires many epochs to converge |
| End-to-end trainable | Weaker on small objects than anchor-based methods |
| Naturally handles a variable number of objects | |

**Variants:** Deformable DETR (faster convergence, better small object detection), DINO-DETR (state-of-the-art on COCO).

**When to use:** when you want a clean, anchor-free detection pipeline and training time is not a constraint.

---

### YOLOv8 / YOLOv10 / YOLO-World

**Core idea:** successive YOLO generations move to anchor-free detection heads, improved backbone designs, and task-unified architectures (detection + segmentation + pose in one model).

| Version | Key improvement |
|---------|-----------------|
| YOLOv8 | Anchor-free head, unified detection/segmentation/pose API |
| YOLOv10 | Dual-label assignment removes NMS at inference — lower latency |
| YOLO-World | Open-vocabulary detection — detect any object described in text, not just trained classes |

**When to use:** YOLOv8 is the current practical default for real-time detection. YOLO-World when you need open-vocabulary detection without retraining.

---

## Image Segmentation

### Mask R-CNN

**Core idea:** extends Faster R-CNN with a third head that predicts a binary segmentation mask for each detected object instance (instance segmentation).

**Architecture:**
```
Faster R-CNN pipeline → RoI Align → Mask head (small FCN per region) → Binary mask per instance
```

RoI Align replaces RoI Pooling — uses bilinear interpolation to avoid quantisation artefacts, which is critical for pixel-level mask accuracy.

| Pros | Cons |
|------|------|
| Instance segmentation — separates overlapping objects | Slow — inherits Faster R-CNN cost |
| Flexible — backbone and neck are swappable | Complex training pipeline |
| Strong baseline for instance segmentation | |

**When to use:** instance segmentation where individual object masks are needed.

---

### U-Net

**Core idea:** encoder-decoder architecture with skip connections between corresponding encoder and decoder layers. Originally designed for biomedical image segmentation where training data is scarce.

**Architecture:**
```
Encoder (contracting path):
  Input → [Conv → Conv → MaxPool] × 4 → Bottleneck

Decoder (expanding path):
  Bottleneck → [Upsample → Concat(skip) → Conv → Conv] × 4 → Output mask

Skip connections copy feature maps from encoder to decoder at each resolution level,
preserving fine spatial detail lost during downsampling.
```

| Pros | Cons |
|------|------|
| Works well with small datasets | Designed for semantic segmentation — not instance |
| Preserves fine spatial detail via skip connections | Less accurate than Mask R-CNN for instance tasks |
| Fast training and inference | |
| Widely used in medical imaging | |

**When to use:** semantic segmentation, medical imaging, any task with limited training data.

---

### Segment Anything Model (SAM / SAM 2)

**Core idea:** a foundation model for image (and video) segmentation. Accepts flexible prompts — a point, a bounding box, or a rough mask — and produces a high-quality segmentation mask for the indicated object.

**Architecture:**
```
Image → ViT image encoder → image embedding
Prompt (point/box/mask) → prompt encoder → prompt embedding
Image embedding + prompt embedding → lightweight mask decoder → mask + confidence
```

The image encoder runs once per image. The prompt encoder and mask decoder are lightweight — enabling interactive segmentation at near-real-time speed after the image is encoded.

SAM 2 extends this to video — propagating masks across frames using a memory mechanism.

| Pros | Cons |
|------|------|
| Zero-shot — segments any object without task-specific training | Large image encoder (ViT-H) is slow without GPU |
| Flexible prompting — point, box, text, or automatic grid | Not optimised for semantic segmentation (no class labels) |
| SAM 2 handles video with temporal consistency | |

**When to use:** interactive annotation tools, zero-shot instance segmentation, video object tracking.

---

## Video Classification

### MoViNet (Mobile Video Networks)

**Core idea:** 3D convolutions with causal streaming — process video frame-by-frame using a buffer state, enabling real-time video classification without loading the full clip into memory.

**Architecture:**
```
Frame stream → Causal 3D Conv (temporal + spatial) → Buffer state → Classification head
```
Causal means each frame only attends to past frames — no future frames needed. This enables online/streaming inference.

| Pros | Cons |
|------|------|
| Streaming — classifies video in real time | Lower accuracy than offline 3D CNNs on long clips |
| Mobile-friendly — small and fast | Limited temporal context per frame |
| Strong accuracy/compute trade-off | |

**When to use:** real-time video classification on mobile or edge devices.

---

### Video Transformers — VideoMAE / TimeSformer

**Core idea:** extend ViT to video by treating the video as a sequence of spatiotemporal patches (tubes). VideoMAE applies masked autoencoding to video — masking ~90% of tubes and reconstructing them, enabling efficient self-supervised pretraining.

| Model | Key idea |
|-------|----------|
| TimeSformer | Divided space-time attention — separate temporal and spatial attention heads |
| VideoMAE | Masked autoencoding on video tubes — very high masking ratio (90%) works well for video |
| Video Swin | Swin Transformer extended to 3D with shifted spatiotemporal windows |

**When to use:** offline video understanding tasks (action recognition, video classification) where accuracy matters more than real-time speed.

---

## Multimodal Models

Models that jointly process multiple modalities (image + text, audio + text) have become a major area of development.

| Model | Modalities | Key capability |
|-------|-----------|----------------|
| CLIP | Image + Text | Contrastive pretraining — zero-shot image classification via text prompts |
| BLIP-2 | Image + Text | Connects frozen image encoder and frozen LLM via a lightweight Q-Former |
| LLaVA | Image + Text | Visual instruction tuning — connects ViT to an LLM for visual question answering |
| Flamingo | Image/Video + Text | Few-shot visual question answering via cross-attention between vision and language |
| Whisper | Audio + Text | Robust speech recognition and translation across many languages |

---

## Data Augmentation

Augmentation artificially expands the training set by applying label-preserving transformations. It is one of the most effective regularisation techniques for vision models.

| Technique | What it does | When to use |
|-----------|-------------|------------|
| Horizontal flip | Mirror image left-right | Most tasks (not text, asymmetric objects) |
| Vertical flip | Mirror image top-bottom | Aerial/satellite imagery, medical scans |
| Rotation | Rotate by random angle | When orientation is not discriminative |
| Scaling / crop | Random resize and crop | Classification — forces model to use partial views |
| Colour jitter | Random brightness, contrast, saturation | When lighting conditions vary |
| Cutout / Random Erasing | Mask random rectangular regions | Forces model to use distributed features |
| MixUp | Blend two images and their labels | Improves calibration and generalisation |
| CutMix | Replace a region with a patch from another image | Stronger than Cutout |
| AutoAugment / RandAugment | Learned or random policy of augmentations | When you want strong augmentation without manual tuning |

```python
import torchvision.transforms as T

train_transform = T.Compose([
    T.RandomResizedCrop(224),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```
