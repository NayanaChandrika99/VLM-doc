# Medical VLM Triage

**Production-oriented page triage system using OlmOCR-2 VLM to classify document pages into 16 classes with calibrated confidence scores**

---

## Why This Project?

Medical document processing pipelines need **reliable, layout-aware page classification** that can route high-confidence predictions automatically while abstaining on ambiguous pages. Manual triage is slow and error-prone; existing classifiers often produce overconfident predictions that break downstream automation.

This system solves that by combining:
- **RVL-CDIP's 16-class taxonomy** for robust page-type recognition
- **DocLayNet layout supervision** to handle tables, figures, and complex layouts
- **Calibrated confidence scores** via temperature scaling for trustworthy thresholds
- **Structured JSON outputs** guaranteed by constrained decoding
- **LoRA adapter infrastructure** for per-tenant customization without full retraining

---

## Visual Flow

```
┌─────────────┐
│ Page Image  │
│ (PNG/JPEG)  │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────┐
│   OlmOCR-2 Backbone (7B)   │
│ richardyoung/olmOCR-2-7B-  │
│      MLX-4bit quantized     │
└──────┬─────────────┬────────┘
       │             │
       │             └─────────────────┐
       ▼                               ▼
┌─────────────────┐        ┌──────────────────────┐
│  Image Embed    │        │  Layout Predictor    │
│    (z_img)      │        │  (trained on         │
│                 │        │   DocLayNet)         │
└────────┬────────┘        └──────────┬───────────┘
         │                            │
         │                            ▼
         │                  ┌──────────────────┐
         │                  │  Layout Features │
         │                  │    (z_lay)       │
         │                  └────────┬─────────┘
         │                           │
         └───────────┬───────────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │   Fusion Layer       │
          │  [z_img ; z_lay]     │
          └──────────┬───────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │   RVL Classifier     │
          │   (16-way head)      │
          └──────────┬───────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │  Calibration Layer   │
          │ (Temperature Scaling)│
          └──────────┬───────────┘
                     │
                     ▼
          ┌──────────────────────────────────┐
          │   Structured JSON Output         │
          │  {                                │
          │    "label": "invoice",            │
          │    "confidence": 0.94,            │
          │    "meta": {                      │
          │      "model_id": "...",           │
          │      "calibration_id": "...",     │
          │      "adapter_id": "global"       │
          │    }                              │
          │  }                                │
          └──────────────────────────────────┘
```

---

## Tech Stack

- **ML Framework**: PyTorch ≥2.2, torchvision ≥0.17
- **Backbone Model**: [OlmOCR-2-7B-MLX-4bit](https://huggingface.co/richardyoung/olmOCR-2-7B-1025-MLX-4bit) (quantized document VLM)
- **Datasets**: [RVL-CDIP](https://adamharley.com/rvl-cdip/) (400k pages, 16 classes), [DocLayNet](https://github.com/DS4SD/DocLayNet) (80k pages, 11 layout classes)
- **Serving**: FastAPI + vLLM (PagedAttention, LoRA hot-swap)
- **Structured Generation**: Outlines (constrained JSON decoding)
- **Adapter Training**: Hugging Face PEFT/LoRA
- **Monitoring**: Prometheus metrics, reliability plots
- **Dependencies**: transformers, datasets, huggingface-hub, fastapi, uvicorn, matplotlib

---

## Features & ML Components

### Core Capabilities
✓ **16-Class Page Classification** — Predicts document type from RVL-CDIP taxonomy (letter, form, email, invoice, resume, etc.)  
✓ **Layout-Aware Features** — Learns table/figure/title distributions from DocLayNet to improve robustness  
✓ **Calibrated Confidence** — Temperature scaling ensures confidence scores reflect true accuracy  
✓ **Abstention Policy** — Returns `label="unknown"` when confidence < threshold (autopilot-safe)  
✓ **Structured Outputs** — 100% schema-valid JSON via Outlines constrained decoding  
✓ **LoRA Adapters** — Per-tenant fine-tuning without full model retraining  
✓ **Observability** — Prometheus metrics (latency, ECE, class distribution, abstention rate)  

### Model Components
- **Backbone**: OlmOCR-2 quantized VLM for vision+text understanding
- **Layout Head** (`gθ`): Predicts 11-class layout descriptors from DocLayNet
- **Classifier Head** (`fφ`): 16-way MLP trained on RVL-CDIP
- **Calibration**: Temperature scaling fitted on validation logits
- **Serving**: vLLM-ready with LoRA adapter registry

---

## Quick Start

### 1. Install Dependencies
```bash
# Ensure Python 3.11+ is installed
pip install -r requirements.txt
```

### 2. Run Tests
```bash
# Unit tests for data adapters, models, calibration
PYTHONPATH=$PWD pytest tests/

# Specific test suites
pytest tests/test_data_adapters.py          # Dataset loaders
pytest tests/test_layout_features.py        # Layout descriptors
pytest tests/test_rvl_training.py          # Classifier training
pytest tests/test_calibration.py           # Calibration fitting
```

### 3. Train Classifier (Dry-Run Mode)
```bash
# Layout head training (uses synthetic embeddings)
PYTHONPATH=$PWD python triage/layout/train_layout_head.py \
  --dry-run \
  --embedding-dim 1024 \
  --epochs 1

# RVL classifier training (uses synthetic embeddings)
PYTHONPATH=$PWD python triage/triage/train_rvl_classifier.py \
  --dry-run \
  --epochs 1

# Evaluate classifier
PYTHONPATH=$PWD python triage/triage/eval_rvl_classifier.py \
  --checkpoint triage/artifacts/rvl_classifier.pt \
  --dry-run
```

### 4. Calibrate & Tune Abstention
```bash
# Fit temperature scaling
PYTHONPATH=$PWD python triage/calibration/fit_temperature.py --dry-run

# Evaluate calibration quality
PYTHONPATH=$PWD python triage/calibration/evaluate_calibration.py \
  --dry-run \
  --plot-dir reports

# Sweep confidence thresholds
PYTHONPATH=$PWD python scripts/tune_abstention.py --dry-run
```

### 5. Start Serving API
```bash
# Local FastAPI server (stub predictor)
uvicorn triage.serving.app:app --reload

# Health check
curl http://127.0.0.1:8000/healthz

# Predict (example)
curl -X POST http://127.0.0.1:8000/triage \
  -F "image=@path/to/page.png"
```

---

## Usage Examples

### Baseline Inference (Prompted)
```bash
# Single-page triage with baseline prompt
python triage_infer.py --image path/to/page.png
# Outputs: {"label": "invoice", "confidence": 0.87, ...}
```

### Training with Real Embeddings (GPU Host)
```bash
# 1. Generate embeddings from OlmOCR-2 backbone
# (requires GPU and full model weights)

# 2. Train layout head with real embeddings
PYTHONPATH=$PWD python triage/layout/train_layout_head.py \
  --embedding-store /path/to/doclaynet_embeddings.npz \
  --embedding-dim 1024 \
  --epochs 5 \
  --batch-size 64

# 3. Train RVL classifier with layout fusion
PYTHONPATH=$PWD python triage/triage/train_rvl_classifier.py \
  --embedding-store /path/to/rvl_embeddings.npz \
  --use-layout \
  --layout-head-checkpoint layout/artifacts/layout_head.pt \
  --epochs 10 \
  --batch-size 64
```

### LoRA Adapter Training (Scaffold)
```bash
# Create synthetic adapter (dry-run)
PYTHONPATH=$PWD python triage/adapters/train_lora_adapter.py \
  --adapter-id demo \
  --dry-run

# On GPU: train real PEFT adapter
PYTHONPATH=$PWD python triage/adapters/train_lora_adapter.py \
  --adapter-id tenant_acme \
  --data-path /path/to/tenant_data.jsonl \
  --epochs 3
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_DATASETS_CACHE` | `~/.cache/huggingface/datasets` | Cache location for RVL-CDIP and DocLayNet |
| `HF_HOME` | `~/.cache/huggingface` | Hugging Face hub cache directory |
| `TRIAGE_SERVING_MODE` | `stub` | `stub` for local dev, `vllm` for production |
| `PYTHONPATH` | (none) | Set to `$PWD` for running scripts from repo root |

---

## File Structure

```
medical_vlm/
├── triage/
│   ├── data/
│   │   ├── rvl_adapter.py          # RVL-CDIP dataset loader
│   │   ├── doclaynet_adapter.py    # DocLayNet dataset loader
│   │   ├── transforms.py           # Image preprocessing
│   │   └── metadata/
│   │       ├── split_guard.py      # Dataset integrity checks
│   │       └── doclaynet_guard.py  # Document-level split validation
│   ├── backbone/
│   │   └── (future: OlmOCR-2 server wrapper)
│   ├── layout/
│   │   ├── features.py             # Layout descriptor computation
│   │   ├── dataset.py              # Embedding+descriptor dataset
│   │   ├── predict_head.py         # Layout prediction MLP
│   │   └── train_layout_head.py    # Training CLI
│   ├── triage/
│   │   ├── dataset.py              # RVL classifier dataset
│   │   ├── model.py                # RVL classifier model
│   │   ├── train_rvl_classifier.py # Training CLI
│   │   ├── eval_rvl_classifier.py  # Evaluation CLI
│   │   └── apply_abstention.py     # Threshold application
│   ├── calibration/
│   │   ├── fit_temperature.py      # Temperature scaling fit
│   │   ├── evaluate_calibration.py # ECE + reliability plots
│   │   ├── temp_scaling.py         # Calibration utilities
│   │   └── plotting.py             # Visualization helpers
│   ├── adapters/
│   │   ├── train_lora_adapter.py   # LoRA training scaffold
│   │   ├── registry.py             # Adapter registry helpers
│   │   └── adapter_registry.json   # Adapter metadata
│   ├── serving/
│   │   ├── app.py                  # FastAPI endpoints
│   │   └── metrics.py              # Prometheus instrumentation
│   ├── io/
│   │   ├── schema.json             # Output JSON schema
│   │   └── structured.py           # Constrained generation
│   └── configs/
│       ├── rvl.yaml                # RVL dataset config
│       ├── doclaynet.yaml          # DocLayNet config
│       ├── train.yaml              # Training hyperparameters
│       └── lora.yaml               # LoRA adapter config
├── tests/
│   ├── test_data_adapters.py
│   ├── test_layout_features.py
│   ├── test_rvl_training.py
│   ├── test_calibration.py
│   └── test_serving_app.py
├── scripts/
│   ├── tune_abstention.py          # Threshold sweep
│   └── run_regression.py           # Test orchestration
├── metrics/
│   ├── baseline.json               # Baseline prompt metrics
│   ├── rvl_classifier.json         # Classifier evaluation
│   ├── calibration.json            # Calibration quality
│   └── abstention.json             # Threshold analysis
├── reports/                         # Visualizations (plots, HTML)
├── plan/                            # Planning documents
├── triage_infer.py                  # Single-page inference CLI
├── requirements.txt
└── README.md
```

---

## Troubleshooting

### DocLayNet Download Issues
**Problem**: `load_dataset("pierreguillou/DocLayNet-base")` fails or hangs  
**Solution**: 
- Ensure Hugging Face auth token is set: `huggingface-cli login`
- Check disk space (~3.2 GB required for DocLayNet base zip)
- Set `HF_DATASETS_OFFLINE=1` and manually download dataset for air-gapped environments

### Embedding Generation on CPU
**Problem**: Real embeddings require GPU but only CPU available  
**Solution**:
- Use `--dry-run` flag for all training scripts (generates synthetic embeddings)
- For production, generate embeddings on GPU-enabled host and transfer `.npz` files

### Calibration Fitting Failures
**Problem**: Temperature scaling produces NaN or very high temperatures  
**Solution**:
- Verify validation logits file exists: `triage/artifacts/rvl_val_logits.npz`
- Check for extreme outliers in logit values
- Ensure validation set is not empty or highly imbalanced

### Test Failures with Real Tesseract
**Problem**: Tests fail when `TENNR_TEST_USE_TESSERACT=1`  
**Solution**:
- Install Tesseract: `brew install tesseract` (macOS) or `apt-get install tesseract-ocr` (Linux)
- Verify installation: `tesseract --version`
- Ensure test images are valid and not corrupted

### vLLM LoRA Adapter Loading
**Problem**: Adapter loading fails with "model does not support LoRA"  
**Solution**:
- Verify base model implements `SupportsLoRA` interface
- Check adapter registry: `cat triage/adapters/adapter_registry.json`
- Ensure adapter checkpoint paths are correct and accessible

### JSON Schema Validation Errors
**Problem**: API returns validation errors on output  
**Solution**:
- Inspect `triage/io/schema.json` for expected format
- Check Outlines installation: `pip show outlines-core`
- Verify constrained decoding is enabled in serving config


---

## Key References

- **RVL-CDIP Dataset**: [Adam Harley](https://adamharley.com/rvl-cdip/) — 400k pages, 16 balanced classes
- **DocLayNet**: [GitHub](https://github.com/DS4SD/DocLayNet) — 80k pages with 11 layout region classes
- **OlmOCR-2 Model**: [Hugging Face](https://huggingface.co/richardyoung/olmOCR-2-7B-1025-MLX-4bit) — Quantized document VLM
- **vLLM Serving**: [Docs](https://docs.vllm.ai/) — PagedAttention and LoRA adapter support
- **Outlines**: [Dottxt AI](https://dottxt-ai.github.io/outlines/) — Structured generation library
- **Calibration Paper**: Guo et al., ["On Calibration of Modern Neural Networks"](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf) — Temperature scaling method

---

## License & Credits

This repository is provided for research and production use.  
Built on RVL-CDIP (Adam Harley), DocLayNet (DS4SD), and OlmOCR-2 (richardyoung).  
Calibration methods inspired by Guo et al. (ICML 2017).

