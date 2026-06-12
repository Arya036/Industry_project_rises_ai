# SilicoSense — AI-Powered Silicosis Detection & Clinical Reporting

> **Industry Project | RISES AI**  
> An end-to-end clinical decision support system for occupational silicosis screening from chest X-rays.

---

## Overview

Silicosis is a progressive, irreversible occupational lung disease caused by crystalline silica dust inhalation — predominantly affecting stone workers, miners, and construction labourers. It remains severely underdiagnosed due to radiologist shortages and its visual overlap with tuberculosis on chest radiographs.

**SilicoSense** automates silicosis risk assessment directly from a chest X-ray (CXR) by combining:

- A deep learning binary risk classifier
- A multi-label radiological finding detector
- GradCAM-based explainability heatmaps
- A fine-tuned multimodal language model for clinical report generation

All components are served through a Flask web application accessible in a browser — delivering a complete diagnosis in under 60 seconds.

---

## Repository Structure

```
Industry_project_rises_ai/
├── silicosis-risk-detector/   ← Main diagnostic web application
│   ├── src/                   ← Flask backend + EfficientNet classifiers + GradCAM
│   ├── kaggle_api/            ← MedGemma API server (runs on Kaggle)
│   ├── models/                ← Model weight files (.pth)
│   ├── pipeline/              ← Training scripts (binary + finding classifiers)
│   ├── data/                  ← Dataset references and preprocessing utilities
│   ├── Dockerfile
│   ├── requirements.txt
│   └── README.md              ← Detailed setup guide for the web app
│
└── medgemma-risk-detector/    ← MedGemma fine-tuning pipeline
    ├── src/                   ← Training loop, LoRA configuration, evaluation
    ├── pipeline/              ← Data preprocessing for medgemma training
    ├── models/                ← Adapter weights (LoRA)
    ├── data/                  ← Silicosis radiology report dataset
    ├── Dockerfile
    ├── requirements.txt
    └── README.md              ← Detailed setup guide for MedGemma fine-tuning
```

---

## System Architecture

```
Chest X-ray Input
       │
       ▼
┌─────────────────────────────┐
│  LungMaskTransform          │  ← Suppress non-lung regions (OpenCV)
└─────────────┬───────────────┘
              │
      ┌───────┴────────┐
      ▼                ▼
┌──────────┐    ┌──────────────────┐
│ Binary   │    │ Multi-label      │
│ Classifier│   │ Finding Classifier│
│ EfficientNet-B4│ EfficientNet-B4 │
│ AUC 0.8888│   │ 8 findings       │
└──────────┘    └──────────────────┘
      │                │
      ▼                ▼
┌──────────────────────────────┐
│  GradCAM Explainability      │  ← Lung-masked attention heatmaps
└──────────────────────────────┘
              │
              ▼
┌──────────────────────────────┐
│  MedGemma 4B (LoRA fine-tuned)│ ← Structured clinical report
│  Hosted on Kaggle via ngrok  │
└──────────────────────────────┘
              │
              ▼
     Flask Web Application
      http://localhost:5000
```

---

## Key Results

| Metric | Value |
|:---|:---|
| Model | EfficientNet-B4 |
| Test AUC | **0.8888** |
| Sensitivity (Recall) | **87.9%** |
| Specificity | **77.5%** |
| Youden Threshold | 0.8423 |
| Task | Silicosis + Silicotuberculosis vs. TB + Normal |
| Training Data | SilicoData + VinDr-CXR (1,500 normal augmentation) |

---

## Radiological Findings Detected

The multi-label finding classifier identifies 8 findings from every CXR:

| Finding | Detection AUC | Threshold |
|:---|:---|:---|
| Consolidation | 0.84 | 65% |
| Cavity | 0.78 | 50% |
| Fibrosis | 0.68 | 55% |
| Multiple Nodules | 0.66 | 50% |
| Bronchiectasis | 0.65 | 65% |
| Pleural Thickening | 0.67 | 60% |
| Ground Glass Opacity | 0.55 | 70% |
| Hilum Abnormality | 0.48 | 90% |

---

## Submodule Guides

### `silicosis-risk-detector`
The main diagnostic web application. Contains the Flask backend, EfficientNet classifiers, GradCAM engine, and frontend UI.  
→ See [`silicosis-risk-detector/README.md`](./silicosis-risk-detector/README.md) for setup and running instructions.

### `medgemma-risk-detector`
The MedGemma 4B fine-tuning pipeline using LoRA adapters on silicosis radiology report data. The trained adapter is deployed as a Flask API on Kaggle and tunnelled via ngrok.  
→ See [`medgemma-risk-detector/README.md`](./medgemma-risk-detector/README.md) for training and deployment instructions.

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/Arya036/Industry_project_rises_ai.git
cd Industry_project_rises_ai

# 2. Set up the diagnostic app
cd silicosis-risk-detector
pip install -r requirements.txt
python src/app.py

# 3. Open the browser
# http://localhost:5000
```

> **MedGemma reports** require the Kaggle notebook to be running and the ngrok URL to be configured in the app settings panel.

---

## Tech Stack

| Component | Technology |
|:---|:---|
| Deep Learning | PyTorch, Torchvision, EfficientNet-B4 |
| Explainability | GradCAM (custom, lung-masked) |
| Language Model | MedGemma 4B + LoRA (via HuggingFace PEFT) |
| Quantization | BitsAndBytes (4-bit NF4) |
| Backend | Flask (Python) |
| Frontend | HTML, CSS, JavaScript |
| Deployment | Kaggle (GPU inference) + ngrok tunnel |
| Containerisation | Docker |

---

## Clinical Disclaimer

> This system is a research prototype intended to assist, not replace, qualified radiologists. All AI-generated reports must be reviewed by a licensed clinician before clinical use. The system is not CE/FDA approved for diagnostic decision-making.
