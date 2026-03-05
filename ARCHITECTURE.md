# Multi-Modal Liveness Detection System — Architecture

## 1. Project Overview

**Objective:** Develop a multi-modal liveness detection system that combines visual (facial) and audio information to distinguish live users from presentation attacks—including replay attacks and AI-generated deepfakes.

**Hypothesis:** Joint analysis of facial movements and speech characteristics will outperform single-modality approaches and reduce demographic bias.

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        MULTI-MODAL LIVENESS DETECTION                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────────────────────────┐ │
│  │   VISUAL     │     │    AUDIO     │     │      FUSION & DECISION             │ │
│  │   STREAM     │     │   STREAM     │     │                                    │ │
│  │              │     │              │     │  ┌─────────────┐  ┌─────────────┐  │ │
│  │ FF-c23 +     │     │ ASVspoof     │     │  │ Cross-Modal │  │ Liveness    │  │ │
│  │ Deepfake     │────▶│ 2019 (Kaggle)│────▶│  │ Consistency │──▶│ Score (0–1) │  │ │
│  │ Detection    │     │              │     │  │ Checker     │  │             │  │ │
│  │ (Kaggle)     │     │ • LFCC/STFT  │     │  └─────────────┘  └─────────────┘  │ │
│  │ • Face crops │     │              │     │         │                  │       │ │
│  └──────────────┘     └──────────────┘     └──────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Data Architecture (200 per Dataset)

All datasets are sourced from **Kaggle**. We use **200 samples from each**.

### 3.1 Visual Stream (Kaggle)

| Dataset | Kaggle Link | Subset | Purpose |
|---------|-------------|--------|---------|
| **FF-c23** (FaceForensics++ C23) | [kaggle.com/datasets/xdxd003/ff-c23](https://www.kaggle.com/datasets/xdxd003/ff-c23) | 200 | Manipulated facial videos (Deepfakes, Face2Face, etc.) |
| **Deepfake Detection Challenge** | [kaggle.com/competitions/deepfake-detection-challenge/data](https://www.kaggle.com/competitions/deepfake-detection-challenge/data) | 200 | Real vs. fake videos (multi-algorithm) |

**Preprocessing:** Extract face crops; C23 for FF-c23; standard video decode for DFDC.

### 3.2 Audio Stream (Kaggle)

| Dataset | Kaggle Link | Subset | Purpose |
|---------|-------------|--------|---------|
| **ASVspoof 2019** | [kaggle.com/datasets/awsaf49/asvpoof-2019-dataset](https://www.kaggle.com/datasets/awsaf49/asvpoof-2019-dataset) | 200 | Bona fide vs. spoofed speech (TTS/VC) |

**Preprocessing:** LFCC, STFT spectrograms, or raw waveform features.

---

## 4. Component Design

### 4.1 Visual Module

- **Input:** Face crops (e.g., 224×224) or short video clips
- **Backbone:** ResNet/EfficientNet or lightweight CNN
- **Output:** Visual liveness score + embeddings for fusion

### 4.2 Audio Module

- **Input:** Spectrograms (LFCC, mel-spectrogram) or raw waveform
- **Backbone:** ResNet, LCNN, or AASIST-style architecture
- **Output:** Audio liveness score + embeddings for fusion

### 4.3 Fusion Module

- **Early fusion:** Concatenate visual + audio features before classifier
- **Late fusion:** Combine visual and audio scores (e.g., weighted average)
- **Cross-modal:** Lip–audio sync check (when both modalities available)

---

## 5. Directory Structure

```
BS4AI/
├── config/
│   ├── data_config.yaml      # n=200 sampling, paths, splits
│   ├── model_config.yaml     # Architecture hyperparameters
│   └── experiment_config.yaml
├── data/
│   ├── raw/                  # Downloaded datasets (gitignored)
│   │   ├── ff_c23/
│   │   ├── deepfake_detection/
│   │   └── asvspoof2019/
│   ├── processed/            # Preprocessed subsets
│   └── sampling/             # Subset indices (n=200)
├── src/
│   ├── data/
│   │   ├── ff_c23_loader.py
│   │   ├── deepfake_loader.py
│   │   ├── asvspoof_loader.py
│   │   └── sampling.py       # n=200 stratified sampling
│   ├── models/
│   │   ├── visual_backbone.py
│   │   ├── audio_backbone.py
│   │   ├── fusion.py
│   │   └── liveness_detector.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── fairness_eval.py
│   └── training/
│       ├── train_visual.py
│       ├── train_audio.py
│       └── train_fusion.py
├── scripts/
│   ├── download_data.sh
│   ├── preprocess.py
│   └── run_experiment.py
├── notebooks/                 # Exploration & visualization
├── tests/
├── ARCHITECTURE.md
├── README.md
├── requirements.txt
├── CONTRIBUTING.md
└── .gitignore
```

---

## 6. Data Flow

1. **Download** → Raw data in `data/raw/`
2. **Sample** → Select 200 from each dataset via `sampling.py`
3. **Preprocess** → Face crops, spectrograms → `data/processed/`
4. **Train** → Visual and audio modules (optionally joint)
5. **Fuse** → Train fusion on combined features
6. **Evaluate** → Overall metrics (EER, t-DCF)

---

## 7. Key Design Decisions

| Decision              | Rationale                                                |
|-----------------------|----------------------------------------------------------|
| 200 per dataset       | Manageable size for collaboration and reproducibility   |
| Kaggle sources        | No manual access forms; `kaggle` CLI for download         |
| C23 compression       | Matches FF-c23 standard; balances quality/size           |
| LFCC for audio        | Common in ASVspoof; good for spoof detection             |

---

## 8. Data Sources (Kaggle)

| Dataset | URL |
|---------|-----|
| FF-c23 | https://www.kaggle.com/datasets/xdxd003/ff-c23 |
| Deepfake Detection Challenge | https://www.kaggle.com/competitions/deepfake-detection-challenge/data |
| ASVspoof 2019 | https://www.kaggle.com/datasets/awsaf49/asvpoof-2019-dataset |

## 9. References

- Rössler et al. (2019). FaceForensics++: Learning to Detect Manipulated Facial Images. ICCV.
- Todisco et al. (2019). ASVspoof 2019: Future Horizons in Spoofed and Fake Audio Detection. Interspeech.
- DFDC: Meta/Facebook Deepfake Detection Challenge.
