# Multi-Modal Liveness Detection

A multi-modal system that combines **visual** (facial) and **audio** information to detect presentation attacks—including replay attacks and AI-generated deepfakes—while reducing demographic bias.

## Motivation

- **Single-modality limitation:** Most liveness systems rely only on visual cues, making them vulnerable to sophisticated spoofing.
- **Cross-modal consistency:** Joint analysis of facial movements and speech can reveal mismatches (e.g., lip–audio desync) common in deepfakes.

## Datasets (200 per Dataset) — Kaggle

We use **200 samples from each dataset**, all from Kaggle:

| Dataset | Kaggle | Use | Subset |
|---------|--------|-----|--------|
| **FF-c23** | [xdxd003/ff-c23](https://www.kaggle.com/datasets/xdxd003/ff-c23) | Visual | 200 |
| **Deepfake Detection Challenge** | [competition](https://www.kaggle.com/competitions/deepfake-detection-challenge/data) | Visual | 200 |
| **ASVspoof 2019** | [awsaf49/asvpoof-2019-dataset](https://www.kaggle.com/datasets/awsaf49/asvpoof-2019-dataset) | Audio | 200 |

See [ARCHITECTURE.md](ARCHITECTURE.md) for full design.

## Quick Start

```bash
# Clone
git clone https://github.com/YOUR_ORG/BS4AI.git
cd BS4AI

# Setup
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# Kaggle API (create token at kaggle.com/account)
pip install kaggle
# Place kaggle.json in ~/.kaggle/

# Download data
python scripts/download_data.py   # or bash scripts/download_data.sh

# Preprocess and run
python scripts/preprocess.py
python scripts/run_experiment.py
```

## Project Structure

```
BS4AI/
├── config/           # data_config.yaml (n=200), model_config.yaml
├── data/             # raw (gitignored), processed, sampling
├── src/
│   ├── data/         # Loaders + sampling.py
│   ├── models/       # Visual, audio, fusion
│   └── evaluation/   # Metrics, fairness_eval.py
├── scripts/
├── notebooks/
└── tests/
```

## References

- Rössler et al. (2019). FaceForensics++: Learning to Detect Manipulated Facial Images. *ICCV*.
- Todisco et al. (2019). ASVspoof 2019: Future Horizons in Spoofed and Fake Audio Detection. *Interspeech*.
- DFDC: Meta Deepfake Detection Challenge.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for collaboration guidelines.

## License

