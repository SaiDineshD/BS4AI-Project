# Contributing to Multi-Modal Liveness Detection

## Collaboration Workflow

### Branching

- `main` — stable, tested code
- `develop` — integration branch for features
- `feature/<name>` — new features (e.g., `feature/audio-backbone`)
- `fix/<name>` — bug fixes

### Workflow

1. **Fork** the repository (or work in a shared org repo).
2. **Create a branch** from `develop`:
   ```bash
   git checkout develop
   git pull
   git checkout -b feature/your-feature
   ```
3. **Implement** your changes. Keep commits focused.
4. **Push** and open a **Pull Request** to `develop`.
5. **Review** — at least one approval before merge.

### Code Style

- Python: follow PEP 8; use `black` for formatting.
- Run `pytest` before pushing.
- Add docstrings for public functions.

### Data Handling

- **Never commit raw data** (videos, audio). Use `data/raw/` (gitignored).
- Commit **sampling indices** (`data/sampling/`) so n=200 subsets are reproducible.
- Document any changes to `config/data_config.yaml`.

### Areas to Contribute

| Area | Description |
|------|-------------|
| Visual backbone | FaceForensics++ loader, CNN for face liveness |
| Audio backbone | ASVspoof loader, LFCC/spectrogram pipeline |
| Fusion | Cross-modal fusion, lip–audio sync |
| Fairness | FairFace evaluation, stratified FRR |
| Preprocessing | Face detection, audio feature extraction |

### Questions

Open an **Issue** for design questions or bugs.
