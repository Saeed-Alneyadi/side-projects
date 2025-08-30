# AI Project Template (Python)

A clean starter for NLP/CV/Tabular projects with training, inference, config, tests, and an optional Streamlit app.

## Features
- `src/` package structure (installable)
- Config via environment variables and `.env`
- Reproducible training script (`scripts/train.py`)
- Inference/serving script (`scripts/infer.py`)
- Streamlit demo app (`app/app.py`)
- Logging + experiment artifacts in `models/`
- Simple tests (`pytest`)
- Dockerfile + GitHub Actions CI

## Quickstart
```bash
# 1) Create environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install
pip install --upgrade pip
pip install -e ".[dev]"

# 3) Train a toy model
python scripts/train.py --dataset iris --model_type rf

# 4) Run inference
python scripts/infer.py --input "[5.1, 3.5, 1.4, 0.2]" --model_path models/latest.joblib

# 5) Launch demo app
streamlit run app/app.py
```

## Project Layout
```text
ai_project/
├── app/
├── data/
│   ├── processed/
│   └── raw/
├── models/
├── notebooks/
├── scripts/
├── src/ai_project/
└── tests/
```