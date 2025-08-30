import typer
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

from ai_project.config import TrainConfig
from ai_project.logging_utils import setup_logger
from ai_project.data_utils import load_dataset
from ai_project.model_utils import build_model

app = typer.Typer(no_args_is_help=True)

@app.command()
def main(dataset: str = "iris", model_type: str = "rf"):
    cfg = TrainConfig(model_type=model_type)
    setup_logger(cfg.model_dir)

    X, y = load_dataset(dataset)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
    )

    model = build_model(cfg.model_type)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print(classification_report(y_test, preds))

    Path(cfg.model_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(cfg.model_dir) / "latest.joblib"
    joblib.dump(model, out_path)
    print(f"Saved model to {out_path}")

if __name__ == "__main__":
    app()