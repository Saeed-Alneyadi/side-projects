import typer, ast, joblib
import numpy as np
from ai_project.config import AppConfig

app = typer.Typer(no_args_is_help=True)

@app.command()
def main(input: str, model_path: str | None = None):
    cfg = AppConfig()
    mp = model_path or cfg.model_path
    model = joblib.load(mp)
    x = np.array(ast.literal_eval(input)).reshape(1, -1)
    pred = model.predict(x)[0]
    print(f"Prediction: {pred}")

if __name__ == "__main__":
    app()