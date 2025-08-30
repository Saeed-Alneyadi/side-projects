from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()

class TrainConfig(BaseModel):
    random_state: int = int(os.getenv("RANDOM_STATE", 42))
    test_size: float = float(os.getenv("TEST_SIZE", 0.2))
    model_dir: str = os.getenv("MODEL_DIR", "models")
    model_type: str = "rf"  # rf | xgb | logreg (extend as needed)

class AppConfig(BaseModel):
    title: str = "AI Project Demo"
    model_path: str = os.getenv("MODEL_PATH", "models/latest.joblib")