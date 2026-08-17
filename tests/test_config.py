from src.config import Settings


def test_settings_loads_with_only_defaults(monkeypatch):
    for var in ("USE_ONNX", "MODEL_PATH", "ONNX_MODEL_PATH", "DATASET_PATH"):
        monkeypatch.delenv(var, raising=False)

    settings = Settings(_env_file=None)

    assert settings.USE_ONNX is False
    assert settings.MODEL_PATH == "models/classifier.joblib"
    assert settings.LABEL_COLUMN == "especialidade"
