"""
Create a mock detector that works without TensorFlow
This patches the PlateDetector to return dummy detections for testing
"""

import os
import sys
from pathlib import Path


def create_mock_model(model_path="./mock_model"):
    """
    Create a directory that looks like a TensorFlow SavedModel.
    We'll patch PlateDetector to recognize this and use mock detections.
    """
    model_dir = Path(model_path)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Create SavedModel directory structure
    (model_dir / "variables").mkdir(exist_ok=True)
    (model_dir / "assets").mkdir(exist_ok=True)
    
    # Create marker files
    (model_dir / "saved_model.pb").touch()
    (model_dir / "variables" / "variables.index").touch()
    
    # Create a config file for the mock
    config = {
        "type": "mock",
        "description": "Mock plate detector for testing",
        "returns_dummy_detections": True
    }
    
    import json
    with open(model_dir / "config.json", "w") as f:
        json.dump(config, f)
    
    print(f"✅ Mock model structure created at: {model_path}")
    return str(model_dir.absolute())


if __name__ == "__main__":
    model_path = create_mock_model("./mock_model")
    print(f"📍 Location: {model_path}")
    print(f"\n✅ Now you need to restart the API for it to work:")
    print(f"   $env:MODEL_PATH='{model_path}'")
    print(f"   .venv\\Scripts\\python.exe app.py")
