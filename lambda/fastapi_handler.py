"""AWS Lambda entrypoint for FastAPI using Mangum.

This allows deploying the same FastAPI app from src/serving/app.py to Lambda
behind API Gateway or Lambda Function URL.
"""

import os
import sys
from pathlib import Path

from mangum import Mangum


# Make project root importable when packaged into lambda/fastapi-package
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


# In Lambda package, src/ and models/ are copied into the package root
os.environ.setdefault("MODEL_DIR", "models")

from src.serving.app import app  # noqa: E402


handler = Mangum(app)
