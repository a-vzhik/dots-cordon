"""Export the local API contract without opening the training database."""

import json
from pathlib import Path

from dots_cordon_ml.audit.web.app import create_app

Path("openapi.json").write_text(json.dumps(create_app().openapi(), indent=2) + "\n")
