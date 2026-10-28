"""Write FastAPI's OpenAPI schema to web/openapi.json (input for the generated TS client).

    make client
"""

import json

from residual_room.api.main import app
from residual_room.corpus.manifest import REPO_ROOT

out = REPO_ROOT / "web" / "openapi.json"
out.write_text(json.dumps(app.openapi(), indent=2) + "\n", encoding="utf-8", newline="\n")
print(f"wrote {out.relative_to(REPO_ROOT)}")
