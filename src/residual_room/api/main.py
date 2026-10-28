from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse
from sqlalchemy import Engine, text
from sqlalchemy.exc import SQLAlchemyError

from residual_room.db.session import get_engine

app = FastAPI(title="Residual Room")


@app.get("/health")
def health(engine: Engine = Depends(get_engine)):
    # Healthy means the database answers AND pgvector is installed, since every
    # later module (ingest, retrieval, checker) needs both.
    try:
        with engine.connect() as conn:
            pgvector = conn.execute(
                text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            ).scalar_one_or_none()
    except SQLAlchemyError:
        # No exception text in the body: it can contain the database host.
        return JSONResponse(status_code=503, content={"status": "error", "database": "unreachable"})
    if pgvector is None:
        return JSONResponse(
            status_code=503, content={"status": "error", "database": "ok", "pgvector": "missing"}
        )
    return {"status": "ok", "database": "ok", "pgvector": pgvector}
