from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import Engine, text
from sqlalchemy.exc import SQLAlchemyError

from residual_room.api.campaigns import router as campaigns_router
from residual_room.api.generation import router as generation_router
from residual_room.api.pieces import router as pieces_router
from residual_room.corpus.manifest import REPO_ROOT
from residual_room.db.session import get_engine
from residual_room.workflow.service import WorkflowError

WEB_DIST = REPO_ROOT / "web" / "dist"

app = FastAPI(title="Residual Room")
app.include_router(pieces_router)
app.include_router(generation_router)
app.include_router(campaigns_router)


@app.exception_handler(WorkflowError)
def workflow_error(request: Request, exc: WorkflowError) -> JSONResponse:
    # 409 for refused moves (wrong state, unresolved blocks), 404 missing, 400 bad input.
    return JSONResponse(status_code=exc.status, content={"detail": str(exc)})


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


# The built review screen (`make web`), served last so /api and /health win.
if WEB_DIST.is_dir():
    app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")
