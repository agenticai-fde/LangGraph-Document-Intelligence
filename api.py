"""FastAPI service for document intelligence.

    uvicorn api:app --port 8002      then open http://127.0.0.1:8002/docs
"""
import os
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, File, HTTPException, Security, UploadFile
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from ollama import ResponseError
from pydantic import BaseModel, Field

from config import get_db
from ingest_graph import ingest_graph
from qa_graph import qa_graph

SUPPORTED = {".txt", ".md", ".pdf"}
MAX_BYTES = 5 * 1024 * 1024
SAMPLE_DIR = Path(__file__).parent / "sample_docs"

API_KEY = os.getenv("API_KEY")  # when set, every request needs header X-API-Key
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(key: Optional[str] = Security(api_key_header)):
    if API_KEY and key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key.")


app = FastAPI(title="Document Intelligence",
              description="Upload invoices, contracts and policies; get extracted fields, "
                          "consistency checks and cited answers.",
              dependencies=[Depends(require_api_key)])


@app.exception_handler(ResponseError)
def llm_unavailable(request, exc: ResponseError):
    return JSONResponse(status_code=503, content={"detail": f"LLM unavailable: {exc.error}"})


class IngestResult(BaseModel):
    document_id: int
    filename: str
    doc_type: str
    summary: str
    chunks: int
    extracted: Optional[dict] = None
    issues: list[str] = []


class AskRequest(BaseModel):
    question: str = Field(description="e.g. 'Are there any problems with the invoices?'")


class AskResponse(BaseModel):
    answer: str
    sources: list[str]


def run_ingest(path: Path) -> IngestResult:
    out = ingest_graph.invoke({"path": str(path)})
    return IngestResult(document_id=out["document_id"], filename=out["filename"],
                        doc_type=out["doc_type"], summary=out["summary"], chunks=out["chunks"],
                        extracted=out.get("extracted"), issues=out.get("issues") or [])


@app.post("/documents", response_model=IngestResult, operation_id="uploadDocument")
def upload_document(file: UploadFile = File(description=".txt, .md or .pdf, max 5 MB")):
    name = Path(file.filename or "").name
    if Path(name).suffix.lower() not in SUPPORTED:
        raise HTTPException(status_code=400, detail=f"Supported types: {', '.join(sorted(SUPPORTED))}")
    content = file.file.read(MAX_BYTES + 1)
    if len(content) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="File larger than 5 MB.")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / name  # keep the original filename: it is the document's key
        path.write_bytes(content)
        try:
            return run_ingest(path)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))


@app.post("/documents/samples", response_model=list[IngestResult], operation_id="ingestSamples")
def ingest_samples():
    return [run_ingest(p) for p in sorted(SAMPLE_DIR.iterdir()) if p.suffix.lower() in SUPPORTED]


@app.get("/documents", operation_id="listDocuments")
def list_documents():
    return (get_db().table("documents").select("id,filename,doc_type,summary,issues,created_at")
            .order("id").execute().data)


@app.get("/documents/{document_id}", operation_id="getDocument")
def get_document(document_id: int):
    rows = get_db().table("documents").select("*").eq("id", document_id).execute().data
    if not rows:
        raise HTTPException(status_code=404, detail="Document not found.")
    return rows[0]


@app.delete("/documents/{document_id}", operation_id="deleteDocument")
def delete_document(document_id: int):
    rows = get_db().table("documents").delete().eq("id", document_id).execute().data
    if not rows:
        raise HTTPException(status_code=404, detail="Document not found.")
    return {"deleted": document_id}


@app.post("/ask", response_model=AskResponse, operation_id="askDocuments")
def ask(data: AskRequest):
    out = qa_graph.invoke({"question": data.question})
    return AskResponse(answer=out["answer"], sources=out["sources"])


@app.get("/health")
def health():
    return {"status": "ok"}
