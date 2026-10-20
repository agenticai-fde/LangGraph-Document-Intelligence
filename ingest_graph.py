"""Document ingestion graph.

load -> classify -> extract -> validate -> store
                 +-> (doc_type "other") ---^
"""
import re
from pathlib import Path
from typing import Literal, Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from config import get_db, get_llm

MAX_LLM_CHARS = 6000


# ---------- Structured outputs ----------

class Classification(BaseModel):
    doc_type: Literal["invoice", "contract", "policy", "other"]
    summary: str = Field(description="One or two sentence summary of the document")


class LineItem(BaseModel):
    description: str
    quantity: float
    unit_price: float
    amount: float


class InvoiceFields(BaseModel):
    invoice_number: Optional[str] = None
    vendor: Optional[str] = None
    bill_to: Optional[str] = None
    invoice_date: Optional[str] = Field(None, description="YYYY-MM-DD")
    due_date: Optional[str] = Field(None, description="YYYY-MM-DD")
    currency: Optional[str] = None
    line_items: list[LineItem] = Field(default_factory=list)
    subtotal: Optional[float] = None
    tax: Optional[float] = None
    total: Optional[float] = None


class ContractFields(BaseModel):
    title: Optional[str] = None
    parties: list[str] = Field(default_factory=list)
    effective_date: Optional[str] = Field(None, description="YYYY-MM-DD")
    end_date: Optional[str] = Field(None, description="YYYY-MM-DD")
    monthly_fee: Optional[float] = None
    payment_terms: Optional[str] = None
    termination_notice_days: Optional[int] = None
    auto_renewal: Optional[bool] = None
    governing_law: Optional[str] = None


class PolicyFields(BaseModel):
    policy_name: Optional[str] = None
    effective_date: Optional[str] = Field(None, description="YYYY-MM-DD")
    key_rules: list[str] = Field(default_factory=list, description="Up to 6 short rules")


SCHEMAS = {"invoice": InvoiceFields, "contract": ContractFields, "policy": PolicyFields}


class IngestState(TypedDict, total=False):
    path: str
    filename: str
    text: str
    doc_type: str
    summary: str
    extracted: dict
    issues: list[str]
    document_id: int
    chunks: int


# ---------- Helpers ----------

def read_document(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        return "\n\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    return path.read_text(encoding="utf-8")


def chunk_text(text: str, size: int = 800, overlap: int = 150) -> list[str]:
    """Paragraph-aware chunks of about `size` characters with a small overlap."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, current = [], ""
    for para in paragraphs:
        if current and len(current) + len(para) + 2 > size:
            chunks.append(current)
            current = current[-overlap:] + "\n\n" + para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current:
        chunks.append(current)

    result = []
    for chunk in chunks:  # split any single paragraph that is far too long
        while len(chunk) > size * 1.5:
            result.append(chunk[:size])
            chunk = chunk[size - overlap:]
        result.append(chunk)
    return result


def close(a: float, b: float, tolerance: float = 1.0) -> bool:
    return abs(a - b) <= tolerance


# ---------- Nodes ----------

def load(state: IngestState) -> dict:
    path = Path(state["path"])
    text = read_document(path).strip()
    if not text:
        raise ValueError(f"No text found in {path.name} (scanned PDFs need OCR first).")
    return {"filename": path.name, "text": text}


def classify(state: IngestState) -> dict:
    result = get_llm().with_structured_output(Classification).invoke(
        "Classify this business document as invoice, contract, policy or other, "
        f"and summarise it.\n\nDOCUMENT:\n{state['text'][:MAX_LLM_CHARS]}"
    )
    return {"doc_type": result.doc_type, "summary": result.summary}


def extract(state: IngestState) -> dict:
    schema = SCHEMAS[state["doc_type"]]
    result = get_llm().with_structured_output(schema).invoke(
        f"Extract the fields of this {state['doc_type']} exactly as written. "
        "Use null for anything not stated. Dates as YYYY-MM-DD, numbers without "
        f"currency symbols or commas.\n\nDOCUMENT:\n{state['text'][:MAX_LLM_CHARS]}"
    )
    return {"extracted": result.model_dump()}


def validate(state: IngestState) -> dict:
    """Deterministic consistency checks on the extracted fields (no LLM)."""
    data, issues = state.get("extracted") or {}, []
    doc_type = state["doc_type"]

    if doc_type == "invoice":
        items = data.get("line_items") or []
        for item in items:
            if not close(item["quantity"] * item["unit_price"], item["amount"]):
                issues.append(f"Line '{item['description']}': {item['quantity']} x "
                              f"{item['unit_price']} != {item['amount']}")
        if items and data.get("subtotal") is not None:
            line_sum = sum(i["amount"] for i in items)
            if not close(line_sum, data["subtotal"]):
                issues.append(f"Line items add up to {line_sum:,.2f} but subtotal is {data['subtotal']:,.2f}")
        if None not in (data.get("subtotal"), data.get("tax"), data.get("total")):
            if not close(data["subtotal"] + data["tax"], data["total"]):
                issues.append(f"Subtotal + tax = {data['subtotal'] + data['tax']:,.2f} "
                              f"but total is {data['total']:,.2f}")
        if not data.get("invoice_number"):
            issues.append("Missing invoice number")
        if data.get("invoice_date") and data.get("due_date") and data["due_date"] < data["invoice_date"]:
            issues.append("Due date is before the invoice date")

    elif doc_type == "contract":
        if len(data.get("parties") or []) < 2:
            issues.append("Fewer than two parties identified")
        if data.get("effective_date") and data.get("end_date") and data["end_date"] < data["effective_date"]:
            issues.append("End date is before the effective date")
        if data.get("auto_renewal") and not data.get("termination_notice_days"):
            issues.append("Auto-renews but no termination notice period found")

    elif doc_type == "policy":
        if not data.get("key_rules"):
            issues.append("No rules could be extracted")

    return {"issues": issues}


def store(state: IngestState) -> dict:
    db = get_db()
    doc = db.table("documents").upsert({
        "filename": state["filename"],
        "doc_type": state["doc_type"],
        "summary": state["summary"],
        "extracted": state.get("extracted"),
        "issues": state.get("issues", []),
        "char_count": len(state["text"]),
    }, on_conflict="filename").execute().data[0]

    db.table("document_chunks").delete().eq("document_id", doc["id"]).execute()
    chunks = chunk_text(state["text"])
    db.table("document_chunks").insert([
        {"document_id": doc["id"], "chunk_index": i, "content": c} for i, c in enumerate(chunks)
    ]).execute()
    return {"document_id": doc["id"], "chunks": len(chunks)}


builder = StateGraph(IngestState)
for name, fn in [("load", load), ("classify", classify), ("extract", extract),
                 ("validate", validate), ("store", store)]:
    builder.add_node(name, fn)

builder.add_edge(START, "load")
builder.add_edge("load", "classify")
builder.add_conditional_edges("classify",
                              lambda s: "extract" if s["doc_type"] in SCHEMAS else "store",
                              ["extract", "store"])
builder.add_edge("extract", "validate")
builder.add_edge("validate", "store")
builder.add_edge("store", END)

ingest_graph = builder.compile()
