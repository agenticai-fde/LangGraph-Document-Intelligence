"""Question-answering graph over ingested documents.

retrieve -> answer        (with citations, extracted fields and validation issues)
         -> not_found
"""
import json
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from config import get_db, get_llm

TOP_K = 4

ANSWER_PROMPT = """Answer the question using ONLY the document material below.
Cite sources inline like [filename #chunk]. Use the extracted fields and validation
issues for questions about amounts, dates, totals or inconsistencies.
If the material does not contain the answer, say so plainly.

DOCUMENT CATALOG (all ingested documents):
{catalog}

RELEVANT PASSAGES:
{passages}

QUESTION: {question}"""


class QAState(TypedDict, total=False):
    question: str
    passages: list[dict]
    catalog: list[dict]
    answer: str
    sources: list[str]


def retrieve(state: QAState) -> dict:
    db = get_db()
    passages = db.rpc("search_document_chunks", {"q": state["question"], "k": TOP_K}).execute().data
    catalog = (db.table("documents").select("filename,doc_type,summary,extracted,issues")
               .order("created_at", desc=True).limit(20).execute().data)
    return {"passages": passages, "catalog": catalog}


def answer(state: QAState) -> dict:
    catalog = "\n".join(
        f"- {d['filename']} ({d['doc_type']}): {d['summary']}\n"
        f"  fields: {json.dumps(d['extracted'], ensure_ascii=False)}\n"
        f"  issues: {d['issues'] or 'none'}"
        for d in state["catalog"]
    )
    passages = "\n\n".join(f"[{p['filename']} #{p['chunk_index']}]\n{p['content']}"
                           for p in state["passages"]) or "(no passage matched the question)"
    response = get_llm().invoke(ANSWER_PROMPT.format(
        catalog=catalog, passages=passages, question=state["question"]))
    sources = sorted({f"{p['filename']} #{p['chunk_index']}" for p in state["passages"]})
    return {"answer": response.content, "sources": sources}


def not_found(state: QAState) -> dict:
    return {"answer": "No documents have been ingested yet. Run: python main.py ingest sample_docs",
            "sources": []}


builder = StateGraph(QAState)
builder.add_node("retrieve", retrieve)
builder.add_node("answer", answer)
builder.add_node("not_found", not_found)
builder.add_edge(START, "retrieve")
builder.add_conditional_edges("retrieve",
                              lambda s: "answer" if s["catalog"] else "not_found",
                              ["answer", "not_found"])
builder.add_edge("answer", END)
builder.add_edge("not_found", END)

qa_graph = builder.compile()
