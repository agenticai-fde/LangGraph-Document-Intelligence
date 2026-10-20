"""Document intelligence CLI.

    python main.py ingest sample_docs            ingest a folder (or individual files)
    python main.py list                          list ingested documents
    python main.py show invoice_techsupply.txt   show extracted fields and issues
    python main.py ask "When does the CloudHost agreement end?"
    python main.py chat                          ask questions interactively
"""
import argparse
import json
from pathlib import Path

from config import get_db
from ingest_graph import ingest_graph
from qa_graph import qa_graph

SUPPORTED = {".txt", ".md", ".pdf"}


def ingest(paths: list[str]) -> None:
    files = []
    for p in map(Path, paths):
        files += sorted(f for f in p.iterdir() if f.suffix.lower() in SUPPORTED) if p.is_dir() else [p]
    for f in files:
        print(f"Ingesting {f.name} ...", flush=True)
        try:
            out = ingest_graph.invoke({"path": str(f)})
        except Exception as e:
            print(f"  failed: {e}")
            continue
        print(f"  type={out['doc_type']}  chunks={out['chunks']}  id={out['document_id']}")
        for issue in out.get("issues") or []:
            print(f"  ! {issue}")


def list_documents() -> None:
    rows = (get_db().table("documents").select("id,filename,doc_type,issues,summary")
            .order("id").execute().data)
    if not rows:
        print("No documents yet.")
    for d in rows:
        flag = f"  [{len(d['issues'])} issue(s)]" if d["issues"] else ""
        print(f"{d['id']:>3}  {d['filename']:<40} {d['doc_type'] or '-':<9}{flag}\n     {d['summary']}")


def show(name: str) -> None:
    query = get_db().table("documents").select("*")
    query = query.eq("id", int(name)) if name.isdigit() else query.eq("filename", name)
    rows = query.execute().data
    if not rows:
        print(f"No document '{name}'.")
        return
    d = rows[0]
    print(f"{d['filename']} ({d['doc_type']})\n{d['summary']}\n")
    print(json.dumps(d["extracted"], indent=2, ensure_ascii=False))
    print("\nIssues:", *(d["issues"] or ["none"]), sep="\n  - ")


def ask(question: str) -> None:
    out = qa_graph.invoke({"question": question})
    print(f"\n{out['answer']}")
    if out["sources"]:
        print("\nSources: " + ", ".join(out["sources"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Document intelligence with LangGraph")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ingest").add_argument("paths", nargs="+")
    sub.add_parser("list")
    sub.add_parser("show").add_argument("document", help="id or filename")
    sub.add_parser("ask").add_argument("question")
    sub.add_parser("chat")
    args = parser.parse_args()

    if args.command == "ingest":
        ingest(args.paths)
    elif args.command == "list":
        list_documents()
    elif args.command == "show":
        show(args.document)
    elif args.command == "ask":
        ask(args.question)
    else:
        print("Ask about your documents. Blank line to quit.")
        while question := input("\nQ: ").strip():
            ask(question)


if __name__ == "__main__":
    main()
