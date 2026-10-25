"""Intelligent RAG Knowledge Assistant - CLI.

    python main.py ingest data/        # build the index
    python main.py ask "your question" # one-shot question
    python main.py chat                # interactive loop
"""
import argparse

from rag import config as C


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid RAG knowledge assistant")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest", help="index a folder").add_argument("folder")
    sub.add_parser("ask", help="ask one question").add_argument("question")
    sub.add_parser("chat", help="interactive chat")
    args = parser.parse_args()

    if args.cmd == "ingest":
        from rag.ingest import ingest

        n = ingest(args.folder)
        print(f"Indexed {n} chunks using provider '{C.PROVIDER}'.")
        return

    from rag.chain import Assistant

    bot = Assistant()
    if args.cmd == "ask":
        answer, sources = bot.ask(args.question)
        print(answer)
        if sources:
            print("\nSources:", ", ".join(sources))
    else:
        print(f"Provider: {C.PROVIDER}. Type 'exit' to quit.")
        while True:
            q = input("\n> ").strip()
            if q.lower() in {"exit", "quit", ""}:
                break
            answer, sources = bot.ask(q)
            print(answer)
            if sources:
                print("Sources:", ", ".join(sources))


if __name__ == "__main__":
    main()
