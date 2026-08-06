"""End-to-end demo of HotdataVectorStore, from empty workspace to a RAG chain.

Stands up a managed database, writes embedded documents into it, then answers a question
with a stock LangChain retrieval chain — `as_retriever()` into a prompt into a model, with
no Hotdata-specific code in the chain itself. That is the point: implementing LangChain's
`VectorStore` interface is what makes the rest of LangChain work against Hotdata unchanged.

    uv run --group demo --env-file .env python demo/vectorstore_demo.py

Embedding needs OPENAI_EMBEDDING_KEY (or OPENAI_API_KEY). The final chain step also needs a
tool-calling model — pass one with --model or DEMO_MODEL — and is skipped without one; every
step before it runs on the embedding key alone. Set LANGSMITH_API_KEY and
LANGSMITH_TRACING=true to trace the run to LangSmith.
"""

from __future__ import annotations

import argparse
import os
from typing import Any

from langchain_core.documents import Document

import hotdata_langchain as hl

#: Display label for the database this demo creates. Not an identifier — `ensure_database`
#: addresses the database by id once it has one.
DATABASE_LABEL = "langchain_vectorstore_demo"
SCHEMA = "public"
TABLE = "documents"

EMBEDDING_MODEL = "text-embedding-3-small"
PROVIDER_KEY_VARS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "google_genai": "GOOGLE_API_KEY",
    "groq": "GROQ_API_KEY",
    "mistralai": "MISTRAL_API_KEY",
    "openai": "OPENAI_API_KEY",
}

# Ids are explicit so a re-run upserts these same rows rather than duplicating them —
# the behaviour the table's `id` key exists for.
CORPUS = [
    Document(
        id="mission-garden",
        page_content=(
            "A quiet studio off Valencia Street with a walled garden, a fig tree and a "
            "small outdoor table. No street noise despite the location."
        ),
        metadata={"neighbourhood": "Mission", "beds": 1, "outdoor": True},
    ),
    Document(
        id="mission-loft",
        page_content=(
            "Bright industrial loft above a bakery in the Mission. Twelve-foot windows, "
            "exposed brick, and the smell of bread in the morning."
        ),
        metadata={"neighbourhood": "Mission", "beds": 2, "outdoor": False},
    ),
    Document(
        id="sunset-cottage",
        page_content=(
            "A foggy-morning cottage three blocks from Ocean Beach. Wood stove, thick "
            "curtains, and a back deck for when the sun finally comes through."
        ),
        metadata={"neighbourhood": "Outer Sunset", "beds": 2, "outdoor": True},
    ),
    Document(
        id="sunset-surf",
        page_content=(
            "Surfer's room in the Outer Sunset with board storage and an outdoor shower. "
            "Steps from the N Judah line."
        ),
        metadata={"neighbourhood": "Outer Sunset", "beds": 1, "outdoor": True},
    ),
    Document(
        id="nob-hill-suite",
        page_content=(
            "Formal one-bedroom on Nob Hill with a bay window, a doorman and a view down "
            "to the cable car line. Very quiet building."
        ),
        metadata={"neighbourhood": "Nob Hill", "beds": 1, "outdoor": False},
    ),
    Document(
        id="soma-studio",
        page_content=(
            "Compact SoMa studio built for working: standing desk, fast wifi, blackout "
            "blinds. Walking distance to Caltrain."
        ),
        metadata={"neighbourhood": "SoMa", "beds": 1, "outdoor": False},
    ),
    Document(
        id="haight-flat",
        page_content=(
            "Victorian flat in the Upper Haight with original mouldings, a piano, and a "
            "shared yard full of nasturtiums."
        ),
        metadata={"neighbourhood": "Haight", "beds": 3, "outdoor": True},
    ),
    Document(
        id="north-beach-walkup",
        page_content=(
            "Fourth-floor walk-up in North Beach over a cafe. Loud, central, and the "
            "espresso downstairs opens at six."
        ),
        metadata={"neighbourhood": "North Beach", "beds": 1, "outdoor": False},
    ),
]

QUESTION = "I want somewhere quiet with outdoor space. Where should I stay, and why?"


def step(message: str) -> None:
    print(f"\n=== {message} ===")


def find_database_by_label(client: hl.HotdataClient, label: str) -> Any | None:
    """Scan the workspace for a database carrying this display label.

    Bootstrap convenience for a re-runnable demo, and the only by-label lookup here.
    Labels are not unique, so this is not how an application should find its database —
    pass `--database-id` to bind one by id instead.
    """
    for db in client.list_managed_databases():
        if db.description == label:
            return db
    return None


def ensure_database(client: hl.HotdataClient, database_id: str | None) -> Any:
    """Return the demo's managed database record, bound by id or created."""
    if database_id:
        db = hl.resolve_database_by_id(client, database_id)
        print(f"Bound managed database {db.id} by id (label={db.description!r})")
        return db

    existing = find_database_by_label(client, DATABASE_LABEL)
    if existing is not None:
        print(f"Reusing managed database {existing.id} (label={DATABASE_LABEL!r})")
        return existing

    # No `tables=` here on purpose: the store declares its own table keyed on `id`, and a
    # table declared without that key would take writes as appends.
    db = client.create_managed_database(description=DATABASE_LABEL, schema=SCHEMA)
    print(f"Created managed database {db.id} (label={DATABASE_LABEL!r})")
    print(f"  Pin it for later runs with --database-id {db.id} (or DEMO_DATABASE_ID)")
    return db


def build_embeddings() -> Any:
    from langchain_openai import OpenAIEmbeddings

    key = os.environ.get("OPENAI_EMBEDDING_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("Set OPENAI_EMBEDDING_KEY (or OPENAI_API_KEY) to run this demo")
    return OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=key)


def print_hits(scored: list[tuple[Document, float]]) -> None:
    for rank, (document, distance) in enumerate(scored, start=1):
        print(f"  {rank}. distance={distance:.4f}  {document.id}  {document.metadata}")
        print(f"     {document.page_content[:100]}…")


def run_chain(store: hl.HotdataVectorStore, *, model: str, k: int) -> None:
    """Answer QUESTION with a stock LangChain retrieval chain over this store."""
    from langchain.chat_models import init_chat_model
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough

    tracing = os.environ.get("LANGSMITH_TRACING", "").lower() in {"1", "true", "yes"}
    print(f"LangSmith tracing={tracing} (project={os.environ.get('LANGSMITH_PROJECT', 'default')})")

    retriever = store.as_retriever(search_kwargs={"k": k})
    prompt = ChatPromptTemplate.from_template(
        "Answer the question using only the listings below. Name the listings you use.\n\n"
        "{context}\n\nQuestion: {question}"
    )

    def format_documents(documents: list[Document]) -> str:
        return "\n\n".join(f"[{d.id}] {d.page_content}" for d in documents)

    chain = (
        {"context": retriever | format_documents, "question": RunnablePassthrough()}
        | prompt
        | init_chat_model(model)
        | StrOutputParser()
    )

    print(f"\nRetrieved for the chain: {[d.id for d in retriever.invoke(QUESTION)]}")
    print("\n--- answer ---")
    print(chain.invoke(QUESTION))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=3, help="how many documents to retrieve")
    parser.add_argument(
        "--model",
        default=os.environ.get("DEMO_MODEL"),
        help="chat model for the final chain, e.g. '<provider>:<model>' (or set DEMO_MODEL); "
        "the chain step is skipped without one",
    )
    parser.add_argument(
        "--database-id",
        default=os.environ.get("DEMO_DATABASE_ID"),
        help="bind an existing managed database by id (or set DEMO_DATABASE_ID); "
        "without one the demo reuses or creates its own and prints the id to pin",
    )
    parser.add_argument("--skip-chain", action="store_true", help="stop after direct searches")
    parser.add_argument(
        "--cleanup", action="store_true", help="delete the demo managed database and exit"
    )
    args = parser.parse_args()

    client = hl.from_env()
    print(f"Connected to {client.host} (workspace={client.workspace_id})")

    if args.cleanup:
        target = (
            hl.resolve_database_by_id(client, args.database_id)
            if args.database_id
            else find_database_by_label(client, DATABASE_LABEL)
        )
        if target is None:
            print(f"No managed database labelled {DATABASE_LABEL!r} to delete")
        else:
            client.delete_managed_database(target)
            print(f"Deleted managed database {target.id} (label={target.description!r})")
        client.close()
        return

    try:
        step("1. Managed database")
        db = ensure_database(client, args.database_id)

        step("2. Vector store")
        store = hl.HotdataVectorStore(
            client,
            build_embeddings(),
            # The resolved record, so the store does not re-look-up what step 1 has.
            database_id=db,
            table=TABLE,
            schema=SCHEMA,
            # Declared so they can be filtered on; all metadata round-trips regardless.
            metadata_columns={"neighbourhood": "string", "beds": "int", "outdoor": "bool"},
        )
        print(f"Store over {store.table_ref} in {store.database.id}")

        step("3. Embed and write")
        written = store.add_documents(CORPUS)
        print(f"Wrote {len(written)} documents; ids are explicit, so a re-run upserts them")

        step("4. Similarity search")
        print(f"Query: {QUESTION!r}")
        print_hits(store.similarity_search_with_score(QUESTION, k=args.k))

        step("5. The same search, filtered")
        print("Filter: outdoor=True, beds=1 — a predicate inside the ranking query")
        print_hits(
            store.similarity_search_with_score(
                QUESTION, k=args.k, filter={"outdoor": True, "beds": 1}
            )
        )

        done = "Everything above ran against the real engine, with no vector index yet."
        if args.skip_chain:
            print("\n--skip-chain set; stopping before the retrieval chain")
            print(done)
            return
        if not args.model:
            print("\nNo model given — skipping the retrieval chain.")
            print("Pass --model '<provider>:<model>' (or set DEMO_MODEL) to run it.")
            print(done)
            return
        provider_key_var = PROVIDER_KEY_VARS.get(args.model.split(":", 1)[0])
        if provider_key_var and not os.environ.get(provider_key_var):
            print(f"\n{provider_key_var} is not set — skipping the retrieval chain.")
            print(done)
            return

        step("6. LangChain retrieval chain over the store")
        run_chain(store, model=args.model, k=args.k)
    finally:
        client.close()


if __name__ == "__main__":
    main()
