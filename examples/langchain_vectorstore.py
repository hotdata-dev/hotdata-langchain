"""HotdataVectorStore as the retrieval backend for a LangChain retriever."""

import os

from langchain_openai import OpenAIEmbeddings

import hotdata_langchain as hl


def main() -> None:
    client = hl.from_env()

    # No tables declared here: the store declares its own, keyed on `id`, which is what
    # makes a repeat write replace a document instead of appending a copy.
    db = client.create_managed_database(description="demo_vectorstore", schema="public")
    print(f"Created managed database {db.id}")

    store = hl.HotdataVectorStore(
        client,
        OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=os.environ.get("OPENAI_EMBEDDING_KEY") or os.environ["OPENAI_API_KEY"],
        ),
        database_id=db,
        table="documents",
        metadata_columns={"city": "string"},
    )

    store.add_texts(
        [
            "A quiet studio with a walled garden and a fig tree.",
            "Bright industrial loft above a bakery.",
            "A foggy-morning cottage three blocks from the beach.",
        ],
        [{"city": "sf"}, {"city": "sf"}, {"city": "sf"}],
        ids=["garden", "loft", "cottage"],
    )

    for document, distance in store.similarity_search_with_score("somewhere calm", k=2):
        print(f"{distance:.4f}  {document.id}  {document.page_content}")

    # Anything built on LangChain's retriever interface now reads from Hotdata.
    retriever = store.as_retriever(search_kwargs={"k": 1})
    print([d.id for d in retriever.invoke("somewhere calm")])

    client.close()


if __name__ == "__main__":
    main()
