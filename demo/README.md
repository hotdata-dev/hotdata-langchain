# Demos

Two runnable end-to-end demos against a real Hotdata workspace.

| Demo | Shows | Script |
|---|---|---|
| [BM25 search](#1-bm25-search) | an agent choosing between full-text search and SQL | `bm25_search_demo.py` |
| [Vector store](#2-vector-store) | `HotdataVectorStore` behind a stock LangChain retrieval chain | `vectorstore_demo.py` |

## 1. BM25 search

Takes a Hotdata workspace from empty to a LangChain agent that full-text searches real
data, in one script. Uses the public San Francisco Airbnb listings fixture (7,535 rows)
and builds a BM25 index over the free-text `description` column.

### Run it

Steps 1–5 need only a Hotdata key and exercise the tool directly, no model involved:

```bash
uv run --group demo --env-file .env python demo/bm25_search_demo.py
```

The agent step (6) runs when you name a tool-calling model. Any provider works — install
its LangChain integration, set its key, and pass `<provider>:<model>`:

```bash
uv run --group demo --env-file .env python demo/bm25_search_demo.py \
    --model '<provider>:<model>'
```

The script is safe to re-run: it reuses the instant database, skips the load when the
table already has rows, and reuses an existing index.

```bash
# bind an existing instant database by id instead of letting the demo find its own
uv run --group demo --env-file .env python demo/bm25_search_demo.py \
    --database-id dbid...

# different search text, more hits
uv run --group demo --env-file .env python demo/bm25_search_demo.py \
    --query "quiet garden studio near the park" --k 10

# stop before the agent step even with a model set
uv run --group demo --env-file .env python demo/bm25_search_demo.py --skip-agent

# tear down the instant database it created
uv run --group demo --env-file .env python demo/bm25_search_demo.py --cleanup

# trace the run into a named LangSmith project
LANGSMITH_TRACING=true LANGSMITH_PROJECT=hotdata-langchain-bm25 \
    uv run --group demo --env-file .env python demo/bm25_search_demo.py
```

### Credentials

| Variable | Needed for | Notes |
|---|---|---|
| `HOTDATA_API_KEY` | steps 1–5 | everything except the agent run |
| `HOTDATA_WORKSPACE` | optional | pins a workspace; first available otherwise |
| `DEMO_DATABASE_ID` | optional | pins the instant database by id (same as `--database-id`) |
| your model provider's key | step 6 | skipped without it, or without `--model` |
| `LANGSMITH_API_KEY` + `LANGSMITH_TRACING=true` | optional | traces the run to LangSmith |
| `LANGSMITH_PROJECT` | optional | project the traces land in (default: `default`) |

BM25 needs no embedding provider — just a string column — so there is no extra
credential for the index itself, unlike a vector index. LangSmith is observability
only; the agent still needs a model provider key of its own.

With tracing on, every tool call becomes a run in the LangSmith project — the search
tool shows up as a `tool` run named `hotdata_search_text` carrying its query and the
ranked JSON it returned, so the generated SQL and the agent's choice between search
and SQL are both inspectable after the fact.

### What each step does

1. **Instant database** — binds the database given by `--database-id`, or else creates one
   labelled `langchain_bm25_demo` with `public.listings` declared up front, so the load
   materialises into it directly. Everything downstream addresses the resolved record, so
   the label never selects the target; the create path prints the new id to pin. Finding a
   previous run's database by its label is the one by-label lookup here, and it exists only
   so the demo is re-runnable without pinning — pass `--database-id` to skip it.
2. **Listings data** — downloads the fixture parquet (cached in the system temp dir) and
   loads it into the managed table.
3. **BM25 index** — creates a `bm25` index on `description` through `IndexesApi` and
   polls until it reports ready. This step is a hard prerequisite: `bm25_search` has no
   brute-force fallback and errors outright when no index exists.
4. **Tools** — reads the real table schema and narrows the projection to a handful of
   useful columns. The listings table is 85 columns wide; returning all of them would
   flood the agent's context.
5. **Direct tool invocation** — prints the generated SQL, then the ranked hits with
   their BM25 scores. Proves the tool against the live engine without an LLM in the loop.
6. **Agent run** — gives a LangChain agent both `hotdata_search_text` and
   `hotdata_execute_sql` and asks a question neither answers alone: find listings whose
   descriptions mention a quiet garden studio, then report listing counts and average
   review scores across *all* listings in those neighbourhoods. The aggregate spans the
   whole dataset, not the handful search returned, so the agent has to use search to
   identify the neighbourhoods and SQL to aggregate over them. The printed tool calls
   show which pathway it picked for which part.

   Set the model with `--model` or `DEMO_MODEL`; any tool-calling model works, and the
   step is skipped when its provider key is absent.

   **What this step demonstrates is the routing, and only the routing.** Across five runs
   the agent called the search tool and the schema tool every time, in either order. That
   part is what the tools are responsible for, and it holds.

   The prose answer is a different matter: the final table matched the true
   whole-neighbourhood figures in three of five runs. The two failures were the model
   aggregating over the handful of listings search returned instead of over every listing
   in those neighbourhoods, and a join that inflated the counts. Five runs on the merged
   `main`, measured the same way against the same ground-truth query, also gave three of
   five — so this is the model's ceiling on a compound question, not something these tools
   introduced or can fix. Read the printed tool calls, not the table.

### What makes the agent run work

**The tool descriptions, not the system prompt.** The demo's system prompt says only who
the agent is — it deliberately says nothing about which tool to use or how the engine
behaves. That guidance lives in the tool descriptions, so any application gets it without
having to teach the model about the query engine.

It matters. An earlier version of this demo had a one-line SQL tool description and a
system prompt spelling out the rules; the model still reached for Postgres full-text
idioms (`to_tsvector`/`plainto_tsquery`), which the engine rejects with
`Invalid function 'to_tsvector'`. With the constraint stated in the SQL tool's own
description instead, the model uses the search tool and feeds its results into SQL —
even with no system-prompt guidance at all.

**Tool errors have to reach the model.** The tools raise on failure, and an exception out
of a tool aborts the whole graph — so the demo wraps them with `hl.with_error_feedback`,
which returns the error as a message instead. That also digs the engine's real message out
of the exception chain: the framework raises `RuntimeError("Bad Request")` while the useful
text sits in the underlying API response body. Descriptions lower the failure rate;
readable errors are what let the model recover from what slips through.

The wrapping used to live in this file. It moved into the package once a second consumer
needed it, and gained the fix the local copy was missing: it wraps the async callable too,
which is the one LangChain actually calls under `langgraph dev` or a deployed Agent Server.
`make_hotdata_tools(..., handle_errors=True)` is the same thing without the extra call.

### Why the generated SQL looks the way it does

Step 5 prints the query. Two details in it are load-bearing:

- **`ORDER BY score DESC`** — the engine returns hits in rowid order, not ranked, so
  ranking has to be asked for explicitly.
- **`k` appearing twice** — once as `bm25_search(...)`'s fourth argument and once as a
  trailing `LIMIT`. The fourth argument is what actually bounds the search, because
  `ORDER BY` blocks limit pushdown; relying on the trailing `LIMIT` alone would let the
  scan fall back to the engine's much larger default bound.

## 2. Vector store

Takes a Hotdata workspace from empty to a LangChain retrieval chain answering a question
from documents it retrieved out of Hotdata. The chain itself is stock LangChain —
`as_retriever()` into a prompt into a model — with no Hotdata-specific code in it, which is
the whole point of implementing the `VectorStore` interface.

The corpus is ten short listing descriptions written into the script, so the demo needs no
fixture download and costs a handful of embedding tokens to run. It is built so the demo
question has three genuinely different good answers — a walled garden, a private deck, a
shared yard — and so that the first of those is written three times over. Similarity search
spends its whole top-3 on the triplet; that is what gives the MMR step something to show.

### Run it

Every step up to the retrieval chain needs a Hotdata key and an embedding key, and involves
no chat model:

```bash
uv run --group demo --env-file .env python demo/vectorstore_demo.py
```

The retrieval chain runs when you name a chat model:

```bash
uv run --group demo --env-file .env python demo/vectorstore_demo.py \
    --model '<provider>:<model>'
```

Safe to re-run: documents carry explicit ids and the table is keyed on `id`, so a second run
upserts the same ten rows rather than duplicating them.

```bash
# bind an existing instant database by id
uv run --group demo --env-file .env python demo/vectorstore_demo.py --database-id dbid...

# retrieve more documents
uv run --group demo --env-file .env python demo/vectorstore_demo.py --k 5

# sweep the MMR balance (1.0 is pure relevance, 0.0 pure variety; the demo defaults to 0.7)
uv run --group demo --env-file .env python demo/vectorstore_demo.py --lambda-mult 1.0

# build the vector index, and print the query plan either side of building it
uv run --group demo --env-file .env python demo/vectorstore_demo.py --create-index

# write into a different managed table, leaving the default one untouched
uv run --group demo --env-file .env python demo/vectorstore_demo.py --table documents_v2

# stop before the chain step even with a model set
uv run --group demo --env-file .env python demo/vectorstore_demo.py --skip-chain

# tear down the instant database it created
uv run --group demo --env-file .env python demo/vectorstore_demo.py --cleanup
```

### Credentials

| Variable | Needed for | Notes |
|---|---|---|
| `HOTDATA_API_KEY` | every step | |
| `OPENAI_EMBEDDING_KEY` | the write onwards | falls back to `OPENAI_API_KEY`; must be embeddings-scoped |
| your model provider's key | the retrieval chain | skipped without it, or without `--model` |
| `DEMO_DATABASE_ID` | optional | pins the instant database by id |

### What each step does

1. **Instant database** — binds `--database-id`, or creates one labelled
   `langchain_vectorstore_demo`. It deliberately declares **no** tables: the store declares
   its own table keyed on `id`, and a table declared without that key would take writes as
   appends, so a re-run would duplicate every document instead of replacing it.
2. **Vector store** — constructs `HotdataVectorStore` over `public.documents`, promoting
   `neighbourhood`/`beds`/`outdoor` to real typed columns so they can be filtered on. The
   resolved database record from step 1 is passed straight in, so no id is looked up twice.
3. **Embed and write** — embeds the ten documents and upserts them in one parquet load.
4. **Vector index** *(only with `--create-index`)* — `EXPLAIN`s the store's own search query,
   builds the index with `store.create_index()`, then `EXPLAIN`s it again. The plan goes from a
   full scan to a `USearchExec` lookup with nothing in the code between the two changing, which
   is the whole argument for the query shape the store emits. Re-running is a no-op: a matching
   index is left alone.
5. **Similarity search** — prints hits with their raw cosine distances. Without
   `--create-index` this runs with **no vector index**: the scalar `cosine_distance` UDF
   brute-forces the table, which is correct from row one. See
   [`docs/engine-contract.md`](../docs/engine-contract.md) for the observed plans.
6. **Diversified search (MMR)** — the same query through
   `max_marginal_relevance_search`, printed next to the nearest-first ids from step 5 so the
   two rankings can be read against each other. Nearest-first spends its top `k` on the
   near-identical garden studios; MMR keeps one and gives the other slots to listings that
   answer the question differently. This is also the one search that reads the stored vectors,
   so its candidate fetch is a full scan even with `--create-index` — `--fetch-k` is what
   bounds it.

   **Why `--lambda-mult` defaults to 0.7 here and not the library's 0.5.** Measured on this
   corpus with `text-embedding-3-small` on 2026-08-08, every cosine distance fell between
   0.6055 and 0.6690. So the relevance term spans about 0.06 across the whole corpus, while
   the redundancy term — near-duplicates score around 0.9 against each other, unrelated
   documents far lower — spans several times that. Weighted equally at 0.5, variety decides
   almost every pick: the run promoted `soma-studio`, which has no outdoor space at all. At
   0.7 and 0.8, identically, it dropped the duplicate `noe-garden` and promoted `sunset-surf`
   instead. At 1.0 it reproduced step 5 exactly, which is the control worth re-running after
   any change to the read path.

   This is one corpus, one query and one embedding model, so it is a reason to sweep
   `lambda_mult` on your own data — not a number to copy.
7. **Filtered search** — the same query with `outdoor=True, beds=1`. The predicate goes into
   the ranking query's `WHERE`, not around its result, so the filter still returns the top `k`
   *matching* rows rather than whatever survives filtering an already-chosen top `k`.
8. **Retrieval chain** — `store.as_retriever()` composed into a prompt and a model with LCEL.
   Nothing in this step knows it is talking to Hotdata.

The script numbers the steps as it runs them, so without `--create-index` step 4 is absent and
everything after it shifts up by one.
