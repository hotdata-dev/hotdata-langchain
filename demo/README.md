# BM25 search demo

Takes a Hotdata workspace from empty to a LangChain agent that full-text searches real
data, in one script. Uses the public San Francisco Airbnb listings fixture (7,535 rows)
and builds a BM25 index over the free-text `description` column.

## Run it

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

The script is safe to re-run: it reuses the managed database, skips the load when the
table already has rows, and reuses an existing index.

```bash
# different search text, more hits
uv run --group demo --env-file .env python demo/bm25_search_demo.py \
    --query "quiet garden studio near the park" --k 10

# stop before the agent step even with a model set
uv run --group demo --env-file .env python demo/bm25_search_demo.py --skip-agent

# tear down the managed database it created
uv run --group demo --env-file .env python demo/bm25_search_demo.py --cleanup

# trace the run into a named LangSmith project
LANGSMITH_TRACING=true LANGSMITH_PROJECT=hotdata-langchain-bm25 \
    uv run --group demo --env-file .env python demo/bm25_search_demo.py
```

## Credentials

| Variable | Needed for | Notes |
|---|---|---|
| `HOTDATA_API_KEY` | steps 1–5 | everything except the agent run |
| `HOTDATA_WORKSPACE` | optional | pins a workspace; first available otherwise |
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

## What each step does

1. **Managed database** — creates `langchain_bm25_demo` with `public.listings` declared
   up front, so the load materialises into it directly.
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

   **What this step reliably demonstrates is the routing**, not the arithmetic. Across
   five runs the agent called the search tool every time and reached for the schema tool
   in four, which is the behaviour the demo exists to show. The final table was right in
   four of five — one run aggregated over the handful of matched listings instead of all
   listings in those neighbourhoods. Answering a compound question correctly is a property
   of the model, not of these tools, so read the printed tool calls as the result and treat
   the prose answer as illustrative.

## What makes the agent run work

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
of a tool aborts the whole graph — so the demo wraps them (`with_error_feedback`) to
return the error as a message instead. The wrapper also digs the engine's real message
out of the exception chain: the framework raises `RuntimeError("Bad Request")` while the
useful text sits in the underlying API response body. Descriptions lower the failure
rate; readable errors are what let the model recover from what slips through.

## Why the generated SQL looks the way it does

Step 5 prints the query. Two details in it are load-bearing:

- **`ORDER BY score DESC`** — the engine returns hits in rowid order, not ranked, so
  ranking has to be asked for explicitly.
- **`k` appearing twice** — once as `bm25_search(...)`'s fourth argument and once as a
  trailing `LIMIT`. The fourth argument is what actually bounds the search, because
  `ORDER BY` blocks limit pushdown; relying on the trailing `LIMIT` alone would let the
  scan fall back to the engine's much larger default bound.
