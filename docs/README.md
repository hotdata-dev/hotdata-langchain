# docs/

Working notes for the AI-native query layer effort, checked in deliberately while the repo is
small so the reasoning stays with the code rather than in chat logs. Expect to prune these once
the roadmap is largely delivered.

| Doc | What it is |
|---|---|
| [`engine-contract.md`](./engine-contract.md) | What the Hotdata SQL and search surface actually does, verified against a live workspace. The source of truth the tool descriptions encode — read this before asserting engine behaviour. |
| [`ai-native-layer-roadmap.md`](./ai-native-layer-roadmap.md) | Tiered plan for the layer, the routing decomposition, and the cross-repo work it surfaced (issues #36–#42). |
| [`vectorstore-plan.md`](./vectorstore-plan.md) | Design for `HotdataVectorStore` — schema, methods, SQL path, testing. Not yet implemented. |

These are point-in-time notes, not specifications. Where one states engine behaviour it carries
the date it was verified; re-check against a live workspace before relying on it.
