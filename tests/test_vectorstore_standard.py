"""LangChain's own conformance suite, run against ``HotdataVectorStore``.

The suite certifies the id/CRUD contract that retrievers and chains depend on:
add-by-id is idempotent and upserts, ids come back in input order, caller documents are
never mutated, deleting unknown ids is a no-op, and an empty store is queryable. Running
the published suite rather than restating its assertions locally is the point — it is
the interface contract as LangChain defines it, not our reading of it.

It runs against :class:`tests.fake_hotdata.FakeHotdataClient`, which stores rows and
ranks them for real, so no workspace or provider credentials are needed.
"""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import MagicMock

import pytest
from hotdata_framework import ManagedDatabase
from langchain_core.vectorstores import VectorStore
from langchain_tests.integration_tests.vectorstores import VectorStoreIntegrationTests

from hotdata_langchain.vectorstore import HotdataVectorStore
from tests.fake_hotdata import FakeHotdataClient


class TestHotdataVectorStore(VectorStoreIntegrationTests):
    @pytest.fixture
    def vectorstore(
        self,
        managed_db: ManagedDatabase,
        databases_api: MagicMock,
    ) -> Iterator[VectorStore]:
        yield HotdataVectorStore(
            FakeHotdataClient(),  # type: ignore[arg-type]
            self.get_embeddings(),
            database_id=managed_db.id,
        )
