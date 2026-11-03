"""20261018_83 builds its index CONCURRENTLY, so a failed build leaves an INVALID index behind. 'It exists' must not
be taken for 'it is done': an invalid one is dropped and rebuilt. The migration is run against a recording stand-in
for ``op`` (no database here); the pg_index lookup is asserted as the query it is."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

PATH = next((Path(__file__).resolve().parents[2] / "migrations" / "versions").glob("20261018_83_*.py"))


def load():
    spec = importlib.util.spec_from_file_location("mig83", PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Recorder:
    def __init__(self, *, existing_index: bool, valid: bool | None, dialect="postgresql"):
        self.calls, self.queries = [], []
        self.existing_index, self.valid = existing_index, valid
        self.bind = SimpleNamespace(dialect=SimpleNamespace(name=dialect), execute=self._execute)

    def _execute(self, statement, params=None):
        self.queries.append((str(statement), params))
        return SimpleNamespace(scalar=lambda: self.valid)

    def get_bind(self):
        return self.bind

    def get_context(self):
        recorder = self

        class _Ctx:
            def autocommit_block(self):
                recorder.calls.append("autocommit:begin")

                class _Block:
                    def __enter__(s):
                        return None

                    def __exit__(s, *a):
                        recorder.calls.append("autocommit:end")
                        return False

                return _Block()

        return _Ctx()

    def add_column(self, *a, **k):
        self.calls.append("add_column")

    def create_index(self, name, table, cols, **kw):
        self.calls.append(("create_index", name, kw.get("postgresql_concurrently")))

    def drop_index(self, name, **kw):
        self.calls.append(("drop_index", name, kw.get("postgresql_concurrently")))


def run(monkeypatch, **kw):
    module = load()
    recorder = Recorder(**kw)
    inspector = SimpleNamespace(
        get_table_names=lambda: ["notification_queue"],
        get_columns=lambda t: [{"name": "next_attempt_at"}],
        get_indexes=lambda t: [{"name": module.INDEX}] if recorder.existing_index else [],
    )
    monkeypatch.setattr(module, "op", recorder)
    monkeypatch.setattr(module, "inspect", lambda bind: inspector)
    module.upgrade()
    return recorder, module


def test_an_invalid_leftover_index_is_dropped_and_rebuilt(monkeypatch):
    recorder, module = run(monkeypatch, existing_index=True, valid=False)
    ops = [c for c in recorder.calls if isinstance(c, tuple)]
    assert ops == [("drop_index", module.INDEX, True), ("create_index", module.INDEX, True)]
    assert "indisvalid" in recorder.queries[0][0] and recorder.queries[0][1] == {"name": module.INDEX}


def test_a_valid_index_is_left_alone(monkeypatch):
    recorder, _ = run(monkeypatch, existing_index=True, valid=True)
    assert [c for c in recorder.calls if isinstance(c, tuple)] == []


def test_a_missing_index_is_created_concurrently_without_a_pg_index_lookup(monkeypatch):
    recorder, module = run(monkeypatch, existing_index=False, valid=None)
    assert [c for c in recorder.calls if isinstance(c, tuple)] == [("create_index", module.INDEX, True)]
    assert recorder.queries == []


def test_another_dialect_has_no_pg_index_to_ask(monkeypatch):
    recorder, _ = run(monkeypatch, existing_index=True, valid=False, dialect="sqlite")
    assert [c for c in recorder.calls if isinstance(c, tuple)] == [] and recorder.queries == []
