#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REGISTRY_PATH = Path("pre_trained_model/registry.yaml")


_MODEL_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_\\-]{2,127}$")


def _fail(msg: str) -> None:
    raise SystemExit(f"ERROR: {msg}")


def _load_yaml_minimal(path: Path) -> dict[str, Any]:
    """
    Minimal YAML loader for our simple registry shape:
    - supports top-level mapping with a 'models' key
    - supports a list of dicts under 'models'

    If PyYAML is installed, use it. Otherwise fall back to a strict, minimal parser.
    """
    if not path.exists():
        _fail(f"Registry not found at {path}")

    text = path.read_text(encoding="utf-8")

    try:
        import yaml  # type: ignore
    except Exception:
        return _parse_registry_minimal(text)
    else:
        data = yaml.safe_load(text) or {}
        if not isinstance(data, dict):
            _fail("registry.yaml must be a YAML mapping at top-level")
        return data


def _parse_registry_minimal(text: str) -> dict[str, Any]:
    # Very small parser that only accepts:
    # models:
    #   - key: value
    #     key2: value2
    # models: []
    lines = [ln.rstrip("\n") for ln in text.splitlines()]
    # strip comments / empty
    raw = []
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        raw.append(ln)

    if not raw:
        return {}

    # allow "models: []"
    if len(raw) == 1 and raw[0].strip() == "models: []":
        return {"models": []}

    if raw[0].strip() != "models:":
        _fail("Minimal parser expects registry.yaml to start with 'models:'")

    models: list[dict[str, Any]] = []
    cur: dict[str, Any] | None = None
    for ln in raw[1:]:
        if ln.startswith("  - "):
            if cur is not None:
                models.append(cur)
            cur = {}
            rest = ln[len("  - ") :].strip()
            if rest:
                k, v = _split_kv(rest)
                cur[k] = _parse_scalar(v)
        elif ln.startswith("    "):
            if cur is None:
                _fail("Found indented key/value before any list item")
            k, v = _split_kv(ln.strip())
            cur[k] = _parse_scalar(v)
        else:
            _fail("Unsupported YAML structure in registry.yaml (install PyYAML to use richer YAML)")

    if cur is not None:
        models.append(cur)
    return {"models": models}


def _split_kv(s: str) -> tuple[str, str]:
    if ":" not in s:
        _fail(f"Expected key: value, got: {s!r}")
    k, v = s.split(":", 1)
    return k.strip(), v.strip()


def _parse_scalar(v: str) -> Any:
    # basic YAML-like scalar parsing: strings (optionally quoted), numbers, booleans, null
    if v in ("null", "Null", "NULL", "~"):
        return None
    if v in ("true", "True", "TRUE"):
        return True
    if v in ("false", "False", "FALSE"):
        return False
    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
        return v[1:-1]
    # int/float
    try:
        if "." in v:
            return float(v)
        return int(v)
    except Exception:
        return v


@dataclass(frozen=True)
class RegistryModel:
    model_id: str
    owner: str
    created_at: str
    task: str
    framework: str
    artifact_uri: str
    dataset: dict[str, Any] | None = None
    metrics: dict[str, Any] | None = None
    notes: str | None = None


def _validate_registry(data: dict[str, Any]) -> list[RegistryModel]:
    if "models" not in data:
        _fail("registry.yaml must contain top-level key 'models'")
    models_raw = data["models"]
    if not isinstance(models_raw, list):
        _fail("'models' must be a list")

    seen: set[str] = set()
    out: list[RegistryModel] = []
    for i, item in enumerate(models_raw):
        if not isinstance(item, dict):
            _fail(f"models[{i}] must be a mapping/object")

        def req_str(key: str) -> str:
            val = item.get(key)
            if not isinstance(val, str) or not val.strip():
                _fail(f"models[{i}].{key} must be a non-empty string")
            return val.strip()

        model_id = req_str("model_id")
        if not _MODEL_ID_RE.match(model_id):
            _fail(
                f"models[{i}].model_id must match {_MODEL_ID_RE.pattern} (got {model_id!r})"
            )
        if model_id in seen:
            _fail(f"Duplicate model_id: {model_id}")
        seen.add(model_id)

        created_at = req_str("created_at")
        # ISO-8601 best-effort check
        try:
            dt.datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        except Exception:
            _fail(f"models[{i}].created_at must be ISO-8601 datetime (got {created_at!r})")

        owner = req_str("owner")
        task = req_str("task")
        framework = req_str("framework")
        artifact_uri = req_str("artifact_uri")

        dataset = item.get("dataset")
        if dataset is not None and not isinstance(dataset, dict):
            _fail(f"models[{i}].dataset must be an object/mapping when provided")

        metrics = item.get("metrics")
        if metrics is not None and not isinstance(metrics, dict):
            _fail(f"models[{i}].metrics must be an object/mapping when provided")

        notes = item.get("notes")
        if notes is not None and not isinstance(notes, str):
            _fail(f"models[{i}].notes must be a string when provided")

        out.append(
            RegistryModel(
                model_id=model_id,
                owner=owner,
                created_at=created_at,
                task=task,
                framework=framework,
                artifact_uri=artifact_uri,
                dataset=dataset,
                metrics=metrics,
                notes=notes,
            )
        )

    return out


def cmd_list(args: argparse.Namespace) -> None:
    data = _load_yaml_minimal(REGISTRY_PATH)
    models = _validate_registry(data)
    if args.json:
        print(json.dumps([m.__dict__ for m in models], indent=2, ensure_ascii=False))
        return
    for m in models:
        print(f"- {m.model_id}  ({m.framework})  owner={m.owner}  artifact={m.artifact_uri}")


def cmd_validate(_: argparse.Namespace) -> None:
    data = _load_yaml_minimal(REGISTRY_PATH)
    models = _validate_registry(data)
    print(f"OK: {REGISTRY_PATH} ({len(models)} models)")


def cmd_add(args: argparse.Namespace) -> None:
    # Require PyYAML for safe write/merge.
    try:
        import yaml  # type: ignore
    except Exception:
        _fail("Add requires PyYAML installed. Run: pip install pyyaml")

    if not REGISTRY_PATH.exists():
        _fail(f"Registry not found at {REGISTRY_PATH}")

    data = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        _fail("registry.yaml must be a YAML mapping at top-level")
    data.setdefault("models", [])
    if not isinstance(data["models"], list):
        _fail("'models' must be a list")

    created_at = args.created_at or dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
    entry: dict[str, Any] = {
        "model_id": args.model_id,
        "owner": args.owner,
        "created_at": created_at,
        "task": args.task,
        "dataset": {"name": args.dataset_name, "version": args.dataset_version},
        "framework": args.framework,
        "artifact_uri": args.artifact_uri,
        "metrics": {},
        "notes": args.notes or "",
    }

    # Merge metrics k=v
    for kv in args.metric or []:
        if "=" not in kv:
            _fail(f"Metric must be key=value, got {kv!r}")
        k, v = kv.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k:
            _fail(f"Metric key is empty in {kv!r}")
        entry["metrics"][k] = _parse_scalar(v)

    data["models"].append(entry)

    # Validate before writing
    _validate_registry(data)
    REGISTRY_PATH.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"Added: {args.model_id}")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Team model registry helper")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="List registered models")
    p_list.add_argument("--json", action="store_true", help="Output JSON")
    p_list.set_defaults(func=cmd_list)

    p_val = sub.add_parser("validate", help="Validate registry format")
    p_val.set_defaults(func=cmd_validate)

    p_add = sub.add_parser("add", help="Add a new model entry (requires PyYAML)")
    p_add.add_argument("--model-id", required=True)
    p_add.add_argument("--owner", required=True)
    p_add.add_argument("--task", required=True)
    p_add.add_argument("--dataset-name", default="SKAB")
    p_add.add_argument("--dataset-version", default="v0.9")
    p_add.add_argument("--framework", required=True)
    p_add.add_argument("--artifact-uri", required=True)
    p_add.add_argument("--created-at", default=None, help="ISO-8601 datetime; default now (UTC)")
    p_add.add_argument("--metric", action="append", help="Metric key=value (repeatable)")
    p_add.add_argument("--notes", default=None)
    p_add.set_defaults(func=cmd_add)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

