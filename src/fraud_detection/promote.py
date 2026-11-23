"""Model registry roles and releases with MLflow aliases: champion, challenger, previous.

A release is a pointer move, not a file copy: every registered version is immutable, and the aliases
say which version plays which role. Rollback moves the pointer back.

    init      give the `champion` alias to the released model (the DVC pipeline's final model)
    register  register the trained challenger (model + its own policy + metadata + comparison +
              approval report) and give it the `challenger` alias; export it to deploy/challenger
              so the API can run it in SHADOW mode
    promote   after human approval: champion -> previous, challenger -> champion, export to
              deploy/champion. Refuses without an approver, and refuses a challenger the automatic
              rules did not recommend unless --override is given (recorded)
    rollback  champion <-> previous, export to deploy/champion
    status    which version holds each alias, with its release tags

Every version carries a "deploy bundle" (final_model.txt, metadata.json, policy.json): exactly the
files the API loads. What was evaluated and approved is what ships; nothing is retrained on promotion.

After promote/rollback, restart the API to load the new files:  docker compose restart api

Run:  python -m fraud_detection.promote status
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import lightgbm as lgb  # noqa: E402  (after the environment setting above)
import mlflow
import mlflow.lightgbm
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from fraud_detection.compare import REPORT_PATH as COMPARISON_PATH
from fraud_detection.config import (
    CHALLENGER_DIR,
    DEPLOY_DIR,
    FINAL_MODEL_PATH,
    METADATA_PATH,
    POLICY_PATH,
    REPORTS_DIR,
    Params,
    load_params,
)
from fraud_detection.retrain import INFO_FILE, METADATA_FILE, MODEL_FILE, POLICY_FILE
from fraud_detection.tracking import _experiment_id

CHAMPION, CHALLENGER, PREVIOUS = "champion", "challenger", "previous"
BUNDLE_DIR = "deploy"                                    # artifact folder of each version's run
BUNDLE_FILES = ("final_model.txt", "metadata.json", "policy.json")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _client(params: Params) -> MlflowClient:
    mlflow.set_tracking_uri(params.mlflow.tracking_uri)
    return MlflowClient()


def alias_version(client: MlflowClient, params: Params, alias: str):
    """The model version holding `alias`, or None."""
    try:
        return client.get_model_version_by_alias(params.mlflow.registered_model, alias)
    except MlflowException:
        return None


def _attach_bundle(client: MlflowClient, run_id: str, files: dict[str, Path]) -> None:
    """Log the deploy bundle (renamed to the serving file names) to an existing run."""
    with tempfile.TemporaryDirectory() as tmp:
        for name, source in files.items():
            shutil.copyfile(source, Path(tmp) / name)
        client.log_artifacts(run_id, tmp, artifact_path=BUNDLE_DIR)


def export(client: MlflowClient, params: Params, alias: str, dest: Path) -> dict:
    """Download the bundle of the version holding `alias` into `dest` (the folder the API reads)."""
    version = alias_version(client, params, alias)
    if version is None:
        raise SystemExit(f"no model version has the alias '{alias}'")
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(mlflow.artifacts.download_artifacts(run_id=version.run_id, artifact_path=BUNDLE_DIR,
                                                          dst_path=tmp))
        missing = [f for f in BUNDLE_FILES if not (local / f).exists()]
        if missing:
            raise SystemExit(f"version {version.version} has an incomplete deploy bundle: missing {missing}")
        dest.mkdir(parents=True, exist_ok=True)
        for f in BUNDLE_FILES:
            shutil.copyfile(local / f, dest / f)
    meta = json.loads((dest / "metadata.json").read_text(encoding="utf-8"))
    return {"alias": alias, "version": str(version.version), "model_version": meta["model_version"], "dest": str(dest)}


def _clear(dest: Path) -> None:
    for f in BUNDLE_FILES:
        (dest / f).unlink(missing_ok=True)


# --------------------------------------------------------------------------- commands
def init_champion(params: Params, metadata_path: Path = METADATA_PATH, models_dir: Path | None = None,
                  deploy_dir: Path = DEPLOY_DIR, refresh: bool = False) -> dict:
    """Give `champion` to the released model's registered version (from metadata.json) and export it.

    Without `refresh` an existing champion is kept. With it, the champion moves to the version the
    pipeline last released (e.g. the same model re-registered with clean lineage after a commit).
    """
    client = _client(params)
    current = alias_version(client, params, CHAMPION)
    if current is None or refresh:
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        version = str(meta["mlflow"]["registered_version"])
        mv = client.get_model_version(params.mlflow.registered_model, version)
        if not client.list_artifacts(mv.run_id, BUNDLE_DIR):
            base = models_dir or metadata_path.parent
            _attach_bundle(client, mv.run_id, {"final_model.txt": base / FINAL_MODEL_PATH.name,
                                               "metadata.json": metadata_path,
                                               "policy.json": base / POLICY_PATH.name})
        client.set_registered_model_alias(params.mlflow.registered_model, CHAMPION, version)
        client.set_model_version_tag(params.mlflow.registered_model, version, "model_version", meta["model_version"])
        client.set_model_version_tag(params.mlflow.registered_model, version, "role_since", _now())
    return export(client, params, CHAMPION, deploy_dir / CHAMPION)


def register_challenger(params: Params, challenger_dir: Path = CHALLENGER_DIR,
                        comparison_path: Path = COMPARISON_PATH, report_path: Path | None = None,
                        deploy_dir: Path = DEPLOY_DIR) -> dict:
    """Register the challenger with all its evidence; alias `challenger`; export it for shadow mode."""
    client = _client(params)
    info = json.loads((challenger_dir / INFO_FILE).read_text(encoding="utf-8"))
    meta = json.loads((challenger_dir / METADATA_FILE).read_text(encoding="utf-8"))
    comparison = json.loads(comparison_path.read_text(encoding="utf-8")) if comparison_path.exists() else None
    report_path = report_path or REPORTS_DIR / f"approval_{info['as_of'][:10]}.md"
    recommendation = comparison["recommendation"]["decision"] if comparison else "not_compared"

    with mlflow.start_run(experiment_id=_experiment_id(params), run_name=meta["model_version"]) as run:
        mlflow.set_tags({"role": CHALLENGER, "model_version": meta["model_version"], "as_of": info["as_of"],
                         "git_commit": info["git_commit"], "data_md5": str(info["data_md5"]),
                         "recommendation": recommendation})
        mlflow.log_params({"n_trees": info["n_trees"], "as_of": info["as_of"],
                           "block_threshold": info["policy"]["block_threshold"],
                           "review_threshold": info["policy"]["review_threshold"],
                           "false_block_budget": info["policy"]["false_block_budget"],
                           **{f"retrain_{k}": v for k, v in info["params"]["retrain"].items()}})
        mlflow.log_metric("best_validation_pr_auc", info["best_validation_pr_auc"])
        if comparison:
            for metric, v in comparison["overall"].items():
                for side in ("champion", "challenger"):
                    if v[side] is not None:
                        mlflow.log_metric(f"holdout_{metric}_{side}", v[side])
            mlflow.log_artifact(str(comparison_path), artifact_path="evidence")
        mlflow.log_artifact(str(challenger_dir / INFO_FILE), artifact_path="evidence")
        if report_path.exists():
            mlflow.log_artifact(str(report_path), artifact_path="evidence")
        model_info = mlflow.lightgbm.log_model(lgb.Booster(model_file=str(challenger_dir / MODEL_FILE)),
                                               name="model", registered_model_name=params.mlflow.registered_model)
        run_id = run.info.run_id
    _attach_bundle(client, run_id, {"final_model.txt": challenger_dir / MODEL_FILE,
                                    "metadata.json": challenger_dir / METADATA_FILE,
                                    "policy.json": challenger_dir / POLICY_FILE})
    version = str(model_info.registered_model_version)
    name = params.mlflow.registered_model
    client.set_registered_model_alias(name, CHALLENGER, version)
    for key, value in {"model_version": meta["model_version"], "recommendation": recommendation,
                       "registered_at": _now()}.items():
        client.set_model_version_tag(name, version, key, value)
    return {"version": version, "recommendation": recommendation,
            "shadow": export(client, params, CHALLENGER, deploy_dir / CHALLENGER)}


def promote(params: Params, approved_by: str, override: bool = False, deploy_dir: Path = DEPLOY_DIR) -> dict:
    """champion -> previous, challenger -> champion; record who approved; export the new champion."""
    if not approved_by.strip():
        raise SystemExit("promotion needs a named approver (--approved-by)")
    client = _client(params)
    name = params.mlflow.registered_model
    challenger, champion = alias_version(client, params, CHALLENGER), alias_version(client, params, CHAMPION)
    if challenger is None:
        raise SystemExit("no challenger to promote: run `register` first")
    if champion is None:
        raise SystemExit("no champion: run `init` first")
    recommendation = challenger.tags.get("recommendation")
    if recommendation != "promote" and not override:
        raise SystemExit(f"the automatic recommendation for version {challenger.version} is '{recommendation}'. "
                         "Promote anyway only with --override (it is recorded).")

    client.set_registered_model_alias(name, PREVIOUS, champion.version)
    client.set_registered_model_alias(name, CHAMPION, challenger.version)
    client.delete_registered_model_alias(name, CHALLENGER)
    now = _now()
    for key, value in {"approved_by": approved_by, "promoted_at": now, "override": str(override)}.items():
        client.set_model_version_tag(name, challenger.version, key, value)
    client.set_model_version_tag(name, champion.version, "demoted_at", now)
    _clear(deploy_dir / CHALLENGER)                         # it no longer runs in shadow: it decides
    exported = export(client, params, CHAMPION, deploy_dir / CHAMPION)
    return {"champion": str(challenger.version), "previous": str(champion.version), "exported": exported}


def rollback(params: Params, reason: str, deploy_dir: Path = DEPLOY_DIR) -> dict:
    """Swap champion and previous; record why; export the restored champion."""
    client = _client(params)
    name = params.mlflow.registered_model
    champion, previous = alias_version(client, params, CHAMPION), alias_version(client, params, PREVIOUS)
    if previous is None:
        raise SystemExit("nothing to roll back to: no version has the 'previous' alias")
    client.set_registered_model_alias(name, CHAMPION, previous.version)
    client.set_registered_model_alias(name, PREVIOUS, champion.version)
    now = _now()
    client.set_model_version_tag(name, champion.version, "rolled_back_at", now)
    client.set_model_version_tag(name, champion.version, "rollback_reason", reason)
    client.set_model_version_tag(name, previous.version, "restored_at", now)
    return {"champion": str(previous.version), "previous": str(champion.version),
            "exported": export(client, params, CHAMPION, deploy_dir / CHAMPION)}


def status(params: Params) -> dict:
    client = _client(params)
    out = {}
    for alias in (CHAMPION, CHALLENGER, PREVIOUS):
        mv = alias_version(client, params, alias)
        out[alias] = None if mv is None else {"version": str(mv.version), **mv.tags}
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m fraud_detection.promote", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    i = sub.add_parser("init", help="give the champion alias to the released model and export it")
    i.add_argument("--refresh", action="store_true",
                   help="move an existing champion alias to the pipeline's latest released version")
    sub.add_parser("register", help="register the challenger (alias challenger) and export it for shadow mode")
    p = sub.add_parser("promote", help="promote the challenger after human approval")
    p.add_argument("--approved-by", required=True)
    p.add_argument("--override", action="store_true", help="promote against the automatic recommendation")
    r = sub.add_parser("rollback", help="restore the previous champion")
    r.add_argument("--reason", required=True)
    sub.add_parser("status", help="which version holds each alias")
    args = parser.parse_args(argv)

    params = load_params()
    if args.command == "init":
        result = init_champion(params, refresh=args.refresh)
    elif args.command == "register":
        result = register_challenger(params)
    elif args.command == "promote":
        result = promote(params, args.approved_by, args.override)
    elif args.command == "rollback":
        result = rollback(params, args.reason)
    else:
        result = status(params)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
