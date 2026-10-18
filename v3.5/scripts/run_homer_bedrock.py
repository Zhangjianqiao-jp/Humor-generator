#!/usr/bin/env python3
"""HOMER anonymous A/B packets through Bedrock Converse; offline by default.

This is an evaluator substitution, not the canonical GPT-5 result. Credentials
are read as JSON, never executed or included in artifacts. No private mapping
is opened. Each successful decision is persisted before the next API call.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

from run_homer_primary_openai import (
    canonical_json, load_existing, load_packets, load_provenance, sha256_text,
)

ROOT = Path(__file__).resolve().parents[1]
PRIMARY = ROOT / "outputs/caption_judgement/pretrained_bridge_comparison_20260911/homer_primary"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_config(path: Path) -> dict:
    if path.stat().st_mode & 0o077:
        raise ValueError("credential file must have permissions 600")
    config = json.loads(path.read_text())
    allowed = {
        "region", "model_id", "auth_mode", "api_key", "aws_access_key_id",
        "aws_secret_access_key", "aws_session_token", "profile", "max_tokens",
    }
    if not isinstance(config, dict) or set(config) - allowed:
        raise ValueError("invalid configuration keys")
    for key in allowed - {"max_tokens"}:
        if not isinstance(config.get(key, ""), str):
            raise ValueError(f"{key} must be a string")
    for key in ("region", "model_id"):
        if not config.get(key, "").strip():
            raise ValueError(f"fill in {key} in the local configuration")
    mode = config.get("auth_mode")
    if mode not in {"bearer", "iam", "profile"}:
        raise ValueError("auth_mode must be bearer, iam or profile")
    required = {"bearer": ["api_key"], "iam": ["aws_access_key_id", "aws_secret_access_key"], "profile": ["profile"]}
    if any(not config.get(key, "").strip() for key in required[mode]):
        raise ValueError(f"fill in credentials for auth_mode={mode}")
    count = config.get("max_tokens", 8)
    if type(count) is not int or count < 1:
        raise ValueError("max_tokens must be a positive integer")
    config["max_tokens"] = count
    return config


def make_client(config: dict):
    # Only called after --execute and all local resume checks have passed.
    import boto3
    from botocore.config import Config

    mode = config["auth_mode"]
    if mode == "bearer":
        os.environ["AWS_BEARER_TOKEN_BEDROCK"] = config["api_key"]
        session = boto3.Session()
    else:
        os.environ.pop("AWS_BEARER_TOKEN_BEDROCK", None)
        if mode == "profile":
            session = boto3.Session(profile_name=config["profile"])
        else:
            session = boto3.Session(
                aws_access_key_id=config["aws_access_key_id"],
                aws_secret_access_key=config["aws_secret_access_key"],
                aws_session_token=config.get("aws_session_token") or None,
            )
    return session.client(
        "bedrock-runtime", region_name=config["region"],
        config=Config(connect_timeout=15, read_timeout=120,
                      retries={"mode": "standard", "total_max_attempts": 3}),
    )


def run(args) -> dict:
    config = load_config(args.credentials)
    provenance = load_provenance(args.provenance)
    prompt_hash = provenance["evaluator"]["prompt_sha256"]
    packets = load_packets(args.packets, expected_system_prompt_hash=prompt_hash)
    if args.output.resolve() in {args.packets.resolve(), args.provenance.resolve(), args.credentials.resolve()}:
        raise ValueError("output must not overwrite an input")
    manifest = {
        "schema_version": 1,
        "track": "homer_bedrock_evaluator_substitution",
        "evaluator_substitution": True,
        "model": config["model_id"], "region": config["region"],
        "temperature": 0, "max_tokens": config["max_tokens"],
        "prompt_sha256": prompt_hash,
        "packets_sha256": digest(args.packets),
        "source_provenance_sha256": digest(args.provenance),
        "script_sha256": digest(Path(__file__)),
        "evaluation_transport": "amazon-bedrock-converse",
    }
    run_hash = sha256_text(canonical_json(manifest))
    expected = {p["packet_id"]: sha256_text(canonical_json(p)) for p in packets}
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")

    def validate_resume():
        if manifest_path.exists():
            if json.loads(manifest_path.read_text()) != manifest:
                raise ValueError("run manifest changed; use a new output path")
        elif args.output.exists() and args.output.stat().st_size:
            raise ValueError("existing judgments have no run manifest")
        existing = load_existing(args.output, model_id=config["model_id"], temperature=0, prompt_hash=prompt_hash)
        for packet_id, row in existing.items():
            if packet_id not in expected or row.get("packet_sha256") != expected[packet_id]:
                raise ValueError("existing packet content differs")
            if row.get("run_sha256") != run_hash or row.get("source_provenance_sha256") != manifest["source_provenance_sha256"]:
                raise ValueError("existing judgment provenance differs")
        return existing

    existing = validate_resume()
    pending = [p for p in packets if p["packet_id"] not in existing]
    if not args.execute:
        return {"status": "offline_validated", "api_calls": 0, "packets": len(packets),
                "remaining": len(pending), "evaluator_substitution": True}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix(args.output.suffix + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)                
        existing = validate_resume()
        pending = [p for p in packets if p["packet_id"] not in existing]
        if not pending:
            return {"status": "already_complete", "new_judgments": 0}
        if args.max_requests is not None:
            pending = pending[:args.max_requests]
        client = make_client(config)
        if not manifest_path.exists():
            temp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
            temp.write_text(canonical_json(manifest) + "\n")
            temp.replace(manifest_path)
        with args.output.open("a", encoding="utf-8") as handle:
            for index, packet in enumerate(pending):
                if index:
                    time.sleep(args.delay_seconds)
                response = client.converse(
                    modelId=config["model_id"],
                    system=[{"text": packet["system_prompt"]}],
                    messages=[{"role": "user", "content": [{"text": packet["user_prompt"]}]}],
                    inferenceConfig={"temperature": 0, "maxTokens": config["max_tokens"]},
                )
                content = response.get("output", {}).get("message", {}).get("content", [])
                raw = "".join(block.get("text", "") for block in content)
                decision = raw.strip()
                if decision not in {"A", "B"} or response.get("stopReason") not in {"end_turn", "max_tokens"}:
                    invalid = args.output.with_suffix(args.output.suffix + ".invalid.jsonl")
                    with invalid.open("a", encoding="utf-8") as bad:
                        bad.write(canonical_json({"packet_id": packet["packet_id"], "raw_text": raw,
                            "stop_reason": response.get("stopReason"), "run_sha256": run_hash}) + "\n")
                    raise ValueError("invalid A/B response saved separately; no default judgment inserted")
                row = {
                    "packet_id": packet["packet_id"], "decision": decision,
                    **{k: manifest[k] for k in ("model", "temperature", "prompt_sha256", "source_provenance_sha256", "evaluation_transport", "evaluator_substitution")},
                    "run_sha256": run_hash, "packet_sha256": expected[packet["packet_id"]],
                    "raw_text": raw, "stop_reason": response.get("stopReason"),
                    "usage": response.get("usage", {}),
                    "request_id": response.get("ResponseMetadata", {}).get("RequestId"),
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "temperature_protocol": "requested-0-bedrock-converse",
                }
                handle.write(canonical_json(row) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                print(json.dumps({"completed": len(existing) + index + 1, "total": len(packets)}), flush=True)
    count = len(existing) + len(pending)
    return {"status": "complete" if count == len(packets) else "partial", "judgments": count}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--credentials", type=Path, default=ROOT / ".secrets/amazon_bedrock.json")
    parser.add_argument("--packets", type=Path, default=PRIMARY / "homer_primary_prompts.jsonl")
    parser.add_argument("--provenance", type=Path, default=PRIMARY / "provenance.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/bedrock_homer/judgments.jsonl")
    parser.add_argument("--execute", action="store_true", help="send paid Bedrock requests; otherwise local validation only")
    parser.add_argument("--max-requests", type=int, help="maximum new packets (SDK may retry transient failures)")
    parser.add_argument("--delay-seconds", type=float, default=1.0)
    args = parser.parse_args()
    if (args.max_requests is not None and args.max_requests < 1) or not 0 <= args.delay_seconds < float("inf"):
        parser.error("max-requests must be positive and delay-seconds finite/non-negative")
    try:
        print(json.dumps(run(args), ensure_ascii=False, indent=2))
    except Exception as exc:
        # SDK exceptions can embed request information. Do not print secrets or payloads.
        if isinstance(exc, ValueError):
            message = str(exc) if not isinstance(exc, json.JSONDecodeError) else "invalid JSON file"
        else:
            message = type(exc).__name__
            code = getattr(exc, "response", {}).get("Error", {}).get("Code")
            if code:
                message += f" ({code})"
        raise SystemExit(f"Stopped: {message}") from None


if __name__ == "__main__":
    main()
