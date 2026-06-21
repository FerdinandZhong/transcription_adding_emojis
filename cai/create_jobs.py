#!/usr/bin/env python3
"""Create or update CAI Jobs from cai/jobs_config.yaml."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import requests
import urllib3
import yaml


def _client() -> tuple[requests.Session, str, str]:
    api_key = os.environ.get("CDSW_APIV2_KEY")
    domain = os.environ.get("CDSW_DOMAIN")
    project_id = os.environ.get("CDSW_PROJECT_ID")
    if not all([api_key, domain, project_id]):
        print("Error: run inside CAI (CDSW_APIV2_KEY, CDSW_DOMAIN, CDSW_PROJECT_ID required)")
        sys.exit(1)
    if not domain.startswith(("http://", "https://")):
        domain = f"https://{domain}"
    session = requests.Session()
    session.headers.update({
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    })
    session.trust_env = False
    session.verify = False
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    return session, domain, project_id


def _load_config() -> Dict[str, Any]:
    path = Path(__file__).with_name("jobs_config.yaml")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _list_jobs(client: requests.Session, domain: str, project_id: str) -> Dict[str, str]:
    url = f"{domain}/api/v2/projects/{project_id}/jobs"
    resp = client.get(url)
    resp.raise_for_status()
    return {j["name"]: j["id"] for j in resp.json().get("jobs", [])}


def _job_payload(cfg: Dict[str, Any]) -> Dict[str, Any]:
    payload = {
        "name": cfg["name"],
        "script": cfg["script"],
        "cpu": cfg.get("cpu", 4),
        "memory": cfg.get("memory", 8),
        "timeout": cfg.get("timeout", 3600),
    }
    if cfg.get("gpu") is not None:
        payload["gpu"] = cfg["gpu"]
    if cfg.get("runtime_identifier"):
        payload["runtime_identifier"] = cfg["runtime_identifier"].replace("\n", "").replace(" ", "")
    if cfg.get("environment"):
        payload["environment"] = cfg["environment"]
    return payload


def _create_or_update(
    client: requests.Session,
    domain: str,
    project_id: str,
    key: str,
    cfg: Dict[str, Any],
    existing: Dict[str, str],
) -> None:
    url = f"{domain}/api/v2/projects/{project_id}/jobs"
    payload = _job_payload(cfg)
    name = cfg["name"]
    if name in existing:
        job_id = existing[name]
        print(f"Updating job {key!r} ({job_id})")
        client.patch(f"{url}/{job_id}", json=payload).raise_for_status()
    else:
        print(f"Creating job {key!r}")
        resp = client.post(url, json=payload)
        resp.raise_for_status()
        print(f"  created id={resp.json().get('id')}")


def main() -> None:
    client, domain, project_id = _client()
    config = _load_config()
    existing = _list_jobs(client, domain, project_id)
    for key, cfg in (config.get("jobs") or {}).items():
        _create_or_update(client, domain, project_id, key, cfg, existing)
    print("\nDone. Run a job from the CAI Jobs UI or API.")


if __name__ == "__main__":
    main()
