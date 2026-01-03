import argparse
import json
import time
from typing import Any, Dict, List, Tuple

import requests


DEFAULT_TESTS: List[str] = [
    "Disallow public network access on storage accounts",
    "Require secure transfer (HTTPS) for storage accounts",
    "Enforce minimum TLS version 1.2 for storage accounts",
    "Disable public blob access on storage accounts",
    "Require tag owner on all resources",
    "App Configuration should use a customer-managed key for encryption",
]


def _normalize_base(url: str) -> str:
    url = (url or "").strip().rstrip("/")
    if url.endswith("/generate"):
        url = url[: -len("/generate")]
    if url.endswith("/health"):
        url = url[: -len("/health")]
    return url


def _get_json(url: str, timeout_s: int = 10) -> Dict[str, Any]:
    resp = requests.get(url, timeout=timeout_s)
    resp.raise_for_status()
    return resp.json()


def _post_json(url: str, payload: Dict[str, Any], timeout_s: int = 180) -> Dict[str, Any]:
    resp = requests.post(url, json=payload, timeout=timeout_s)
    resp.raise_for_status()
    return resp.json()


def _is_non_empty_if(policy: Any) -> bool:
    if not isinstance(policy, dict):
        return False
    props = policy.get("properties")
    if not isinstance(props, dict):
        return False
    pr = props.get("policyRule")
    if not isinstance(pr, dict):
        return False
    if_block = pr.get("if")
    if not isinstance(if_block, dict):
        return False
    if if_block == {}:
        return False
    all_of = if_block.get("allOf")
    any_of = if_block.get("anyOf")
    if isinstance(all_of, list) and len(all_of) == 0:
        return False
    if isinstance(any_of, list) and len(any_of) == 0:
        return False
    return True


def _has_effect_parameter(policy: Any) -> bool:
    if not isinstance(policy, dict):
        return False
    props = policy.get("properties")
    if not isinstance(props, dict):
        return False
    params = props.get("parameters")
    if not isinstance(params, dict):
        return False
    eff = params.get("effect")
    return isinstance(eff, dict) and eff.get("type") == "String"


def _then_effect_parameterized(policy: Any) -> bool:
    if not isinstance(policy, dict):
        return False
    props = policy.get("properties")
    if not isinstance(props, dict):
        return False
    pr = props.get("policyRule")
    if not isinstance(pr, dict):
        return False
    then = pr.get("then")
    if not isinstance(then, dict):
        return False
    return then.get("effect") == "[parameters('effect')]"


def score_one(result: Dict[str, Any]) -> Tuple[bool, List[str]]:
    issues: List[str] = []
    fixed_policy = result.get("fixed_policy")
    if fixed_policy is None:
        # backward compatibility
        fixed_policy = result.get("policy")

    if not isinstance(fixed_policy, dict):
        issues.append("missing_fixed_policy")
        return False, issues

    if "properties" not in fixed_policy:
        issues.append("missing_properties")
    if not _is_non_empty_if(fixed_policy):
        issues.append("empty_if")
    if not _has_effect_parameter(fixed_policy):
        issues.append("missing_effect_parameter")
    if not _then_effect_parameterized(fixed_policy):
        issues.append("then_effect_not_parameterized")

    ok = len(issues) == 0
    return ok, issues


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", required=True, help="Base API URL (ngrok), e.g. https://xxxx.ngrok-free.app")
    parser.add_argument("--out", default="eval_results.jsonl", help="Where to write JSONL results")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--tests", nargs="*", default=DEFAULT_TESTS)
    args = parser.parse_args()

    base = _normalize_base(args.api)
    health_url = f"{base}/health"
    gen_url = f"{base}/generate"

    print("API:", base)
    print("Checking health...")
    try:
        health = _get_json(health_url)
        print("Health:", json.dumps(health, indent=2))
        if not health.get("model_loaded"):
            raise SystemExit("Model not loaded on server. Run the notebook model-load cell, then restart the API.")
    except Exception as e:
        raise SystemExit(f"Health check failed: {e}")

    ok_count = 0
    total = 0

    with open(args.out, "w", encoding="utf-8") as f:
        for t in args.tests:
            total += 1
            started = time.time()
            try:
                result = _post_json(gen_url, {"instruction": t}, timeout_s=args.timeout)
            except Exception as e:
                rec = {"instruction": t, "error": str(e)}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(f"[{total}] FAIL request: {t} -> {e}")
                continue

            passed, issues = score_one(result)
            elapsed = round(time.time() - started, 2)
            rec = {
                "instruction": t,
                "passed": passed,
                "issues": issues,
                "retry": result.get("retry"),
                "meta": result.get("meta"),
                "elapsed_s": elapsed,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if passed:
                ok_count += 1
                print(f"[{total}] PASS ({elapsed}s) {t}")
            else:
                print(f"[{total}] FAIL ({elapsed}s) {t} -> {issues}")

    print(f"\nSummary: {ok_count}/{total} passed")
    print("Wrote:", args.out)


if __name__ == "__main__":
    main()
