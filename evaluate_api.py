import argparse
import json
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import requests


# ---------------------------------------------------------------------------
# Test suite — 14 instructions covering multiple resource types.
# `reference_if` is the expected policyRule.if block for Field-Level F1.
# Tests without reference_if skip F1 scoring for that case.
# ---------------------------------------------------------------------------

DEFAULT_TESTS: List[Dict[str, Any]] = [
    # --- Storage (original 4) ---
    {
        "instruction": "Disallow public network access on storage accounts",
        "expected_type": "Microsoft.Storage/storageAccounts",
        "reference_if": {
            "allOf": [
                {"field": "type", "equals": "Microsoft.Storage/storageAccounts"},
                {"field": "Microsoft.Storage/storageAccounts/publicNetworkAccess", "notEquals": "Disabled"},
            ]
        },
    },
    {
        "instruction": "Require secure transfer (HTTPS) for storage accounts",
        "expected_type": "Microsoft.Storage/storageAccounts",
        "reference_if": {
            "allOf": [
                {"field": "type", "equals": "Microsoft.Storage/storageAccounts"},
                {"field": "Microsoft.Storage/storageAccounts/supportsHttpsTrafficOnly", "equals": False},
            ]
        },
    },
    {
        "instruction": "Enforce minimum TLS version 1.2 for storage accounts",
        "expected_type": "Microsoft.Storage/storageAccounts",
        "reference_if": {
            "allOf": [
                {"field": "type", "equals": "Microsoft.Storage/storageAccounts"},
                {
                    "anyOf": [
                        {"field": "Microsoft.Storage/storageAccounts/minimumTlsVersion", "exists": False},
                        {"field": "Microsoft.Storage/storageAccounts/minimumTlsVersion", "notEquals": "TLS1_2"},
                    ]
                },
            ]
        },
    },
    {
        "instruction": "Disable public blob access on storage accounts",
        "expected_type": "Microsoft.Storage/storageAccounts",
        "reference_if": {
            "allOf": [
                {"field": "type", "equals": "Microsoft.Storage/storageAccounts"},
                {"field": "Microsoft.Storage/storageAccounts/allowBlobPublicAccess", "equals": True},
            ]
        },
    },
    # --- Tags (original 1) ---
    {
        "instruction": "Require tag owner on all resources",
        "expected_type": None,
        "reference_if": {
            "allOf": [
                {"field": "tags['owner']", "exists": False},
            ]
        },
    },
    # --- App Configuration (original 1) ---
    {
        "instruction": "App Configuration should use a customer-managed key for encryption",
        "expected_type": "Microsoft.AppConfiguration/configurationStores",
        "reference_if": {
            "allOf": [
                {"field": "type", "equals": "Microsoft.AppConfiguration/configurationStores"},
                {
                    "anyOf": [
                        {"field": "Microsoft.AppConfiguration/configurationStores/encryption.keyVaultProperties.keyIdentifier", "exists": False},
                        {"field": "Microsoft.AppConfiguration/configurationStores/encryption.keyVaultProperties.keyIdentifier", "equals": ""},
                    ]
                },
            ]
        },
    },
    # --- New: Key Vault ---
    {
        "instruction": "Ensure Azure Key Vault has soft delete enabled",
        "expected_type": "Microsoft.KeyVault/vaults",
    },
    {
        "instruction": "Key Vault should have purge protection enabled",
        "expected_type": "Microsoft.KeyVault/vaults",
    },
    # --- New: SQL ---
    {
        "instruction": "SQL servers should have an Azure Active Directory administrator provisioned",
        "expected_type": "Microsoft.Sql/servers",
    },
    {
        "instruction": "Enforce TLS 1.2 on Azure SQL Database",
        "expected_type": "Microsoft.Sql/servers",
    },
    # --- New: Virtual Machines ---
    {
        "instruction": "Virtual machines should have disk encryption enabled",
        "expected_type": "Microsoft.Compute/virtualMachines",
    },
    # --- New: App Service ---
    {
        "instruction": "Web apps should require HTTPS only",
        "expected_type": "Microsoft.Web/sites",
    },
    # --- New: Paraphrased (tests generalisation beyond exact keywords) ---
    {
        "instruction": "Block unencrypted connections to Azure blob storage",
        "expected_type": "Microsoft.Storage/storageAccounts",
    },
    {
        "instruction": "Enforce encrypted transit for all storage account traffic",
        "expected_type": "Microsoft.Storage/storageAccounts",
    },
]


# ---------------------------------------------------------------------------
# Scoring weights — structural + semantic checks (total = 100 pts)
# ---------------------------------------------------------------------------

WEIGHTS = {
    "has_properties":              20,   # structural: properties envelope exists
    "non_empty_if":                25,   # structural: if block has real conditions
    "has_effect_parameter":        15,   # structural: effect param defined
    "then_effect_parameterized":   15,   # structural: then.effect uses parameter
    "condition_has_field":         15,   # semantic: if block contains a real field check
    "resource_type_correct":       10,   # semantic: policy targets the right resource type
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Structural checks
# ---------------------------------------------------------------------------

def _check_has_properties(policy: Any) -> Tuple[bool, Optional[str]]:
    if not isinstance(policy, dict) or "properties" not in policy:
        return False, "Top-level 'properties' key is missing"
    return True, None


def _check_non_empty_if(policy: Any) -> Tuple[bool, Optional[str]]:
    props = (policy or {}).get("properties", {})
    pr = (props or {}).get("policyRule", {})
    if_block = (pr or {}).get("if")
    if not isinstance(if_block, dict) or if_block == {}:
        return False, "policyRule.if is missing or empty"
    if isinstance(if_block.get("allOf"), list) and len(if_block["allOf"]) == 0:
        return False, "policyRule.if.allOf is an empty list"
    if isinstance(if_block.get("anyOf"), list) and len(if_block["anyOf"]) == 0:
        return False, "policyRule.if.anyOf is an empty list"
    return True, None


def _check_has_effect_parameter(policy: Any) -> Tuple[bool, Optional[str]]:
    params = (policy or {}).get("properties", {}).get("parameters", {})
    eff = (params or {}).get("effect")
    if not isinstance(eff, dict) or eff.get("type") != "String":
        return False, "parameters.effect is missing or has wrong type"
    return True, None


def _check_then_effect_parameterized(policy: Any) -> Tuple[bool, Optional[str]]:
    pr = (policy or {}).get("properties", {}).get("policyRule", {})
    then = (pr or {}).get("then", {})
    if then.get("effect") != "[parameters('effect')]":
        return False, f"then.effect is '{then.get('effect')}', expected \"[parameters('effect')]\""
    return True, None


# ---------------------------------------------------------------------------
# Semantic check 1 — if block contains a real field expression
# ---------------------------------------------------------------------------

_MAX_DEPTH = 20


def _collect_fields(obj: Any, found: List[str], _depth: int = 0) -> None:
    if _depth > _MAX_DEPTH:
        return
    if isinstance(obj, dict):
        if "field" in obj:
            found.append(obj["field"])
        for v in obj.values():
            _collect_fields(v, found, _depth + 1)
    elif isinstance(obj, list):
        for item in obj:
            _collect_fields(item, found, _depth + 1)


def _check_condition_has_field(policy: Any) -> Tuple[bool, Optional[str]]:
    pr = (policy or {}).get("properties", {}).get("policyRule", {})
    if_block = pr.get("if", {})

    fields: List[str] = []
    _collect_fields(if_block, fields)

    if not fields:
        return False, "policyRule.if contains no 'field' expressions at all"

    non_type_fields = [f for f in fields if f.lower() != "type"]
    if not non_type_fields:
        return False, (
            "policyRule.if only checks resource 'type' — no actual compliance "
            "condition (e.g. property value check) found"
        )

    return True, None


# ---------------------------------------------------------------------------
# Semantic check 2 — resource type matches instruction intent
# ---------------------------------------------------------------------------

KEYWORD_TO_TYPE: Dict[str, str] = {
    "storage account":               "Microsoft.Storage/storageAccounts",
    "storage accounts":              "Microsoft.Storage/storageAccounts",
    "blob storage":                  "Microsoft.Storage/storageAccounts",
    "app configuration":             "Microsoft.AppConfiguration/configurationStores",
    "appconfiguration":              "Microsoft.AppConfiguration/configurationStores",
    "key vault":                     "Microsoft.KeyVault/vaults",
    "keyvault":                      "Microsoft.KeyVault/vaults",
    "sql server":                    "Microsoft.Sql/servers",
    "sql servers":                   "Microsoft.Sql/servers",
    "azure sql":                     "Microsoft.Sql/servers",
    "virtual machine":               "Microsoft.Compute/virtualMachines",
    "virtual machines":              "Microsoft.Compute/virtualMachines",
    "web app":                       "Microsoft.Web/sites",
    "web apps":                      "Microsoft.Web/sites",
    "app service":                   "Microsoft.Web/sites",
    "function app":                  "Microsoft.Web/sites",
}


def _expected_type_from_instruction(instruction: str) -> Optional[str]:
    lower = instruction.lower()
    for keyword, rtype in KEYWORD_TO_TYPE.items():
        if keyword in lower:
            return rtype
    return None


def _extract_type_from_policy(policy: Any) -> Optional[str]:
    pr = (policy or {}).get("properties", {}).get("policyRule", {})
    if_block = pr.get("if", {})
    candidates: List[str] = []

    def _scan(obj: Any) -> None:
        if isinstance(obj, dict):
            if obj.get("field", "").lower() == "type" and "equals" in obj:
                candidates.append(obj["equals"])
            for v in obj.values():
                _scan(v)
        elif isinstance(obj, list):
            for item in obj:
                _scan(item)

    _scan(if_block)
    return candidates[0] if candidates else None


def _check_resource_type_correct(
    policy: Any, expected_type: Optional[str], instruction: str
) -> Tuple[bool, Optional[str]]:
    if expected_type is None:
        expected_type = _expected_type_from_instruction(instruction)
    if expected_type is None:
        return True, None

    actual_type = _extract_type_from_policy(policy)
    if actual_type is None:
        return False, (
            f"Expected resource type '{expected_type}' but policy has no "
            f"'field':'type' condition in policyRule.if"
        )
    if actual_type.lower() != expected_type.lower():
        return False, (
            f"Wrong resource type: got '{actual_type}', expected '{expected_type}'"
        )
    return True, None


# ---------------------------------------------------------------------------
# Field-Level F1 — measures condition accuracy against a reference if-block
#
# Each condition node is represented as a frozenset of (key, value) pairs.
# e.g. {"field": "type", "equals": "Microsoft.Storage/storageAccounts"}
#   -> frozenset({("field","type"), ("equals","Microsoft.Storage/storageAccounts")})
#
# Precision = matched / generated_conditions
# Recall    = matched / reference_conditions
# F1        = harmonic mean of precision and recall
# ---------------------------------------------------------------------------

def _extract_leaf_conditions(obj: Any, results: List[Set], _depth: int = 0) -> None:
    """Recursively extract leaf condition dicts (those containing 'field') as sets."""
    if _depth > _MAX_DEPTH:
        return
    if isinstance(obj, dict):
        if "field" in obj:
            # This is a leaf condition node — convert to frozenset for comparison
            results.append(frozenset((k, str(v)) for k, v in obj.items()))
        else:
            for v in obj.values():
                _extract_leaf_conditions(v, results, _depth + 1)
    elif isinstance(obj, list):
        for item in obj:
            _extract_leaf_conditions(item, results, _depth + 1)


def field_f1(generated_policy: Any, reference_if: Any) -> Optional[float]:
    """
    Compute Field-Level F1 between a generated policy and a reference if-block.
    Returns None if either side has no extractable conditions.
    """
    if not isinstance(generated_policy, dict) or not isinstance(reference_if, dict):
        return None

    gen_if = (generated_policy.get("properties", {})
              .get("policyRule", {})
              .get("if", {}))

    gen_conditions: List = []
    ref_conditions: List = []

    _extract_leaf_conditions(gen_if, gen_conditions)
    _extract_leaf_conditions(reference_if, ref_conditions)

    if not gen_conditions or not ref_conditions:
        return None

    gen_set = set(gen_conditions)
    ref_set = set(ref_conditions)

    matched = len(gen_set & ref_set)
    precision = matched / len(gen_set)
    recall    = matched / len(ref_set)

    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 3)


# ---------------------------------------------------------------------------
# Master scorer
# ---------------------------------------------------------------------------

def score_one(
    result: Dict[str, Any],
    expected_type: Optional[str] = None,
    instruction: str = "",
    policy_key: str = "fixed_policy",
) -> Dict[str, Any]:
    """
    Score a policy from an API response.
    policy_key: 'fixed_policy' for pipeline output, 'raw_policy' for model-only output.
    """
    policy = result.get(policy_key)
    if policy is None and policy_key == "fixed_policy":
        policy = result.get("policy")  # backward compat

    checks: Dict[str, Dict] = {}
    issues: List[str] = []
    total_score = 0

    if not isinstance(policy, dict):
        return {
            "score": 0,
            "passed": False,
            "checks": {"missing_policy": {"passed": False, "detail": f"No '{policy_key}' object in response"}},
            "issues": ["missing_policy"],
        }

    check_fns = [
        ("has_properties",            lambda: _check_has_properties(policy)),
        ("non_empty_if",              lambda: _check_non_empty_if(policy)),
        ("has_effect_parameter",      lambda: _check_has_effect_parameter(policy)),
        ("then_effect_parameterized", lambda: _check_then_effect_parameterized(policy)),
        ("condition_has_field",       lambda: _check_condition_has_field(policy)),
        ("resource_type_correct",     lambda: _check_resource_type_correct(policy, expected_type, instruction)),
    ]

    for name, fn in check_fns:
        ok, detail = fn()
        checks[name] = {"passed": ok, "detail": detail or "ok"}
        if ok:
            total_score += WEIGHTS[name]
        else:
            issues.append(name)

    return {
        "score": total_score,
        "passed": total_score == 100,
        "checks": checks,
        "issues": issues,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", required=True, help="Base API URL (ngrok)")
    parser.add_argument("--out", default="eval_results.jsonl")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--tests", nargs="*", default=None,
                        help="Override test instructions (plain strings, no expected_type)")
    args = parser.parse_args()

    base = _normalize_base(args.api)
    print("API:", base)
    print("Checking health...")
    try:
        health = _get_json(f"{base}/health")
        print("Health:", json.dumps(health, indent=2))
        if not health.get("model_loaded"):
            print("ERROR: Model not loaded. Run the notebook model-load cell first.")
            sys.exit(1)
    except SystemExit:
        raise
    except Exception as e:
        print(f"ERROR: Health check failed: {e}")
        sys.exit(1)

    if args.tests is not None:
        tests = [{"instruction": t, "expected_type": None} for t in args.tests]
    else:
        tests = DEFAULT_TESTS

    results = []
    latencies = []

    with open(args.out, "w", encoding="utf-8") as f:
        for idx, test in enumerate(tests, 1):
            instruction   = test["instruction"]
            expected_type = test.get("expected_type")
            reference_if  = test.get("reference_if")

            started = time.time()
            try:
                result = _post_json(f"{base}/generate", {"instruction": instruction}, timeout_s=args.timeout)
            except Exception as e:
                rec = {"instruction": instruction, "error": str(e)}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(f"[{idx}] ERROR: {instruction} → {e}")
                continue

            elapsed = round(time.time() - started, 2)
            latencies.append(elapsed)

            # --- Pipeline score (after check_and_fix) ---
            pipeline_scored = score_one(
                result, expected_type=expected_type, instruction=instruction,
                policy_key="fixed_policy",
            )

            # --- Model-only score (raw model output, before post-processing) ---
            model_scored = score_one(
                result, expected_type=expected_type, instruction=instruction,
                policy_key="raw_policy",
            )

            # --- Field-Level F1 (pipeline output vs reference) ---
            f1_score = None
            if reference_if is not None:
                f1_score = field_f1(
                    result.get("fixed_policy") or result.get("policy"),
                    reference_if,
                )

            rec = {
                "instruction":       instruction,
                # Pipeline metrics
                "pipeline_score":    pipeline_scored["score"],
                "pipeline_passed":   pipeline_scored["passed"],
                "pipeline_checks":   pipeline_scored["checks"],
                "pipeline_issues":   pipeline_scored["issues"],
                # Model-only metrics (before post-processing)
                "model_score":       model_scored["score"],
                "model_passed":      model_scored["passed"],
                "model_issues":      model_scored["issues"],
                # Field-Level F1
                "field_f1":          f1_score,
                # Metadata
                "retry":             result.get("retry"),
                "meta":              result.get("meta"),
                "elapsed_s":         elapsed,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            results.append(rec)

            p_status = "PASS" if pipeline_scored["passed"] else f"PARTIAL({pipeline_scored['score']}/100)"
            m_status = "PASS" if model_scored["passed"] else f"FAIL({model_scored['score']}/100)"
            f1_str   = f"  F1={f1_score:.3f}" if f1_score is not None else ""

            if pipeline_scored["issues"]:
                failed_details = "; ".join(
                    f"{c}: {pipeline_scored['checks'][c]['detail']}"
                    for c in pipeline_scored["issues"]
                )
                print(f"[{idx}] pipeline={p_status}  model={m_status}{f1_str} ({elapsed}s)")
                print(f"       {instruction}")
                print(f"       ↳ {failed_details}")
            else:
                print(f"[{idx}] pipeline={p_status}  model={m_status}{f1_str} ({elapsed}s)  {instruction}")

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------
    total              = len(results)
    pipeline_full_pass = sum(1 for r in results if r["pipeline_passed"])
    model_full_pass    = sum(1 for r in results if r["model_passed"])
    avg_pipeline_score = round(sum(r["pipeline_score"] for r in results) / total, 1) if total else 0
    avg_model_score    = round(sum(r["model_score"]    for r in results) / total, 1) if total else 0
    retry_count        = sum(1 for r in results if r.get("retry"))
    fallback_count     = sum(1 for r in results if (r.get("meta") or {}).get("fallback_used"))
    avg_lat            = round(sum(latencies) / len(latencies), 2) if latencies else 0
    min_lat            = min(latencies) if latencies else 0
    max_lat            = max(latencies) if latencies else 0

    # Average F1 over tests that had a reference
    f1_scores = [r["field_f1"] for r in results if r["field_f1"] is not None]
    avg_f1    = round(sum(f1_scores) / len(f1_scores), 3) if f1_scores else None

    # Per-check pass rate (pipeline)
    check_pass: Dict[str, int] = {k: 0 for k in WEIGHTS}
    for r in results:
        for check, detail in r.get("pipeline_checks", {}).items():
            if detail.get("passed"):
                check_pass[check] = check_pass.get(check, 0) + 1

    print(f"""
=== SUMMARY ===
                        Pipeline    Model-only
Full pass (100/100):    {pipeline_full_pass}/{total}          {model_full_pass}/{total}
Average score:          {avg_pipeline_score}/100       {avg_model_score}/100
---
Field-Level F1 (avg):   {avg_f1 if avg_f1 is not None else "n/a"} (over {len(f1_scores)} tests with reference)
Retry triggered:        {retry_count}/{total}
Fallback used:          {fallback_count}/{total}
Latency:                avg={avg_lat}s  min={min_lat}s  max={max_lat}s

Per-check pass rate (pipeline):""")
    for check, weight in WEIGHTS.items():
        n = check_pass.get(check, 0)
        print(f"  {check:<30} {n}/{total}  (worth {weight} pts)")

    print(f"\nResults written to: {args.out}")


if __name__ == "__main__":
    main()
