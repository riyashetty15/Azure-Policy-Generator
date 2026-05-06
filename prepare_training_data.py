import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


BOOL_STRINGS = {
    "true": True,
    "false": False,
}


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _normalize_obj(obj: Any) -> Any:
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            nk = "anyOf" if k == "anyof" else k
            out[nk] = _normalize_obj(v)
        return out
    if isinstance(obj, list):
        return [_normalize_obj(v) for v in obj]
    if isinstance(obj, str):
        s = obj.strip()
        low = s.lower()
        if low in BOOL_STRINGS:
            return BOOL_STRINGS[low]
        return s
    return obj


def _parse_target(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        obj = raw
    elif isinstance(raw, str):
        obj = json.loads(raw)
    else:
        raise ValueError("Unsupported target format")

    obj = _normalize_obj(obj)
    if "properties" not in obj and isinstance(obj, dict):
        obj = {
            "properties": {
                "displayName": "Generated Azure Policy",
                "description": "Generated Azure Policy",
                "parameters": {},
                "policyRule": obj.get("policyRule", obj),
                "mode": obj.get("mode", "All"),
            }
        }
    props = obj.setdefault("properties", {})
    props.setdefault("displayName", "Generated Azure Policy")
    props.setdefault("description", props.get("displayName", "Generated Azure Policy"))
    props.setdefault("parameters", {})
    props.setdefault("policyRule", {"if": {"allOf": []}, "then": {"effect": "[parameters('effect')]"}})
    props.setdefault("mode", "All")
    return obj


def clean_pairs(rows: Iterable[Dict[str, Any]]) -> List[Tuple[str, str]]:
    cleaned: List[Tuple[str, str]] = []
    for row in rows:
        instruction = (row.get("instruction") or "").strip()
        if not instruction:
            continue

        raw_target = row.get("target")
        if raw_target is None:
            raw_target = row.get("output")
        if raw_target is None:
            continue

        try:
            target_obj = _parse_target(raw_target)
        except Exception:
            continue

        target_obj["properties"]["displayName"] = instruction
        target_obj["properties"]["description"] = instruction

        cleaned.append((instruction, json.dumps(target_obj, ensure_ascii=False, separators=(",", ":"))))
    return cleaned


def _paraphrase_templates(instruction: str) -> List[str]:
    core = instruction.strip().rstrip(".")
    core_lower = core[:1].lower() + core[1:] if core else core
    variants = [
        core,
        f"Create an Azure Policy to {core_lower}",
        f"Write a policy definition that ensures: {core}",
        f"Generate a compliant Azure Policy for this requirement: {core}",
        f"Configure Azure Policy to {core_lower}",
        f"Define a policy rule that enforces: {core}",
        f"Add an Azure Policy to {core_lower}",
        f"Create a policy definition to {core_lower}",
    ]

    replacement_rules = [
        (r"\bshould\b", "must"),
        (r"\bshould\b", "needs to"),
        (r"\bdisallow\b", "block"),
        (r"\bdisallow\b", "prevent"),
        (r"\brequire\b", "enforce"),
        (r"\brequire\b", "mandate"),
        (r"\bdisable\b", "turn off"),
        (r"\bdisable\b", "deactivate"),
        (r"\baudit\b", "monitor"),
        (r"\baudit\b", "verify"),
        (r"\brestrict\b", "limit"),
        (r"\bblock\b", "prevent"),
        (r"\benforce\b", "mandate"),
    ]
    for pattern, repl in replacement_rules:
        if re.search(pattern, core, flags=re.IGNORECASE):
            variants.append(re.sub(pattern, repl, core, flags=re.IGNORECASE))

    dedup: List[str] = []
    seen = set()
    for v in variants:
        norm = v.strip()
        if not norm:
            continue
        key = norm.lower()
        if key in seen:
            continue
        seen.add(key)
        dedup.append(norm)
    return dedup


def augment_pairs(pairs: List[Tuple[str, str]], max_variants_per_sample: int, seed: int) -> List[Tuple[str, str]]:
    rng = random.Random(seed)
    augmented: List[Tuple[str, str]] = []
    seen = set()

    for instruction, target in pairs:
        variants = _paraphrase_templates(instruction)
        if len(variants) > max_variants_per_sample:
            # Keep original instruction, sample remaining variants.
            sampled = [variants[0]] + rng.sample(variants[1:], k=max_variants_per_sample - 1)
            variants = sampled

        for variant in variants:
            key = (variant.lower(), target)
            if key in seen:
                continue
            seen.add(key)
            augmented.append((variant, target))
    return augmented


def write_outputs(pairs: List[Tuple[str, str]], out_base: Path) -> None:
    out_train = out_base / "train_augmented.jsonl"
    out_supervised = out_base / "azure_policies_ft_supervised_augmented.jsonl"
    out_chat = out_base / "azure_policies_ft_chat_augmented.jsonl"

    with out_train.open("w", encoding="utf-8") as f_train, out_supervised.open("w", encoding="utf-8") as f_sup, out_chat.open("w", encoding="utf-8") as f_chat:
        for instruction, target in pairs:
            f_train.write(json.dumps({"instruction": instruction, "target": target}, ensure_ascii=False) + "\n")

            prompt = (
                "You are an assistant that generates Azure Policy JSON.\\n"
                f"Instruction: {instruction}\\n"
                "Return ONLY a single JSON object with `properties` containing displayName, "
                "description, parameters, and policyRule (if/then)."
            )
            f_sup.write(json.dumps({"prompt": prompt, "completion": target}, ensure_ascii=False) + "\n")

            chat = {
                "messages": [
                    {"role": "system", "content": "You are an assistant that generates valid Azure Policy JSON. Return ONLY JSON."},
                    {
                        "role": "user",
                        "content": (
                            f"Instruction: {instruction}\\n"
                            "Return a full Azure Policy object with properties.displayName, properties.description, "
                            "properties.parameters (define any referenced parameters), and properties.policyRule (with if/then)."
                        ),
                    },
                    {"role": "assistant", "content": target},
                ]
            }
            f_chat.write(json.dumps(chat, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean and augment Azure Policy training data.")
    parser.add_argument("--input", default="azure_policy_dataset_clean.json", help="Input JSONL file")
    parser.add_argument("--out-dir", default=".", help="Output directory")
    parser.add_argument("--max-variants", type=int, default=3, help="Max instruction variants per sample (including original)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(input_path)
    cleaned = clean_pairs(rows)
    augmented = augment_pairs(cleaned, max_variants_per_sample=max(1, args.max_variants), seed=args.seed)
    write_outputs(augmented, out_dir)

    print(f"Loaded samples: {len(rows)}")
    print(f"Clean samples: {len(cleaned)}")
    print(f"Augmented samples: {len(augmented)}")
    print(f"Wrote: {out_dir / 'train_augmented.jsonl'}")
    print(f"Wrote: {out_dir / 'azure_policies_ft_supervised_augmented.jsonl'}")
    print(f"Wrote: {out_dir / 'azure_policies_ft_chat_augmented.jsonl'}")


if __name__ == "__main__":
    main()