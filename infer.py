# infer.py
#
# DEPRECATED: This script uses FLAN-T5 (Seq2Seq) and is NOT part of the
# current Qwen 2.5 + QLoRA inference path. It is kept for reference only.
# Use the Colab notebook API (FastAPI + /generate) or gradio_app.py instead.

import json
import sys
import re

try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
except ImportError:
    print("ERROR: transformers is not installed. Run: pip install transformers")
    sys.exit(1)

# Path to the fine-tuned model you saved in train.py
MODEL_DIR = "./finetuned-flan-t5-azure-policy"

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
except OSError as e:
    print(
        f"ERROR: Could not load model from '{MODEL_DIR}'.\n"
        f"  {e}\n\n"
        f"NOTE: This script is DEPRECATED. The current project uses\n"
        f"Qwen 2.5 7B + QLoRA served via the Colab notebook API.\n"
        f"See README.md for the recommended workflow."
    )
    sys.exit(1)


def generate_policy(instruction: str, max_new_tokens=512, num_beams=4):
    """Generate Azure policy JSON from natural language instruction"""
    inputs = tokenizer(instruction, return_tensors="pt", truncation=True, max_length=256)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=num_beams)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text


def extract_json(text: str):
    # Ensure it starts/ends with curly braces
    if not text.strip().startswith("{"):
        text = "{" + text
    if not text.strip().endswith("}"):
        text = text + "}"

    # Try to insert missing quotes around keys
    text = re.sub(r'(\w+):', r'"\1":', text)

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print("JSON parsing failed:", e)
        return None


if __name__ == "__main__":
    test_instruction = "Disallow public IPs on storage accounts"
    gen = generate_policy(test_instruction)
    print("\n=== Raw model output ===\n", gen)

    maybe_json = extract_json(gen)
    if maybe_json is not None:
        print("\n=== Parsed JSON (pretty) ===")
        print(json.dumps(maybe_json, indent=2))
    else:
        print("\n(Output was not valid JSON. Consider adding post-processing or refining training.)")
