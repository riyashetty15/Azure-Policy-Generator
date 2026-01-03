# infer.py
import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re

# Path to the fine-tuned model you saved in train.py
MODEL_DIR = "./finetuned-flan-t5-azure-policy"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)

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
    except Exception as e:
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
