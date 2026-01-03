# Azure Policy Generator (QLoRA + Colab API + Local UI)

This repo contains a workflow to fine-tune an instruction-following LLM to generate **valid Azure Policy JSON**, then serve it from a **Colab GPU runtime** (FastAPI + ngrok) and consume it locally via a **Gradio UI** and a **local evaluation runner**.

## Architecture (end-to-end)

**Training / data prep (Colab notebook):**
- Source dataset: `azure_policy_dataset_clean.json` (JSONL: one JSON object per line).
- Conversion outputs:
  - `azure_policies_ft_supervised.jsonl` (HF SFT prompt/completion format)
  - `azure_policies_ft_chat.jsonl` (OpenAI-style chat messages format)
- Fine-tuning method: **QLoRA** (4-bit quantized base model + LoRA adapter weights).

**Serving (Colab notebook):**
- Loads base model `Qwen/Qwen2.5-7B-Instruct`.
- Loads LoRA adapter from Google Drive folder: `azure-policy-qlora-adapter/`.
- Exposes HTTP endpoints:
  - `GET /health` (sanity check that model/tokenizer are loaded)
  - `POST /generate` (generate raw output, then normalize/validate it)
- Uses **ngrok** to publish the Colab runtime to a public HTTPS URL.

**Local consumption (your laptop):**
- `gradio_app.py`: simple UI calling Colab `/generate`.
- `evaluate_api.py`: CLI runner that calls `/health`, then runs a set of test prompts and scores responses.

## Repo layout

- `Azure Policy AI.ipynb`: primary notebook (data prep, training, validation helpers, model load, API server).
- `azure_policy_dataset_clean.json`: JSONL dataset with `{instruction, target}`.
- `azure_policies_ft_supervised.jsonl`: HF supervised fine-tuning file generated from the dataset.
- `azure_policies_ft_chat.jsonl`: chat-format file generated from the dataset.
- `azure-policy-qlora-adapter/`: saved LoRA adapter + tokenizer artifacts (what you ship/use for inference).
- `azure-policy-qlora/`: training outputs/checkpoints from QLoRA runs.
- `gradio_app.py`: local UI.
- `evaluate_api.py`: evaluation harness.
- `infer.py`: legacy FLAN-T5 inference script (not part of the Qwen+QLoRA+API path).

## Data format

`azure_policy_dataset_clean.json` is **JSONL** (one JSON object per line). Each row looks like:

```json
{"instruction": "...", "target": "{\"properties\": {...}}"}
```

Notes:
- `target` is a JSON *string* containing the policy JSON.
- The notebook converts this into trainable JSONL formats.

## Fine-tuning: QLoRA details

The QLoRA approach used in `Azure Policy AI.ipynb` is:

- **Base model:** `Qwen/Qwen2.5-7B-Instruct`
- **Quantization:** bitsandbytes 4-bit (`BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, ...)`)
- **Adapter method:** PEFT LoRA (`LoraConfig(...)`, `get_peft_model(...)`)
- **Result artifact:** the adapter folder (`azure-policy-qlora-adapter/`) containing:
  - `adapter_model.safetensors`
  - `adapter_config.json`
  - tokenizer files

What this means:
- You do **not** save a full fine-tuned 7B model.
- You save **small adapter weights**, and at inference time you load:
  1) the base Qwen model (quantized), then
  2) apply the LoRA adapter (`PeftModel.from_pretrained(...)`).

## Inference + validation strategy

The API returns both raw and “fixed” outputs:

- The model is prompted to return **ONLY one JSON object**.
- The response is parsed with `extract_last_json(...)` to recover the last complete JSON object.
- `check_and_fix(...)` normalizes the object into an Azure Policy envelope and applies guardrails:
  - Ensures `properties.displayName`, `properties.description`, `properties.parameters`, `properties.policyRule` exist.
  - Normalizes `then.effect` to `"[parameters('effect')]"`.
  - Drops unexpected root-level keys (e.g. a stray top-level `allOf` from the model).
  - Detects empty `policyRule.if` (e.g. `{ "allOf": [] }`).
  - If needed, retries once with strict feedback.
  - If still empty, may apply a small heuristic fallback `if` for common instructions.

This is intentionally a “WIP but usable” path: it prioritizes valid, non-empty policies while you iterate on model quality.

## How to run (recommended)

### 1) Start the Colab server (GPU)

In `Azure Policy AI.ipynb`, run in order:

1. **Mount Drive** (so the adapter path resolves).
2. **Setup/Imports**.
3. **Validation helpers** (the cell with `extract_last_json`, `check_and_fix`, etc.).
4. **Load base + adapter for inference** (creates `tok` and `model`).
5. **API cell** (FastAPI + ngrok). It prints a public URL.

Make sure you set the secret in Colab:
- `NGROK_AUTH_TOKEN` (Colab: Runtime → Secrets)

You should be able to hit:
- `<public_url>/health`

### 2) Run the local Gradio UI

From your machine:

```bash
cd "/Users/{local_location}"
python3 -m pip install gradio requests
python3 gradio_app.py
```

Paste the **public ngrok URL** (base URL, not `/generate`) into the UI.

Optional: set an env var to prefill the URL:

```bash
export COLAB_API_URL="https://<your-ngrok>.ngrok-free.app"
python3 gradio_app.py
```

### 3) Run the evaluation harness

```bash
cd "/Users/{local_location}"
python3 -m pip install requests
python3 evaluate_api.py --api "https://<your-ngrok>.ngrok-free.app"
```

Useful flags:
- Custom output file:
  - `python3 evaluate_api.py --api "https://..." --out results.jsonl`
- Longer timeout:
  - `python3 evaluate_api.py --api "https://..." --timeout 300`
- Custom tests:
  - `python3 evaluate_api.py --api "https://..." --tests "Disallow public network access on storage accounts" "Require tag owner on all resources"`

## Troubleshooting

### zsh: unknown file attribute: h

This happens if you paste a markdown link into the terminal, like:

- `python [evaluate_api.py](http://...) --api https://...`

Run it without the brackets/parentheses:

- `python evaluate_api.py --api "https://..."`

### ngrok URL “offline” / HTML error

ngrok URLs change when the Colab runtime restarts.

Fix:
1. Re-run the API cell in Colab.
2. Copy the newly printed **PUBLIC API URL**.
3. Paste that new URL into the UI / evaluator.

### /health says model_loaded: false

You started the API before loading the model.

Fix:
1. Run the model-load cell (the one that creates `tok` and `model`).
2. Restart/re-run the API cell.

## Notes / limitations

- The dataset includes some inconsistent policy fragments (e.g., booleans as strings). Cleaning those upstream can improve training quality.
- `infer.py` is a separate/legacy approach (FLAN-T5) and does not reflect the Qwen + QLoRA inference path.
