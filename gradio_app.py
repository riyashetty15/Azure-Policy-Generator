import os
import json
from typing import Any, Dict, Optional, Tuple

import gradio as gr
import requests


def _post_generate(api_base_url: str, instruction: str, timeout_s: int = 180) -> Tuple[str, str]:
    api_base_url = (api_base_url or "").strip()
    instruction = (instruction or "").strip()

    if not api_base_url:
        return "", "ERROR: Missing Colab API URL"
    if not instruction:
        return "", "ERROR: Instruction is empty"

    # Be forgiving if the user pastes the full endpoint URL.
    api_base_url = api_base_url.rstrip("/")
    if api_base_url.endswith("/generate"):
        api_base_url = api_base_url[: -len("/generate")]

    # Optional health check (helps catch "model not loaded" early).
    health_url = f"{api_base_url}/health"
    try:
        h = requests.get(health_url, timeout=8)
        if h.status_code == 200:
            health = h.json()
            if isinstance(health, dict) and health.get("model_loaded") is False:
                return "", "ERROR: Colab API is up but model is not loaded.\nRun the notebook model-load cell, then restart the API cell."
    except Exception:
        # Ignore health errors; server may not expose /health.
        pass

    url = f"{api_base_url}/generate"
    try:
        resp = requests.post(url, json={"instruction": instruction}, timeout=timeout_s)
    except requests.RequestException as e:
        return "", f"ERROR: Request failed: {e}"

    if resp.status_code != 200:
        content_type = resp.headers.get("content-type", "")
        body = resp.text or ""
        # ngrok offline pages are HTML and can be very noisy; detect and provide a helpful hint.
        if "text/html" in content_type and ("ERR_NGROK_" in body or "ngrok" in body.lower()):
            hint = (
                "ERROR: Colab/ngrok endpoint is offline.\n\n"
                "Most common causes:\n"
                "- Your Colab runtime disconnected or the API cell stopped\n"
                "- You restarted Colab and ngrok gave you a NEW public URL\n\n"
                "Fix: Re-run the Colab API cell, copy the NEW PUBLIC API URL, and paste it here."
            )
            return "", hint
        return "", f"ERROR: {resp.status_code} {body[:2000]}"

    try:
        payload: Dict[str, Any] = resp.json()
    except Exception as e:
        return "", f"ERROR: Non-JSON response from server: {e}\n\nRaw:\n{resp.text[:2000]}"

    # Backward/forward compatible keys
    fixed_policy = payload.get("fixed_policy")
    if fixed_policy is None:
        fixed_policy = payload.get("policy")

    raw = payload.get("raw_output")
    meta = payload.get("meta")
    retry = payload.get("retry")

    pretty_policy = json.dumps(fixed_policy, indent=2, ensure_ascii=False) if isinstance(fixed_policy, (dict, list)) else str(fixed_policy)

    header_lines = []
    if retry is not None:
        header_lines.append(f"retry: {retry}")
    if isinstance(meta, dict):
        # Only surface the most useful WIP signals.
        if "fallback_used" in meta:
            header_lines.append(f"fallback_used: {meta.get('fallback_used')}")
        if "empty_if_before_fix" in meta:
            header_lines.append(f"empty_if_before_fix: {meta.get('empty_if_before_fix')}")

    header = ("\n".join(header_lines) + "\n\n") if header_lines else ""
    raw_text = header + (raw if isinstance(raw, str) else "")
    return pretty_policy, raw_text


def build_ui() -> gr.Blocks:
    default_api = os.getenv("COLAB_API_URL", "")

    with gr.Blocks() as demo:
        gr.Markdown("# Azure Policy Generator (Local UI)\nThis UI calls your Colab GPU runtime via HTTP.")

        api_url = gr.Textbox(
            label="Colab API URL",
            placeholder="https://<your-ngrok-or-cloudflared-url>",
            value=default_api,
        )

        instruction = gr.Textbox(
            label="Instruction",
            placeholder="Example: Disallow public IPs on storage accounts",
            lines=3,
        )

        generate_btn = gr.Button("Generate")

        policy_json = gr.Code(label="Validated Policy JSON", language="json")
        raw_output = gr.Textbox(label="Raw Model Output", lines=10)

        generate_btn.click(
            fn=_post_generate,
            inputs=[api_url, instruction],
            outputs=[policy_json, raw_output],
        )

    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="127.0.0.1", server_port=7860)
