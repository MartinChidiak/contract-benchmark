"""
run_experiment.py
-----------------
Replaces main.py. Receives a RunConfig and executes one full
inference pass, saving results to config.output_dir.

Usage:
    python run_experiment.py --config experiments/my_config.json

Or import and call directly from run_all.py.

Changelog vs original:
- build_prompt() now returns a list[dict] (messages) instead of a
  hardcoded Llama-3 special-token string.
- apply_chat_template() converts messages → final string using the
  model's own tokenizer. Falls back to a minimal generic format if
  the tokenizer has no chat_template defined, with an explicit warning.
- load_llm_with_fallback() now also loads and returns the tokenizer,
  which is passed through to build_prompt callers.
- experiment_config.py: optional 'chat_template_override' field added
  to RunConfig for fully custom templates (e.g. future models).
"""

from __future__ import annotations
import argparse
import json
import os
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

from experiment_config import RunConfig, PROMPT_VERSIONS

# vLLM imports — only imported if actually running inference
try:
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from schema import RealEstateContract


# ---------------------------------------------------------------------------
# Constants (non-configurable)
# ---------------------------------------------------------------------------
TOKEN_FILE = "token.txt"

NULLABLE_FIELDS = {
    "agreement_date", "effective_date", "expiration_date",
    "governing_law", "renewal_term", "notice_period_to_terminate_renewal",
}
YES_NO_FIELDS = {
    "audit_rights", "cap_on_liability",
    "termination_for_convenience", "liquidated_damages",
}
ANTI_ASSIGNMENT_MAP = {
    "prohibited": "Prohibited",
    "allowed with consent": "Allowed with consent",
    "not mentioned": "Not Mentioned",
}
YES_NO_MAP = {"yes": "Yes", "no": "No"}

# Few-shot examples (same as original main.py — kept here for single source of truth)
FEW_SHOT_EXAMPLES = """
--- EXAMPLE 1 ---
CONTRACT FRAGMENT:
"Unless either party gives written notice of termination of this Agreement at least 60 days
prior to the end of the Initial Term, or any successive three-year term, this Agreement shall
automatically renew for successive additional three-year terms."

CORRECT JSON OUTPUT:
{
  "parties": null,
  "agreement_date": null,
  "effective_date": null,
  "expiration_date": null,
  "governing_law": null,
  "anti_assignment": "Not Mentioned",
  "renewal_term": "3 years",
  "notice_period_to_terminate_renewal": "60 days",
  "audit_rights": "No",
  "cap_on_liability": "No",
  "termination_for_convenience": "No",
  "liquidated_damages": "No"
}

--- EXAMPLE 2 ---
CONTRACT FRAGMENT:
"This agreement shall remain in effect until the end of the current calendar year and shall be
automatically renewed for successive one (1) year periods unless otherwise terminated.
This Agreement may be terminated by either party upon thirty (30) days written notice."

CORRECT JSON OUTPUT:
{
  "parties": null,
  "agreement_date": null,
  "effective_date": null,
  "expiration_date": null,
  "governing_law": null,
  "anti_assignment": "Not Mentioned",
  "renewal_term": "1 year",
  "notice_period_to_terminate_renewal": "30 days",
  "audit_rights": "No",
  "cap_on_liability": "No",
  "termination_for_convenience": "No",
  "liquidated_damages": "No"
}

--- EXAMPLE 3 ---
CONTRACT FRAGMENT:
"Neither party may assign this Agreement without the prior written consent of the other party.
This Agreement shall be governed by the laws of the State of California."

CORRECT JSON OUTPUT:
{
  "parties": null,
  "agreement_date": null,
  "effective_date": null,
  "expiration_date": null,
  "governing_law": "California",
  "anti_assignment": "Allowed with consent",
  "renewal_term": null,
  "notice_period_to_terminate_renewal": null,
  "audit_rights": "No",
  "cap_on_liability": "No",
  "termination_for_convenience": "No",
  "liquidated_damages": "No"
}

--- EXAMPLE 4 ---
CONTRACT FRAGMENT:
"Either party may terminate this Agreement at any time upon thirty (30) days prior written notice.
In no event shall either party's total aggregate liability exceed the total fees paid during the
twelve (12) months immediately preceding the claim.
This Agreement shall be governed by the laws of the State of New York."

CORRECT JSON OUTPUT:
{
  "parties": null,
  "agreement_date": null,
  "effective_date": null,
  "expiration_date": null,
  "governing_law": "New York",
  "anti_assignment": "Not Mentioned",
  "renewal_term": null,
  "notice_period_to_terminate_renewal": null,
  "audit_rights": "No",
  "cap_on_liability": "Yes",
  "termination_for_convenience": "Yes",
  "liquidated_damages": "No"
}
"""


# ---------------------------------------------------------------------------
# Chat template logic
# ---------------------------------------------------------------------------

def build_messages(config: RunConfig, text: str, chunk_info: str = "") -> list[dict]:
    """
    Returns a messages list in the standard OpenAI/HuggingFace format:
        [{"role": "system", "content": "..."},
         {"role": "user",   "content": "..."}]

    This format is model-agnostic. The actual special tokens
    (<|im_start|>, <|begin_of_text|>, etc.) are applied later by
    apply_chat_template(), which uses the tokenizer of each specific model.
    """
    context_note = (
        f"\nNOTE: You are analyzing {chunk_info} of the full contract. "
        "Extract only what is present in this fragment."
        if chunk_info else ""
    )
    few_shot_block = (
        f"\nThe following are examples of correct extractions from real contract fragments:\n{FEW_SHOT_EXAMPLES}"
        if config.use_few_shot else ""
    )

    system_content = f"{config.system_prompt}{context_note}{few_shot_block}"
    user_content = f"Now analyze this contract and return the JSON:\n{text}"

    return [
        {"role": "system", "content": system_content},
        {"role": "user",   "content": user_content},
    ]


def apply_chat_template(
    messages: list[dict],
    tokenizer,
    config: RunConfig,
) -> str:
    """
    Converts a messages list to the final prompt string using the
    model's own tokenizer.

    Priority order:
    1. config.chat_template_override — fully custom Jinja2 template string
       defined in RunConfig (for exotic models not covered by HuggingFace).
    2. tokenizer.chat_template — the model's native template from its
       tokenizer_config.json on HuggingFace. Covers Llama-3, Qwen2.5,
       Qwen3, Mistral, Gemma, Phi, etc. automatically.
    3. Generic fallback — a minimal System/User/Assistant format that works
       reasonably well with any instruction-tuned model. Emits a warning
       so you know the fallback was triggered.

    The add_generation_prompt=True argument appends the assistant turn
    opener (e.g. "<|start_header_id|>assistant<|end_header_id|>\n\n" for
    Llama, or "<|im_start|>assistant\n" for Qwen) so the model continues
    from there.
    """
    # 1. Custom override defined in RunConfig
    if config.chat_template_override:
        print(f"   📋 Using custom chat_template_override from RunConfig")
        return tokenizer.apply_chat_template(
            messages,
            chat_template=config.chat_template_override,
            tokenize=False,
            add_generation_prompt=True,
        )

    # 2. Native template from the tokenizer
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    # 3. Generic fallback — warn explicitly
    print(
        f"   ⚠️  WARNING: No chat_template found for {config.model_id}. "
        "Using generic System/User/Assistant fallback. "
        "Consider adding a chat_template_override to RunConfig for this model."
    )
    system = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
    user   = next((m["content"] for m in messages if m["role"] == "user"), "")
    return (
        f"System: {system}\n\n"
        f"User: {user}\n\n"
        f"Assistant:"
    )


# ---------------------------------------------------------------------------
# Utility functions (unchanged from original)
# ---------------------------------------------------------------------------

def load_hf_token(token_path: str = TOKEN_FILE) -> str | None:
    """
    Loads the HuggingFace token if present.
    - Gated models (Meta/Llama): require token in HF_TOKEN env var.
    - Public models (Qwen): token ignored if present, no failure if absent.
    Returns the token string or None.
    """
    if not os.path.exists(token_path):
        print(f"⚠️  {token_path} not found — only public models available.")
        return None

    with open(token_path) as f:
        token = f.read().strip()

    if not token:
        print(f"⚠️  {token_path} is empty — only public models available.")
        return None

    os.environ["HF_TOKEN"] = token
    print(f"✅ Token loaded from {token_path}")
    return token


def estimate_char_limit(max_tokens: int, lang_factor: float) -> int:
    return int(max_tokens * lang_factor)


def compute_lang_factor(tokenizer, sample_texts: list[str], max_sample_chars: int = 2000) -> float:
    """Estimate chars-per-token ratio from sample texts using the loaded tokenizer.
    Falls back to 3.5 if tokenization fails or no samples are provided."""
    total_chars = total_tokens = 0
    for text in sample_texts:
        sample = text[:max_sample_chars]
        try:
            tokens = tokenizer.encode(sample, add_special_tokens=False)
            total_chars += len(sample)
            total_tokens += len(tokens)
        except Exception:
            continue
    if total_tokens == 0:
        return 3.5
    return total_chars / total_tokens


def split_into_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def normalize_extracted_data(data: dict) -> dict:
    normalized = dict(data)
    for key, value in list(normalized.items()):
        if isinstance(value, str):
            value = value.strip()
        if key in NULLABLE_FIELDS:
            if value in {"", "null", "NULL", "None", "none", "N/A", "n/a"}:
                normalized[key] = None
                continue
        if key == "anti_assignment" and isinstance(value, str):
            mapped = ANTI_ASSIGNMENT_MAP.get(value.strip().lower())
            if mapped is not None:
                normalized[key] = mapped
                continue
        if key in YES_NO_FIELDS and isinstance(value, str):
            mapped = YES_NO_MAP.get(value.strip().lower())
            if mapped is not None:
                normalized[key] = mapped
                continue
        normalized[key] = value

    if normalized.get("parties") in {None, "", "null", "NULL", "None", "none", "N/A", "n/a"}:
        normalized["parties"] = "Not Mentioned"
    if normalized.get("anti_assignment") in {None, "", "null", "NULL", "None", "none"}:
        normalized["anti_assignment"] = "Not Mentioned"
    for f in YES_NO_FIELDS:
        if normalized.get(f) in {None, "", "null", "NULL", "None", "none"}:
            normalized[f] = "No"
    return normalized


def merge_results(results: list[dict]) -> dict:
    EMPTY = {None, "Not Mentioned", "No"}
    text_fields = {"parties", "renewal_term", "notice_period_to_terminate_renewal", "governing_law"}
    categorical_fields = YES_NO_FIELDS | {"anti_assignment"}

    merged = {}

    # Categorical fields: majority voting, positional tiebreak (earlier chunk wins).
    for key in categorical_fields:
        non_empty = [r[key] for r in results if r.get(key) not in EMPTY]
        if not non_empty:
            merged[key] = results[0].get(key) if results else None
        else:
            counts = Counter(non_empty)
            max_count = max(counts.values())
            candidates = {v for v, c in counts.items() if c == max_count}
            # Tiebreak: first non-empty value by chunk position.
            merged[key] = next(v for v in non_empty if v in candidates)

    # Text and other fields: first non-null wins; concatenate if text field differs.
    for result in results:
        for key, value in result.items():
            if key in categorical_fields:
                continue
            if key not in merged:
                merged[key] = value
            else:
                current = merged[key]
                if current in EMPTY and value not in EMPTY:
                    merged[key] = value
                elif (key in text_fields
                      and current not in EMPTY
                      and value not in EMPTY
                      and value != current):
                    merged[key] = f"{current} | {value}"
    return merged


def parse_output(text: str) -> dict | None:
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _extract_estimated_max_len(error_text: str) -> int | None:
    match = re.search(r"estimated maximum model length is\s+(\d+)", error_text, re.IGNORECASE)
    if not match:
        return None
    estimated = int(match.group(1))
    return (estimated // 256) * 256


def load_llm_with_fallback(config: RunConfig):
    """
    Returns (llm, tokenizer, actual_max_context_tokens).

    The tokenizer is loaded separately from the LLM so it can be used
    by apply_chat_template() without going through vLLM internals.
    Both use the same model_id and HF_TOKEN, so no extra download occurs
    if the model is already cached.
    """
    requested  = config.max_context_tokens
    gpu_util   = config.gpu_memory_utilization
    token      = os.environ.get("HF_TOKEN")

    # Load tokenizer once — lightweight, no GPU memory used
    print(f"   Loading tokenizer for {config.model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id,
        token=token,
        trust_remote_code=True,   # needed for some Qwen variants
    )

    if not VLLM_AVAILABLE:
        raise RuntimeError("vLLM no está disponible — verificá la instalación en el container.")

    print(f"   Attempt 1: max_model_len={requested}, gpu_memory_utilization={gpu_util:.2f}")
    try:
        llm = LLM(
            model=config.model_id,
            dtype="bfloat16",
            max_model_len=requested,
            gpu_memory_utilization=gpu_util,
            disable_log_stats=False,
        )
        return llm, tokenizer, requested
    except Exception as first_error:
        # The real ValueError lives in the EngineCore subprocess; str(first_error)
        # often just says "Engine core initialization failed." without the number.
        # Try to parse it anyway, then fall back to 65% of requested (accounts for
        # CUDAGraph buffer overhead that wasn't needed with enforce_eager=True).
        error_text = str(first_error)
        fallback_len = _extract_estimated_max_len(error_text)
        if fallback_len is None:
            fallback_len = (int(requested * 0.65) // 256) * 256
        if fallback_len >= requested:
            fallback_len = requested - 2048

        print(f"   ⚠️  KV cache limit hit. Retrying with max_model_len={fallback_len}.")
        try:
            llm = LLM(
                model=config.model_id,
                dtype="bfloat16",
                max_model_len=fallback_len,
                gpu_memory_utilization=gpu_util,
                disable_log_stats=False,
            )
            return llm, tokenizer, fallback_len
        except Exception as second_error:
            error_text2 = str(second_error)
            fallback_len2 = _extract_estimated_max_len(error_text2)
            if fallback_len2 is None:
                fallback_len2 = (fallback_len * 3 // 4 // 256) * 256
            print(f"   ⚠️  Still too large. Retrying with max_model_len={fallback_len2}.")
            llm = LLM(
                model=config.model_id,
                dtype="bfloat16",
                max_model_len=fallback_len2,
                gpu_memory_utilization=gpu_util,
                disable_log_stats=False,
            )
            return llm, tokenizer, fallback_len2


def process_batch(llm, sampling_params, items: list[tuple]) -> list[str]:
    if not items:
        return []
    prompts = [item[1] for item in items]
    outputs = llm.generate(prompts, sampling_params)
    return [out.outputs[0].text.strip() for out in outputs]


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run(config: RunConfig) -> dict:
    """
    Execute one full inference pass according to config.
    Returns a summary dict with counts (for use by run_all.py).
    """
    start_time = datetime.now()
    print(f"\n{'='*60}")
    print(f"🚀 RUN: {config.name}")
    print(f"   model            : {config.model_id}")
    print(f"   prompt_version   : {config.prompt_version}")
    print(f"   use_few_shot     : {config.use_few_shot}")
    print(f"   temperature      : {config.temperature}")
    print(f"   overlap_chars    : {config.overlap_chars}")
    print(f"   max_context_tokens: {config.max_context_tokens}")
    print(f"{'='*60}")

    os.makedirs(config.output_dir, exist_ok=True)

    # Save config alongside results for full reproducibility
    config.save(os.path.join(config.output_dir, "config.json"))

    load_hf_token()

    if not os.path.exists(config.input_dir):
        raise FileNotFoundError(f"Input dir not found: {config.input_dir}")

    filenames = [f for f in os.listdir(config.input_dir) if f.endswith(".txt")]
    print(f"📊 Found {len(filenames)} .txt files in input_dir")

    # Filter by split (dev / holdout) if a split CSV and filter are configured
    if config.split_filter and config.split_csv and os.path.exists(config.split_csv):
        import csv as _csv
        with open(config.split_csv, encoding="utf-8") as _f:
            split_set = {
                row["filename"]
                for row in _csv.DictReader(_f)
                if row["split"] == config.split_filter
            }
        before = len(filenames)
        filenames = [f for f in filenames if f in split_set]
        print(f"📂 Split filter '{config.split_filter}': {before} → {len(filenames)} contracts")
    elif config.split_filter:
        print(f"⚠️  split_filter='{config.split_filter}' set but split_csv not found — processing all contracts")

    pending = [
        f for f in filenames
        if not os.path.exists(os.path.join(config.output_dir, f"{f}.json"))
    ]
    if not pending:
        print("✅ All files already processed.")
        return {"skipped": True}

    print(f"📋 Pending: {len(pending)}")

    # llm, tokenizer, active_ctx — tokenizer added here
    llm, tokenizer, active_ctx = load_llm_with_fallback(config)
    input_budget = active_ctx - config.max_output_tokens - 512
    sample_texts = [Path(config.input_dir, f).read_text(encoding="utf-8") for f in pending[:5]]
    lang_factor  = compute_lang_factor(tokenizer, sample_texts)
    print(f"📐 lang_factor: {lang_factor:.2f} chars/token (computed from {len(sample_texts)} samples, config default was {config.lang_factor})")
    char_limit   = estimate_char_limit(input_budget, lang_factor)
    print(f"📏 Max chunk: {char_limit:,} chars (~{input_budget:,} tokens)")

    sampling_params = SamplingParams(
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_output_tokens,
        structured_outputs=StructuredOutputsParams(
            json=RealEstateContract.model_json_schema()
        ),
    )

    # ------------------------------------------------------------------
    # Helper: messages → final prompt string for this model
    # Defined here (not at module level) so tokenizer is in closure scope.
    # ------------------------------------------------------------------
    def make_prompt(text: str, chunk_info: str = "") -> str:
        messages = build_messages(config, text, chunk_info)
        return apply_chat_template(messages, tokenizer, config)

    # Log chat template once per run
    if tokenizer.chat_template:
        template_preview = tokenizer.chat_template[:60].replace("\n", " ")
        print(f"   📋 Chat template: native ({config.model_id.split('/')[-1]}) — '{template_preview}...'")
    elif not config.chat_template_override:
        print(f"   ⚠️  No chat_template found for {config.model_id}. Using generic fallback.")

    short_contracts, long_contracts = [], []
    for f in pending:
        content = Path(config.input_dir, f).read_text(encoding="utf-8")
        if len(content) <= char_limit:
            short_contracts.append((f, content))
        else:
            chunks = split_into_chunks(content, char_limit, config.overlap_chars)
            long_contracts.append((f, chunks))
            print(f"✂️  {f}: {len(content):,} chars → {len(chunks)} chunks")

    exitosos = fallidos = 0

    # --- Short contracts ---
    if short_contracts:
        print(f"\n--- Short contracts: {len(short_contracts)} ---")
        items   = [(f, make_prompt(content)) for f, content in short_contracts]
        outputs = process_batch(llm, sampling_params, items)
        for (filename, _), generated in zip(short_contracts, outputs):
            data_dict = parse_output(generated)
            if data_dict is None:
                print(f"❌ Invalid JSON: {filename}")
                fallidos += 1
                Path(config.output_dir, f"ERROR_{filename}.raw").write_text(generated)
                continue
            try:
                normalized = normalize_extracted_data(data_dict)
                validated  = RealEstateContract(**normalized)
                out = Path(config.output_dir, f"{filename}.json")
                out.write_text(json.dumps(validated.model_dump(), indent=4, ensure_ascii=False))
                print(f"✅ {filename}")
                exitosos += 1
            except Exception as e:
                print(f"❌ Validation error {filename}: {e}")
                fallidos += 1

    # --- Long contracts ---
    if long_contracts:
        print(f"\n--- Long contracts: {len(long_contracts)} ---")
        all_chunks = []
        for filename, chunks in long_contracts:
            total = len(chunks)
            for i, chunk in enumerate(chunks):
                prompt = make_prompt(chunk, f"part {i+1} of {total}")
                all_chunks.append((filename, i, total, prompt))

        print(f"   Total chunks: {len(all_chunks)}")
        batch_items = [(x[0], x[3]) for x in all_chunks]
        outputs     = process_batch(llm, sampling_params, batch_items)

        chunks_by_file: dict[str, list[dict]] = {}
        for (filename, chunk_idx, total, _), generated in zip(all_chunks, outputs):
            data_dict = parse_output(generated)
            if data_dict is None:
                print(f"⚠️  Chunk {chunk_idx+1}/{total} invalid in {filename}")
                # Guardar output crudo para diagnóstico
                raw_path = Path(config.output_dir, f"CHUNK_ERROR_{chunk_idx+1}of{total}_{filename}.raw")
                raw_path.write_text(generated)
                continue            
            normalized = normalize_extracted_data(data_dict)
            chunks_by_file.setdefault(filename, []).append(normalized)

        for filename, chunk_results in chunks_by_file.items():
            if not chunk_results:
                fallidos += 1
                continue
            try:
                merged    = merge_results(chunk_results)
                validated = RealEstateContract(**merged)
                out = Path(config.output_dir, f"{filename}.json")
                out.write_text(json.dumps(validated.model_dump(), indent=4, ensure_ascii=False))
                print(f"✅ {filename} ({len(chunk_results)} chunks merged)")
                exitosos += 1
            except Exception as e:
                print(f"❌ Merge error {filename}: {e}")
                fallidos += 1

    elapsed = (datetime.now() - start_time).total_seconds()
    summary = {
        "run_name":        config.name,
        "total":           len(pending),
        "successful":      exitosos,
        "failed":          fallidos,
        "elapsed_seconds": round(elapsed, 1),
    }
    Path(config.output_dir, "inference_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    print(f"\n{'='*60}")
    print(f"📊 {config.name} → ✅ {exitosos}/{len(pending)}  ❌ {fallidos}  ⏱ {elapsed:.0f}s")
    print(f"{'='*60}\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run one inference experiment")
    parser.add_argument("--config", required=True, help="Path to RunConfig JSON file")
    args   = parser.parse_args()
    config = RunConfig.from_json(args.config)
    run(config)


if __name__ == "__main__":
    main()
