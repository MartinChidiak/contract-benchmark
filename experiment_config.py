"""
experiment_config.py
--------------------
Defines RunConfig: the single source of truth for every experiment parameter.
Serialize with .to_dict() / load with RunConfig.from_dict() for full reproducibility.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional

import json
import hashlib


# ---------------------------------------------------------------------------
# Model registry
# To add a new model: add one ModelConfig entry to MODELS. Nothing else.
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    hf_id: str                              # HuggingFace model ID
    max_context: int                        # max tokens (respects max_position_embeddings + KV budget)
    prompt_override: Optional[str] = None  # replaces "v3_full" for this model (e.g. disable CoT)
    gpu_memory_utilization: float = 0.90   # fraction of VRAM for vLLM
    max_output_tokens: int = 4096          # max tokens to generate per call


MODELS: dict[str, ModelConfig] = {
    # ── Llama ─────────────────────────────────────────────────────────────
    "llama31_8b": ModelConfig(
        hf_id="meta-llama/Llama-3.1-8B-Instruct",
        max_context=28000,              # ~29k available with CUDAGraphs at gpu_util=0.90
    ),
    # ── Qwen 2.5 ──────────────────────────────────────────────────────────
    "qwen25_3b": ModelConfig(
        hf_id="Qwen/Qwen2.5-3B-Instruct",
        max_context=30000,              # max_position_embeddings=32768
    ),
    "qwen25_7b": ModelConfig(
        hf_id="Qwen/Qwen2.5-7B-Instruct",
        max_context=28000,              # ~29k available with CUDAGraphs at gpu_util=0.90
    ),
    "qwen25_14b_awq": ModelConfig(
        hf_id="Qwen/Qwen2.5-14B-Instruct-AWQ",
        max_context=30000,              # max_position_embeddings=32768
    ),
    "qwen25_32b_awq": ModelConfig(
        hf_id="Qwen/Qwen2.5-32B-Instruct-AWQ",
        max_context=8000,               # ~18GB weights AWQ; KV cache constrained
        gpu_memory_utilization=0.95,
    ),
    # ── Qwen 3 ────────────────────────────────────────────────────────────
    "qwen3_8b": ModelConfig(
        hf_id="Qwen/Qwen3-8B",
        max_context=28000,              # ~29k available with CUDAGraphs at gpu_util=0.90
        prompt_override="v3_full_qwen3",  # disables thinking mode (/no_think)
    ),
}

# Derived dicts — used by run_all.py, app.py, run_experiment.py
MODEL_IDS             = {k: v.hf_id           for k, v in MODELS.items()}
MODEL_MAX_CONTEXT     = {k: v.max_context      for k, v in MODELS.items()}
MODEL_PROMPT_OVERRIDE = {k: v.prompt_override  for k, v in MODELS.items() if v.prompt_override}

# ---------------------------------------------------------------------------
# Prompt version registry
# Each key is a human-readable tag; the value is the system prompt text.
# Add new prompt variants here without touching run_experiments.py.
# ---------------------------------------------------------------------------
PROMPT_VERSIONS: dict[str, str] = {
    "v1_baseline": """\
You are an expert legal contract analyst.
Your task is to extract information from the contract and return it as a single JSON object
that STRICTLY follows the provided schema.

Rules:
- Extract ALL fields you can find in the text.
- Missing value policy:
    - For parties: extract legal party names whenever they appear (from title, preamble, or signature blocks).
        Use "Not Mentioned" only when no party names are present in this fragment.
    - Use null for: agreement_date, effective_date, expiration_date, governing_law,
        renewal_term, notice_period_to_terminate_renewal.
    - Use "Not Mentioned" for anti_assignment when absent.
    - Use "No" for audit_rights, cap_on_liability, termination_for_convenience,
        liquidated_damages when absent.
- Respond with JSON ONLY, no additional text, no markdown.""",

    "v2_with_date_rules": """\
You are an expert legal contract analyst.
Your task is to extract information from the contract and return it as a single JSON object
that STRICTLY follows the provided schema.

Rules:
- Extract ALL fields you can find in the text.
- Missing value policy:
    - For parties: extract legal party names whenever they appear.
        Use "Not Mentioned" only when no party names are present in this fragment.
    - Use null for: agreement_date, effective_date, expiration_date, governing_law,
        renewal_term, notice_period_to_terminate_renewal.
    - Use "Not Mentioned" for anti_assignment when absent.
    - Use "No" for audit_rights, cap_on_liability, termination_for_convenience,
        liquidated_damages when absent.
- Respond with JSON ONLY, no additional text, no markdown.

DATE EXTRACTION RULES (critical):
- All dates MUST be converted to ISO 8601 format (YYYY-MM-DD).
  Example: "March 29, 2004" -> "2004-03-29" | "29th day of March, 2004" -> "2004-03-29"
  If only month and year are known, use the first day: "March 2004" -> "2004-03-01"
- agreement_date: the date the contract was SIGNED.
  Look for keywords: "dated", "entered into as of", "executed on", "as of the ___ day of".
- effective_date: the date obligations BEGIN.
  Look for keywords: "effective as of", "effective date", "commencing on", "shall commence".
  NOTE: this is often DIFFERENT from agreement_date. Do not copy agreement_date here unless explicitly stated.
- expiration_date: the date the contract ENDS naturally.
  Look for keywords: "expire", "expiration", "terminate on", "through and including", "ending on".""",

    "v3_full": """\
You are an expert legal contract analyst.
Your task is to extract information from the contract and return it as a single JSON object
that STRICTLY follows the provided schema.

Rules:
- Extract ALL fields you can find in the text.
- Missing value policy:
    - For parties: extract legal party names whenever they appear (from title, preamble, or signature blocks).
        Use "Not Mentioned" only when no party names are present in this fragment.
    - Use null for: agreement_date, effective_date, expiration_date, governing_law,
        renewal_term, notice_period_to_terminate_renewal.
    - Use "Not Mentioned" for anti_assignment when absent.
    - Use "No" for audit_rights, cap_on_liability, termination_for_convenience,
        liquidated_damages when absent.
- Respond with JSON ONLY, no additional text, no markdown.

DATE EXTRACTION RULES (critical):
- All dates MUST be converted to ISO 8601 format (YYYY-MM-DD).
  Example: "March 29, 2004" -> "2004-03-29" | "29th day of March, 2004" -> "2004-03-29"
  If only month and year are known, use the first day: "March 2004" -> "2004-03-01"
- agreement_date: the date the contract was SIGNED.
  Look for keywords: "dated", "entered into as of", "executed on", "as of the ___ day of".
- effective_date: the date obligations BEGIN.
  Look for keywords: "effective as of", "effective date", "commencing on", "shall commence".
  NOTE: this is often DIFFERENT from agreement_date. Do not copy agreement_date here unless explicitly stated.
- expiration_date: the date the contract ENDS naturally.
  Look for keywords: "expire", "expiration", "terminate on", "through and including", "ending on".

ANTI-ASSIGNMENT RULES (critical):
- Use 'Prohibited' when assignment is EXPLICITLY FORBIDDEN with no consent option.
  Keywords: 'may not assign', 'shall not assign', 'cannot be assigned', 'non-assignable'.
- Use 'Allowed with consent' when assignment IS POSSIBLE but REQUIRES the other party's approval.
  Keywords: 'prior written consent', 'with the consent of', 'consent shall not be unreasonably withheld'.
- Use 'Not Mentioned' ONLY when the contract contains no assignment clause at all.

YES/NO CLAUSE EXTRACTION RULES (critical — do NOT default to 'No' without checking):
- audit_rights: Answer 'Yes' if ANY party has the right to inspect or audit the other party's books.
  Keywords: 'right to audit', 'audit the books', 'inspect the records', 'audit rights'.
- cap_on_liability: Answer 'Yes' if the contract sets a MAXIMUM dollar limit on liability.
  Keywords: 'shall not exceed', 'aggregate liability', 'in no event shall', 'liability cap'.
- termination_for_convenience: Answer 'Yes' if either party may end WITHOUT needing to prove breach.
  Keywords: 'terminate for convenience', 'terminate at any time', 'without cause'.
- liquidated_damages: Answer 'Yes' if the contract specifies a PRE-DETERMINED monetary penalty.
  Keywords: 'liquidated damages', 'agreed damages', 'per day for each day'.

RENEWAL AND NOTICE PERIOD RULES (critical):
- renewal_term: the duration of each automatic renewal period, NOT the initial contract term.
  Keywords: "automatically renew", "successive periods of", "renewal term", "renewed for".
  Extract ONLY the duration (e.g. "1 year", "6 months").
- notice_period_to_terminate_renewal: advance notice required to PREVENT automatic renewal.
  Keywords: "prior written notice", "days before", "days prior to expiration", "notice of non-renewal".
  Extract ONLY the period (e.g. "30 days", "60 days").
  NOTE: this field is often in the SAME paragraph as renewal_term.""",
}

# Qwen3 needs /no_think to suppress chain-of-thought and return JSON directly.
PROMPT_VERSIONS["v3_full_qwen3"] = PROMPT_VERSIONS["v3_full"] + "\n/no_think"


@dataclass
class RunConfig:
    """
    All parameters that define a single experiment run.

    Fields are grouped by concern:
    - Identity: name, description
    - Paths: input_dir, output_dir, ground_truth_csv
    - Model: model_id, max_context_tokens, max_output_tokens, gpu_memory_utilization
    - Sampling: temperature, top_p
    - Chunking: overlap_chars, lang_factor
    - Prompt: prompt_version, use_few_shot, chat_template_override
    """

    # --- Identity ---
    name: str = "run_001"
    description: str = ""

    # --- Paths ---
    input_dir: str = "./Dataset_CUAD_Completo"
    output_dir: str = "./experiments/run_001/results"
    ground_truth_csv: str = "./ground_truth.csv"
    split_csv: str = "./split.csv"       # path to the dev/holdout split record
    split_filter: str = "dev"            # "dev" | "holdout" | "" (empty = all contracts)

    # --- Model ---
    model_id: str = "meta-llama/Llama-3.1-8B-Instruct"
    max_context_tokens: int = 46080
    max_output_tokens: int = 4096
    gpu_memory_utilization: float = 0.90

    # --- Sampling ---
    temperature: float = 0.0
    top_p: float = 1.0

    # --- Chunking ---
    overlap_chars: int = 800
    lang_factor: float = 3.5          # chars-per-token estimate for char_limit calculation

    # --- Prompt ---
    prompt_version: str = "v3_full"   # must be a key in PROMPT_VERSIONS
    use_few_shot: bool = True

    # --- Chat template (optional override) ---
    # Leave as None to use the model's native template from its tokenizer_config.json.
    # This covers Llama-3, Qwen2.5, Qwen3, Mistral, Gemma, Phi, etc. automatically.
    #
    # Set to a Jinja2 template string ONLY for models whose tokenizer does not ship
    # a chat_template, or when you need to test a non-standard formatting variant.
    #
    # Example for a hypothetical model with a simple format:
    #   chat_template_override = (
    #       "{% for m in messages %}"
    #       "{% if m.role == 'system' %}<|system|>\n{{ m.content }}\n{% endif %}"
    #       "{% if m.role == 'user' %}<|user|>\n{{ m.content }}\n{% endif %}"
    #       "{% endfor %}<|assistant|>\n"
    #   )
    chat_template_override: Optional[str] = None

    # --- Auto-generated metadata ---
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def __post_init__(self):
        if self.prompt_version not in PROMPT_VERSIONS:
            raise ValueError(
                f"Unknown prompt_version '{self.prompt_version}'. "
                f"Available: {list(PROMPT_VERSIONS.keys())}"
            )

    @property
    def system_prompt(self) -> str:
        return PROMPT_VERSIONS[self.prompt_version]

    @property
    def run_id(self) -> str:
        """Short deterministic hash useful for directory names."""
        payload = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(payload.encode()).hexdigest()[:8]

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str) -> None:
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> RunConfig:
        # Drop auto-generated fields that should not be set from outside
        data = {k: v for k, v in data.items() if k != "created_at"}
        return cls(**data)

    @classmethod
    def from_json(cls, path: str) -> RunConfig:
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
