import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


FIELD_MAPPING = {
    "parties": "Parties-Answer",
    "agreement_date": "Agreement Date-Answer",
    "effective_date": "Effective Date-Answer",
    "expiration_date": "Expiration Date-Answer",
    "renewal_term": "Renewal Term-Answer",
    "notice_period_to_terminate_renewal": "Notice Period To Terminate Renewal- Answer",
    "governing_law": "Governing Law-Answer",
    "anti_assignment": "Anti-Assignment-Answer",
    "audit_rights": "Audit Rights-Answer",
    "cap_on_liability": "Cap On Liability-Answer",
    "termination_for_convenience": "Termination For Convenience-Answer",
    "liquidated_damages": "Liquidated Damages-Answer",
}

YES_NO_FIELDS = {
    "anti_assignment",
    "audit_rights",
    "cap_on_liability",
    "termination_for_convenience",
    "liquidated_damages",
}

DATE_FIELDS = {"agreement_date", "effective_date", "expiration_date"}
DURATION_FIELDS = {"renewal_term", "notice_period_to_terminate_renewal"}

PARTY_NOISE_TOKENS = {
    "party",
    "parties",
    "company",
    "vendor",
    "owner",
    "operator",
    "affiliate",
    "affiliates",
    "service",
    "provider",
    "recipient",
    "client",
    "customer",
    "issuer",
    "depositor",
    "servicer",
    "marketing",
    "agent",
    "custodian",
    "seller",
    "buyer",
    "trust",
    "together",
    "collectively",
    "hereinafter",
    "referred",
    "individually",
    "plural",
    "singular",
    "as",
    "or",
    "the",
    "a",
    "an",
}


def normalize_filename_key(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"\.(pdf|txt|json)$", "", name)
    name = re.sub(r"\s+", " ", name)
    return name


def parse_ground_truth_rows(csv_path: Path) -> dict[str, dict[str, str]]:
    rows_by_key: dict[str, dict[str, str]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row.get("Filename", "")
            if not filename:
                continue
            key = normalize_filename_key(filename)
            rows_by_key[key] = row
    return rows_by_key


def parse_result_files(results_dir: Path) -> dict[str, dict[str, Any]]:
    parsed: dict[str, dict[str, Any]] = {}
    for fp in sorted(results_dir.glob("*.json")):
        # Expected pattern: <contract>.txt.json
        stem = fp.name
        stem = re.sub(r"\.json$", "", stem)
        key = normalize_filename_key(stem)
        try:
            parsed[key] = json.loads(fp.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            parsed[key] = {}
    return parsed


def clean_string(value: Any) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    if s.lower() in {"", "none", "null", "nan", "not mentioned", "[]"}:
        return ""
    return s


def normalize_text(value: Any) -> str:
    s = clean_string(value).lower()
    s = s.replace("&", " and ")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_governing_law(value: Any) -> str:
    s = normalize_text(value)
    if not s:
        return ""
    s = re.sub(r"\b(state|commonwealth|republic|laws?)\b", " ", s)
    s = re.sub(r"\bof\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_parties_tokens(value: Any) -> set[str]:
    s = clean_string(value)
    if not s:
        return set()

    # Remove aliases/roles usually placed in parenthesis or quoted labels.
    s = s.replace("“", '"').replace("”", '"').replace("’", "'")
    s = re.sub(r"\([^)]*\)", " ", s)
    s = re.sub(r'"[^"]*"', " ", s)

    normalized = normalize_text(s)
    tokens = {t for t in normalized.split() if t and t not in PARTY_NOISE_TOKENS}
    return tokens


def parties_match(pred_value: Any, gt_value: Any) -> tuple[str, str, bool]:
    pred_tokens = normalize_parties_tokens(pred_value)
    gt_tokens = normalize_parties_tokens(gt_value)

    pred_norm = " ".join(sorted(pred_tokens))
    gt_norm = " ".join(sorted(gt_tokens))

    if not pred_tokens and not gt_tokens:
        return pred_norm, gt_norm, True
    if not pred_tokens or not gt_tokens:
        return pred_norm, gt_norm, False

    overlap = len(pred_tokens & gt_tokens)
    min_size = min(len(pred_tokens), len(gt_tokens))

    # Tolerant entity-level match: allow extra role words/aliases, require strong overlap.
    match = overlap >= 3 and (overlap / min_size) >= 0.80
    return pred_norm, gt_norm, match


def normalize_yes_no(value: Any) -> str:
    s = clean_string(value).lower()
    if not s:
        return "No"
    if "yes" in s:
        return "Yes"
    if "no" in s:
        return "No"
    if s in {"prohibited", "allowed with consent"}:
        return "Yes"
    if s == "not mentioned":
        return "No"
    return "Yes"


def parse_date(text: str) -> str:
    text = clean_string(text)
    if not text:
        return ""

    # If already YYYY-MM-DD
    if re.match(r"^\d{4}-\d{2}-\d{2}$", text):
        return text

    # Handle m/d/yy or m/d/yyyy (including placeholders like []/[]/2020)
    mdY = re.match(r"^(\d{1,2}|\[\])/(\d{1,2}|\[\])/(\d{2,4})$", text)
    if mdY:
        month_raw, day_raw, year_raw = mdY.groups()
        if month_raw == "[]" or day_raw == "[]":
            return ""
        month = int(month_raw)
        day = int(day_raw)
        year = int(year_raw)
        if year < 100:
            year += 2000 if year <= 29 else 1900
        try:
            return datetime(year=year, month=month, day=day).strftime("%Y-%m-%d")
        except ValueError:
            return ""

    return ""


def normalize_duration(value: Any) -> str:
    s = normalize_text(value)
    if not s:
        return ""

    if "perpetual" in s or "indefinite" in s:
        return "perpetual"

    # Common format: "successive 1 year", "1 years", "60 days"
    m = re.search(r"(\d+)\s*(day|days|month|months|year|years)", s)
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        unit = {
            "day": "day",
            "days": "day",
            "month": "month",
            "months": "month",
            "year": "year",
            "years": "year",
        }[unit]
        plural = "" if n == 1 else "s"
        return f"{n} {unit}{plural}"

    # Some values are words only, keep normalized fallback.
    return s


def compare_field(field: str, pred_value: Any, gt_value: Any) -> tuple[str, str, bool]:
    if field == "parties":
        return parties_match(pred_value, gt_value)
    if field in DATE_FIELDS:
        pred_norm = parse_date(str(pred_value) if pred_value is not None else "")
        gt_norm = parse_date(str(gt_value) if gt_value is not None else "")
    elif field in DURATION_FIELDS:
        pred_norm = normalize_duration(pred_value)
        gt_norm = normalize_duration(gt_value)
    elif field == "governing_law":
        pred_norm = normalize_governing_law(pred_value)
        gt_norm = normalize_governing_law(gt_value)
    elif field in YES_NO_FIELDS:
        pred_norm = normalize_yes_no(pred_value)
        gt_norm = normalize_yes_no(gt_value)
    else:
        pred_norm = normalize_text(pred_value)
        gt_norm = normalize_text(gt_value)

    match = pred_norm == gt_norm
    return pred_norm, gt_norm, match


def benchmark(results_dir: Path, ground_truth_csv: Path, output_csv: Path) -> None:
    gt_rows = parse_ground_truth_rows(ground_truth_csv)
    predictions = parse_result_files(results_dir)

    total_files = len([k for k in gt_rows if k in predictions])
    files_with_predictions = 0
    fully_matched_files = 0

    per_field_counts = {
        field: {"correct": 0, "total": 0}
        for field in FIELD_MAPPING
    }

    detailed_rows = []

    for key, gt_row in gt_rows.items():
        if key not in predictions:
            continue
        pred = predictions.get(key)
        if pred is not None:
            files_with_predictions += 1

        file_all_correct = True
        for field, gt_col in FIELD_MAPPING.items():
            gt_val = gt_row.get(gt_col, "")
            pred_val = pred.get(field) if pred else None

            pred_norm, gt_norm, is_match = compare_field(field, pred_val, gt_val)
            per_field_counts[field]["total"] += 1
            if is_match:
                per_field_counts[field]["correct"] += 1
            else:
                file_all_correct = False

            detailed_rows.append(
                {
                    "file_key": key,
                    "field": field,
                    "pred_raw": "" if pred_val is None else str(pred_val),
                    "gt_raw": "" if gt_val is None else str(gt_val),
                    "pred_norm": pred_norm,
                    "gt_norm": gt_norm,
                    "match": "1" if is_match else "0",
                }
            )

        if file_all_correct:
            fully_matched_files += 1

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file_key",
                "field",
                "pred_raw",
                "gt_raw",
                "pred_norm",
                "gt_norm",
                "match",
            ],
        )
        writer.writeheader()
        writer.writerows(detailed_rows)

    print("=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Ground truth files:          {len(gt_rows)}")
    print(f"Files evaluated:             {total_files}")
    print(f"Files with predictions:      {files_with_predictions}")
    print(f"Missing prediction files:    {len(gt_rows) - files_with_predictions}")
    print(f"Fully matched files:         {fully_matched_files}/{total_files}")
    print()

    macro_acc_values = []
    for field, stats in per_field_counts.items():
        total = stats["total"]
        correct = stats["correct"]
        acc = (correct / total) if total else 0.0
        macro_acc_values.append(acc)
        print(f"{field:35s} {correct:4d}/{total:<4d}  acc={acc:.4f}")

    micro_correct = sum(v["correct"] for v in per_field_counts.values())
    micro_total = sum(v["total"] for v in per_field_counts.values())
    micro_acc = (micro_correct / micro_total) if micro_total else 0.0
    macro_acc = (sum(macro_acc_values) / len(macro_acc_values)) if macro_acc_values else 0.0

    print()
    print(f"Micro accuracy (all cells):  {micro_acc:.4f}")
    print(f"Macro accuracy (avg fields): {macro_acc:.4f}")
    print(f"Detailed report:             {output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark extraction JSON files in Resultados against ground_truth.csv"
    )
    parser.add_argument(
        "--results-dir",
        default="./Resultados",
        help="Directory containing result JSON files (default: ./Resultados)",
    )
    parser.add_argument(
        "--ground-truth",
        default="./ground_truth.csv",
        help="Path to ground truth CSV (default: ./ground_truth.csv)",
    )
    parser.add_argument(
        "--output-csv",
        default="./Resultados/benchmark_detailed.csv",
        help="Path to output detailed CSV report",
    )
    args = parser.parse_args()

    benchmark(
        results_dir=Path(args.results_dir),
        ground_truth_csv=Path(args.ground_truth),
        output_csv=Path(args.output_csv),
    )


if __name__ == "__main__":
    main()