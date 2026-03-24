"""
app.py — Streamlit UI for the contract extraction benchmark.

Run inside the container with:
    streamlit run app.py --server.port 8501 --server.address 0.0.0.0
Then open http://localhost:8501 in your browser.
"""

import contextlib
import csv
import io
import json
import shutil
import tempfile
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from experiment_config import (
    RunConfig, MODEL_IDS, PROMPT_VERSIONS,
    MODEL_MAX_CONTEXT, MODEL_PROMPT_OVERRIDE,
)
from run_experiment import run as run_inference
from benchmark import benchmark as run_benchmark, normalize_filename_key, YES_NO_FIELDS
from run_all import load_benchmark_summary, save_comparison_csv, FIELDS

# ── Constants ─────────────────────────────────────────────────────────────────
DEFAULT_INPUT_DIR = "./Dataset_Filtrado_Tesis"
GROUND_TRUTH_CSV  = "./ground_truth.csv"
EXPERIMENTS_DIR   = Path("./experiments")
COMPARISON_CSV    = EXPERIMENTS_DIR / "comparison.csv"

BASE_CONFIGS: list[dict] = [
    dict(
        name="v1_no_fewshot",
        description="Baseline — minimal prompt, no few-shot",
        prompt_version="v1_baseline",
        use_few_shot=False,
        temperature=0.0,
        overlap_chars=800,
    ),
    dict(
        name="v2_date_rules",
        description="Adds explicit date extraction rules, no few-shot",
        prompt_version="v2_with_date_rules",
        use_few_shot=False,
        temperature=0.0,
        overlap_chars=800,
    ),
    dict(
        name="v3_full_fewshot",
        description="Full rules + few-shot examples (recommended)",
        prompt_version="v3_full",
        use_few_shot=True,
        temperature=0.0,
        overlap_chars=800,
    ),
    dict(
        name="v3_overlap1600",
        description="Full + few-shot, overlap doubled to 1600 chars",
        prompt_version="v3_full",
        use_few_shot=True,
        temperature=0.0,
        overlap_chars=1600,
    ),
    dict(
        name="v3_temp02",
        description="Full + few-shot, temperature=0.2",
        prompt_version="v3_full",
        use_few_shot=True,
        temperature=0.2,
        overlap_chars=800,
    ),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_existing_comparison() -> list[dict]:
    """Load existing comparison.csv rows, return empty list if not present."""
    if not COMPARISON_CSV.exists():
        return []
    with COMPARISON_CSV.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Contract Extraction Benchmark",
    page_icon="📄",
    layout="wide",
)
st.title("📄 Contract Extraction Benchmark")

tab_run, tab_results, tab_comparison, tab_overview, tab_manage = st.tabs([
    "⚙️ Configure & Run",
    "📄 Results",
    "📊 Comparison",
    "📈 Overview",
    "🗑️ Manage",
])


# ── Tab 1: Configure & Run ────────────────────────────────────────────────────

# Persistent state for the run lifecycle
for _k, _v in [
    ("running",      False),
    ("run_params",   None),
    ("run_logs",     []),
    ("run_messages", []),
    ("run_complete", False),
]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

with tab_run:
    is_running = st.session_state.running
    left, right = st.columns([1, 2])

    # ── Left: configuration (locked while running) ────────────────────────────
    with left:
        st.subheader("1. Contracts")
        source = st.radio(
            "Input source",
            ["Use existing dataset", "Upload contracts"],
            disabled=is_running,
        )
        uploaded_files = None
        if source == "Upload contracts":
            uploaded_files = st.file_uploader(
                "Upload .txt contract files",
                type=["txt"],
                accept_multiple_files=True,
                disabled=is_running,
            )
            if uploaded_files:
                st.caption(f"{len(uploaded_files)} file(s) ready")

        st.subheader("2. Models")
        selected_models = {
            tag: st.checkbox(tag, value=True, key=f"m_{tag}", disabled=is_running)
            for tag in MODEL_IDS
        }

        st.subheader("3. Experiment configs")
        selected_cfgs = {
            cfg["name"]: st.checkbox(
                cfg["name"],
                value=(cfg["name"] == "v3_full_fewshot"),
                help=cfg["description"],
                key=f"c_{cfg['name']}",
                disabled=is_running,
            )
            for cfg in BASE_CONFIGS
        }

        run_btn = st.button(
            "⏳ Running…" if is_running else "🚀 Run",
            type="primary",
            use_container_width=True,
            disabled=is_running,
        )
        bench_btn = st.button(
            "📊 Run benchmark only",
            use_container_width=True,
            disabled=is_running,
            help="Re-evaluate existing results without re-running inference.",
        )
        if is_running:
            st.info("Run in progress. Configuration locked until it completes.")

    # ── Right: progress & logs ────────────────────────────────────────────────
    with right:
        st.subheader("Progress & Logs")

        if (run_btn or bench_btn) and not is_running:
            # ── Step 1: validate, store params, save uploads, trigger rerun ──
            active_models = [t for t, checked in selected_models.items() if checked]
            active_cfgs   = [c for c in BASE_CONFIGS if selected_cfgs.get(c["name"])]

            if not active_models:
                st.error("Select at least one model.")
            elif not active_cfgs:
                st.error("Select at least one experiment config.")
            else:
                # Save uploaded files NOW — file_uploader is cleared after rerun
                tmp_dir = None
                if source == "Upload contracts" and uploaded_files:
                    tmp_dir = tempfile.mkdtemp()
                    for f in uploaded_files:
                        (Path(tmp_dir) / f.name).write_bytes(f.read())
                    input_dir = tmp_dir
                else:
                    input_dir = DEFAULT_INPUT_DIR

                st.session_state.run_params   = {
                    "active_models":  active_models,
                    "active_cfgs":    active_cfgs,
                    "input_dir":      input_dir,
                    "tmp_dir":        tmp_dir,
                    "benchmark_only": bool(bench_btn),
                }
                st.session_state.run_logs     = []
                st.session_state.run_messages = []
                st.session_state.run_complete = False
                st.session_state.running      = True
                st.rerun()

        elif is_running:
            # ── Step 2: widgets disabled, run inference from stored params ────
            params         = st.session_state.run_params
            active_models  = params["active_models"]
            active_cfgs    = params["active_cfgs"]
            input_dir      = params["input_dir"]
            tmp_dir        = params["tmp_dir"]
            benchmark_only = params.get("benchmark_only", False)

            total    = len(active_models) * len(active_cfgs)
            progress = st.progress(0, text="Starting…")
            log_box  = st.empty()
            step     = 0

            comparison_rows = load_existing_comparison()

            for model_tag in active_models:
                for cfg in active_cfgs:
                    step     += 1
                    run_name  = f"{model_tag}__{cfg['name']}"
                    progress.progress(step / total, text=f"Running {run_name}…")

                    prompt_ver = cfg["prompt_version"]
                    if prompt_ver == "v3_full" and model_tag in MODEL_PROMPT_OVERRIDE:
                        prompt_ver = MODEL_PROMPT_OVERRIDE[model_tag]

                    config = RunConfig(
                        name               = run_name,
                        description        = f"[{model_tag}] {cfg['description']}",
                        model_id           = MODEL_IDS[model_tag],
                        input_dir          = input_dir,
                        output_dir         = str(EXPERIMENTS_DIR / run_name / "results"),
                        ground_truth_csv   = GROUND_TRUTH_CSV,
                        prompt_version     = prompt_ver,
                        use_few_shot       = cfg["use_few_shot"],
                        temperature        = cfg["temperature"],
                        overlap_chars      = cfg["overlap_chars"],
                        max_context_tokens = MODEL_MAX_CONTEXT[model_tag],
                    )

                    buf = io.StringIO()
                    try:
                        prev_elapsed = 0.0
                        prev_summary_path = Path(config.output_dir) / "benchmark_summary.json"
                        if prev_summary_path.exists():
                            try:
                                prev_data = json.loads(prev_summary_path.read_text(encoding="utf-8"))
                                prev_elapsed = float(prev_data.get("elapsed_seconds") or 0)
                            except (json.JSONDecodeError, ValueError):
                                pass

                        if benchmark_only:
                            summary = {"skipped": True}
                        else:
                            with contextlib.redirect_stdout(buf):
                                summary = run_inference(config)

                        detailed_csv = Path(config.output_dir) / "benchmark_detailed.csv"
                        with contextlib.redirect_stdout(buf):
                            run_benchmark(
                                Path(config.output_dir),
                                Path(GROUND_TRUTH_CSV),
                                detailed_csv,
                            )

                        metrics = load_benchmark_summary(config.output_dir)
                        if metrics:
                            new_elapsed = float(metrics.get("elapsed_seconds") or 0)
                            if summary.get("skipped"):
                                metrics["elapsed_seconds"] = prev_elapsed
                            else:
                                metrics["elapsed_seconds"] = round(prev_elapsed + new_elapsed, 1)

                            new_row = {"run_name": run_name, **metrics}
                            comparison_rows = [
                                r for r in comparison_rows
                                if r.get("run_name") != run_name
                            ]
                            comparison_rows.append(new_row)
                            Path(config.output_dir, "benchmark_summary.json").write_text(
                                json.dumps(new_row, indent=2)
                            )

                        mac = metrics.get("macro_acc", "?") if metrics else "?"
                        if benchmark_only:
                            msg = f"✅ {run_name} — benchmark refreshed. macro_acc={mac}"
                            st.session_state.run_messages.append(("success", msg))
                        elif summary.get("skipped"):
                            msg = f"ℹ️ {run_name} — all contracts already processed, benchmark refreshed. macro_acc={mac}"
                            st.session_state.run_messages.append(("info", msg))
                        else:
                            ok  = summary.get("successful", "?")
                            tot = summary.get("total", "?")
                            msg = f"✅ {run_name} — {ok}/{tot} contracts, macro_acc={mac}"
                            st.session_state.run_messages.append(("success", msg))

                    except Exception as e:
                        msg = f"❌ {run_name}: {e}"
                        st.session_state.run_messages.append(("error", msg))
                        buf.write(f"\nERROR: {e}\n")

                    st.session_state.run_logs.append(
                        f"{'='*50}\n{run_name}\n{'='*50}\n{buf.getvalue()}"
                    )
                    log_box.text_area("Logs", "\n".join(st.session_state.run_logs), height=400)

            save_comparison_csv(comparison_rows, str(COMPARISON_CSV))

            if tmp_dir:
                shutil.rmtree(tmp_dir, ignore_errors=True)

            progress.progress(1.0, text="All done!")
            st.session_state.run_complete = True
            st.session_state.running      = False
            st.rerun()

        elif st.session_state.run_logs:
            # ── Step 3: show preserved results from the completed run ─────────
            if st.session_state.run_complete:
                st.success("Run completed. Switch to the 📊 Comparison tab to see results.")
            for level, msg in st.session_state.run_messages:
                getattr(st, level)(msg)
            st.text_area("Logs", "\n".join(st.session_state.run_logs), height=400)


# ── Load comparison data once (shared by sidebar, Tab 3, and Tab 4) ──────────
_comparison_df = pd.read_csv(COMPARISON_CSV) if COMPARISON_CSV.exists() else None

# ── Sidebar filters ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🔍 Filters")
    st.caption("Applied to Results, Comparison and Overview tabs.")
    if _comparison_df is not None:
        _all_runs_s    = _comparison_df["run_name"].tolist()
        _avail_models  = sorted({r.split("__")[0] for r in _all_runs_s if "__" in r})
        _avail_configs = sorted({r.split("__", 1)[1] for r in _all_runs_s if "__" in r})
        sb_models  = st.multiselect("Models",  options=_avail_models,  default=_avail_models,  key="sb_models")
        sb_configs = st.multiselect("Configs", options=_avail_configs, default=_avail_configs, key="sb_configs")
    else:
        sb_models  = []
        sb_configs = []
        st.info("No experiments yet.")

    st.divider()
    with st.expander("📖 Metric definitions"):
        st.markdown("""
**macro_acc** — Macro Accuracy
Average per-field accuracy across all 12 fields.
Includes correct *absent-absent* predictions (model says "Not Mentioned", GT is empty).
*Tends to be optimistic on sparse fields.*

---

**macro_precision** — Macro Precision
When the model extracts a value, how often is it correct?
`TP / (TP + FP)` averaged across fields.
*Low precision → model hallucinates values that aren't there.*

---

**macro_recall** — Macro Recall
When a field is present in the contract, how often does the model find it?
`TP / (TP + FN)` averaged across fields.
*Low recall → model misses or fails to extract present fields.*

---

**macro_f1** — Macro F1
Harmonic mean of Precision and Recall.
Best single metric when both extraction and correctness matter equally.
`2 · P · R / (P + R)` averaged across fields.

---

**micro_acc** — Micro Accuracy
Accuracy computed **only** over cases where the ground truth is present
(excludes absent-absent matches).
More demanding than macro_acc; reveals true extraction ability.
*A large gap between macro_acc and micro_acc means the model
benefits heavily from predicting "Not Mentioned" correctly.*

---

**elapsed_seconds**
Total inference time in seconds, accumulated across partial re-runs
(new contracts added to an existing experiment).
""")


# ── Tab 2: Results ────────────────────────────────────────────────────────────
with tab_results:
    def _matches_sidebar(name: str) -> bool:
        if "__" not in name:
            return True
        model, cfg = name.split("__", 1)
        model_ok  = (not sb_models)  or (model in sb_models)
        config_ok = (not sb_configs) or (cfg   in sb_configs)
        return model_ok and config_ok

    exp_dirs = (
        sorted([
            d for d in EXPERIMENTS_DIR.iterdir()
            if d.is_dir() and (d / "results").exists() and _matches_sidebar(d.name)
        ])
        if EXPERIMENTS_DIR.exists()
        else []
    )

    if not exp_dirs:
        st.info("No results yet. Run an experiment first.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            exp_name = st.selectbox("Experiment run", [d.name for d in exp_dirs])

        results_path = EXPERIMENTS_DIR / exp_name / "results"
        result_files = sorted([
            f for f in results_path.glob("*.json")
            if f.name not in {"config.json", "inference_summary.json"}
        ])

        with col2:
            if result_files:
                file_name = st.selectbox("Contract file", [f.name for f in result_files])
            else:
                st.warning("No result files found for this experiment.")
                file_name = None

        if file_name:
            data = json.loads((results_path / file_name).read_text(encoding="utf-8"))
            st.json(data)

            # ── Feature 1: Predicted vs. Ground Truth table ───────────────
            detailed_csv_path = results_path / "benchmark_detailed.csv"
            if detailed_csv_path.exists():
                det_df = pd.read_csv(detailed_csv_path)
                # Strip .json first, then normalize (handles "contract.txt.json" → "contract")
                fkey = normalize_filename_key(file_name.removesuffix(".json"))
                contract_rows = det_df[det_df["file_key"] == fkey]
                if not contract_rows.empty:
                    st.subheader("Predicted vs. Ground Truth")
                    disp = contract_rows[["field", "pred_raw", "gt_raw", "match"]].copy()
                    disp["match"] = disp["match"].map({1: "✅", 0: "❌", "1": "✅", "0": "❌"})
                    disp.columns = ["Field", "Predicted", "Ground Truth", "Match"]
                    st.dataframe(disp.set_index("Field"), use_container_width=True)

        # Show inference summary if available
        summary_path = results_path / "inference_summary.json"
        if summary_path.exists():
            with st.expander("Inference summary (last run only)"):
                st.caption("Stats below reflect only the most recent inference run for this experiment, not the cumulative total.")
                st.json(json.loads(summary_path.read_text(encoding="utf-8")))

        # ── Feature 2: Download benchmark_detailed.csv ────────────────────
        detailed_csv_dl = results_path / "benchmark_detailed.csv"
        if detailed_csv_dl.exists():
            st.download_button(
                "⬇️ Download benchmark_detailed.csv",
                data=detailed_csv_dl.read_bytes(),
                file_name=f"{exp_name}_detailed.csv",
                mime="text/csv",
            )



# ── Metric config (shared across Tab 3 and Tab 4) ─────────────────────────────
METRIC_OPTIONS = {
    "Accuracy":  "acc_",
    "Precision": "prec_",
    "Recall":    "rec_",
    "F1":        "f1_",
}
MACRO_METRIC_COLS = {
    "macro_acc":       "Accuracy",
    "macro_precision": "Precision",
    "macro_recall":    "Recall",
    "macro_f1":        "F1",
}


# ── Tab 3: Comparison ─────────────────────────────────────────────────────────
with tab_comparison:
    if _comparison_df is None:
        st.info("No comparison data yet. Run experiments first.")
    else:
        df       = _comparison_df
        all_runs = df["run_name"].tolist()

        # ── Feature 2: Download comparison.csv ────────────────────────────
        st.download_button(
            "⬇️ Download comparison.csv",
            data=COMPARISON_CSV.read_bytes(),
            file_name="comparison.csv",
            mime="text/csv",
        )

        # Field names derived from acc_ columns (all metrics share the same fields)
        field_acc_cols = [f"acc_{f}" for f in FIELDS if f"acc_{f}" in df.columns]
        field_names    = [c.replace("acc_", "") for c in field_acc_cols]

        # ── Per-tab filters: Fields and Metric (Models/Configs are in sidebar) ──
        fcol1, fcol2 = st.columns(2)
        with fcol1:
            selected_fields = st.multiselect(
                "Fields",
                options=field_names,
                default=field_names,
            )
        with fcol2:
            selected_metric = st.selectbox(
                "Metric",
                options=list(METRIC_OPTIONS.keys()),
                index=0,
                key="cmp_metric",
            )

        prefix = METRIC_OPTIONS[selected_metric]

        # Derive selected runs from sidebar model + config selection
        selected_runs = [
            r for r in all_runs
            if "__" in r
            and r.split("__")[0] in sb_models
            and r.split("__", 1)[1] in sb_configs
        ]

        if not selected_runs:
            st.warning("Select at least one experiment.")
        else:
            filtered = df[df["run_name"].isin(selected_runs)]

            # ── Overall metrics table ──────────────────────────────────────
            st.subheader("Overall metrics")
            macro_display_cols = [c for c in MACRO_METRIC_COLS if c in filtered.columns]
            summary_cols = [c for c in ["run_name", "files_evaluated"] + macro_display_cols + ["micro_acc", "elapsed_seconds"] if c in filtered.columns]
            summary_df = filtered[summary_cols].copy()
            summary_df.insert(0, "model",  summary_df["run_name"].apply(lambda x: x.split("__")[0] if "__" in x else x))
            summary_df.insert(1, "config", summary_df["run_name"].apply(lambda x: x.split("__", 1)[1] if "__" in x else ""))
            summary_df = summary_df.drop(columns="run_name").set_index(["model", "config"])
            _col_help = {
                "files_evaluated": st.column_config.NumberColumn(
                    "files_evaluated",
                    help="Number of contracts evaluated against the ground truth.",
                ),
                "macro_acc": st.column_config.NumberColumn(
                    "macro_acc",
                    help="Macro Accuracy: average per-field accuracy across all 12 fields. Includes correct 'Not Mentioned' predictions.",
                ),
                "macro_precision": st.column_config.NumberColumn(
                    "macro_precision",
                    help="Macro Precision: when the model extracts a value, how often is it correct? Average across fields.",
                ),
                "macro_recall": st.column_config.NumberColumn(
                    "macro_recall",
                    help="Macro Recall: when a field is present in the contract, how often does the model find it? Average across fields.",
                ),
                "macro_f1": st.column_config.NumberColumn(
                    "macro_f1",
                    help="Macro F1: harmonic mean of Precision and Recall. Best single metric when both matter equally.",
                ),
                "micro_acc": st.column_config.NumberColumn(
                    "micro_acc",
                    help="Micro Accuracy: accuracy computed only over cases where the ground truth field is present (excludes absent-absent matches).",
                ),
                "elapsed_seconds": st.column_config.NumberColumn(
                    "elapsed_seconds",
                    help="Total inference time in seconds, accumulated across partial re-runs.",
                ),
            }
            st.dataframe(
                summary_df.style.background_gradient(
                    cmap="RdYlGn",
                    subset=[c for c in macro_display_cols + ["micro_acc"] if c in summary_df.columns],
                ),
                column_config={k: v for k, v in _col_help.items() if k in summary_df.columns},
                use_container_width=True,
            )

            # ── Per-field table (selected metric) ─────────────────────────
            field_metric_cols = [f"{prefix}{f}" for f in FIELDS if f"{prefix}{f}" in df.columns]
            if field_metric_cols and selected_fields:
                st.subheader(f"Per-field {selected_metric.lower()}")
                chosen_cols = [f"{prefix}{f}" for f in selected_fields if f"{prefix}{f}" in filtered.columns]
                field_df = filtered[["run_name"] + chosen_cols].set_index("run_name")
                field_df.columns = [c.replace(prefix, "") for c in field_df.columns]
                st.dataframe(
                    field_df.style.background_gradient(cmap="RdYlGn", axis=None),
                    use_container_width=True,
                )

                # ── Grouped bar chart ─────────────────────────────────────
                st.subheader(f"Per-field {selected_metric.lower()} chart")
                melted = field_df.reset_index().melt(
                    id_vars="run_name",
                    var_name="field",
                    value_name="value",
                )
                fig = px.bar(
                    melted,
                    x="field",
                    y="value",
                    color="run_name",
                    barmode="group",
                    range_y=[0, 1],
                    labels={"value": selected_metric, "field": "Field", "run_name": "Run"},
                    height=500,
                )
                fig.update_layout(
                    xaxis_tickangle=-35,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                )
                st.plotly_chart(fig, use_container_width=True)

            # ── Quality vs. time scatter ───────────────────────────────────
            _metric_to_macro = {
                "Accuracy":  "macro_acc",
                "Precision": "macro_precision",
                "Recall":    "macro_recall",
                "F1":        "macro_f1",
            }
            macro_col = _metric_to_macro.get(selected_metric, "macro_acc")
            if macro_col in filtered.columns and "elapsed_seconds" in filtered.columns:
                st.subheader(f"Quality vs. time — {selected_metric}")
                st.caption(
                    "Each point is one experiment run. "
                    "Top-left = fast and accurate. Hover for details."
                )
                qt_df = filtered[["run_name", "elapsed_seconds", macro_col]].dropna().copy()
                qt_df["minutes"] = (qt_df["elapsed_seconds"] / 60).round(1)
                qt_df["model"]  = qt_df["run_name"].apply(lambda x: x.split("__")[0] if "__" in x else x)
                qt_df["config"] = qt_df["run_name"].apply(lambda x: x.split("__", 1)[1] if "__" in x else x)
                fig_qt = px.scatter(
                    qt_df,
                    x="minutes",
                    y=macro_col,
                    color="model",
                    hover_name="run_name",
                    hover_data={"minutes": True, macro_col: ":.3f", "model": False, "config": True},
                    range_y=[0, 1],
                    labels={
                        "minutes": "Elapsed time (min)",
                        macro_col: selected_metric,
                        "model":   "Model",
                        "config":  "Config",
                    },
                    height=420,
                )
                fig_qt.update_traces(marker=dict(size=13))
                fig_qt.update_layout(
                    xaxis=dict(showgrid=True),
                    yaxis=dict(showgrid=True),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                )
                st.plotly_chart(fig_qt, use_container_width=True)


# ── Tab 4: Overview ───────────────────────────────────────────────────────────
with tab_overview:
    if _comparison_df is None:
        st.info("No comparison data yet. Run experiments first.")
    else:
        df_ov = _comparison_df

        macro_cols_available = [c for c in MACRO_METRIC_COLS if c in df_ov.columns]
        if not macro_cols_available:
            st.info("Re-run benchmark to compute the full metric set (Precision, Recall, F1).")
        else:
            # Derive selected runs from sidebar filters (same logic as Tab 3)
            selected_ov_runs = [
                r for r in df_ov["run_name"].tolist()
                if "__" in r
                and r.split("__")[0] in sb_models
                and r.split("__", 1)[1] in sb_configs
            ]

            if not selected_ov_runs:
                st.warning("Select at least one model and config in the sidebar.")
            else:
                ov_df = df_ov[df_ov["run_name"].isin(selected_ov_runs)]

                # ── 1. Grouped bar chart: macro metrics per run ───────────
                st.subheader("Macro metrics comparison")
                st.caption("Higher is better for all metrics.")
                macro_melt = (
                    ov_df[["run_name"] + macro_cols_available]
                    .melt(id_vars="run_name", var_name="metric", value_name="score")
                )
                macro_melt["metric"] = macro_melt["metric"].map(MACRO_METRIC_COLS)
                fig_bar = px.bar(
                    macro_melt,
                    x="run_name",
                    y="score",
                    color="metric",
                    barmode="group",
                    range_y=[0, 1],
                    labels={"score": "Score", "run_name": "Run", "metric": "Metric"},
                    height=420,
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                fig_bar.update_layout(
                    xaxis_tickangle=-35,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                )
                st.plotly_chart(fig_bar, use_container_width=True)

                # ── 3. Heatmap: per-field F1 across runs ─────────────────
                f1_field_cols = [f"f1_{f}" for f in FIELDS if f"f1_{f}" in ov_df.columns]
                if f1_field_cols:
                    st.subheader("Per-field F1 heatmap")
                    st.caption("Cells show F1 score per field per run. Reveals which fields each model handles best.")
                    hm_df = ov_df[["run_name"] + f1_field_cols].set_index("run_name")
                    hm_df.columns = [c.replace("f1_", "") for c in hm_df.columns]
                    # Sort columns by average F1 ascending (hardest field on the left)
                    hm_df = hm_df[hm_df.mean().sort_values().index]
                    fig_hm = px.imshow(
                        hm_df,
                        color_continuous_scale="RdYlGn",
                        zmin=0, zmax=1,
                        aspect="auto",
                        text_auto=".2f",
                        labels={"x": "Field", "y": "Run", "color": "F1"},
                        height=max(300, 60 * len(selected_ov_runs) + 100),
                    )
                    fig_hm.update_layout(
                        xaxis_tickangle=-35,
                        coloraxis_colorbar=dict(title="F1"),
                    )
                    fig_hm.update_traces(textfont=dict(size=11))
                    st.plotly_chart(fig_hm, use_container_width=True)

                # ── 4. Contract difficulty ranking ────────────────────────
                st.subheader("Contract difficulty ranking")
                st.caption("Average accuracy per contract across all selected runs. Lower = harder.")
                all_detailed_dfs = []
                for rn in selected_ov_runs:
                    det_path = EXPERIMENTS_DIR / rn / "results" / "benchmark_detailed.csv"
                    if det_path.exists():
                        _d = pd.read_csv(det_path)
                        _d["run_name"] = rn
                        all_detailed_dfs.append(_d)
                if all_detailed_dfs:
                    combined_det = pd.concat(all_detailed_dfs, ignore_index=True)
                    combined_det["match_int"] = combined_det["match"].map(
                        {1: 1, 0: 0, "1": 1, "0": 0}
                    ).fillna(0)
                    difficulty = (
                        combined_det.groupby("file_key")["match_int"]
                        .mean()
                        .reset_index()
                        .rename(columns={"file_key": "Contract", "match_int": "Avg Accuracy"})
                        .sort_values("Avg Accuracy")
                        .reset_index(drop=True)
                    )
                    difficulty["Avg Accuracy"] = difficulty["Avg Accuracy"].round(3)

                    # Dot plot — full distribution, contract name in hover
                    fig_dot = go.Figure(go.Scatter(
                        x=difficulty["Avg Accuracy"],
                        y=difficulty.index,
                        mode="markers",
                        marker=dict(
                            color=difficulty["Avg Accuracy"],
                            colorscale="RdYlGn",
                            cmin=0, cmax=1,
                            size=10,
                            colorbar=dict(title="Accuracy"),
                        ),
                        text=difficulty["Contract"],
                        hovertemplate="<b>%{text}</b><br>Avg accuracy: %{x:.3f}<extra></extra>",
                    ))
                    fig_dot.update_layout(
                        height=320,
                        xaxis=dict(title="Avg accuracy", range=[0, 1], showgrid=True),
                        yaxis=dict(visible=False),
                        margin=dict(l=10, r=10, t=10, b=40),
                    )
                    st.plotly_chart(fig_dot, use_container_width=True)

                    # Table: N hardest contracts
                    top_n = st.slider("Show N hardest contracts", 5, len(difficulty), 10, key="diff_top_n")
                    st.dataframe(
                        difficulty.head(top_n).set_index("Contract").style.background_gradient(
                            cmap="RdYlGn", vmin=0, vmax=1
                        ),
                        use_container_width=True,
                    )
                else:
                    st.info("No benchmark_detailed.csv found for selected runs.")

                # ── 5. Confusion matrix — binary fields ───────────────────
                binary_fields = sorted(YES_NO_FIELDS)
                bin_det_dfs = [
                    d for d in (
                        pd.read_csv(EXPERIMENTS_DIR / rn / "results" / "benchmark_detailed.csv")
                        .assign(run_name=rn)
                        for rn in selected_ov_runs
                        if (EXPERIMENTS_DIR / rn / "results" / "benchmark_detailed.csv").exists()
                    )
                ]
                if bin_det_dfs:
                    st.subheader("Prediction breakdown — binary fields")
                    st.caption(
                        "Each bar shows how predictions break down per field. "
                        "Green = correct (TP/TN), red/orange = errors (FP/FN). "
                        "Positive class = Yes."
                    )
                    bin_run = st.selectbox(
                        "Run", options=selected_ov_runs, key="cm_run_select"
                    )
                    bin_df = next(
                        (d for d in bin_det_dfs if d["run_name"].iloc[0] == bin_run), None
                    )
                    if bin_df is not None:
                        bin_df = bin_df[bin_df["field"].isin(binary_fields)].copy()
                        cm_rows = []
                        for field in binary_fields:
                            fdf = bin_df[bin_df["field"] == field]
                            tp = int(((fdf["pred_norm"] == "Yes") & (fdf["gt_norm"] == "Yes")).sum())
                            fp = int(((fdf["pred_norm"] == "Yes") & (fdf["gt_norm"] == "No")).sum())
                            fn = int(((fdf["pred_norm"] == "No")  & (fdf["gt_norm"] == "Yes")).sum())
                            tn = int(((fdf["pred_norm"] == "No")  & (fdf["gt_norm"] == "No")).sum())
                            cm_rows.append({"field": field, "TP": tp, "FP": fp, "FN": fn, "TN": tn})
                        cm_df = pd.DataFrame(cm_rows)

                        # Stacked horizontal bar: one bar per field, segments colored by outcome
                        _CM_COLORS = {
                            "TP": "#27ae60",   # green  — predicted Yes, was Yes
                            "FN": "#e67e22",   # orange — predicted No,  was Yes  (missed)
                            "FP": "#e74c3c",   # red    — predicted Yes, was No   (hallucinated)
                            "TN": "#95a5a6",   # gray   — predicted No,  was No
                        }
                        _CM_LABELS = {
                            "TP": "TP — correctly found (Yes→Yes)",
                            "FN": "FN — missed clause (No→Yes)",
                            "FP": "FP — hallucinated (Yes→No)",
                            "TN": "TN — correctly absent (No→No)",
                        }
                        fig_cm = go.Figure()
                        for cat in ["TP", "FN", "FP", "TN"]:
                            fig_cm.add_trace(go.Bar(
                                name=_CM_LABELS[cat],
                                y=cm_df["field"],
                                x=cm_df[cat],
                                orientation="h",
                                marker_color=_CM_COLORS[cat],
                                text=cm_df[cat],
                                textposition="inside",
                                insidetextanchor="middle",
                            ))
                        fig_cm.update_layout(
                            barmode="stack",
                            height=max(300, 60 * len(binary_fields) + 120),
                            xaxis_title="Number of contracts",
                            yaxis_title="",
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                            margin=dict(l=10, r=10, t=10, b=10),
                        )
                        st.plotly_chart(fig_cm, use_container_width=True)


# ── Tab 5: Manage experiments ─────────────────────────────────────────────────
with tab_manage:
    st.subheader("Manage experiments")

    all_exp_dirs = (
        sorted([d for d in EXPERIMENTS_DIR.iterdir() if d.is_dir()])
        if EXPERIMENTS_DIR.exists()
        else []
    )

    if not all_exp_dirs:
        st.info("No experiments found.")
    else:
        # ── Quick-select helpers ───────────────────────────────────────────
        all_names   = [d.name for d in all_exp_dirs]
        all_models  = sorted({n.split("__")[0] for n in all_names if "__" in n})
        all_configs = sorted({n.split("__", 1)[1] for n in all_names if "__" in n})

        st.caption("Use the filters to pre-select a subset, then adjust individual checkboxes freely.")
        fcol1, fcol2, fcol3, fcol4, fcol5 = st.columns([2, 2, 1, 1, 1], vertical_alignment="bottom")
        with fcol1:
            filter_models  = st.multiselect("Models",  all_models,  default=[], key="mgmt_models",  placeholder="All models")
        with fcol2:
            filter_configs = st.multiselect("Configs", all_configs, default=[], key="mgmt_configs", placeholder="All configs")
        with fcol3:
            if st.button("Apply", use_container_width=True, help="Select experiments matching the filters above"):
                active_models  = filter_models  or all_models
                active_configs = filter_configs or all_configs
                for n in all_names:
                    model = n.split("__")[0] if "__" in n else ""
                    cfg   = n.split("__", 1)[1] if "__" in n else ""
                    if model in active_models and cfg in active_configs:
                        st.session_state[f"del_chk_{n}"] = True
        with fcol4:
            if st.button("☑️ All", use_container_width=True, help="Select all experiments"):
                for n in all_names:
                    st.session_state[f"del_chk_{n}"] = True
        with fcol5:
            if st.button("☐ None", use_container_width=True, help="Deselect all"):
                for n in all_names:
                    st.session_state[f"del_chk_{n}"] = False

        st.divider()

        # ── Checkbox list ─────────────────────────────────────────────────
        for exp_dir in all_exp_dirs:
            n = exp_dir.name
            model_tag = n.split("__")[0] if "__" in n else "—"
            cfg_tag   = n.split("__", 1)[1] if "__" in n else "—"
            size_mb   = sum(f.stat().st_size for f in exp_dir.rglob("*") if f.is_file()) / 1e6
            label     = f"**{model_tag}** · `{cfg_tag}` · {size_mb:.0f} MB"
            st.checkbox(label, key=f"del_chk_{n}", value=st.session_state.get(f"del_chk_{n}", False))

        st.divider()

        selected_to_delete = [n for n in all_names if st.session_state.get(f"del_chk_{n}", False)]
        st.caption(f"{len(selected_to_delete)} of {len(all_names)} experiment(s) selected.")

        if selected_to_delete:
            confirmed_del = st.checkbox(
                f"Confirm: permanently delete **{len(selected_to_delete)}** experiment(s).",
                key="mgmt_confirm",
            )
            if st.button(
                f"🗑️ Delete {len(selected_to_delete)} experiment(s)",
                type="primary",
                disabled=not confirmed_del,
            ):
                for n in selected_to_delete:
                    shutil.rmtree(EXPERIMENTS_DIR / n, ignore_errors=True)
                    # Remove from comparison data
                    if COMPARISON_CSV.exists():
                        rows = load_existing_comparison()
                        rows = [r for r in rows if r.get("run_name") != n]
                        save_comparison_csv(rows, str(COMPARISON_CSV))
                st.success(f"Deleted: {', '.join(selected_to_delete)}")
                st.rerun()
