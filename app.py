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

from experiment_config import RunConfig, MODEL_IDS, PROMPT_VERSIONS
from run_experiment import run as run_inference
from benchmark import benchmark as run_benchmark
from run_all import load_benchmark_summary, save_comparison_csv, FIELDS

# Register Qwen3 prompt variant (mirrors run_all.py)
if "v3_full_qwen3" not in PROMPT_VERSIONS:
    PROMPT_VERSIONS["v3_full_qwen3"] = PROMPT_VERSIONS["v3_full"] + "\n/no_think"

# ── Constants ─────────────────────────────────────────────────────────────────
DEFAULT_INPUT_DIR = "./Dataset_Filtrado_Tesis"
GROUND_TRUTH_CSV  = "./ground_truth.csv"
EXPERIMENTS_DIR   = Path("./experiments")
COMPARISON_CSV    = EXPERIMENTS_DIR / "comparison.csv"

MODEL_MAX_CONTEXT: dict[str, int] = {
    "llama31_8b": 45000,
    "qwen25_7b":  30000,
    "qwen3_8b":   30000,
}

MODEL_PROMPT_OVERRIDE: dict[str, str] = {
    "qwen3_8b": "v3_full_qwen3",
}

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

tab_run, tab_results, tab_comparison, tab_overview = st.tabs([
    "⚙️ Configure & Run",
    "📄 Results",
    "📊 Comparison",
    "📈 Overview",
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


# ── Tab 2: Results ────────────────────────────────────────────────────────────
with tab_results:
    exp_dirs = (
        sorted([d for d in EXPERIMENTS_DIR.iterdir() if d.is_dir() and (d / "results").exists()])
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

        # Show inference summary if available
        summary_path = results_path / "inference_summary.json"
        if summary_path.exists():
            with st.expander("Inference summary (last run only)"):
                st.caption("Stats below reflect only the most recent inference run for this experiment, not the cumulative total.")
                st.json(json.loads(summary_path.read_text(encoding="utf-8")))


# ── Load comparison data once (shared by sidebar, Tab 3, and Tab 4) ──────────
_comparison_df = pd.read_csv(COMPARISON_CSV) if COMPARISON_CSV.exists() else None

# ── Sidebar filters ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🔍 Filters")
    st.caption("Applied to Comparison and Overview tabs.")
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
            summary_df   = filtered[summary_cols].set_index("run_name")
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

                # ── 2. Radar chart: metric profile per run ────────────────
                st.subheader("Metric profile (radar)")
                st.caption("Shows the balance between Accuracy, Precision, Recall, and F1 per run.")
                radar_labels = [MACRO_METRIC_COLS[c] for c in macro_cols_available]
                fig_radar = go.Figure()
                for _, row in ov_df.iterrows():
                    values = [float(row.get(c) or 0) for c in macro_cols_available]
                    values += values[:1]  # close polygon
                    fig_radar.add_trace(go.Scatterpolar(
                        r=values,
                        theta=radar_labels + [radar_labels[0]],
                        fill="toself",
                        name=row["run_name"],
                        opacity=0.55,
                    ))
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    height=520,
                    legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
                )
                st.plotly_chart(fig_radar, use_container_width=True)

                # ── 3. Heatmap: per-field F1 across runs ─────────────────
                f1_field_cols = [f"f1_{f}" for f in FIELDS if f"f1_{f}" in ov_df.columns]
                if f1_field_cols:
                    st.subheader("Per-field F1 heatmap")
                    st.caption("Cells show F1 score per field per run. Reveals which fields each model handles best.")
                    hm_df = ov_df[["run_name"] + f1_field_cols].set_index("run_name")
                    hm_df.columns = [c.replace("f1_", "") for c in hm_df.columns]
                    fig_hm = px.imshow(
                        hm_df,
                        color_continuous_scale="RdYlGn",
                        zmin=0, zmax=1,
                        aspect="auto",
                        labels={"x": "Field", "y": "Run", "color": "F1"},
                        height=max(300, 60 * len(selected_ov_runs) + 100),
                    )
                    fig_hm.update_layout(
                        xaxis_tickangle=-35,
                        coloraxis_colorbar=dict(title="F1"),
                    )
                    st.plotly_chart(fig_hm, use_container_width=True)
