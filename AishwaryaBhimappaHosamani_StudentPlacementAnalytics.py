"""
AishwaryaBhimappaHosamani_StudentPlacementAnalytics.py
=======================================================
Student Placement Analytics and Decision Support System
IBM SkillsBuild Data Analytics with AI Internship 2026

Student  : Aishwarya Bhimappa Hosamani
USN      : 2BA23CS006
Type     : Descriptive and Diagnostic Analytics
Dataset  : train.csv (45,000 rows) + test.csv (5,000 rows) = 50,000 student records

This single file consolidates the complete project code:
  1. Imports
  2. Data loading and validation
  3. Data preparation
  4. EDA / analysis functions
  5. Insight generation
  6. Streamlit dashboard
  7. Main execution
"""

# =============================================================================
# SECTION 1 — IMPORTS
# =============================================================================

import os
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import pointbiserialr


# =============================================================================
# SECTION 2 — DATA LOADING AND VALIDATION
# =============================================================================

# ---------------------------------------------------------------------------
# Paths (resolved relative to this file so the script runs from any directory)
# ---------------------------------------------------------------------------
_FILE_DIR    = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH   = os.path.join(_FILE_DIR, "train.csv")
TEST_PATH    = os.path.join(_FILE_DIR, "test.csv")
DATA_DIR     = os.path.join(_FILE_DIR, "data")
OUTPUT_PATH  = os.path.join(DATA_DIR, "processed_data.csv")

# ---------------------------------------------------------------------------
# Expected schema constants
# ---------------------------------------------------------------------------
EXPECTED_COLUMNS = [
    "Student_ID", "Age", "Gender", "Degree", "Branch", "CGPA",
    "Internships", "Projects", "Coding_Skills", "Communication_Skills",
    "Aptitude_Test_Score", "Soft_Skills_Rating", "Certifications",
    "Backlogs", "Placement_Status",
]

EXPECTED_CATEGORICAL = {
    "Gender":           {"Male", "Female"},
    "Degree":           {"B.Tech", "B.Sc", "BCA", "MCA"},
    "Branch":           {"CSE", "IT", "ECE", "ME", "Civil"},
    "Placement_Status": {"Placed", "Not Placed"},
}

EXPECTED_NUMERIC_RANGES = {
    "Age":                  (18,   24),
    "CGPA":                 (4.50, 9.80),
    "Internships":          (0,    3),
    "Projects":             (1,    6),
    "Coding_Skills":        (1,    10),
    "Communication_Skills": (1,    10),
    "Aptitude_Test_Score":  (35,   100),
    "Soft_Skills_Rating":   (1,    10),
    "Certifications":       (0,    3),
    "Backlogs":             (0,    3),
}


def _section(title: str) -> None:
    """Print a section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def load_raw_files() -> tuple:
    """Load train.csv and test.csv without modifying them."""
    print(f"\nLoading files from: {_FILE_DIR}")
    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)
    print(f"  train.csv loaded  ->  {train.shape[0]:,} rows  x  {train.shape[1]} columns")
    print(f"  test.csv  loaded  ->  {test.shape[0]:,} rows  x  {test.shape[1]} columns")
    return train, test


def check_schema(train: pd.DataFrame, test: pd.DataFrame) -> list:
    """Verify both files have identical column names in the same order."""
    _section("Schema Consistency Check")
    issues = []
    if list(train.columns) == list(test.columns) == EXPECTED_COLUMNS:
        print("\n  [OK]  Column names and order match the expected schema in both files.")
    else:
        issues.append("Column order mismatch detected between train.csv and/or test.csv")
    return issues


def validate_categoricals(df: pd.DataFrame, name: str) -> list:
    """Check that categorical columns contain only expected values."""
    issues = []
    for col, allowed in EXPECTED_CATEGORICAL.items():
        if col not in df.columns:
            issues.append(f"{name} | Column '{col}' is missing")
            continue
        found_values = set(df[col].dropna().unique())
        unexpected   = found_values - allowed
        if unexpected:
            issues.append(f"{name} | '{col}' contains unexpected values: {unexpected}")
    return issues


def validate_numeric_ranges(df: pd.DataFrame, name: str) -> list:
    """Check that numeric columns stay within documented ranges."""
    issues = []
    for col, (lo, hi) in EXPECTED_NUMERIC_RANGES.items():
        if col not in df.columns:
            issues.append(f"{name} | Column '{col}' is missing")
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        out_of_range = ((series < lo) | (series > hi)).sum()
        if out_of_range > 0:
            issues.append(f"{name} | '{col}' has {out_of_range} row(s) outside [{lo}, {hi}]")
    return issues


def check_duplicate_ids(df: pd.DataFrame, name: str) -> list:
    """Check for duplicate Student_ID values within a single file."""
    issues = []
    dupes = df[df.duplicated(subset=["Student_ID"], keep=False)]
    if not dupes.empty:
        count = df["Student_ID"].duplicated().sum()
        issues.append(f"{name} | {count} duplicate Student_ID(s) found")
    return issues


def check_id_overlap(train: pd.DataFrame, test: pd.DataFrame) -> list:
    """Check whether any Student_ID appears in both train.csv and test.csv."""
    issues = []
    overlap = set(train["Student_ID"]) & set(test["Student_ID"])
    if overlap:
        issues.append(f"{len(overlap)} Student_ID(s) overlap between train.csv and test.csv")
    return issues


def run_validation(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """
    Run all validation checks and print a summary.
    Used at startup to confirm dataset quality before analysis.
    """
    _section("DATA VALIDATION")
    all_issues = []
    all_issues += check_schema(train, test)
    all_issues += validate_categoricals(train, "train.csv")
    all_issues += validate_categoricals(test,  "test.csv")
    all_issues += validate_numeric_ranges(train, "train.csv")
    all_issues += validate_numeric_ranges(test,  "test.csv")
    all_issues += check_duplicate_ids(train, "train.csv")
    all_issues += check_duplicate_ids(test,  "test.csv")
    all_issues += check_id_overlap(train, test)

    if not all_issues:
        print("\n  [OK]  No issues found. Both files passed all validation checks.")
    else:
        print(f"\n  {len(all_issues)} issue(s) detected:")
        for i, issue in enumerate(all_issues, start=1):
            print(f"  [{i}] {issue}")
    print("\n" + "=" * 60 + "\n")


# =============================================================================
# SECTION 3 — DATA PREPARATION
# =============================================================================

# CGPA band definitions
CGPA_BINS   = [0.0,  6.0,  7.0,  8.0,  9.0, 10.0]
CGPA_LABELS = ["< 6.0", "6.0-6.9", "7.0-7.9", "8.0-8.9", "9.0+"]
CGPA_BAND_ORDER = ["< 6.0", "6.0-6.9", "7.0-7.9", "8.0-8.9", "9.0+"]


def prepare_data() -> pd.DataFrame:
    """
    Full data preparation pipeline:
      1. Load train.csv and test.csv, tag with 'split' column.
      2. Combine into a single DataFrame.
      3. Strip whitespace from string columns.
      4. Standardise categorical casing.
      5. Drop Student_ID (surrogate key).
      6. Add derived columns: Placement_Binary, CGPA_Band, Skills_Composite.
      7. Save to data/processed_data.csv.
      8. Return the processed DataFrame.

    Returns
    -------
    pd.DataFrame  (50,000 rows, 18 columns)
    """
    _section("DATA PREPARATION")

    # --- Load and tag splits ---
    train = pd.read_csv(TRAIN_PATH)
    train["split"] = "train"
    test  = pd.read_csv(TEST_PATH)
    test["split"]  = "test"
    df = pd.concat([train, test], ignore_index=True)
    print(f"  Combined: {len(df):,} rows")

    # --- Strip whitespace ---
    for col in df.select_dtypes(include=["object", "str"]).columns:
        df[col] = df[col].str.strip()

    # --- Standardise categorical casing ---
    casing_map = {
        "Gender":           {v.lower(): v for v in ["Male", "Female"]},
        "Degree":           {v.lower(): v for v in ["B.Tech", "B.Sc", "BCA", "MCA"]},
        "Branch":           {v.lower(): v for v in ["CSE", "IT", "ECE", "ME", "Civil"]},
        "Placement_Status": {v.lower(): v for v in ["Placed", "Not Placed"]},
    }
    for col, mapping in casing_map.items():
        if col in df.columns:
            df[col] = df[col].str.strip().str.lower().map(mapping).fillna(df[col])

    # --- Drop Student_ID ---
    df = df.drop(columns=["Student_ID"])

    # --- Add derived columns ---
    df["Placement_Binary"] = df["Placement_Status"].map({"Placed": 1, "Not Placed": 0})
    df["CGPA_Band"] = pd.cut(
        df["CGPA"],
        bins=CGPA_BINS,
        labels=CGPA_LABELS,
        right=False,
        include_lowest=True,
    )
    df["Skills_Composite"] = (
        (df["Coding_Skills"] + df["Communication_Skills"] + df["Soft_Skills_Rating"]) / 3
    ).round(2)

    # --- Save processed dataset ---
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"  Processed dataset saved -> {OUTPUT_PATH}")
    print(f"  Final shape: {df.shape[0]:,} rows x {df.shape[1]} columns\n")

    return df


def load_processed_data(split: str = "all") -> pd.DataFrame:
    """
    Load data/processed_data.csv and optionally filter by split.

    Parameters
    ----------
    split : "train" | "test" | "all"

    Returns
    -------
    pd.DataFrame
    """
    df = pd.read_csv(OUTPUT_PATH)
    df["CGPA_Band"] = pd.Categorical(df["CGPA_Band"], categories=CGPA_BAND_ORDER, ordered=True)
    if split in ("train", "test"):
        df = df[df["split"] == split].reset_index(drop=True)
    return df


# =============================================================================
# SECTION 4 — EDA / ANALYSIS FUNCTIONS
# =============================================================================

# ---------------------------------------------------------------------------
# Colour palette (applied consistently across all charts)
# ---------------------------------------------------------------------------
PLACED_COLOR     = "#2ecc71"   # green
NOT_PLACED_COLOR = "#e74c3c"   # red
COLOR_MAP        = {"Placed": PLACED_COLOR, "Not Placed": NOT_PLACED_COLOR}


# --- Placement Overview ---

def placement_kpis(df: pd.DataFrame) -> dict:
    """Return a dict of key performance indicators computed from df."""
    total        = len(df)
    placed_mask  = df["Placement_Status"] == "Placed"
    total_placed = int(placed_mask.sum())
    total_not    = total - total_placed
    rate         = round(total_placed / total * 100, 1) if total > 0 else 0.0
    return {
        "total_students":      total,
        "total_placed":        total_placed,
        "total_not_placed":    total_not,
        "placement_rate_pct":  rate,
        "avg_cgpa_placed":     round(df.loc[placed_mask,  "CGPA"].mean(), 2),
        "avg_cgpa_not_placed": round(df.loc[~placed_mask, "CGPA"].mean(), 2),
        "avg_cgpa_overall":    round(df["CGPA"].mean(), 2),
    }


def plot_placement_distribution(df: pd.DataFrame) -> go.Figure:
    """Side-by-side pie chart and bar chart: Placed vs Not Placed."""
    counts = df["Placement_Status"].value_counts().reset_index()
    counts.columns = ["Placement_Status", "Count"]
    counts["Percentage"] = (counts["Count"] / len(df) * 100).round(1)

    fig = go.Figure()
    fig.add_trace(go.Pie(
        labels=counts["Placement_Status"],
        values=counts["Count"],
        marker_colors=[COLOR_MAP.get(s, "#999") for s in counts["Placement_Status"]],
        textinfo="label+percent",
        hovertemplate="%{label}: %{value:,} students (%{percent})<extra></extra>",
        domain={"x": [0, 0.45]},
        name="Pie",
    ))
    fig.add_trace(go.Bar(
        x=counts["Placement_Status"],
        y=counts["Count"],
        marker_color=[COLOR_MAP.get(s, "#999") for s in counts["Placement_Status"]],
        text=[f"{r['Count']:,}<br>({r['Percentage']}%)" for _, r in counts.iterrows()],
        textposition="outside",
        hovertemplate="%{x}: %{y:,} students<extra></extra>",
        xaxis="x2", yaxis="y2", name="Bar",
    ))
    fig.update_layout(
        title="Overall Placement Distribution",
        xaxis2={"domain": [0.55, 1.0], "anchor": "y2"},
        yaxis2={"anchor": "x2", "title": "Number of Students"},
        showlegend=False, height=400,
    )
    return fig


# --- Academic Performance ---

def plot_cgpa_distribution(df: pd.DataFrame) -> go.Figure:
    """Histogram of CGPA coloured by Placement_Status."""
    fig = px.histogram(
        df, x="CGPA", color="Placement_Status",
        color_discrete_map=COLOR_MAP, nbins=40,
        barmode="overlay", opacity=0.7,
        title="CGPA Distribution by Placement Status",
        labels={"CGPA": "CGPA", "count": "Number of Students"},
    )
    fig.update_layout(xaxis_title="CGPA", yaxis_title="Number of Students",
                      legend_title="Placement Status", height=400)
    return fig


def plot_cgpa_band_vs_placement(df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart: Placed/Not Placed count per CGPA band + rate line."""
    agg = (df.groupby(["CGPA_Band", "Placement_Status"], observed=True)
             .size().reset_index(name="Count"))
    rate = (df.groupby("CGPA_Band", observed=True)["Placement_Binary"]
              .mean().mul(100).round(1).reset_index())
    rate.columns = ["CGPA_Band", "Placement_Rate"]

    fig = go.Figure()
    for status in ["Placed", "Not Placed"]:
        subset = agg[agg["Placement_Status"] == status]
        fig.add_trace(go.Bar(
            x=subset["CGPA_Band"].astype(str), y=subset["Count"],
            name=status, marker_color=COLOR_MAP[status],
            hovertemplate=f"{status}: %{{y:,}} students<extra></extra>",
        ))
    fig.add_trace(go.Scatter(
        x=rate["CGPA_Band"].astype(str), y=rate["Placement_Rate"],
        mode="lines+markers+text", name="Placement Rate %",
        line={"color": "#2c3e50", "width": 2, "dash": "dot"}, marker={"size": 8},
        text=[f"{v:.1f}%" for v in rate["Placement_Rate"]],
        textposition="top center", yaxis="y2",
        hovertemplate="Rate: %{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        title="Placement by CGPA Band (Count + Rate)",
        xaxis_title="CGPA Band", yaxis_title="Number of Students",
        yaxis2={"overlaying": "y", "side": "right",
                "title": "Placement Rate (%)", "range": [0, 100]},
        barmode="group", legend_title="", height=450,
    )
    return fig


def plot_placement_by_degree(df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart: Placed/Not Placed per Degree + rate line."""
    agg  = df.groupby(["Degree", "Placement_Status"]).size().reset_index(name="Count")
    rate = (df.groupby("Degree")["Placement_Binary"].mean().mul(100).round(1)
              .reset_index(name="Rate"))
    fig = go.Figure()
    for status in ["Placed", "Not Placed"]:
        s = agg[agg["Placement_Status"] == status]
        fig.add_trace(go.Bar(x=s["Degree"], y=s["Count"],
                             name=status, marker_color=COLOR_MAP[status],
                             hovertemplate=f"{status}: %{{y:,}}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=rate["Degree"], y=rate["Rate"], mode="lines+markers+text",
        name="Placement Rate %",
        line={"color": "#2c3e50", "width": 2, "dash": "dot"},
        text=[f"{v:.1f}%" for v in rate["Rate"]], textposition="top center",
        yaxis="y2", hovertemplate="Rate: %{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        title="Placement by Degree Type (Count + Rate)",
        xaxis_title="Degree", yaxis_title="Number of Students",
        yaxis2={"overlaying": "y", "side": "right",
                "title": "Placement Rate (%)", "range": [0, 100]},
        barmode="group", legend_title="", height=450,
    )
    return fig


def plot_placement_by_branch(df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart: Placed/Not Placed per Branch + rate line."""
    agg  = df.groupby(["Branch", "Placement_Status"]).size().reset_index(name="Count")
    rate = (df.groupby("Branch")["Placement_Binary"].mean().mul(100).round(1)
              .reset_index(name="Rate"))
    fig = go.Figure()
    for status in ["Placed", "Not Placed"]:
        s = agg[agg["Placement_Status"] == status]
        fig.add_trace(go.Bar(x=s["Branch"], y=s["Count"],
                             name=status, marker_color=COLOR_MAP[status],
                             hovertemplate=f"{status}: %{{y:,}}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=rate["Branch"], y=rate["Rate"], mode="lines+markers+text",
        name="Placement Rate %",
        line={"color": "#2c3e50", "width": 2, "dash": "dot"},
        text=[f"{v:.1f}%" for v in rate["Rate"]], textposition="top center",
        yaxis="y2", hovertemplate="Rate: %{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        title="Placement by Branch (Count + Rate)",
        xaxis_title="Branch", yaxis_title="Number of Students",
        yaxis2={"overlaying": "y", "side": "right",
                "title": "Placement Rate (%)", "range": [0, 100]},
        barmode="group", legend_title="", height=450,
    )
    return fig


def plot_backlogs_vs_placement(df: pd.DataFrame) -> go.Figure:
    """Bar chart: placement rate per backlogs count."""
    agg = (df.groupby("Backlogs")
             .agg(Total=("Placement_Binary", "count"), Placed=("Placement_Binary", "sum"))
             .reset_index())
    agg["Rate"] = (agg["Placed"] / agg["Total"] * 100).round(1)
    fig = go.Figure(go.Bar(
        x=agg["Backlogs"].astype(str), y=agg["Rate"], marker_color="#3498db",
        text=[f"{r:.1f}%<br>(n={t:,})" for r, t in zip(agg["Rate"], agg["Total"])],
        textposition="outside",
        hovertemplate="Backlogs=%{x}: %{y:.1f}% placed<extra></extra>",
    ))
    fig.update_layout(title="Placement Rate by Number of Backlogs",
                      xaxis_title="Number of Backlogs", yaxis_title="Placement Rate (%)",
                      yaxis_range=[0, 100], height=400)
    return fig


# --- Demographic Analysis ---

def plot_placement_by_gender(df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart: Placed/Not Placed per Gender + rate annotations."""
    agg  = df.groupby(["Gender", "Placement_Status"]).size().reset_index(name="Count")
    rate = (df.groupby("Gender")["Placement_Binary"].mean().mul(100).round(1)
              .reset_index(name="Rate"))
    fig = go.Figure()
    for status in ["Placed", "Not Placed"]:
        s = agg[agg["Placement_Status"] == status]
        fig.add_trace(go.Bar(x=s["Gender"], y=s["Count"],
                             name=status, marker_color=COLOR_MAP[status],
                             hovertemplate=f"{status}: %{{y:,}}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=rate["Gender"], y=rate["Rate"], mode="markers+text",
        name="Placement Rate %",
        marker={"size": 12, "color": "#2c3e50", "symbol": "diamond"},
        text=[f"{v:.1f}%" for v in rate["Rate"]], textposition="top center",
        yaxis="y2", hovertemplate="Rate: %{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        title="Placement by Gender (Count + Rate)",
        xaxis_title="Gender", yaxis_title="Number of Students",
        yaxis2={"overlaying": "y", "side": "right",
                "title": "Placement Rate (%)", "range": [0, 100]},
        barmode="group", legend_title="", height=400,
    )
    return fig


def plot_age_vs_placement(df: pd.DataFrame) -> go.Figure:
    """Box plot: Age distribution for Placed vs Not Placed."""
    fig = px.box(df, x="Placement_Status", y="Age",
                 color="Placement_Status", color_discrete_map=COLOR_MAP,
                 title="Age Distribution by Placement Status",
                 labels={"Placement_Status": "Placement Status", "Age": "Age"},
                 points="outliers")
    fig.update_layout(xaxis_title="Placement Status", yaxis_title="Age",
                      showlegend=False, height=400)
    return fig


# --- Skills Analysis ---

def _plot_skill_vs_placement_rate(df: pd.DataFrame, col: str, title: str) -> go.Figure:
    """Helper: bar chart of placement rate for each integer skill score value."""
    agg = (df.groupby(col)
             .agg(Total=("Placement_Binary", "count"), Placed=("Placement_Binary", "sum"))
             .reset_index())
    agg["Rate"] = (agg["Placed"] / agg["Total"] * 100).round(1)
    fig = go.Figure(go.Bar(
        x=agg[col].astype(str), y=agg["Rate"], marker_color="#9b59b6",
        text=[f"{r:.1f}%<br>(n={t:,})" for r, t in zip(agg["Rate"], agg["Total"])],
        textposition="outside",
        hovertemplate=f"{col}=%{{x}}: %{{y:.1f}}% placed<extra></extra>",
    ))
    fig.update_layout(title=title, xaxis_title=f"{col} Score (1-10)",
                      yaxis_title="Placement Rate (%)", yaxis_range=[0, 100], height=400)
    return fig


def plot_coding_skills_vs_placement(df: pd.DataFrame) -> go.Figure:
    return _plot_skill_vs_placement_rate(df, "Coding_Skills",
                                         "Placement Rate by Coding Skills Score")


def plot_communication_skills_vs_placement(df: pd.DataFrame) -> go.Figure:
    return _plot_skill_vs_placement_rate(df, "Communication_Skills",
                                         "Placement Rate by Communication Skills Score")


def plot_soft_skills_vs_placement(df: pd.DataFrame) -> go.Figure:
    return _plot_skill_vs_placement_rate(df, "Soft_Skills_Rating",
                                         "Placement Rate by Soft Skills Rating")


def plot_aptitude_vs_placement(df: pd.DataFrame) -> go.Figure:
    """Box plot of Aptitude_Test_Score for Placed vs Not Placed."""
    median_score = df["Aptitude_Test_Score"].median()
    fig = px.box(df, x="Placement_Status", y="Aptitude_Test_Score",
                 color="Placement_Status", color_discrete_map=COLOR_MAP,
                 title="Aptitude Test Score by Placement Status",
                 labels={"Placement_Status": "Placement Status",
                         "Aptitude_Test_Score": "Aptitude Test Score"},
                 points="outliers")
    fig.add_hline(y=median_score, line_dash="dash", line_color="#7f8c8d",
                  annotation_text=f"Overall median: {median_score:.0f}",
                  annotation_position="top right")
    fig.update_layout(xaxis_title="Placement Status", yaxis_title="Aptitude Test Score",
                      showlegend=False, height=400)
    return fig


def plot_skills_composite_vs_placement(df: pd.DataFrame) -> go.Figure:
    """Box plot of Skills_Composite for Placed vs Not Placed."""
    fig = px.box(df, x="Placement_Status", y="Skills_Composite",
                 color="Placement_Status", color_discrete_map=COLOR_MAP,
                 title="Skills Composite Score by Placement Status",
                 labels={"Placement_Status": "Placement Status",
                         "Skills_Composite": "Skills Composite (avg of 3 skill scores)"},
                 points="outliers")
    fig.update_layout(xaxis_title="Placement Status", yaxis_title="Skills Composite Score",
                      showlegend=False, height=400)
    return fig


# --- Activity Analysis ---

def _plot_activity_vs_placement(df: pd.DataFrame, col: str, title: str,
                                 x_label: str) -> go.Figure:
    """Helper: grouped bar + rate line for Internships, Projects, Certifications."""
    agg  = df.groupby([col, "Placement_Status"]).size().reset_index(name="Count")
    rate = (df.groupby(col)["Placement_Binary"].mean().mul(100).round(1)
              .reset_index(name="Rate"))
    fig = go.Figure()
    for status in ["Placed", "Not Placed"]:
        s = agg[agg["Placement_Status"] == status]
        fig.add_trace(go.Bar(x=s[col].astype(str), y=s["Count"],
                             name=status, marker_color=COLOR_MAP[status],
                             hovertemplate=f"{status}: %{{y:,}}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=rate[col].astype(str), y=rate["Rate"], mode="lines+markers+text",
        name="Placement Rate %",
        line={"color": "#2c3e50", "width": 2, "dash": "dot"}, marker={"size": 8},
        text=[f"{v:.1f}%" for v in rate["Rate"]], textposition="top center",
        yaxis="y2", hovertemplate="Rate: %{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(title=title, xaxis_title=x_label, yaxis_title="Number of Students",
                      yaxis2={"overlaying": "y", "side": "right",
                              "title": "Placement Rate (%)", "range": [0, 100]},
                      barmode="group", legend_title="", height=450)
    return fig


def plot_internships_vs_placement(df: pd.DataFrame) -> go.Figure:
    return _plot_activity_vs_placement(df, "Internships",
                                       "Placement by Number of Internships (Count + Rate)",
                                       "Number of Internships")


def plot_projects_vs_placement(df: pd.DataFrame) -> go.Figure:
    return _plot_activity_vs_placement(df, "Projects",
                                       "Placement by Number of Projects (Count + Rate)",
                                       "Number of Projects")


def plot_certifications_vs_placement(df: pd.DataFrame) -> go.Figure:
    return _plot_activity_vs_placement(df, "Certifications",
                                       "Placement by Number of Certifications (Count + Rate)",
                                       "Number of Certifications")


# --- Correlation and Multi-factor ---

def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Pearson correlation heatmap for all numeric columns."""
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if df[c].nunique() > 1]
    corr = df[numeric_cols].corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
        colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
        text=np.round(corr.values, 2), texttemplate="%{text}",
        hovertemplate="x: %{x}<br>y: %{y}<br>r = %{z:.3f}<extra></extra>",
        colorbar={"title": "Pearson r"},
    ))
    fig.update_layout(title="Correlation Heatmap (Numeric Features)",
                      height=550, xaxis={"tickangle": -45})
    return fig


def plot_top_factors(df: pd.DataFrame):
    """
    Horizontal bar chart of point-biserial correlation with Placement_Binary.
    Returns (fig, results_df).
    """
    candidate_cols = [
        "Age", "CGPA", "Internships", "Projects",
        "Coding_Skills", "Communication_Skills",
        "Aptitude_Test_Score", "Soft_Skills_Rating",
        "Certifications", "Backlogs", "Skills_Composite",
    ]
    results = []
    binary = df["Placement_Binary"].dropna()
    for col in candidate_cols:
        series = df.loc[binary.index, col].dropna()
        idx = binary.index.intersection(series.index)
        if len(idx) < 10 or series.loc[idx].std() == 0:
            continue
        r, p = pointbiserialr(binary.loc[idx], series.loc[idx])
        results.append({"Feature": col, "Correlation": round(r, 4), "p_value": round(p, 4)})

    results_df = pd.DataFrame(results)
    results_df = results_df.reindex(
        results_df["Correlation"].abs().sort_values(ascending=True).index)

    bar_colors = [PLACED_COLOR if r >= 0 else NOT_PLACED_COLOR
                  for r in results_df["Correlation"]]
    fig = go.Figure(go.Bar(
        x=results_df["Correlation"], y=results_df["Feature"], orientation="h",
        marker_color=bar_colors,
        text=[f"{r:+.3f}" for r in results_df["Correlation"]],
        textposition="outside",
        hovertemplate="%{y}<br>r = %{x:.4f}<extra></extra>",
    ))
    fig.add_vline(x=0, line_color="#7f8c8d", line_width=1)
    fig.update_layout(
        title="Top Factors Associated with Placement (Point-Biserial Correlation)",
        xaxis_title="Point-Biserial Correlation with Placement (Placed=1)",
        yaxis_title="Feature", height=500, xaxis_range=[-1, 1],
    )
    return fig, results_df


def plot_cgpa_vs_aptitude_scatter(df: pd.DataFrame) -> go.Figure:
    """Scatter plot: CGPA vs Aptitude_Test_Score coloured by Placement_Status."""
    sample = df.sample(min(3000, len(df)), random_state=42)
    fig = px.scatter(
        sample, x="CGPA", y="Aptitude_Test_Score",
        color="Placement_Status", color_discrete_map=COLOR_MAP, opacity=0.6,
        title="CGPA vs Aptitude Test Score (coloured by Placement Status)",
        labels={"CGPA": "CGPA", "Aptitude_Test_Score": "Aptitude Test Score"},
        hover_data=["Degree", "Branch", "Internships"],
    )
    fig.update_layout(xaxis_title="CGPA", yaxis_title="Aptitude Test Score",
                      legend_title="Placement Status", height=450)
    return fig


# =============================================================================
# SECTION 5 — INSIGHT GENERATION
# =============================================================================

def _placement_rate(series: pd.Series) -> float:
    """Return the placement rate (%) for a boolean/binary Series."""
    return round(series.mean() * 100, 1)


def compute_insight_data(df: pd.DataFrame) -> dict:
    """Compute all statistics needed for structured insight generation."""
    d = {}
    band_order = ["< 6.0", "6.0-6.9", "7.0-7.9", "8.0-8.9", "9.0+"]

    cgpa_rates = (df.groupby("CGPA_Band")["Placement_Binary"].mean().mul(100).round(1)
                    .reindex([b for b in band_order if b in df["CGPA_Band"].unique()]))
    d["cgpa_band_rates"]       = cgpa_rates.to_dict()
    d["cgpa_lowest_band"]      = cgpa_rates.idxmin()
    d["cgpa_lowest_rate"]      = cgpa_rates.min()
    d["cgpa_highest_band"]     = cgpa_rates.idxmax()
    d["cgpa_highest_rate"]     = cgpa_rates.max()
    cgpa_mean = df.groupby("Placement_Status")["CGPA"].mean().round(2)
    d["cgpa_mean_placed"]      = cgpa_mean.get("Placed", float("nan"))
    d["cgpa_mean_not_placed"]  = cgpa_mean.get("Not Placed", float("nan"))

    bl_rates = (df.groupby("Backlogs")["Placement_Binary"].mean().mul(100).round(1).sort_index())
    d["backlogs_rates"]  = bl_rates.to_dict()
    d["backlogs_rate_0"] = bl_rates.get(0, float("nan"))
    d["backlogs_rate_3"] = bl_rates.get(3, float("nan"))

    intern_rates = (df.groupby("Internships")["Placement_Binary"].mean().mul(100).round(1).sort_index())
    d["intern_rates"]  = intern_rates.to_dict()
    d["intern_rate_0"] = intern_rates.get(0, float("nan"))
    d["intern_rate_3"] = intern_rates.get(3, float("nan"))

    proj_rates = (df.groupby("Projects")["Placement_Binary"].mean().mul(100).round(1).sort_index())
    d["proj_rates"]     = proj_rates.to_dict()
    d["proj_rate_min"]  = proj_rates.min()
    d["proj_rate_max"]  = proj_rates.max()
    d["proj_count_min"] = proj_rates.idxmin()
    d["proj_count_max"] = proj_rates.idxmax()

    cert_rates = (df.groupby("Certifications")["Placement_Binary"].mean().mul(100).round(1).sort_index())
    d["cert_rates"]  = cert_rates.to_dict()
    d["cert_rate_0"] = cert_rates.get(0, float("nan"))
    d["cert_rate_3"] = cert_rates.get(3, float("nan"))

    coding_mean = df.groupby("Placement_Status")["Coding_Skills"].mean().round(2)
    d["coding_mean_placed"]     = coding_mean.get("Placed", float("nan"))
    d["coding_mean_not_placed"] = coding_mean.get("Not Placed", float("nan"))
    d["coding_rate_low"]  = _placement_rate(df.loc[df["Coding_Skills"] <= 4, "Placement_Binary"])
    d["coding_rate_high"] = _placement_rate(df.loc[df["Coding_Skills"] >= 8, "Placement_Binary"])

    apt_mean = df.groupby("Placement_Status")["Aptitude_Test_Score"].mean().round(1)
    d["aptitude_mean_placed"]     = apt_mean.get("Placed", float("nan"))
    d["aptitude_mean_not_placed"] = apt_mean.get("Not Placed", float("nan"))

    comm_mean = df.groupby("Placement_Status")["Communication_Skills"].mean().round(2)
    d["comm_mean_placed"]     = comm_mean.get("Placed", float("nan"))
    d["comm_mean_not_placed"] = comm_mean.get("Not Placed", float("nan"))

    soft_mean = df.groupby("Placement_Status")["Soft_Skills_Rating"].mean().round(2)
    d["soft_mean_placed"]     = soft_mean.get("Placed", float("nan"))
    d["soft_mean_not_placed"] = soft_mean.get("Not Placed", float("nan"))

    degree_rates = (df.groupby("Degree")["Placement_Binary"].mean().mul(100).round(1)
                      .sort_values(ascending=False))
    d["degree_rates"]      = degree_rates.to_dict()
    d["degree_best"]       = degree_rates.idxmax()
    d["degree_best_rate"]  = degree_rates.max()
    d["degree_worst"]      = degree_rates.idxmin()
    d["degree_worst_rate"] = degree_rates.min()

    branch_rates = (df.groupby("Branch")["Placement_Binary"].mean().mul(100).round(1)
                      .sort_values(ascending=False))
    d["branch_rates"]      = branch_rates.to_dict()
    d["branch_best"]       = branch_rates.idxmax()
    d["branch_best_rate"]  = branch_rates.max()
    d["branch_worst"]      = branch_rates.idxmin()
    d["branch_worst_rate"] = branch_rates.min()

    gender_rates = (df.groupby("Gender")["Placement_Binary"].mean().mul(100).round(1))
    d["gender_rates"]        = gender_rates.to_dict()
    d["gender_male_rate"]    = gender_rates.get("Male", float("nan"))
    d["gender_female_rate"]  = gender_rates.get("Female", float("nan"))

    d["overall_placement_rate"] = round(df["Placement_Binary"].mean() * 100, 1)
    sc_mean = df.groupby("Placement_Status")["Skills_Composite"].mean().round(2)
    d["skills_comp_mean_placed"]     = sc_mean.get("Placed", float("nan"))
    d["skills_comp_mean_not_placed"] = sc_mean.get("Not Placed", float("nan"))

    high_cgpa_intern = df[(df["CGPA"] >= 8.0) & (df["Internships"] >= 1)]
    d["high_cgpa_intern_rate"]  = round(high_cgpa_intern["Placement_Binary"].mean() * 100, 1)
    d["high_cgpa_intern_count"] = len(high_cgpa_intern)

    return d


def get_insights(df: pd.DataFrame) -> list:
    """
    Generate 12 structured, data-grounded insights with recommendations.
    All numbers are computed directly from df.
    """
    c = compute_insight_data(df)
    insights = []

    insights.append({"id": "cgpa_01", "category": "CGPA",
        "observation": (
            f"Students in the highest CGPA band ({c['cgpa_highest_band']}) showed a placement "
            f"rate of {c['cgpa_highest_rate']}%, compared to {c['cgpa_lowest_rate']}% for the "
            f"'{c['cgpa_lowest_band']}' band. Mean CGPA: placed {c['cgpa_mean_placed']} vs "
            f"not placed {c['cgpa_mean_not_placed']}."),
        "recommendation": (
            "Students should prioritise maintaining or improving their CGPA. Faculty can use "
            "CGPA-band breakdowns in mid-semester reviews to identify at-risk cohorts."),
        "audience": "Students, Faculty/Departments"})

    insights.append({"id": "backlogs_01", "category": "Backlogs",
        "observation": (
            f"Students with 0 backlogs had a placement rate of {c['backlogs_rate_0']}%; those "
            f"with 3 backlogs had {c['backlogs_rate_3']}%. Breakdown: "
            + ", ".join(f"{k} backlog(s)={v}%" for k, v in sorted(c["backlogs_rates"].items()))
            + "."),
        "recommendation": (
            "Students should address backlogs proactively. The Placement Cell may flag students "
            "with 2+ backlogs for targeted academic support."),
        "audience": "Students, Placement Cell"})

    insights.append({"id": "internships_01", "category": "Internships",
        "observation": (
            f"0 internships: {c['intern_rate_0']}% placed; 3 internships: {c['intern_rate_3']}%. "
            + ", ".join(f"{k} internship(s)={v}%" for k, v in sorted(c["intern_rates"].items()))
            + "."),
        "recommendation": (
            "Students are encouraged to pursue at least one internship. Institutions should "
            "strengthen industry tie-ups and structured internship guidance."),
        "audience": "Students, Institution"})

    insights.append({"id": "projects_01", "category": "Projects",
        "observation": (
            f"{c['proj_count_max']} project(s) = highest placement rate {c['proj_rate_max']}%; "
            f"{c['proj_count_min']} project(s) = lowest at {c['proj_rate_min']}%. Breakdown: "
            + ", ".join(f"{k} project(s)={v}%" for k, v in sorted(c["proj_rates"].items()))
            + "."),
        "recommendation": (
            "Students should complete meaningful projects. Faculty should integrate project-based "
            "learning to ensure all students have relevant experience before placement."),
        "audience": "Students, Faculty/Departments"})

    insights.append({"id": "certifications_01", "category": "Certifications",
        "observation": (
            f"0 certifications: {c['cert_rate_0']}% placed; 3 certifications: {c['cert_rate_3']}%. "
            + ", ".join(f"{k} cert(s)={v}%" for k, v in sorted(c["cert_rates"].items()))
            + "."),
        "recommendation": (
            "Students should pursue industry certifications. Placement Cells can organise "
            "certification drives or subsidised access to online platforms."),
        "audience": "Students, Placement Cell"})

    insights.append({"id": "coding_01", "category": "Coding",
        "observation": (
            f"Mean Coding_Skills: placed {c['coding_mean_placed']} vs not placed "
            f"{c['coding_mean_not_placed']}. Score <= 4: {c['coding_rate_low']}% placed; "
            f"score >= 8: {c['coding_rate_high']}% placed."),
        "recommendation": (
            "Students should invest in coding proficiency through practice platforms. Faculty "
            "can incorporate programming-intensive coursework to raise the coding baseline."),
        "audience": "Students, Faculty/Departments"})

    insights.append({"id": "aptitude_01", "category": "Aptitude",
        "observation": (
            f"Mean Aptitude_Test_Score: placed {c['aptitude_mean_placed']} vs not placed "
            f"{c['aptitude_mean_not_placed']}. Aptitude is associated with placement outcomes."),
        "recommendation": (
            "Students should practise aptitude well before placement season. The Placement Cell "
            "should schedule mock aptitude drives in the penultimate semester."),
        "audience": "Students, Placement Cell"})

    insights.append({"id": "comm_soft_01", "category": "Communication/Soft",
        "observation": (
            f"Mean Communication_Skills: placed {c['comm_mean_placed']} vs not placed "
            f"{c['comm_mean_not_placed']}. Soft_Skills_Rating: placed {c['soft_mean_placed']} "
            f"vs not placed {c['soft_mean_not_placed']}."),
        "recommendation": (
            "Students should develop communication skills through group discussions and mock "
            "interviews. Institutions should incorporate personality development from early semesters."),
        "audience": "Students, Institution"})

    insights.append({"id": "degree_01", "category": "Degree",
        "observation": (
            f"{c['degree_best']} had the highest placement rate ({c['degree_best_rate']}%); "
            f"{c['degree_worst']} had the lowest ({c['degree_worst_rate']}%). "
            + ", ".join(f"{k}={v}%" for k, v in sorted(c["degree_rates"].items(), key=lambda x: -x[1]))
            + "."),
        "recommendation": (
            "Departments with lower placement rates should examine curriculum alignment. The "
            "Placement Cell should design targeted preparation for weaker-placing degrees."),
        "audience": "Placement Cell, Faculty/Departments"})

    insights.append({"id": "branch_01", "category": "Branch",
        "observation": (
            f"{c['branch_best']} branch had the highest rate ({c['branch_best_rate']}%); "
            f"{c['branch_worst']} branch had the lowest ({c['branch_worst_rate']}%). "
            + ", ".join(f"{k}={v}%" for k, v in sorted(c["branch_rates"].items(), key=lambda x: -x[1]))
            + "."),
        "recommendation": (
            "Branches with lower placement rates may benefit from additional industry interaction, "
            "skill-development workshops, and targeted recruiting events."),
        "audience": "Institution, Placement Cell"})

    male_rate   = c["gender_male_rate"]
    female_rate = c["gender_female_rate"]
    diff        = abs(round(male_rate - female_rate, 1))
    direction   = (
        "Male students showed a slightly higher placement rate"
        if male_rate > female_rate else
        ("Female students showed a slightly higher placement rate"
         if female_rate > male_rate else
         "Male and female students showed identical placement rates")
    )
    insights.append({"id": "gender_01", "category": "Gender",
        "observation": (
            f"Male: {male_rate}% placed; Female: {female_rate}% placed. {direction} "
            f"(difference: {diff} pp). Figures describe the sample only."),
        "recommendation": (
            "The Placement Cell should monitor gender-wise outcomes each year to detect persistent "
            "disparity and ensure equal access to preparation resources."),
        "audience": "Placement Cell, Institution"})

    insights.append({"id": "multifactor_01", "category": "Multi-factor",
        "observation": (
            f"Overall placement rate: {c['overall_placement_rate']}%. Students with CGPA >= 8.0 "
            f"and >= 1 internship (n={c['high_cgpa_intern_count']}) had a combined rate of "
            f"{c['high_cgpa_intern_rate']}%. Mean Skills_Composite: placed "
            f"{c['skills_comp_mean_placed']} vs not placed {c['skills_comp_mean_not_placed']}."),
        "recommendation": (
            "No single factor explains placement outcomes. Students should build a balanced "
            "profile: solid CGPA, internship experience, and strong skills. The Institution "
            "should design holistic programmes addressing all three dimensions simultaneously."),
        "audience": "Students, Institution, Placement Cell"})

    insights.sort(key=lambda x: x["category"])
    return insights


# =============================================================================
# SECTION 6 — STREAMLIT DASHBOARD
# =============================================================================

AUDIENCE_COLORS = {
    "Students":            "#3498db",
    "Placement Cell":      "#e67e22",
    "Faculty/Departments": "#9b59b6",
    "Institution":         "#1abc9c",
}


@st.cache_data
def _load_full_data() -> pd.DataFrame:
    """Load data/processed_data.csv once and cache it in the Streamlit session."""
    df = pd.read_csv(OUTPUT_PATH)
    df["CGPA_Band"] = pd.Categorical(df["CGPA_Band"], categories=CGPA_BAND_ORDER, ordered=True)
    return df


def _apply_filters(df: pd.DataFrame, split: str, degrees: list,
                   branches: list, genders: list) -> pd.DataFrame:
    if split != "All":
        df = df[df["split"] == split.lower()]
    if degrees:
        df = df[df["Degree"].isin(degrees)]
    if branches:
        df = df[df["Branch"].isin(branches)]
    if genders:
        df = df[df["Gender"].isin(genders)]
    return df.reset_index(drop=True)


def _empty_state(message: str = "No data matches the current filters."):
    st.warning(message)


def _build_sidebar(df_full: pd.DataFrame) -> pd.DataFrame:
    with st.sidebar:
        st.title("🎓 Student Placement Analytics")
        st.markdown("Descriptive and diagnostic analytics on student placement outcomes.")
        st.divider()

        split = st.selectbox("Dataset Split", options=["All", "Train", "Test"], index=0,
                             help="Filter by training set, test set, or the combined dataset.")

        all_degrees = sorted(df_full["Degree"].unique().tolist())
        degrees = st.multiselect("Degree", options=all_degrees, default=all_degrees,
                                 help="Filter by degree programme.")

        all_branches = sorted(df_full["Branch"].unique().tolist())
        branches = st.multiselect("Branch", options=all_branches, default=all_branches,
                                  help="Filter by academic branch.")

        all_genders = sorted(df_full["Gender"].unique().tolist())
        genders = st.multiselect("Gender", options=all_genders, default=all_genders,
                                 help="Filter by gender.")

        st.divider()
        st.caption("Filters apply to all tabs.")

    return _apply_filters(df_full, split, degrees, branches, genders)


def _tab_overview(df: pd.DataFrame):
    st.subheader("Placement Overview")
    if df.empty:
        _empty_state(); return

    kpis = placement_kpis(df)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Students",     f"{kpis['total_students']:,}")
    c2.metric("Placed",             f"{kpis['total_placed']:,}")
    c3.metric("Not Placed",         f"{kpis['total_not_placed']:,}")
    c4.metric("Placement Rate",     f"{kpis['placement_rate_pct']}%")
    c5.metric("Avg CGPA (Overall)", str(kpis["avg_cgpa_overall"]))
    st.divider()

    with st.spinner("Rendering placement distribution..."):
        st.plotly_chart(plot_placement_distribution(df), use_container_width=True,
                        key="overview_placement_distribution")
    st.divider()

    col_l, col_r = st.columns(2)
    with col_l:
        with st.spinner("Rendering CGPA Band chart..."):
            st.plotly_chart(plot_cgpa_band_vs_placement(df), use_container_width=True,
                            key="overview_cgpa_band")
    with col_r:
        with st.spinner("Rendering Branch chart..."):
            st.plotly_chart(plot_placement_by_branch(df), use_container_width=True,
                            key="overview_branch")

    with st.spinner("Rendering Gender chart..."):
        st.plotly_chart(plot_placement_by_gender(df), use_container_width=True,
                        key="overview_gender")


def _tab_academic(df: pd.DataFrame):
    st.subheader("Academic Analysis")
    if df.empty:
        _empty_state(); return

    col_l, col_r = st.columns(2)
    with col_l:
        with st.spinner("Rendering CGPA distribution..."):
            st.plotly_chart(plot_cgpa_distribution(df), use_container_width=True,
                            key="academic_cgpa_distribution")
    with col_r:
        with st.spinner("Rendering CGPA Band vs Placement..."):
            st.plotly_chart(plot_cgpa_band_vs_placement(df), use_container_width=True,
                            key="academic_cgpa_band")
    st.divider()

    col_l2, col_r2 = st.columns(2)
    with col_l2:
        with st.spinner("Rendering Backlogs chart..."):
            st.plotly_chart(plot_backlogs_vs_placement(df), use_container_width=True,
                            key="academic_backlogs")
    with col_r2:
        with st.spinner("Rendering Aptitude chart..."):
            st.plotly_chart(plot_aptitude_vs_placement(df), use_container_width=True,
                            key="academic_aptitude")
    st.divider()

    with st.spinner("Rendering CGPA vs Aptitude scatter..."):
        st.plotly_chart(plot_cgpa_vs_aptitude_scatter(df), use_container_width=True,
                        key="academic_cgpa_aptitude")

    with st.spinner("Rendering Degree chart..."):
        st.plotly_chart(plot_placement_by_degree(df), use_container_width=True,
                        key="academic_degree")


def _tab_skills(df: pd.DataFrame):
    st.subheader("Skills Analysis")
    if df.empty:
        _empty_state(); return

    placed     = df[df["Placement_Status"] == "Placed"]
    not_placed = df[df["Placement_Status"] == "Not Placed"]
    if not placed.empty and not not_placed.empty:
        skill_cols = ["Coding_Skills", "Communication_Skills",
                      "Soft_Skills_Rating", "Aptitude_Test_Score", "Skills_Composite"]
        summary = pd.DataFrame({
            "Placed (mean)":     placed[skill_cols].mean().round(2),
            "Not Placed (mean)": not_placed[skill_cols].mean().round(2),
        })
        summary["Difference"] = (summary["Placed (mean)"] - summary["Not Placed (mean)"]).round(2)
        st.markdown("##### Average Skill Scores: Placed vs Not Placed")
        st.dataframe(summary, use_container_width=True)
        st.divider()

    col_l, col_r = st.columns(2)
    with col_l:
        with st.spinner("Rendering Coding Skills chart..."):
            st.plotly_chart(plot_coding_skills_vs_placement(df), use_container_width=True,
                            key="skills_coding")
    with col_r:
        with st.spinner("Rendering Communication Skills chart..."):
            st.plotly_chart(plot_communication_skills_vs_placement(df), use_container_width=True,
                            key="skills_communication")

    col_l2, col_r2 = st.columns(2)
    with col_l2:
        with st.spinner("Rendering Soft Skills chart..."):
            st.plotly_chart(plot_soft_skills_vs_placement(df), use_container_width=True,
                            key="skills_soft")
    with col_r2:
        with st.spinner("Rendering Aptitude chart..."):
            st.plotly_chart(plot_aptitude_vs_placement(df), use_container_width=True,
                            key="skills_aptitude")

    with st.spinner("Rendering Skills Composite chart..."):
        st.plotly_chart(plot_skills_composite_vs_placement(df), use_container_width=True,
                        key="skills_composite")


def _tab_activity(df: pd.DataFrame):
    st.subheader("Activity Analysis")
    if df.empty:
        _empty_state(); return

    col_l, col_r = st.columns(2)
    with col_l:
        with st.spinner("Rendering Internships chart..."):
            st.plotly_chart(plot_internships_vs_placement(df), use_container_width=True,
                            key="activity_internships")
    with col_r:
        with st.spinner("Rendering Projects chart..."):
            st.plotly_chart(plot_projects_vs_placement(df), use_container_width=True,
                            key="activity_projects")

    with st.spinner("Rendering Certifications chart..."):
        st.plotly_chart(plot_certifications_vs_placement(df), use_container_width=True,
                        key="activity_certifications")
    st.divider()

    with st.spinner("Computing top placement factors..."):
        fig_factors, factors_df = plot_top_factors(df)
        st.plotly_chart(fig_factors, use_container_width=True, key="activity_factors")
    with st.expander("View correlation values table"):
        st.dataframe(
            factors_df.sort_values("Correlation", key=abs, ascending=False)
                      .reset_index(drop=True),
            use_container_width=True,
        )
    st.divider()

    with st.spinner("Rendering correlation heatmap..."):
        st.plotly_chart(plot_correlation_heatmap(df), use_container_width=True,
                        key="activity_heatmap")


def _tab_insights(df: pd.DataFrame, df_full: pd.DataFrame):
    st.subheader("Insights & Recommendations")
    st.markdown(
        "The observations below are computed directly from the data. "
        "Each insight is linked to an actionable recommendation for the audience shown."
    )

    with st.spinner("Generating insights..."):
        insights = get_insights(df_full)

    all_audiences = ["All", "Students", "Placement Cell", "Faculty/Departments", "Institution"]
    audience_choice = st.selectbox("Filter by Audience", options=all_audiences, index=0)
    st.divider()

    shown = 0
    for ins in insights:
        if audience_choice != "All" and audience_choice not in ins["audience"]:
            continue
        shown += 1
        cat = ins["category"]
        with st.expander(f"[{cat}]  {ins['observation'][:100]}...", expanded=False):
            st.markdown("**Observation**")
            st.info(ins["observation"])
            st.markdown("**Recommendation**")
            st.success(ins["recommendation"])
            badge_html = " ".join(
                f'<span style="background:{AUDIENCE_COLORS.get(a.strip(), "#999")};'
                f'color:white;padding:2px 8px;border-radius:4px;'
                f'font-size:0.8em;margin-right:4px;">{a.strip()}</span>'
                for a in ins["audience"].split(",")
            )
            st.markdown(f"**Audience:** {badge_html}", unsafe_allow_html=True)

    if shown == 0:
        st.info(f"No insights found for audience: {audience_choice}")


def run_dashboard():
    """Entry point for the Streamlit dashboard."""
    # Page config must be first Streamlit call
    st.set_page_config(
        page_title="Student Placement Analytics",
        layout="wide",
        page_icon="🎓",
    )

    st.title("🎓 Student Placement Analytics and Decision Support System")
    st.markdown(
        "Descriptive and diagnostic analytics on placement outcomes for 50,000 students. "
        "Use the sidebar to filter by dataset split, degree, branch, or gender."
    )
    st.divider()

    with st.spinner("Loading dataset..."):
        df_full = _load_full_data()

    df = _build_sidebar(df_full)

    if len(df) < len(df_full):
        st.caption(
            f"Showing **{len(df):,}** of **{len(df_full):,}** students based on current filters."
        )

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview",
        "🎓 Academic Analysis",
        "💡 Skills Analysis",
        "🏆 Activity Analysis",
        "💬 Insights & Recommendations",
    ])

    with tab1: _tab_overview(df)
    with tab2: _tab_academic(df)
    with tab3: _tab_skills(df)
    with tab4: _tab_activity(df)
    with tab5: _tab_insights(df, df_full)

    st.divider()
    st.caption(
        "Student Placement Analytics and Decision Support System  |  "
        "Descriptive & Diagnostic Analytics  |  Dataset: train.csv + test.csv (50,000 students)"
    )


# =============================================================================
# SECTION 7 — MAIN EXECUTION
# =============================================================================

def main():
    """
    Run the full project pipeline:
      1. Load and validate raw data.
      2. Prepare and save processed_data.csv.
      3. Run all EDA functions as a quick self-test.
      4. Generate and print all insights.
    """
    print("=" * 60)
    print("  Student Placement Analytics and Decision Support System")
    print("  Aishwarya Bhimappa Hosamani | USN: 2BA23CS006")
    print("=" * 60)

    # --- Validate raw data ---
    train, test = load_raw_files()
    run_validation(train, test)

    # --- Data preparation ---
    df = prepare_data()

    # --- Quick EDA self-test ---
    _section("EDA SELF-TEST")
    eda_functions = [
        ("placement_kpis",                      lambda: placement_kpis(df)),
        ("plot_placement_distribution",          lambda: plot_placement_distribution(df)),
        ("plot_cgpa_distribution",               lambda: plot_cgpa_distribution(df)),
        ("plot_cgpa_band_vs_placement",          lambda: plot_cgpa_band_vs_placement(df)),
        ("plot_placement_by_degree",             lambda: plot_placement_by_degree(df)),
        ("plot_placement_by_branch",             lambda: plot_placement_by_branch(df)),
        ("plot_backlogs_vs_placement",           lambda: plot_backlogs_vs_placement(df)),
        ("plot_placement_by_gender",             lambda: plot_placement_by_gender(df)),
        ("plot_age_vs_placement",                lambda: plot_age_vs_placement(df)),
        ("plot_coding_skills_vs_placement",      lambda: plot_coding_skills_vs_placement(df)),
        ("plot_communication_skills_vs_placement", lambda: plot_communication_skills_vs_placement(df)),
        ("plot_soft_skills_vs_placement",        lambda: plot_soft_skills_vs_placement(df)),
        ("plot_aptitude_vs_placement",           lambda: plot_aptitude_vs_placement(df)),
        ("plot_skills_composite_vs_placement",   lambda: plot_skills_composite_vs_placement(df)),
        ("plot_internships_vs_placement",        lambda: plot_internships_vs_placement(df)),
        ("plot_projects_vs_placement",           lambda: plot_projects_vs_placement(df)),
        ("plot_certifications_vs_placement",     lambda: plot_certifications_vs_placement(df)),
        ("plot_correlation_heatmap",             lambda: plot_correlation_heatmap(df)),
        ("plot_top_factors",                     lambda: plot_top_factors(df)),
        ("plot_cgpa_vs_aptitude_scatter",        lambda: plot_cgpa_vs_aptitude_scatter(df)),
    ]

    print(f"\n  {'Function':<46} {'Status'}")
    print("  " + "-" * 56)
    for name, fn in eda_functions:
        try:
            result = fn()
            fig = result[0] if isinstance(result, tuple) else result
            if name == "placement_kpis":
                assert isinstance(result, dict)
            else:
                assert hasattr(fig, "data"), "Not a Plotly figure"
            print(f"  {name:<46}  [OK]")
        except Exception as exc:
            print(f"  {name:<46}  [FAIL] {exc}")

    # --- Insights ---
    _section("INSIGHTS AND RECOMMENDATIONS")
    insights = get_insights(df)
    print(f"\n  Total insights generated: {len(insights)}\n")
    for ins in insights:
        print(f"  [{ins['id']}] {ins['category']}")
        print(f"    Observation    : {ins['observation'][:120]}...")
        print(f"    Recommendation : {ins['recommendation'][:100]}...")
        print(f"    Audience       : {ins['audience']}\n")

    print("=" * 60)
    print("  Pipeline complete.")
    print("  To launch the dashboard, run:")
    print("    streamlit run AishwaryaBhimappaHosamani_StudentPlacementAnalytics.py")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # When run under Streamlit, the module is imported, not executed via __main__.
    # When run directly with `python ...`, launch the pipeline.
    # When run with `streamlit run ...`, Streamlit imports the module but does
    # not call main(); run_dashboard() must be invoked at module level.
    import sys as _sys
    if any("streamlit" in arg for arg in _sys.argv):
        run_dashboard()
    else:
        main()
