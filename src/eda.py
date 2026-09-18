"""
eda.py
------
Sub-Task 3: Exploratory Data Analysis
Student Placement Analytics and Decision Support System

All functions accept a pandas DataFrame and return a Plotly figure or a
plain dict/DataFrame.  No global state is used.  Nothing is saved to disk.
The Streamlit dashboard imports these functions directly.

Colour convention used throughout:
  Placed     -> PLACED_COLOR  (green)
  Not Placed -> NOT_PLACED_COLOR (red/orange)
"""

import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import pointbiserialr

# ---------------------------------------------------------------------------
# Colour palette (applied consistently across all charts)
# ---------------------------------------------------------------------------
PLACED_COLOR     = "#2ecc71"   # green
NOT_PLACED_COLOR = "#e74c3c"   # red
COLOR_MAP        = {"Placed": PLACED_COLOR, "Not Placed": NOT_PLACED_COLOR}

# CGPA band order for correct x-axis sorting
CGPA_BAND_ORDER = ["< 6.0", "6.0-6.9", "7.0-7.9", "8.0-8.9", "9.0+"]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_PATH   = os.path.join(PROJECT_ROOT, "data", "processed_data.csv")


# ===========================================================================
# SETUP: Data loader
# ===========================================================================

def load_data(split: str = "all") -> pd.DataFrame:
    """
    Load data/processed_data.csv and optionally filter by split.

    Parameters
    ----------
    split : "train" | "test" | "all"
        "all" returns all 50,000 rows.

    Returns
    -------
    pd.DataFrame
    """
    df = pd.read_csv(PROCESSED_PATH)

    # Restore the categorical order for CGPA_Band so charts sort correctly
    df["CGPA_Band"] = pd.Categorical(
        df["CGPA_Band"], categories=CGPA_BAND_ORDER, ordered=True
    )

    if split in ("train", "test"):
        df = df[df["split"] == split].reset_index(drop=True)

    return df


# ===========================================================================
# PLACEMENT OVERVIEW
# ===========================================================================

def placement_kpis(df: pd.DataFrame) -> dict:
    """
    Return a dict of key performance indicators computed from df.

    Keys
    ----
    total_students      : int
    total_placed        : int
    total_not_placed    : int
    placement_rate_pct  : float  (rounded to 1 dp)
    avg_cgpa_placed     : float
    avg_cgpa_not_placed : float
    avg_cgpa_overall    : float
    """
    total          = len(df)
    placed_mask    = df["Placement_Status"] == "Placed"
    total_placed   = int(placed_mask.sum())
    total_not      = total - total_placed
    rate           = round(total_placed / total * 100, 1) if total > 0 else 0.0

    avg_cgpa_p  = round(df.loc[placed_mask,  "CGPA"].mean(), 2)
    avg_cgpa_np = round(df.loc[~placed_mask, "CGPA"].mean(), 2)
    avg_cgpa    = round(df["CGPA"].mean(), 2)

    return {
        "total_students":      total,
        "total_placed":        total_placed,
        "total_not_placed":    total_not,
        "placement_rate_pct":  rate,
        "avg_cgpa_placed":     avg_cgpa_p,
        "avg_cgpa_not_placed": avg_cgpa_np,
        "avg_cgpa_overall":    avg_cgpa,
    }


def plot_placement_distribution(df: pd.DataFrame) -> go.Figure:
    """
    Side-by-side pie chart and bar chart showing Placed vs Not Placed
    counts and percentages.
    """
    counts = df["Placement_Status"].value_counts().reset_index()
    counts.columns = ["Placement_Status", "Count"]
    counts["Percentage"] = (counts["Count"] / len(df) * 100).round(1)

    fig = go.Figure()

    # Pie chart (left)
    fig.add_trace(go.Pie(
        labels=counts["Placement_Status"],
        values=counts["Count"],
        marker_colors=[COLOR_MAP.get(s, "#999") for s in counts["Placement_Status"]],
        textinfo="label+percent",
        hovertemplate="%{label}: %{value:,} students (%{percent})<extra></extra>",
        domain={"x": [0, 0.45]},
        name="Pie",
    ))

    # Bar chart (right)
    fig.add_trace(go.Bar(
        x=counts["Placement_Status"],
        y=counts["Count"],
        marker_color=[COLOR_MAP.get(s, "#999") for s in counts["Placement_Status"]],
        text=[f"{r['Count']:,}<br>({r['Percentage']}%)" for _, r in counts.iterrows()],
        textposition="outside",
        hovertemplate="%{x}: %{y:,} students<extra></extra>",
        xaxis="x2",
        yaxis="y2",
        name="Bar",
    ))

    fig.update_layout(
        title="Overall Placement Distribution",
        xaxis2={"domain": [0.55, 1.0], "anchor": "y2"},
        yaxis2={"anchor": "x2", "title": "Number of Students"},
        showlegend=False,
        height=400,
    )
    return fig


# ===========================================================================
# ACADEMIC PERFORMANCE
# ===========================================================================

def plot_cgpa_distribution(df: pd.DataFrame) -> go.Figure:
    """
    Histogram of CGPA values coloured by Placement_Status.
    Shows how CGPA is distributed for placed vs not-placed students.
    """
    fig = px.histogram(
        df,
        x="CGPA",
        color="Placement_Status",
        color_discrete_map=COLOR_MAP,
        nbins=40,
        barmode="overlay",
        opacity=0.7,
        title="CGPA Distribution by Placement Status",
        labels={"CGPA": "CGPA", "count": "Number of Students"},
    )
    fig.update_layout(
        xaxis_title="CGPA",
        yaxis_title="Number of Students",
        legend_title="Placement Status",
        height=400,
    )
    return fig


def plot_cgpa_band_vs_placement(df: pd.DataFrame) -> go.Figure:
    """
    Grouped bar chart: count of Placed / Not Placed in each CGPA band,
    with a line showing the placement rate (%) per band.
    """
    # Aggregate counts per band × status
    agg = (
        df.groupby(["CGPA_Band", "Placement_Status"], observed=True)
          .size()
          .reset_index(name="Count")
    )

    # Placement rate per band
    rate = (
        df.groupby("CGPA_Band", observed=True)["Placement_Binary"]
          .mean()
          .mul(100)
          .round(1)
          .reset_index()
    )
    rate.columns = ["CGPA_Band", "Placement_Rate"]

    fig = go.Figure()

    for status in ["Placed", "Not Placed"]:
        subset = agg[agg["Placement_Status"] == status]
        fig.add_trace(go.Bar(
            x=subset["CGPA_Band"].astype(str),
            y=subset["Count"],
            name=status,
            marker_color=COLOR_MAP[status],
            hovertemplate=f"{status}: %{{y:,}} students<extra></extra>",
        ))

    # Placement rate line (secondary y-axis)
    fig.add_trace(go.Scatter(
        x=rate["CGPA_Band"].astype(str),
        y=rate["Placement_Rate"],
        mode="lines+markers+text",
        name="Placement Rate %",
        line={"color": "#2c3e50", "width": 2, "dash": "dot"},
        marker={"size": 8},
        text=[f"{v:.1f}%" for v in rate["Placement_Rate"]],
        textposition="top center",
        yaxis="y2",
        hovertemplate="Rate: %{y:.1f}%<extra></extra>",
    ))

    fig.update_layout(
        title="Placement by CGPA Band (Count + Rate)",
        xaxis_title="CGPA Band",
        yaxis_title="Number of Students",
        yaxis2={"overlaying": "y", "side": "right", "title": "Placement Rate (%)", "range": [0, 100]},
        barmode="group",
        legend_title="",
        height=450,
    )
    return fig


def plot_placement_by_degree(df: pd.DataFrame) -> go.Figure:
    """
    Grouped bar chart: Placed / Not Placed count for each Degree type,
    with placement rate annotations.
    """
    agg = (
        df.groupby(["Degree", "Placement_Status"])
          .size()
          .reset_index(name="Count")
    )
    rate = (
        df.groupby("Degree")["Placement_Binary"]
          .mean().mul(100).round(1)
          .reset_index(name="Rate")
    )

    fig = go.Figure()
    for status in ["Placed", "Not Placed"]:
        s = agg[agg["Placement_Status"] == status]
        fig.add_trace(go.Bar(
            x=s["Degree"], y=s["Count"],
            name=status, marker_color=COLOR_MAP[status],
            hovertemplate=f"{status}: %{{y:,}}<extra></extra>",
        ))

    # Rate line
    fig.add_trace(go.Scatter(
        x=rate["Degree"], y=rate["Rate"],
        mode="lines+markers+text",
        name="Placement Rate %",
        line={"color": "#2c3e50", "width": 2, "dash": "dot"},
        text=[f"{v:.1f}%" for v in rate["Rate"]],
        textposition="top center",
        yaxis="y2",
        hovertemplate="Rate: %{y:.1f}%<extra></extra>",
    ))

    fig.update_layout(
        title="Placement by Degree Type (Count + Rate)",
        xaxis_title="Degree", yaxis_title="Number of Students",
        yaxis2={"overlaying": "y", "side": "right", "title": "Placement Rate (%)", "range": [0, 100]},
        barmode="group", legend_title="", height=450,
    )
    return fig


def plot_placement_by_branch(df: pd.DataFrame) -> go.Figure:
    """
    Grouped bar chart: Placed / Not Placed count for each Branch,
    with placement rate annotations.
    """
    agg = (
        df.groupby(["Branch", "Placement_Status"])
          .size()
          .reset_index(name="Count")
    )
    rate = (
        df.groupby("Branch")["Placement_Binary"]
          .mean().mul(100).round(1)
          .reset_index(name="Rate")
    )

    fig = go.Figure()
    for status in ["Placed", "Not Placed"]:
        s = agg[agg["Placement_Status"] == status]
        fig.add_trace(go.Bar(
            x=s["Branch"], y=s["Count"],
            name=status, marker_color=COLOR_MAP[status],
            hovertemplate=f"{status}: %{{y:,}}<extra></extra>",
        ))

    fig.add_trace(go.Scatter(
        x=rate["Branch"], y=rate["Rate"],
        mode="lines+markers+text",
        name="Placement Rate %",
        line={"color": "#2c3e50", "width": 2, "dash": "dot"},
        text=[f"{v:.1f}%" for v in rate["Rate"]],
        textposition="top center",
        yaxis="y2",
        hovertemplate="Rate: %{y:.1f}%<extra></extra>",
    ))

    fig.update_layout(
        title="Placement by Branch (Count + Rate)",
        xaxis_title="Branch", yaxis_title="Number of Students",
        yaxis2={"overlaying": "y", "side": "right", "title": "Placement Rate (%)", "range": [0, 100]},
        barmode="group", legend_title="", height=450,
    )
    return fig


def plot_backlogs_vs_placement(df: pd.DataFrame) -> go.Figure:
    """
    Bar chart showing placement rate (%) for each backlogs count (0-3),
    with the number of students in each group as annotation.
    """
    agg = (
        df.groupby("Backlogs")
          .agg(Total=("Placement_Binary", "count"),
               Placed=("Placement_Binary", "sum"))
          .reset_index()
    )
    agg["Rate"] = (agg["Placed"] / agg["Total"] * 100).round(1)

    fig = go.Figure(go.Bar(
        x=agg["Backlogs"].astype(str),
        y=agg["Rate"],
        marker_color="#3498db",
        text=[f"{r:.1f}%<br>(n={t:,})" for r, t in zip(agg["Rate"], agg["Total"])],
        textposition="outside",
        hovertemplate="Backlogs=%{x}: %{y:.1f}% placed<extra></extra>",
    ))

    fig.update_layout(
        title="Placement Rate by Number of Backlogs",
        xaxis_title="Number of Backlogs",
        yaxis_title="Placement Rate (%)",
        yaxis_range=[0, 100],
        height=400,
    )
    return fig


# ===========================================================================
# DEMOGRAPHIC ANALYSIS
# ===========================================================================

def plot_placement_by_gender(df: pd.DataFrame) -> go.Figure:
    """
    Grouped bar chart: Placed / Not Placed count for each Gender,
    with placement rate annotations.
    """
    agg = (
        df.groupby(["Gender", "Placement_Status"])
          .size()
          .reset_index(name="Count")
    )
    rate = (
        df.groupby("Gender")["Placement_Binary"]
          .mean().mul(100).round(1)
          .reset_index(name="Rate")
    )

    fig = go.Figure()
    for status in ["Placed", "Not Placed"]:
        s = agg[agg["Placement_Status"] == status]
        fig.add_trace(go.Bar(
            x=s["Gender"], y=s["Count"],
            name=status, marker_color=COLOR_MAP[status],
            hovertemplate=f"{status}: %{{y:,}}<extra></extra>",
        ))

    fig.add_trace(go.Scatter(
        x=rate["Gender"], y=rate["Rate"],
        mode="markers+text",
        name="Placement Rate %",
        marker={"size": 12, "color": "#2c3e50", "symbol": "diamond"},
        text=[f"{v:.1f}%" for v in rate["Rate"]],
        textposition="top center",
        yaxis="y2",
        hovertemplate="Rate: %{y:.1f}%<extra></extra>",
    ))

    fig.update_layout(
        title="Placement by Gender (Count + Rate)",
        xaxis_title="Gender", yaxis_title="Number of Students",
        yaxis2={"overlaying": "y", "side": "right", "title": "Placement Rate (%)", "range": [0, 100]},
        barmode="group", legend_title="", height=400,
    )
    return fig


def plot_age_vs_placement(df: pd.DataFrame) -> go.Figure:
    """
    Box plot showing Age distribution for Placed vs Not Placed students.
    """
    fig = px.box(
        df, x="Placement_Status", y="Age",
        color="Placement_Status",
        color_discrete_map=COLOR_MAP,
        title="Age Distribution by Placement Status",
        labels={"Placement_Status": "Placement Status", "Age": "Age"},
        points="outliers",
    )
    fig.update_layout(
        xaxis_title="Placement Status",
        yaxis_title="Age",
        showlegend=False,
        height=400,
    )
    return fig


# ===========================================================================
# SKILLS ANALYSIS
# ===========================================================================

def _plot_skill_vs_placement_rate(df: pd.DataFrame, col: str, title: str) -> go.Figure:
    """
    Internal helper: bar chart of placement rate (%) for each integer
    skill score value, with student count as annotation.
    Used by coding, communication, and soft skills functions.
    """
    agg = (
        df.groupby(col)
          .agg(Total=("Placement_Binary", "count"),
               Placed=("Placement_Binary", "sum"))
          .reset_index()
    )
    agg["Rate"] = (agg["Placed"] / agg["Total"] * 100).round(1)

    fig = go.Figure(go.Bar(
        x=agg[col].astype(str),
        y=agg["Rate"],
        marker_color="#9b59b6",
        text=[f"{r:.1f}%<br>(n={t:,})" for r, t in zip(agg["Rate"], agg["Total"])],
        textposition="outside",
        hovertemplate=f"{col}=%{{x}}: %{{y:.1f}}% placed<extra></extra>",
    ))

    fig.update_layout(
        title=title,
        xaxis_title=f"{col} Score (1-10)",
        yaxis_title="Placement Rate (%)",
        yaxis_range=[0, 100],
        height=400,
    )
    return fig


def plot_coding_skills_vs_placement(df: pd.DataFrame) -> go.Figure:
    return _plot_skill_vs_placement_rate(
        df, "Coding_Skills",
        "Placement Rate by Coding Skills Score"
    )


def plot_communication_skills_vs_placement(df: pd.DataFrame) -> go.Figure:
    return _plot_skill_vs_placement_rate(
        df, "Communication_Skills",
        "Placement Rate by Communication Skills Score"
    )


def plot_soft_skills_vs_placement(df: pd.DataFrame) -> go.Figure:
    return _plot_skill_vs_placement_rate(
        df, "Soft_Skills_Rating",
        "Placement Rate by Soft Skills Rating"
    )


def plot_aptitude_vs_placement(df: pd.DataFrame) -> go.Figure:
    """
    Box plot of Aptitude_Test_Score for Placed vs Not Placed.
    Adds a horizontal reference line at the overall median score.
    """
    median_score = df["Aptitude_Test_Score"].median()

    fig = px.box(
        df, x="Placement_Status", y="Aptitude_Test_Score",
        color="Placement_Status",
        color_discrete_map=COLOR_MAP,
        title="Aptitude Test Score by Placement Status",
        labels={"Placement_Status": "Placement Status",
                "Aptitude_Test_Score": "Aptitude Test Score"},
        points="outliers",
    )

    # Reference line at overall median
    fig.add_hline(
        y=median_score,
        line_dash="dash",
        line_color="#7f8c8d",
        annotation_text=f"Overall median: {median_score:.0f}",
        annotation_position="top right",
    )

    fig.update_layout(
        xaxis_title="Placement Status",
        yaxis_title="Aptitude Test Score",
        showlegend=False,
        height=400,
    )
    return fig


def plot_skills_composite_vs_placement(df: pd.DataFrame) -> go.Figure:
    """
    Box plot of Skills_Composite (avg of Coding, Communication, Soft Skills)
    for Placed vs Not Placed.
    """
    fig = px.box(
        df, x="Placement_Status", y="Skills_Composite",
        color="Placement_Status",
        color_discrete_map=COLOR_MAP,
        title="Skills Composite Score by Placement Status",
        labels={"Placement_Status": "Placement Status",
                "Skills_Composite": "Skills Composite (avg of 3 skill scores)"},
        points="outliers",
    )
    fig.update_layout(
        xaxis_title="Placement Status",
        yaxis_title="Skills Composite Score",
        showlegend=False,
        height=400,
    )
    return fig


# ===========================================================================
# ACTIVITY ANALYSIS
# ===========================================================================

def _plot_activity_vs_placement(df: pd.DataFrame, col: str, title: str,
                                  x_label: str) -> go.Figure:
    """
    Internal helper: grouped bar (count) + rate line for an integer
    activity column (Internships, Projects, Certifications).
    """
    agg = (
        df.groupby([col, "Placement_Status"])
          .size()
          .reset_index(name="Count")
    )
    rate = (
        df.groupby(col)["Placement_Binary"]
          .mean().mul(100).round(1)
          .reset_index(name="Rate")
    )

    fig = go.Figure()
    for status in ["Placed", "Not Placed"]:
        s = agg[agg["Placement_Status"] == status]
        fig.add_trace(go.Bar(
            x=s[col].astype(str), y=s["Count"],
            name=status, marker_color=COLOR_MAP[status],
            hovertemplate=f"{status}: %{{y:,}}<extra></extra>",
        ))

    fig.add_trace(go.Scatter(
        x=rate[col].astype(str), y=rate["Rate"],
        mode="lines+markers+text",
        name="Placement Rate %",
        line={"color": "#2c3e50", "width": 2, "dash": "dot"},
        marker={"size": 8},
        text=[f"{v:.1f}%" for v in rate["Rate"]],
        textposition="top center",
        yaxis="y2",
        hovertemplate="Rate: %{y:.1f}%<extra></extra>",
    ))

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title="Number of Students",
        yaxis2={"overlaying": "y", "side": "right",
                "title": "Placement Rate (%)", "range": [0, 100]},
        barmode="group", legend_title="", height=450,
    )
    return fig


def plot_internships_vs_placement(df: pd.DataFrame) -> go.Figure:
    return _plot_activity_vs_placement(
        df, "Internships",
        "Placement by Number of Internships (Count + Rate)",
        "Number of Internships"
    )


def plot_projects_vs_placement(df: pd.DataFrame) -> go.Figure:
    return _plot_activity_vs_placement(
        df, "Projects",
        "Placement by Number of Projects (Count + Rate)",
        "Number of Projects"
    )


def plot_certifications_vs_placement(df: pd.DataFrame) -> go.Figure:
    return _plot_activity_vs_placement(
        df, "Certifications",
        "Placement by Number of Certifications (Count + Rate)",
        "Number of Certifications"
    )


# ===========================================================================
# CORRELATION & MULTI-FACTOR
# ===========================================================================

def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """
    Pearson correlation heatmap for all numeric columns in the dataset.
    Excludes 'Placement_Binary' from the row/column labels to avoid
    confusion with the string target, but it is included so analysts can
    see which numeric features correlate with the outcome.
    """
    # Select only numeric columns that have more than one unique value
    numeric_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if df[c].nunique() > 1
    ]

    corr = df[numeric_cols].corr()

    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale="RdBu",
        zmid=0,
        zmin=-1, zmax=1,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        hovertemplate="x: %{x}<br>y: %{y}<br>r = %{z:.3f}<extra></extra>",
        colorbar={"title": "Pearson r"},
    ))

    fig.update_layout(
        title="Correlation Heatmap (Numeric Features)",
        height=550,
        xaxis={"tickangle": -45},
    )
    return fig


def plot_top_factors(df: pd.DataFrame) -> go.Figure:
    """
    Horizontal bar chart of point-biserial correlation between each
    numeric feature and Placement_Binary, sorted by absolute value.

    Point-biserial correlation is the standard measure for the
    relationship between a continuous variable and a binary outcome.
    """
    # Features to test (exclude the binary target itself and Skills_Composite
    # which is a derived combination — include it as well for reference)
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
        # Align both series on the same index
        idx = binary.index.intersection(series.index)
        if len(idx) < 10:
            continue   # not enough data to compute correlation
        if series.loc[idx].std() == 0:
            continue   # zero variance — skip

        r, p = pointbiserialr(binary.loc[idx], series.loc[idx])
        results.append({"Feature": col, "Correlation": round(r, 4), "p_value": round(p, 4)})

    results_df = pd.DataFrame(results)
    results_df = results_df.reindex(
        results_df["Correlation"].abs().sort_values(ascending=True).index
    )

    # Colour bars by direction of correlation
    bar_colors = [PLACED_COLOR if r >= 0 else NOT_PLACED_COLOR
                  for r in results_df["Correlation"]]

    fig = go.Figure(go.Bar(
        x=results_df["Correlation"],
        y=results_df["Feature"],
        orientation="h",
        marker_color=bar_colors,
        text=[f"{r:+.3f}" for r in results_df["Correlation"]],
        textposition="outside",
        hovertemplate=(
            "%{y}<br>r = %{x:.4f}<extra></extra>"
        ),
    ))

    fig.add_vline(x=0, line_color="#7f8c8d", line_width=1)

    fig.update_layout(
        title="Top Factors Associated with Placement (Point-Biserial Correlation)",
        xaxis_title="Point-Biserial Correlation with Placement (Placed=1)",
        yaxis_title="Feature",
        height=500,
        xaxis_range=[-1, 1],
    )
    return fig, results_df   # also return the table for verification


def plot_cgpa_vs_aptitude_scatter(df: pd.DataFrame) -> go.Figure:
    """
    Scatter plot: CGPA (x) vs Aptitude_Test_Score (y),
    coloured by Placement_Status.  Uses a random sample of up to 3,000
    points to keep the chart responsive.
    """
    sample = df.sample(min(3000, len(df)), random_state=42)

    fig = px.scatter(
        sample,
        x="CGPA",
        y="Aptitude_Test_Score",
        color="Placement_Status",
        color_discrete_map=COLOR_MAP,
        opacity=0.6,
        title="CGPA vs Aptitude Test Score (coloured by Placement Status)",
        labels={
            "CGPA": "CGPA",
            "Aptitude_Test_Score": "Aptitude Test Score",
        },
        hover_data=["Degree", "Branch", "Internships"],
    )
    fig.update_layout(
        xaxis_title="CGPA",
        yaxis_title="Aptitude Test Score",
        legend_title="Placement Status",
        height=450,
    )
    return fig


# ===========================================================================
# SELF-TEST: run all functions and print a results summary
# ===========================================================================

def _self_test():
    """
    Load processed_data.csv and call every EDA function.
    Print a concise pass/fail table so we can confirm everything works.
    """
    print("=" * 60)
    print("  EDA Self-Test — Sub-Task 3")
    print("=" * 60)

    df = load_data(split="all")
    print(f"\n  Loaded dataset: {len(df):,} rows x {len(df.columns)} columns\n")

    # --- KPIs ---
    kpis = placement_kpis(df)
    print("  placement_kpis():")
    for k, v in kpis.items():
        print(f"    {k:<28} = {v}")

    # --- Each chart function ---
    chart_functions = [
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
        ("plot_cgpa_vs_aptitude_scatter",        lambda: plot_cgpa_vs_aptitude_scatter(df)),
    ]

    print("\n  Chart functions:")
    print(f"  {'Function':<46} {'Status'}")
    print("  " + "-" * 56)

    for name, fn in chart_functions:
        try:
            result = fn()
            # Some functions return (fig, df) tuples
            fig = result[0] if isinstance(result, tuple) else result
            assert hasattr(fig, "data"), "Return value is not a Plotly figure"
            print(f"  {name:<46}  [OK]")
        except Exception as exc:
            print(f"  {name:<46}  [FAIL] {exc}")

    # plot_top_factors returns (fig, df)
    print("\n  plot_top_factors():")
    try:
        fig, factors_df = plot_top_factors(df)
        print(f"  {'plot_top_factors':<46}  [OK]")
        print("\n  Point-biserial correlations with Placement_Binary:")
        sorted_df = factors_df.reindex(
            factors_df["Correlation"].abs().sort_values(ascending=False).index
        )
        for _, row in sorted_df.iterrows():
            direction = "+" if row["Correlation"] >= 0 else "-"
            print(f"    {row['Feature']:<28} r = {row['Correlation']:+.4f}  (p={row['p_value']:.4f})")
    except Exception as exc:
        print(f"  [FAIL] plot_top_factors: {exc}")

    # Quick placement rate summary per key groupings
    print("\n  Placement rates by key groupings:")
    for grp_col in ["CGPA_Band", "Degree", "Branch", "Gender", "Internships", "Backlogs"]:
        rates = (
            df.groupby(grp_col, observed=True)["Placement_Binary"]
              .mean().mul(100).round(1)
        )
        print(f"\n    {grp_col}:")
        for val, rate in rates.items():
            print(f"      {str(val):<14} -> {rate:.1f}%")

    print("\n" + "=" * 60)
    print("  EDA self-test complete.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    _self_test()
