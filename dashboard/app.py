"""
dashboard/app.py
----------------
Sub-Task 5 — Streamlit Dashboard
Student Placement Analytics and Decision Support System

Run from the project root:
    streamlit run dashboard/app.py
"""

import os
import sys

# ---------------------------------------------------------------------------
# Make src/ importable when running from the project root
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pandas as pd
import streamlit as st

from src import eda
from src.insights import get_insights

# ---------------------------------------------------------------------------
# Page config — must be the very first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Student Placement Analytics",
    layout="wide",
    page_icon="🎓",
)

# ---------------------------------------------------------------------------
# Colour constants (keep consistent with eda.py)
# ---------------------------------------------------------------------------
PLACED_COLOR     = "#2ecc71"
NOT_PLACED_COLOR = "#e74c3c"

AUDIENCE_COLORS = {
    "Students":        "#3498db",
    "Placement Cell":  "#e67e22",
    "Faculty/Departments": "#9b59b6",
    "Institution":     "#1abc9c",
}

# ---------------------------------------------------------------------------
# Data loading (cached so filters don't reload from disk every render)
# ---------------------------------------------------------------------------
@st.cache_data
def load_full_data() -> pd.DataFrame:
    """Load data/processed_data.csv once and cache it."""
    path = os.path.join(_ROOT, "data", "processed_data.csv")
    df = pd.read_csv(path)
    band_order = ["< 6.0", "6.0-6.9", "7.0-7.9", "8.0-8.9", "9.0+"]
    df["CGPA_Band"] = pd.Categorical(df["CGPA_Band"], categories=band_order, ordered=True)
    return df


def apply_filters(df: pd.DataFrame, split: str, degrees: list,
                  branches: list, genders: list) -> pd.DataFrame:
    """Apply sidebar filter selections and return the filtered DataFrame."""
    if split != "All":
        df = df[df["split"] == split.lower()]
    if degrees:
        df = df[df["Degree"].isin(degrees)]
    if branches:
        df = df[df["Branch"].isin(branches)]
    if genders:
        df = df[df["Gender"].isin(genders)]
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Helper — safe guard against empty filtered DataFrames
# ---------------------------------------------------------------------------
def _empty_state(message: str = "No data matches the current filters."):
    st.warning(message)


# ===========================================================================
# Sidebar
# ===========================================================================
def build_sidebar(df_full: pd.DataFrame):
    """Render sidebar filters and return the filtered DataFrame."""
    with st.sidebar:
        st.title("🎓 Student Placement Analytics")
        st.markdown(
            "Descriptive and diagnostic analytics on student placement outcomes."
        )
        st.divider()

        split = st.selectbox(
            "Dataset Split",
            options=["All", "Train", "Test"],
            index=0,
            help="Filter by training set, test set, or the combined dataset.",
        )

        all_degrees = sorted(df_full["Degree"].unique().tolist())
        degrees = st.multiselect(
            "Degree", options=all_degrees, default=all_degrees,
            help="Filter by degree programme.",
        )

        all_branches = sorted(df_full["Branch"].unique().tolist())
        branches = st.multiselect(
            "Branch", options=all_branches, default=all_branches,
            help="Filter by academic branch.",
        )

        all_genders = sorted(df_full["Gender"].unique().tolist())
        genders = st.multiselect(
            "Gender", options=all_genders, default=all_genders,
            help="Filter by gender.",
        )

        st.divider()
        st.caption("Filters apply to all tabs.")

    return apply_filters(df_full, split, degrees, branches, genders)


# ===========================================================================
# Tab 1 — Overview
# ===========================================================================
def tab_overview(df: pd.DataFrame):
    st.subheader("Placement Overview")

    if df.empty:
        _empty_state()
        return

    kpis = eda.placement_kpis(df)

    # KPI metric cards
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Students",   f"{kpis['total_students']:,}")
    c2.metric("Placed",           f"{kpis['total_placed']:,}")
    c3.metric("Not Placed",       f"{kpis['total_not_placed']:,}")
    c4.metric("Placement Rate",   f"{kpis['placement_rate_pct']}%")
    c5.metric("Avg CGPA (Overall)", str(kpis["avg_cgpa_overall"]))

    st.divider()

    # Placement distribution chart
    with st.spinner("Rendering placement distribution..."):
        fig_dist = eda.plot_placement_distribution(df)
        st.plotly_chart(fig_dist, use_container_width=True, key="overview_placement_distribution")

    st.divider()

    # Placement by CGPA Band  +  Placement by Branch (side by side)
    col_left, col_right = st.columns(2)
    with col_left:
        with st.spinner("Rendering CGPA Band chart..."):
            fig_band = eda.plot_cgpa_band_vs_placement(df)
            st.plotly_chart(fig_band, use_container_width=True, key="overview_cgpa_band")

    with col_right:
        with st.spinner("Rendering Branch chart..."):
            fig_branch = eda.plot_placement_by_branch(df)
            st.plotly_chart(fig_branch, use_container_width=True, key="overview_branch")

    # Placement by Gender
    with st.spinner("Rendering Gender chart..."):
        fig_gender = eda.plot_placement_by_gender(df)
        st.plotly_chart(fig_gender, use_container_width=True, key="overview_gender")


# ===========================================================================
# Tab 2 — Academic Analysis
# ===========================================================================
def tab_academic(df: pd.DataFrame):
    st.subheader("Academic Analysis")

    if df.empty:
        _empty_state()
        return

    # CGPA distribution + CGPA Band vs Rate (side by side)
    col_left, col_right = st.columns(2)
    with col_left:
        with st.spinner("Rendering CGPA distribution..."):
            st.plotly_chart(eda.plot_cgpa_distribution(df), use_container_width=True, key="academic_cgpa_distribution")

    with col_right:
        with st.spinner("Rendering CGPA Band vs Placement..."):
            st.plotly_chart(eda.plot_cgpa_band_vs_placement(df), use_container_width=True, key="academic_cgpa_band")

    st.divider()

    # Backlogs + Aptitude (side by side)
    col_left2, col_right2 = st.columns(2)
    with col_left2:
        with st.spinner("Rendering Backlogs chart..."):
            st.plotly_chart(eda.plot_backlogs_vs_placement(df), use_container_width=True, key="academic_backlogs")

    with col_right2:
        with st.spinner("Rendering Aptitude chart..."):
            st.plotly_chart(eda.plot_aptitude_vs_placement(df), use_container_width=True, key="academic_aptitude")

    st.divider()

    # CGPA vs Aptitude Scatter
    with st.spinner("Rendering CGPA vs Aptitude scatter plot..."):
        st.plotly_chart(eda.plot_cgpa_vs_aptitude_scatter(df), use_container_width=True, key="academic_cgpa_aptitude")

    # Degree breakdown
    with st.spinner("Rendering Degree chart..."):
        st.plotly_chart(eda.plot_placement_by_degree(df), use_container_width=True, key="academic_degree")


# ===========================================================================
# Tab 3 — Skills Analysis
# ===========================================================================
def tab_skills(df: pd.DataFrame):
    st.subheader("Skills Analysis")

    if df.empty:
        _empty_state()
        return

    # Mean skill scores comparison table
    placed    = df[df["Placement_Status"] == "Placed"]
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

    # Coding + Communication side by side
    col_l, col_r = st.columns(2)
    with col_l:
        with st.spinner("Rendering Coding Skills chart..."):
            st.plotly_chart(eda.plot_coding_skills_vs_placement(df), use_container_width=True, key="skills_coding")
    with col_r:
        with st.spinner("Rendering Communication Skills chart..."):
            st.plotly_chart(eda.plot_communication_skills_vs_placement(df), use_container_width=True, key="skills_communication")

    # Soft Skills + Aptitude side by side
    col_l2, col_r2 = st.columns(2)
    with col_l2:
        with st.spinner("Rendering Soft Skills chart..."):
            st.plotly_chart(eda.plot_soft_skills_vs_placement(df), use_container_width=True, key="skills_soft")
    with col_r2:
        with st.spinner("Rendering Aptitude chart..."):
            st.plotly_chart(eda.plot_aptitude_vs_placement(df), use_container_width=True, key="skills_aptitude")

    # Skills Composite box plot
    with st.spinner("Rendering Skills Composite chart..."):
        st.plotly_chart(eda.plot_skills_composite_vs_placement(df), use_container_width=True, key="skills_composite")


# ===========================================================================
# Tab 4 — Activity Analysis
# ===========================================================================
def tab_activity(df: pd.DataFrame):
    st.subheader("Activity Analysis")

    if df.empty:
        _empty_state()
        return

    # Internships + Projects side by side
    col_l, col_r = st.columns(2)
    with col_l:
        with st.spinner("Rendering Internships chart..."):
            st.plotly_chart(eda.plot_internships_vs_placement(df), use_container_width=True, key="activity_internships")
    with col_r:
        with st.spinner("Rendering Projects chart..."):
            st.plotly_chart(eda.plot_projects_vs_placement(df), use_container_width=True, key="activity_projects")

    # Certifications
    with st.spinner("Rendering Certifications chart..."):
        st.plotly_chart(eda.plot_certifications_vs_placement(df), use_container_width=True, key="activity_certifications")

    st.divider()

    # Top factors (point-biserial correlation)
    with st.spinner("Computing top placement factors..."):
        fig_factors, factors_df = eda.plot_top_factors(df)
        st.plotly_chart(fig_factors, use_container_width=True, key="activity_factors")

    with st.expander("View correlation values table"):
        st.dataframe(
            factors_df.sort_values("Correlation", key=abs, ascending=False)
                      .reset_index(drop=True),
            use_container_width=True,
        )

    st.divider()

    # Correlation heatmap
    with st.spinner("Rendering correlation heatmap..."):
        st.plotly_chart(eda.plot_correlation_heatmap(df), use_container_width=True, key="activity_heatmap")


# ===========================================================================
# Tab 5 — Insights & Recommendations
# ===========================================================================
def tab_insights(df: pd.DataFrame, df_full: pd.DataFrame):
    """
    Insights are always computed on the full dataset (consistent, pre-computed
    observations).  The audience filter just controls which cards are shown.
    """
    st.subheader("Insights & Recommendations")
    st.markdown(
        "The observations below are computed directly from the data. "
        "Each insight is linked to an actionable recommendation for the audience shown."
    )

    # Load insights from the full dataset (observations should not change with filters)
    with st.spinner("Generating insights..."):
        insights = get_insights(df_full)

    # Audience filter
    all_audiences = ["All", "Students", "Placement Cell", "Faculty/Departments", "Institution"]
    audience_choice = st.selectbox("Filter by Audience", options=all_audiences, index=0)

    st.divider()

    shown = 0
    for ins in insights:
        # Audience matching: an insight may list multiple audiences (comma-separated)
        if audience_choice != "All":
            if audience_choice not in ins["audience"]:
                continue

        shown += 1
        # Category badge colour cycling
        cat = ins["category"]
        with st.expander(f"[{cat}]  {ins['observation'][:100]}...", expanded=False):
            st.markdown(f"**Observation**")
            st.info(ins["observation"])
            st.markdown(f"**Recommendation**")
            st.success(ins["recommendation"])
            # Audience badges
            badge_html = " ".join(
                f'<span style="background:{AUDIENCE_COLORS.get(a.strip(), "#999")};'
                f'color:white;padding:2px 8px;border-radius:4px;'
                f'font-size:0.8em;margin-right:4px;">{a.strip()}</span>'
                for a in ins["audience"].split(",")
            )
            st.markdown(f"**Audience:** {badge_html}", unsafe_allow_html=True)

    if shown == 0:
        st.info(f"No insights found for audience: {audience_choice}")


# ===========================================================================
# Main entry point
# ===========================================================================
def main():
    # Page title
    st.title("🎓 Student Placement Analytics and Decision Support System")
    st.markdown(
        "Descriptive and diagnostic analytics on placement outcomes for 50,000 students. "
        "Use the sidebar to filter by dataset split, degree, branch, or gender."
    )
    st.divider()

    # Load data
    with st.spinner("Loading dataset..."):
        df_full = load_full_data()

    # Build sidebar and get filtered DataFrame
    df = build_sidebar(df_full)

    # Filtered row count info
    if len(df) < len(df_full):
        st.caption(
            f"Showing **{len(df):,}** of **{len(df_full):,}** students based on current filters."
        )

    # Five tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview",
        "🎓 Academic Analysis",
        "💡 Skills Analysis",
        "🏆 Activity Analysis",
        "💬 Insights & Recommendations",
    ])

    with tab1:
        tab_overview(df)

    with tab2:
        tab_academic(df)

    with tab3:
        tab_skills(df)

    with tab4:
        tab_activity(df)

    with tab5:
        tab_insights(df, df_full)

    # Footer
    st.divider()
    st.caption(
        "Student Placement Analytics and Decision Support System  |  "
        "Descriptive & Diagnostic Analytics  |  Dataset: train.csv + test.csv (50,000 students)"
    )


if __name__ == "__main__":
    main()
