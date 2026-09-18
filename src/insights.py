"""
src/insights.py
---------------
Sub-Task 4 — Observations and Insights Module

Converts EDA findings into structured, data-driven insights and actionable
recommendations.  All numbers in the observation text are calculated directly
from the dataset; nothing is hard-coded.

Usage
-----
    from src.insights import get_insights
    import pandas as pd
    df = pd.read_csv("data/processed_data.csv")
    insights = get_insights(df)
"""

import pandas as pd


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _placement_rate(series: pd.Series) -> float:
    """Return the placement rate (%) for a boolean/binary Series."""
    return round(series.mean() * 100, 1)


# ---------------------------------------------------------------------------
# Step 1 — Aggregate all key statistics needed for insight generation
# ---------------------------------------------------------------------------

def compute_insight_data(df: pd.DataFrame) -> dict:
    """
    Run all aggregations needed for the insights.

    Parameters
    ----------
    df : pd.DataFrame
        The processed dataset (data/processed_data.csv).

    Returns
    -------
    dict
        A flat dictionary of computed statistics.  Keys are used directly
        inside get_insights() to build observation text.
    """
    d = {}  # collector for all computed values

    # -----------------------------------------------------------------------
    # CGPA — placement rate per CGPA band
    # We use the CGPA_Band column created during data preparation.
    # -----------------------------------------------------------------------
    # Define band order so results are always sorted low → high
    band_order = ["< 6.0", "6.0-6.9", "7.0-7.9", "8.0-8.9", "9.0+"]
    cgpa_rates = (
        df.groupby("CGPA_Band")["Placement_Binary"]
        .mean()
        .mul(100)
        .round(1)
        .reindex([b for b in band_order if b in df["CGPA_Band"].unique()])
    )
    d["cgpa_band_rates"] = cgpa_rates.to_dict()

    # Lowest and highest CGPA-band placement rates
    d["cgpa_lowest_band"] = cgpa_rates.idxmin()
    d["cgpa_lowest_rate"] = cgpa_rates.min()
    d["cgpa_highest_band"] = cgpa_rates.idxmax()
    d["cgpa_highest_rate"] = cgpa_rates.max()

    # Mean CGPA for placed vs not placed
    cgpa_mean = df.groupby("Placement_Status")["CGPA"].mean().round(2)
    d["cgpa_mean_placed"] = cgpa_mean.get("Placed", float("nan"))
    d["cgpa_mean_not_placed"] = cgpa_mean.get("Not Placed", float("nan"))

    # -----------------------------------------------------------------------
    # Backlogs — placement rate per backlog count (0–3)
    # -----------------------------------------------------------------------
    backlogs_rates = (
        df.groupby("Backlogs")["Placement_Binary"]
        .mean()
        .mul(100)
        .round(1)
        .sort_index()
    )
    d["backlogs_rates"] = backlogs_rates.to_dict()
    # Rate at 0 backlogs vs 3 backlogs
    d["backlogs_rate_0"] = backlogs_rates.get(0, float("nan"))
    d["backlogs_rate_3"] = backlogs_rates.get(3, float("nan"))

    # -----------------------------------------------------------------------
    # Internships — placement rate per internship count (0–3)
    # -----------------------------------------------------------------------
    intern_rates = (
        df.groupby("Internships")["Placement_Binary"]
        .mean()
        .mul(100)
        .round(1)
        .sort_index()
    )
    d["intern_rates"] = intern_rates.to_dict()
    d["intern_rate_0"] = intern_rates.get(0, float("nan"))
    d["intern_rate_3"] = intern_rates.get(3, float("nan"))

    # -----------------------------------------------------------------------
    # Projects — placement rate per project count
    # -----------------------------------------------------------------------
    proj_rates = (
        df.groupby("Projects")["Placement_Binary"]
        .mean()
        .mul(100)
        .round(1)
        .sort_index()
    )
    d["proj_rates"] = proj_rates.to_dict()
    d["proj_rate_min"] = proj_rates.min()
    d["proj_rate_max"] = proj_rates.max()
    d["proj_count_min"] = proj_rates.idxmin()
    d["proj_count_max"] = proj_rates.idxmax()

    # -----------------------------------------------------------------------
    # Certifications — placement rate per certification count (0–3)
    # -----------------------------------------------------------------------
    cert_rates = (
        df.groupby("Certifications")["Placement_Binary"]
        .mean()
        .mul(100)
        .round(1)
        .sort_index()
    )
    d["cert_rates"] = cert_rates.to_dict()
    d["cert_rate_0"] = cert_rates.get(0, float("nan"))
    d["cert_rate_3"] = cert_rates.get(3, float("nan"))

    # -----------------------------------------------------------------------
    # Coding Skills — mean coding score for placed vs not placed
    # -----------------------------------------------------------------------
    coding_mean = df.groupby("Placement_Status")["Coding_Skills"].mean().round(2)
    d["coding_mean_placed"] = coding_mean.get("Placed", float("nan"))
    d["coding_mean_not_placed"] = coding_mean.get("Not Placed", float("nan"))

    # Placement rate at coding skill levels <= 4 and >= 8
    d["coding_rate_low"] = _placement_rate(
        df.loc[df["Coding_Skills"] <= 4, "Placement_Binary"]
    )
    d["coding_rate_high"] = _placement_rate(
        df.loc[df["Coding_Skills"] >= 8, "Placement_Binary"]
    )

    # -----------------------------------------------------------------------
    # Aptitude Test Score — mean for placed vs not placed
    # -----------------------------------------------------------------------
    aptitude_mean = df.groupby("Placement_Status")["Aptitude_Test_Score"].mean().round(1)
    d["aptitude_mean_placed"] = aptitude_mean.get("Placed", float("nan"))
    d["aptitude_mean_not_placed"] = aptitude_mean.get("Not Placed", float("nan"))

    # -----------------------------------------------------------------------
    # Communication Skills — mean for placed vs not placed
    # -----------------------------------------------------------------------
    comm_mean = df.groupby("Placement_Status")["Communication_Skills"].mean().round(2)
    d["comm_mean_placed"] = comm_mean.get("Placed", float("nan"))
    d["comm_mean_not_placed"] = comm_mean.get("Not Placed", float("nan"))

    # -----------------------------------------------------------------------
    # Soft Skills Rating — mean for placed vs not placed
    # -----------------------------------------------------------------------
    soft_mean = df.groupby("Placement_Status")["Soft_Skills_Rating"].mean().round(2)
    d["soft_mean_placed"] = soft_mean.get("Placed", float("nan"))
    d["soft_mean_not_placed"] = soft_mean.get("Not Placed", float("nan"))

    # -----------------------------------------------------------------------
    # Degree — placement rate per degree type
    # -----------------------------------------------------------------------
    degree_rates = (
        df.groupby("Degree")["Placement_Binary"]
        .mean()
        .mul(100)
        .round(1)
        .sort_values(ascending=False)
    )
    d["degree_rates"] = degree_rates.to_dict()
    d["degree_best"] = degree_rates.idxmax()
    d["degree_best_rate"] = degree_rates.max()
    d["degree_worst"] = degree_rates.idxmin()
    d["degree_worst_rate"] = degree_rates.min()

    # -----------------------------------------------------------------------
    # Branch — placement rate per branch
    # -----------------------------------------------------------------------
    branch_rates = (
        df.groupby("Branch")["Placement_Binary"]
        .mean()
        .mul(100)
        .round(1)
        .sort_values(ascending=False)
    )
    d["branch_rates"] = branch_rates.to_dict()
    d["branch_best"] = branch_rates.idxmax()
    d["branch_best_rate"] = branch_rates.max()
    d["branch_worst"] = branch_rates.idxmin()
    d["branch_worst_rate"] = branch_rates.min()

    # -----------------------------------------------------------------------
    # Gender — placement rate per gender
    # -----------------------------------------------------------------------
    gender_rates = (
        df.groupby("Gender")["Placement_Binary"]
        .mean()
        .mul(100)
        .round(1)
    )
    d["gender_rates"] = gender_rates.to_dict()
    d["gender_male_rate"] = gender_rates.get("Male", float("nan"))
    d["gender_female_rate"] = gender_rates.get("Female", float("nan"))

    # -----------------------------------------------------------------------
    # Multi-factor — overall placement rate and Skills_Composite comparison
    # -----------------------------------------------------------------------
    d["overall_placement_rate"] = round(df["Placement_Binary"].mean() * 100, 1)

    skills_comp_mean = (
        df.groupby("Placement_Status")["Skills_Composite"].mean().round(2)
    )
    d["skills_comp_mean_placed"] = skills_comp_mean.get("Placed", float("nan"))
    d["skills_comp_mean_not_placed"] = skills_comp_mean.get("Not Placed", float("nan"))

    # Among students with CGPA >= 8 AND at least 1 internship -- placement rate
    high_cgpa_intern = df[(df["CGPA"] >= 8.0) & (df["Internships"] >= 1)]
    d["high_cgpa_intern_rate"] = round(
        high_cgpa_intern["Placement_Binary"].mean() * 100, 1
    )
    d["high_cgpa_intern_count"] = len(high_cgpa_intern)

    return d


# ---------------------------------------------------------------------------
# Step 2 — Build the list of structured insight dictionaries
# ---------------------------------------------------------------------------

def get_insights(df: pd.DataFrame) -> list:
    """
    Generate structured insights from the processed dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The processed dataset (data/processed_data.csv).

    Returns
    -------
    list of dict
        Each dict has keys: id, category, observation, recommendation, audience.
        Sorted by category for consistent rendering.
    """
    # Compute all statistics first
    c = compute_insight_data(df)

    insights = []

    # ------------------------------------------------------------------
    # 1 — CGPA
    # ------------------------------------------------------------------
    insights.append({
        "id": "cgpa_01",
        "category": "CGPA",
        "observation": (
            f"In this dataset, students in the highest CGPA band ({c['cgpa_highest_band']}) "
            f"showed a placement rate of {c['cgpa_highest_rate']}%, compared to "
            f"{c['cgpa_lowest_rate']}% for students in the '{c['cgpa_lowest_band']}' band. "
            f"The mean CGPA of placed students was {c['cgpa_mean_placed']} vs "
            f"{c['cgpa_mean_not_placed']} for unplaced students."
        ),
        "recommendation": (
            "Students should prioritise maintaining or improving their CGPA, particularly "
            "aiming to move into a higher band where placement rates are noticeably stronger. "
            "Faculty can use CGPA-band breakdowns in mid-semester reviews to identify at-risk cohorts early."
        ),
        "audience": "Students, Faculty/Departments",
    })

    # ------------------------------------------------------------------
    # 2 — Backlogs
    # ------------------------------------------------------------------
    insights.append({
        "id": "backlogs_01",
        "category": "Backlogs",
        "observation": (
            f"Students with 0 backlogs had a placement rate of {c['backlogs_rate_0']}% in this dataset, "
            f"while students with 3 backlogs had a placement rate of {c['backlogs_rate_3']}%. "
            f"Placement rates by backlog count: "
            + ", ".join(f"{k} backlog(s)={v}%" for k, v in sorted(c["backlogs_rates"].items()))
            + "."
        ),
        "recommendation": (
            "Students should address pending backlogs proactively. "
            "The Placement Cell may consider flagging students with 2+ backlogs for targeted academic support, "
            "as they appear underrepresented among placed students in this data."
        ),
        "audience": "Students, Placement Cell",
    })

    # ------------------------------------------------------------------
    # 3 — Internships
    # ------------------------------------------------------------------
    insights.append({
        "id": "internships_01",
        "category": "Internships",
        "observation": (
            f"Students with no internships had a placement rate of {c['intern_rate_0']}%, "
            f"whereas students with 3 internships had a placement rate of {c['intern_rate_3']}%. "
            f"Placement rates across internship counts: "
            + ", ".join(f"{k} internship(s)={v}%" for k, v in sorted(c["intern_rates"].items()))
            + "."
        ),
        "recommendation": (
            "Students are encouraged to pursue at least one internship before graduation. "
            "Institutions should strengthen industry tie-ups and offer structured internship guidance, "
            "particularly for branches where internship participation is lower."
        ),
        "audience": "Students, Institution",
    })

    # ------------------------------------------------------------------
    # 4 — Projects
    # ------------------------------------------------------------------
    insights.append({
        "id": "projects_01",
        "category": "Projects",
        "observation": (
            f"Among students grouped by project count, the group with "
            f"{c['proj_count_max']} project(s) had the highest placement rate of "
            f"{c['proj_rate_max']}%, while the group with {c['proj_count_min']} project(s) "
            f"had the lowest at {c['proj_rate_min']}%. "
            f"Full breakdown: "
            + ", ".join(f"{k} project(s)={v}%" for k, v in sorted(c["proj_rates"].items()))
            + "."
        ),
        "recommendation": (
            "Students should aim to complete meaningful projects that demonstrate practical skills. "
            "Faculty should integrate project-based learning into the curriculum to ensure all "
            "students have relevant project experience before entering the placement process."
        ),
        "audience": "Students, Faculty/Departments",
    })

    # ------------------------------------------------------------------
    # 5 — Certifications
    # ------------------------------------------------------------------
    insights.append({
        "id": "certifications_01",
        "category": "Certifications",
        "observation": (
            f"Students with 0 certifications had a placement rate of {c['cert_rate_0']}%, "
            f"while students with 3 certifications had a rate of {c['cert_rate_3']}%. "
            f"Breakdown: "
            + ", ".join(f"{k} certification(s)={v}%" for k, v in sorted(c["cert_rates"].items()))
            + "."
        ),
        "recommendation": (
            "Students should consider obtaining relevant industry certifications to strengthen "
            "their profiles. Placement Cells can organise certification drives or subsidised "
            "access to recognised online platforms."
        ),
        "audience": "Students, Placement Cell",
    })

    # ------------------------------------------------------------------
    # 6 — Coding Skills
    # ------------------------------------------------------------------
    insights.append({
        "id": "coding_01",
        "category": "Coding",
        "observation": (
            f"Placed students had a mean Coding_Skills score of {c['coding_mean_placed']} "
            f"compared to {c['coding_mean_not_placed']} for unplaced students in this dataset. "
            f"Students with a coding skill score of 4 or below had a placement rate of {c['coding_rate_low']}%, "
            f"while those scoring 8 or above had a placement rate of {c['coding_rate_high']}%."
        ),
        "recommendation": (
            "Students should invest time in improving coding proficiency through practice platforms "
            "and coding contests. Faculty can incorporate more programming-intensive coursework and "
            "lab sessions to raise the overall coding baseline across batches."
        ),
        "audience": "Students, Faculty/Departments",
    })

    # ------------------------------------------------------------------
    # 7 — Aptitude
    # ------------------------------------------------------------------
    insights.append({
        "id": "aptitude_01",
        "category": "Aptitude",
        "observation": (
            f"The mean Aptitude_Test_Score for placed students was {c['aptitude_mean_placed']} "
            f"compared to {c['aptitude_mean_not_placed']} for unplaced students. "
            "This suggests that aptitude test performance is associated with placement outcomes "
            "in this dataset."
        ),
        "recommendation": (
            "Students should practise quantitative aptitude, logical reasoning, and verbal ability "
            "well before placement season. The Placement Cell can schedule mock aptitude drives "
            "in the penultimate semester to identify and support low scorers."
        ),
        "audience": "Students, Placement Cell",
    })

    # ------------------------------------------------------------------
    # 8 — Communication / Soft Skills
    # ------------------------------------------------------------------
    insights.append({
        "id": "comm_soft_01",
        "category": "Communication/Soft",
        "observation": (
            f"Placed students had a mean Communication_Skills score of {c['comm_mean_placed']} "
            f"vs {c['comm_mean_not_placed']} for unplaced students, and a mean Soft_Skills_Rating "
            f"of {c['soft_mean_placed']} vs {c['soft_mean_not_placed']}. "
            "Both communication and soft-skill scores showed higher averages among placed students."
        ),
        "recommendation": (
            "Students should develop communication skills through group discussions, presentations, "
            "and mock interview sessions. Institutions should incorporate communication and "
            "personality development programmes into the curriculum from early semesters."
        ),
        "audience": "Students, Institution",
    })

    # ------------------------------------------------------------------
    # 9 — Degree
    # ------------------------------------------------------------------
    insights.append({
        "id": "degree_01",
        "category": "Degree",
        "observation": (
            f"Among degree types, {c['degree_best']} students had the highest placement rate "
            f"({c['degree_best_rate']}%), while {c['degree_worst']} students had the lowest "
            f"({c['degree_worst_rate']}%). "
            f"Breakdown: "
            + ", ".join(f"{k}={v}%" for k, v in sorted(c["degree_rates"].items(), key=lambda x: -x[1]))
            + "."
        ),
        "recommendation": (
            "Departments with lower placement rates should examine whether curriculum alignment "
            "with industry requirements needs improvement. The Placement Cell should design "
            "targeted preparation programmes for degrees that show weaker placement outcomes."
        ),
        "audience": "Placement Cell, Faculty/Departments",
    })

    # ------------------------------------------------------------------
    # 10 — Branch
    # ------------------------------------------------------------------
    insights.append({
        "id": "branch_01",
        "category": "Branch",
        "observation": (
            f"The {c['branch_best']} branch had the highest placement rate in this dataset "
            f"({c['branch_best_rate']}%), while the {c['branch_worst']} branch had the lowest "
            f"({c['branch_worst_rate']}%). "
            f"Branch-wise breakdown: "
            + ", ".join(f"{k}={v}%" for k, v in sorted(c["branch_rates"].items(), key=lambda x: -x[1]))
            + "."
        ),
        "recommendation": (
            "Branches with lower placement rates may benefit from additional industry interaction, "
            "skill-development workshops, and targeted recruiting events. The Institution should "
            "investigate whether certain branches have structural gaps in placement preparation."
        ),
        "audience": "Institution, Placement Cell",
    })

    # ------------------------------------------------------------------
    # 11 — Gender
    # ------------------------------------------------------------------
    male_rate = c["gender_male_rate"]
    female_rate = c["gender_female_rate"]
    diff = abs(round(male_rate - female_rate, 1))
    direction = (
        "Male students showed a slightly higher placement rate"
        if male_rate > female_rate
        else (
            "Female students showed a slightly higher placement rate"
            if female_rate > male_rate
            else "Male and female students showed identical placement rates"
        )
    )
    insights.append({
        "id": "gender_01",
        "category": "Gender",
        "observation": (
            f"In this dataset, male students had a placement rate of {male_rate}% and "
            f"female students had a placement rate of {female_rate}%. "
            f"{direction} (difference: {diff} percentage points). "
            "These figures describe the sample and do not imply any causal relationship with gender."
        ),
        "recommendation": (
            "The Placement Cell should monitor gender-wise placement outcomes each year to detect "
            "any persistent disparity and ensure equal access to placement preparation resources "
            "and opportunities for all students."
        ),
        "audience": "Placement Cell, Institution",
    })

    # ------------------------------------------------------------------
    # 12 — Multi-factor
    # ------------------------------------------------------------------
    insights.append({
        "id": "multifactor_01",
        "category": "Multi-factor",
        "observation": (
            f"The overall placement rate in this dataset is {c['overall_placement_rate']}%. "
            f"Students with a CGPA of 8.0 or above and at least one internship (n={c['high_cgpa_intern_count']}) "
            f"had a combined placement rate of {c['high_cgpa_intern_rate']}%, noticeably higher "
            f"than the dataset average. "
            f"The mean Skills_Composite (average of Coding, Communication, and Soft Skills) was "
            f"{c['skills_comp_mean_placed']} for placed students versus "
            f"{c['skills_comp_mean_not_placed']} for unplaced students."
        ),
        "recommendation": (
            "No single factor fully explains placement outcomes - students should focus on a "
            "balanced profile: solid CGPA, relevant internship experience, and strong skills. "
            "The Institution should design holistic preparation programmes that address academics, "
            "practical experience, and soft skills simultaneously rather than treating each in isolation."
        ),
        "audience": "Students, Institution, Placement Cell",
    })

    # ------------------------------------------------------------------
    # Sort by category for consistent rendering
    # ------------------------------------------------------------------
    insights.sort(key=lambda x: x["category"])

    return insights


# ---------------------------------------------------------------------------
# Executable test section
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    # Resolve path relative to this file so the script can be run from any
    # working directory (e.g.  python src/insights.py  or  cd src && python insights.py)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "processed_data.csv")

    print(f"Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Dataset loaded - {len(df):,} rows, {df.shape[1]} columns.\n")

    insights = get_insights(df)

    print(f"{'='*70}")
    print(f"  STUDENT PLACEMENT ANALYTICS - Insights & Recommendations")
    print(f"  Total insights generated: {len(insights)}")
    print(f"{'='*70}\n")

    for ins in insights:
        print(f"[{ins['id']}]  Category: {ins['category']}")
        print(f"  Observation    : {ins['observation']}")
        print(f"  Recommendation : {ins['recommendation']}")
        print(f"  Audience       : {ins['audience']}")
        print()
