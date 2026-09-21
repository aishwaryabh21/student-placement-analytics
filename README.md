# Student Placement Analytics and Decision Support System

> Descriptive and diagnostic analytics on student placement outcomes — built as an IBM Virtual Internship project.

---

## Project Overview

The **Student Placement Analytics and Decision Support System** is a pure analytics project that applies descriptive and diagnostic data analytics to a labelled dataset of 50,000 student records. The system uncovers the factors associated with campus placement outcomes, translates findings into structured, data-driven insights, and presents everything through an interactive Streamlit dashboard designed for institutional decision-making.

The project does **not** include predictive modelling or machine learning. The focus is entirely on understanding patterns already present in the data and communicating them clearly to students, faculty, placement cells, and institutional leadership.

---

## Problem Statement

Campus placement is one of the most important outcomes for students and institutions alike. Despite collecting large amounts of student profile data, many institutions lack systematic tools to analyse placement patterns across cohorts, identify at-risk groups, or communicate findings in a decision-ready format. This project addresses that gap by building a transparent, data-driven analytics system that answers: *Which student profile characteristics are associated with placement outcomes in this dataset?*

---

## Objectives

1. Validate and profile the raw dataset to confirm data quality.
2. Clean and enrich the data with derived analytical columns.
3. Conduct comprehensive exploratory data analysis (EDA) across all key factors.
4. Convert findings into structured, audience-targeted insights and recommendations.
5. Present all analyses through an interactive Streamlit dashboard with sidebar filters.
6. Produce clear documentation suitable for institutional and academic submission.

---

## Dataset Description

| Property | Detail |
|---|---|
| Source | Provided synthetic dataset — `train.csv` and `test.csv` |
| Total rows | 50,000 (45,000 train + 5,000 test) |
| Columns | 15 (shared schema across both files) |
| Target column | `Placement_Status` (`"Placed"` / `"Not Placed"`) |
| Class distribution | ~36.2% Placed / ~63.8% Not Placed |

### Column Reference

| Column | Type | Range / Values |
|---|---|---|
| `Student_ID` | int | Unique identifier — excluded from analysis |
| `Age` | int | 18–24 |
| `Gender` | str | `Male`, `Female` |
| `Degree` | str | `B.Tech`, `B.Sc`, `BCA`, `MCA` |
| `Branch` | str | `CSE`, `IT`, `ECE`, `ME`, `Civil` |
| `CGPA` | float | 4.50–9.80 |
| `Internships` | int | 0–3 |
| `Projects` | int | 1–6 |
| `Coding_Skills` | int | 1–10 |
| `Communication_Skills` | int | 1–10 |
| `Aptitude_Test_Score` | int | 35–100 |
| `Soft_Skills_Rating` | int | 1–10 |
| `Certifications` | int | 0–3 |
| `Backlogs` | int | 0–3 |
| `Placement_Status` | str | `Placed`, `Not Placed` — target column |

No missing values were found. Both files share an identical 15-column schema.

---

## Data Preparation Steps

Implemented in `src/data_preparation.py`:

1. Load `train.csv` and `test.csv` separately; tag each row with a `split` column (`"train"` / `"test"`).
2. Combine into a single 50,000-row DataFrame.
3. Strip leading/trailing whitespace from all string columns.
4. Standardise categorical casing (e.g., confirm `"Placed"` not `"placed"`).
5. Drop `Student_ID` — it is a surrogate key with no analytical value.
6. Add `Placement_Binary`: 1 if Placed, 0 if Not Placed.
7. Add `CGPA_Band`: fixed interpretable ranges — `"< 6.0"`, `"6.0-6.9"`, `"7.0-7.9"`, `"8.0-8.9"`, `"9.0+"`.
8. Add `Skills_Composite`: average of `Coding_Skills`, `Communication_Skills`, and `Soft_Skills_Rating`.
9. Save to `data/processed_data.csv` (original files are never modified).

---

## EDA and Analytics Performed

Implemented in `src/eda.py` using Plotly:

- Placement distribution (pie + bar chart)
- CGPA histogram (overlaid by placement status)
- CGPA Band vs Placement Rate (grouped bar + rate line)
- Placement by Degree, Branch, and Gender
- Backlogs vs Placement Rate
- Age distribution by Placement Status
- Coding Skills, Communication Skills, Soft Skills Rating vs Placement Rate
- Aptitude Test Score box plot by Placement Status
- Skills Composite box plot
- Internships, Projects, Certifications vs Placement Rate
- Pearson correlation heatmap (all numeric features)
- Top placement-associated factors (point-biserial correlation)
- CGPA vs Aptitude scatter plot (coloured by placement)

---

## Key Verified Findings

All values below are computed directly from `data/processed_data.csv` by `src/insights.py`.

| Factor | Finding |
|---|---|
| **Overall placement rate** | 36.2% placed, 63.8% not placed |
| **CGPA** | `9.0+` band: 73.8% placed; `< 6.0` band: 0.0%. Mean CGPA placed: 7.65 vs not placed: 6.63 |
| **Backlogs** | 0 backlogs: 59.2% placed; 2+ backlogs: 0.0% placed |
| **Internships** | 0 internships: 21.0% placed; 3 internships: 67.6% placed |
| **Projects** | 1 project: 0.0% placed; 6 projects: 74.3% placed |
| **Certifications** | 0 certifications: 0.0% placed; 3 certifications: 67.9% placed |
| **Coding Skills** | Mean score placed: 6.86 vs not placed: 5.03; score >= 8: 61.1% placed |
| **Aptitude Score** | Mean placed: 76.6 vs not placed: 65.3 |
| **Communication** | Mean placed: 6.16 vs not placed: 5.13 |
| **Branch** | CSE highest (42.4%); Civil lowest (28.0%) |
| **Degree** | MCA highest (36.9%); B.Tech/BCA lowest (35.8%) |
| **Gender** | Male: 36.3%; Female: 36.2% (negligible difference) |
| **Multi-factor** | CGPA >= 8.0 AND >= 1 internship (n=6,478): 73.9% placed |

> **Note:** All findings describe associations in this dataset. They do not establish causal relationships.

---

## Dashboard Features

Implemented in `dashboard/app.py` using Streamlit and Plotly.

**Run with:**
```bash
streamlit run dashboard/app.py
```

### Sidebar Filters (apply to all tabs)
- Dataset Split: All / Train / Test
- Degree: multi-select
- Branch: multi-select
- Gender: multi-select

### Tab 1 — Overview
- 5 KPI metric cards: Total Students, Placed, Not Placed, Placement Rate, Avg CGPA
- Placement distribution chart (pie + bar)
- Placement by CGPA Band and by Branch (side by side)
- Placement by Gender

### Tab 2 — Academic Analysis
- CGPA distribution histogram
- CGPA Band vs Placement Rate
- Backlogs vs Placement Rate
- Aptitude Test Score box plot
- CGPA vs Aptitude scatter plot
- Placement by Degree

### Tab 3 — Skills Analysis
- Average skill score comparison table (Placed vs Not Placed)
- Coding Skills vs Placement Rate
- Communication Skills vs Placement Rate
- Soft Skills Rating vs Placement Rate
- Aptitude Test Score box plot
- Skills Composite box plot

### Tab 4 — Activity Analysis
- Internships vs Placement Rate
- Projects vs Placement Rate
- Certifications vs Placement Rate
- Top placement-associated factors (point-biserial correlation bar chart)
- Pearson correlation heatmap

### Tab 5 — Insights & Recommendations
- 12 structured insight cards, each with: Observation, Recommendation, Audience badge
- Audience filter: All / Students / Placement Cell / Faculty/Departments / Institution
- Insights computed from data — no hard-coded findings

---

## Technology Stack

| Tool | Purpose | Version |
|---|---|---|
| Python | Core language | 3.10+ recommended |
| pandas | Data loading, cleaning, aggregation | >=2.0,<3.0 |
| NumPy | Numeric operations | >=1.24,<2.0 |
| Plotly | Interactive charts | >=5.18,<6.0 |
| Streamlit | Dashboard web framework | >=1.32,<2.0 |
| SciPy | Point-biserial correlation | >=1.11,<2.0 |

---

## Project Structure

```
Student_Placement_Analytics/
|-- train.csv                    # Original training data (45,000 rows) — never modified
|-- test.csv                     # Original test data (5,000 rows) — never modified
|-- data/
|   `-- processed_data.csv       # Cleaned + enriched combined dataset (50,000 rows)
|-- src/
|   |-- data_validation.py       # Sub-Task 1: validate and profile raw data
|   |-- data_preparation.py      # Sub-Task 2: clean, engineer features, save processed data
|   |-- eda.py                   # Sub-Task 3: all EDA chart functions (Plotly)
|   `-- insights.py              # Sub-Task 4: structured observations and recommendations
|-- dashboard/
|   `-- app.py                   # Sub-Task 5: Streamlit dashboard (5 tabs)
|-- requirements.txt             # Python package dependencies
|-- README.md                    # This file
|-- PROJECT_REPORT.md            # Full project report
`-- project_plan.md              # Approved project plan
```

---

## Installation

**Requirements:** Python 3.10 or later.

```bash
# 1. Clone or download the repository
git clone https://github.com/aishwaryabh21/student-placement-analytics
cd student-placement-analytics

# 2. (Recommended) Create and activate a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## How to Run

### Launch the Dashboard
```bash
streamlit run dashboard/app.py
```
The dashboard opens at `http://localhost:8501` in your browser.

### Run Individual Scripts
```bash
# Step 1 — Validate raw data
python src/data_validation.py

# Step 2 — Prepare and save processed_data.csv
python src/data_preparation.py

# Step 3 — Run EDA self-test (generates all charts, prints summary)
python src/eda.py

# Step 4 — Run insights self-test (prints all 12 insights with computed values)
python src/insights.py
```

---

## Project Limitations

- The dataset is synthetic. Findings reflect patterns within this specific dataset and may not generalise to real institutional data.
- The dataset is class-imbalanced (~36% Placed / ~64% Not Placed). Raw accuracy would be a misleading metric; the project uses placement rates (percentages) throughout.
- All reported relationships are associations, not causal claims. Many confounding factors are not captured in this dataset.
- The dataset does not include attendance, socioeconomic background, or external examination scores, which are important real-world placement factors.
- This project is descriptive and diagnostic only. No prediction or recommendation engine is implemented.

---

## Dataset Source

`train.csv` and `test.csv` are provided as part of the IBM Virtual Internship project dataset. Both files are included in this repository:

**GitHub Repository:** [https://github.com/aishwaryabh21/student-placement-analytics](https://github.com/aishwaryabh21/student-placement-analytics)

The dataset files (`train.csv`, `test.csv`) can be found directly at:
- `https://github.com/aishwaryabh21/student-placement-analytics/blob/main/train.csv`
- `https://github.com/aishwaryabh21/student-placement-analytics/blob/main/test.csv`

---

## Acknowledgements

This project was developed as part of the **IBM Virtual Internship Programme**. The dataset, problem statement, and project guidelines were provided through the internship framework. The analytics implementation, dashboard, insights module, and documentation are original work.
