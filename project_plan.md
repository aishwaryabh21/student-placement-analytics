# Student Placement Analytics and Decision Support System — Project Plan

## Overview

**Goal:** Build a descriptive and diagnostic analytics project that uncovers what factors drive student placement outcomes, translates findings into actionable insights for institutions and students, and presents everything in an interactive Streamlit dashboard.

**Scope:**
- Dataset: `train.csv` (45,000 rows) and `test.csv` (5,000 rows) — 15 shared columns, pre-labelled.
- Primary focus: Descriptive + diagnostic analytics (distributions, comparisons, correlations, insights).
- Deliverables: cleaned data layer, EDA module, insights module, Streamlit dashboard, README, requirements.txt.
- Original CSV files are never modified.

**Non-Goals:**
- Any prediction modelling or machine learning.
- Real-time data ingestion.
- User authentication or multi-tenant features.

**Dataset Key Facts (discovered from files):**
- 15 columns: Student_ID, Age, Gender, Degree, Branch, CGPA, Internships, Projects, Coding_Skills, Communication_Skills, Aptitude_Test_Score, Soft_Skills_Rating, Certifications, Backlogs, Placement_Status.
- Target: `Placement_Status` — string values `"Placed"` / `"Not Placed"` (not binary int).
- Class imbalance: ~36% Placed / 64% Not Placed in both splits.
- `Student_ID` is a surrogate key — must be excluded from analytics.
- No nulls or blank rows detected. Data is clean but CGPA reaches the maximum boundary (9.8) in many rows.
- Categorical encoding needed for: Gender (binary), Degree (4 values), Branch (5 values).

---

## Project Structure (Target)

```
Student_Placement_Analytics/
├── train.csv                        # Original — never modified
├── test.csv                         # Original — never modified
├── data/
│   └── processed_data.csv           # Cleaned, feature-engineered dataset
├── src/
│   ├── data_validation.py           # Step 1: validate & profile raw data
│   ├── data_preparation.py          # Step 2: clean, encode, save processed data
│   ├── eda.py                       # Step 3: all EDA charts (reusable functions)
│   └── insights.py                  # Step 4: observation + recommendation text
├── dashboard/
│   └── app.py                       # Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## Sub-Tasks

---

### Sub-Task 1 — Data Validation and Profiling

**Status:** [x] done

**Intent:**
Systematically validate both CSV files to build a clear understanding of data quality before any cleaning or analysis. This step produces a data-quality report that informs cleaning decisions in Sub-Task 2.

**Expected Outcomes:**
- A printed/logged profile for every column: dtype, unique value counts, min/max/mean, null count, and sample values.
- A list of any quality issues found: outliers, unexpected values, duplicate Student_IDs, or inconsistent categoricals.
- Confirmation that both files share an identical schema.
- A clear decision log: which issues (if any) need to be fixed in cleaning, and how.

**Todo List:**
1. Create `src/data_validation.py`.
2. Load both `train.csv` and `test.csv` with pandas (do not modify originals).
3. Print shape, column names, dtypes, and null counts for each file.
4. Check that both files share identical column names and column order.
5. Profile categorical columns (Gender, Degree, Branch, Placement_Status): list all unique values and counts — flag any value not in the expected set.
6. Profile numeric columns (Age, CGPA, Internships, Projects, Coding_Skills, Communication_Skills, Aptitude_Test_Score, Soft_Skills_Rating, Certifications, Backlogs): report min, max, mean, std — flag values outside the documented ranges.
7. Check for duplicate Student_ID values within each file.
8. Check for Student_ID overlap between train and test files.
9. Check for completely blank rows.
10. Print a clean summary table of all issues found (or "No issues" if clean).

**Relevant Context:**
- `train.csv` / `test.csv` — raw source files.
- Expected ranges per AGENTS.md: Age 18–24, CGPA 4.50–9.80, Internships 0–3, Projects 1–6, Coding_Skills 1–10, Communication_Skills 1–10, Aptitude_Test_Score 35–100, Soft_Skills_Rating 1–10, Certifications 0–3, Backlogs 0–3.
- CGPA 9.8 is a valid boundary value — do not flag it as an error.
- No nulls or blank rows were detected in the pre-analysis, so validation is expected to pass cleanly.

---

### Sub-Task 2 — Data Preparation and Cleaning

**Status:** [x] done

**Intent:**
Produce a clean, analysis-ready dataset saved to `data/processed_data.csv` without touching the originals. This dataset is used by all downstream modules (EDA and insights).

**Expected Outcomes:**
- `data/processed_data.csv` created — combines train and test, cleaned and with new derived columns.
- A separate flag column (`split`) marks each row as `"train"` or `"test"` so downstream code can filter.
- `Placement_Status` is preserved as a string and also encoded as a binary integer column (`Placed`: 1, `Not Placed`: 0) named `Placement_Binary`.
- Original string columns (Gender, Degree, Branch, Placement_Status) are preserved unchanged alongside any derived columns.
- A brief printed summary confirms row counts and column list of the processed file.

**Todo List:**
1. Create `src/data_preparation.py`.
2. Load `train.csv` and `test.csv`; add a `split` column (`"train"` / `"test"`) to each before combining.
3. Combine into a single DataFrame for unified processing.
4. Strip leading/trailing whitespace from all string columns (defensive).
5. Standardize categorical values to consistent casing (e.g., ensure `"Placed"` not `"placed"`).
6. Drop `Student_ID` from the analysis-ready dataset (retain it only as an index if needed for lookups).
7. Add derived columns:
   - `Placement_Binary`: 1 if Placed, 0 if Not Placed.
   - `CGPA_Band`: fixed interpretable CGPA ranges — `"< 6.0"`, `"6.0–6.9"`, `"7.0–7.9"`, `"8.0–8.9"`, `"9.0+"` — for categorical EDA charts.
   - `Skills_Composite`: average of Coding_Skills, Communication_Skills, and Soft_Skills_Rating — a composite skills score.
8. Create `data/` directory if it does not exist.
9. Save the processed DataFrame to `data/processed_data.csv`.
10. Print shape, column names, and sample rows of the saved file as a sanity check.

**Relevant Context:**
- `Placement_Status` is `"Placed"` / `"Not Placed"` (strings) — `Placement_Binary` mapping: `{"Placed": 1, "Not Placed": 0}`.
- `Student_ID` is confirmed to be a surrogate key with no analytical value.
- CGPA range 4.50–9.80. The five CGPA bands are fixed interpretable ranges — group sizes will vary based on the data distribution.
- `Skills_Composite` enables a single-axis comparison of overall skill level vs placement.

---

### Sub-Task 3 — Exploratory Data Analysis (EDA)

**Status:** [x] done

**Intent:**
Produce all analytical charts and numerical summaries needed for the dashboard and insights. Functions are written to return figures (not save them), so the dashboard can embed them directly. This is the analytical heart of the project.

**Expected Outcomes:**
- `src/eda.py` containing one clearly named function per analysis topic.
- Every function accepts the processed DataFrame as input and returns a Plotly figure (for dashboard use) or a pandas summary table.
- Coverage of all required EDA topics listed in the project objective.
- Charts are clearly labelled (title, axis labels, legend, percentage annotations where applicable).

**Todo List:**

*Setup*
1. Create `src/eda.py` with imports: pandas, numpy, plotly.express, plotly.graph_objects.
2. Load processed data from `data/processed_data.csv` in a `load_data()` helper function (accepts optional `split` filter parameter: `"train"`, `"test"`, or `"all"`).

*Placement Overview*
3. `plot_placement_distribution(df)` — pie chart + bar chart of Placed vs Not Placed with counts and percentages.
4. `placement_kpis(df)` — returns a dict of: total students, total placed, total not placed, placement rate (%), average CGPA of placed vs not placed.

*Academic Performance*
5. `plot_cgpa_distribution(df)` — histogram of CGPA coloured by Placement_Status; overlay KDE.
6. `plot_cgpa_band_vs_placement(df)` — grouped bar chart: CGPA_Band × Placement_Status counts; add placement rate line.
7. `plot_placement_by_degree(df)` — grouped bar chart: Degree × Placement_Status; show both count and rate.
8. `plot_placement_by_branch(df)` — grouped bar chart: Branch × Placement_Status.
9. `plot_backlogs_vs_placement(df)` — bar chart: Backlogs count (0–3) × placement rate.

*Demographic Analysis*
10. `plot_placement_by_gender(df)` — grouped bar chart: Gender × Placement_Status.
11. `plot_age_vs_placement(df)` — box plot: Age distribution by Placement_Status.

*Skills Analysis*
12. `plot_coding_skills_vs_placement(df)` — bar/box chart: Coding_Skills scores (1–10) × placement rate.
13. `plot_communication_skills_vs_placement(df)` — same pattern for Communication_Skills.
14. `plot_soft_skills_vs_placement(df)` — same pattern for Soft_Skills_Rating.
15. `plot_aptitude_vs_placement(df)` — box plot: Aptitude_Test_Score by Placement_Status; add threshold line.
16. `plot_skills_composite_vs_placement(df)` — box plot of Skills_Composite by Placement_Status.

*Activity Analysis*
17. `plot_internships_vs_placement(df)` — grouped bar chart: Internships count (0–3) × placement rate.
18. `plot_projects_vs_placement(df)` — grouped bar chart: Projects count × placement rate.
19. `plot_certifications_vs_placement(df)` — grouped bar chart: Certifications (0–3) × placement rate.

*Correlation and Multi-factor*
20. `plot_correlation_heatmap(df)` — heatmap of Pearson correlation of all numeric columns (placed students highlighted separately if useful).
21. `plot_top_factors(df)` — horizontal bar chart of point-biserial correlation of each feature with `Placement_Binary`, sorted by absolute magnitude. Shows top placement predictors at a glance.
22. `plot_cgpa_vs_aptitude_scatter(df)` — scatter plot: CGPA vs Aptitude_Test_Score, coloured by Placement_Status.

**Relevant Context:**
- Use Plotly (not matplotlib/seaborn) so charts render interactively in Streamlit without saving to files.
- All functions must be self-contained (accept df, return fig) — no global state.
- `CGPA_Band` and `Placement_Binary` columns are available from the processed dataset.
- Class imbalance means raw counts can mislead — always include a placement *rate* (%) alongside counts.

---

### Sub-Task 4 — Observations and Insights Module

**Status:** [ ] pending

**Intent:**
Convert analytical findings into structured, human-readable observations and actionable recommendations. These are stored as structured data (list of dicts) so the dashboard can render them in a dedicated section.

**Expected Outcomes:**
- `src/insights.py` with a `get_insights()` function that returns a list of insight dicts.
- Each insight has: `category`, `observation`, `recommendation`, and `audience` (Student / Placement Cell / Institution).
- At least one insight per major EDA topic.
- Insights are grounded in actual data patterns (not generic advice) — values in the observation text must reference real numbers computed from the dataset.

**Todo List:**
1. Create `src/insights.py`.
2. Define a `compute_insight_data(df)` function that runs the key aggregations needed (placement rates by CGPA band, by internship count, by backlogs, etc.) and returns a dict of findings.
3. Define `get_insights(df)` that calls `compute_insight_data` and returns a list of insight dicts, one per finding, with keys: `id`, `category`, `observation`, `recommendation`, `audience`.
4. Write insights for each of these categories (minimum one insight each):
   - CGPA & Academic Performance
   - Backlogs
   - Internships
   - Projects
   - Certifications
   - Coding Skills
   - Aptitude Test Score
   - Communication / Soft Skills
   - Degree Type
   - Branch
   - Gender
   - Multi-factor (combined pattern)
5. Ensure observation text embeds computed values (e.g., "Students with 2+ internships have a placement rate of X% vs Y% for those with none.").
6. Return insights sorted by category for consistent rendering.

**Relevant Context:**
- `Placement_Binary` column simplifies rate computation: `df.groupby('feature')['Placement_Binary'].mean() * 100`.
- Audience tagging enables the dashboard to filter insights by who they are relevant for.
- Observations must be factual and grounded — do not invent thresholds; compute them from the data.

---

### Sub-Task 5 — Streamlit Dashboard

**Status:** [ ] pending

**Intent:**
Build a clean, interactive, single-file Streamlit application that ties together all modules. The dashboard is the primary deliverable for college viva presentation.

**Expected Outcomes:**
- `dashboard/app.py` that runs with `streamlit run dashboard/app.py` from the project root.
- Exactly five tabs using `st.tabs()`: Overview, Academic Analysis, Skills Analysis, Activity Analysis, Insights & Recommendations.
- All charts are Plotly-based and render interactively.
- Global sidebar filters (split selector, degree filter, branch filter, gender filter) applied to all sections.
- A final Insights & Recommendations tab with audience-filtered cards.

**Todo List:**

*Setup & Layout*
1. Create `dashboard/app.py`.
2. Configure page: `st.set_page_config(page_title="Student Placement Analytics", layout="wide", page_icon="🎓")`.
3. Add sidebar with:
   - Project title and description.
   - Dataset selector: `Train`, `Test`, `Combined` (filters `split` column).
   - Degree filter (multi-select, default all).
   - Branch filter (multi-select, default all).
   - Gender filter (multi-select, default all).
4. Load processed data via `src/eda.load_data()` and apply sidebar filters reactively.
5. Use `st.tabs()` to create exactly five tabs: `Overview`, `Academic Analysis`, `Skills Analysis`, `Activity Analysis`, `Insights & Recommendations`.

*Tab 1 — Overview*
6. Display 4 KPI metric cards in a `st.columns(4)` row: Total Students, Placed, Not Placed, Placement Rate %.
7. Show placement distribution pie chart.
8. Show placement by degree and by branch (side by side).
9. Show placement by gender.

*Tab 2 — Academic Analysis*
10. CGPA distribution histogram (coloured by placement).
11. Placement rate by CGPA band (bar + line combo).
12. Backlogs vs placement rate.
13. Degree and Branch grouped bar charts.
14. Age box plot vs placement.

*Tab 3 — Skills Analysis*
15. Coding skills vs placement rate.
16. Communication skills vs placement rate.
17. Soft skills vs placement rate.
18. Aptitude test score box plot.
19. Skills composite box plot.
20. Correlation heatmap of all numeric features.

*Tab 4 — Activity Analysis*
21. Internships vs placement rate.
22. Projects vs placement rate.
23. Certifications vs placement rate.
24. Top placement factors horizontal bar chart (point-biserial correlations).
25. CGPA vs Aptitude scatter plot coloured by placement.

*Tab 5 — Insights & Recommendations*
26. Add audience filter buttons (All / Students / Placement Cell / Institution).
27. Render each insight as a styled `st.expander` card showing: Observation, Recommendation, and Audience badge.
28. Add a brief summary paragraph at the top of the tab.

*Polish*
29. Add `st.spinner` wrappers around any heavy computations.
30. Add footer: dataset source note and project name.
31. Test that the app runs without errors from the project root with `streamlit run dashboard/app.py`.

**Relevant Context:**
- Import from `src/` using `sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))` at the top of `app.py`.
- All chart functions return Plotly figures — render with `st.plotly_chart(fig, use_container_width=True)`.
- Sidebar filters must be applied before passing `df` to any chart or insight function.
- Keep colour palette consistent: use a single primary colour for "Placed" and another for "Not Placed" across all charts (e.g., `#2ecc71` green / `#e74c3c` red).

---

### Sub-Task 6 — requirements.txt and README.md

**Status:** [ ] pending

**Intent:**
Provide complete setup documentation and dependency pinning so the project runs reproducibly in any Python environment.

**Expected Outcomes:**
- `requirements.txt` with pinned versions for all dependencies.
- `README.md` covering all required sections from the project objective.

**Todo List:**

*requirements.txt*
1. Create `requirements.txt` in the project root.
2. Include only packages actually required by the implemented project:
   - `pandas>=2.0,<3.0`
   - `numpy>=1.24,<2.0`
   - `plotly>=5.18,<6.0`
   - `streamlit>=1.32,<2.0`
   - `scipy>=1.11,<2.0` (used for point-biserial correlation in `plot_top_factors`)
3. Do NOT include scikit-learn or joblib — no prediction model is implemented.
4. Verify no conflicting version constraints.

*README.md*
5. Create `README.md` in the project root with the following sections:
   - **Project Title and Badge Line**
   - **Project Overview** — what the system does, emphasis on descriptive and diagnostic analytics.
   - **Problem Statement** — why placement analytics matters for institutions.
   - **Dataset** — source note, file names, row counts, column list with brief descriptions.
   - **Features** — what the dashboard includes (KPIs, charts, insights and recommendations).
   - **Technologies Used** — Python, pandas, NumPy, Plotly, Streamlit.
   - **Project Structure** — directory tree with one-line descriptions per file.
   - **Installation** — step-by-step: clone, create venv, `pip install -r requirements.txt`.
   - **How to Run** — `streamlit run dashboard/app.py` from the project root.
   - **Dataset Source** — note that `train.csv` and `test.csv` are provided in the repository.
   - **Acknowledgements** — brief note.

**Relevant Context:**
- Python 3.10+ is recommended (Streamlit 1.32 requires it).
- README should be suitable for a college project submission — clear, structured, no jargon.
- Do not mention prediction or machine learning anywhere in the README.

---

## Dependency Map Between Sub-Tasks

```
Sub-Task 1 (Validation)
        |
        v
Sub-Task 2 (Preparation) --> produces data/processed_data.csv
        |
        +-----------> Sub-Task 3 (EDA)
        |                   |
        +-----------> Sub-Task 4 (Insights)
                            |
                    Both feed into:
                    Sub-Task 5 (Dashboard)
                            |
                    Sub-Task 6 (Docs) runs in parallel
```

Sub-Tasks 3 and 4 both depend on the processed data from Sub-Task 2 and can be built in any order. Sub-Task 5 integrates all of them. Sub-Task 6 is independent and can be written alongside any other sub-task.

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Plotly instead of matplotlib/seaborn | Interactive in Streamlit without saving to disk; better UX for viva demo |
| Combined train+test with `split` column | Enables unified analysis while preserving the ability to filter by split |
| Insights as structured dicts, not hardcoded text | Makes them filterable by audience in the dashboard |
| `Placement_Binary` derived column | Simplifies placement rate computation via `.mean()` |
| `CGPA_Band` derived column | Enables categorical comparison charts for CGPA using fixed, interpretable ranges |
| No prediction model in scope | Keeps focus on descriptive/diagnostic analytics and actionable recommendations |
| scipy for point-biserial correlation only | Lightweight; no ML framework needed for a correlation statistic |
| sys.path manipulation in app.py | Avoids packaging complexity for a college project |
