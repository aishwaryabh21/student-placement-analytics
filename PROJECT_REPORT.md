# Project Report: Student Placement Analytics and Decision Support System

**Programme:** IBM Virtual Internship  
**Domain:** Data Analytics  
**Type:** Descriptive and Diagnostic Analytics  
**Dataset:** train.csv (45,000 rows) + test.csv (5,000 rows) = 50,000 student records

---

## 1. Title

**Student Placement Analytics and Decision Support System**

---

## 2. Abstract

Campus placement is a high-stakes outcome for students, faculty, and institutions. This project develops a descriptive and diagnostic analytics system that analyses 50,000 student records to identify which profile characteristics are associated with placement outcomes. The system consists of a data validation module, a data preparation pipeline, an exploratory data analysis (EDA) library, a structured insights module, and an interactive Streamlit dashboard. All findings are presented as associations within this dataset; no causal claims are made and no predictive modelling is applied. The overall placement rate in the dataset is 36.2%. Key factors showing strong associations with placement include CGPA, number of internships, project count, certifications, coding skill scores, and aptitude test scores. The dashboard allows placement cells, faculty, and institutional leadership to explore these patterns interactively and filter results by degree, branch, gender, and dataset split.

---

## 3. Introduction

Higher education institutions invest considerable resources in campus placement preparation, yet analysis of placement data typically remains ad hoc or limited to summary statistics. A systematic, data-driven approach can reveal patterns that are not visible from aggregate reports alone — for example, how placement rates change across CGPA bands, how having backlogs affects placement odds, or which branches have persistently lower outcomes.

This project was developed under the IBM Virtual Internship Programme to demonstrate how a structured analytics pipeline can be built on a realistic student dataset using open-source Python tools. The project is deliberately scoped to descriptive and diagnostic analytics: it describes what the data shows and explores relationships between variables, without attempting to predict individual placement outcomes.

---

## 4. Problem Statement

Campus placement cells and institutional leadership often face the following challenges:

- No systematic view of which student profile factors are associated with placement success.
- Difficulty identifying cohorts at risk of low placement rates in time to intervene.
- Inability to filter and compare placement patterns across degree programmes, branches, or demographic groups.
- Findings locked in spreadsheets or one-off reports that cannot be updated as new data arrives.

This project addresses these gaps by providing a reusable, data-driven analytics system with an interactive dashboard that makes placement data exploration accessible without requiring programming knowledge.

---

## 5. Objectives

1. Validate both raw data files (train.csv, test.csv) to confirm data quality and schema consistency.
2. Produce a clean, enriched processed dataset with derived analytical columns, without modifying the originals.
3. Perform comprehensive EDA across all 14 analytical columns using interactive Plotly charts.
4. Generate 12 structured, data-grounded insights with audience-targeted recommendations.
5. Build a five-tab interactive Streamlit dashboard with sidebar filters for institutional exploration.
6. Document the project fully for academic and professional submission.

---

## 6. Dataset Description

### Source
`train.csv` and `test.csv` are provided as part of the IBM Virtual Internship dataset. Both files share an identical 15-column schema and contain pre-labelled placement outcomes.

### Size
| File | Rows | Purpose |
|---|---|---|
| `train.csv` | 45,000 | Primary analytical dataset |
| `test.csv` | 5,000 | Held-out evaluation set |
| **Combined** | **50,000** | Used for all dashboard analysis |

### Class Distribution
- Placed: 18,124 students (36.2%)
- Not Placed: 31,876 students (63.8%)

The dataset is class-imbalanced. All placement comparisons in this report use placement rates (percentages) rather than raw counts to avoid misleading conclusions.

### Schema

| Column | Type | Range / Values | Notes |
|---|---|---|---|
| `Student_ID` | int | Unique | Excluded from analysis — surrogate key |
| `Age` | int | 18–24 | Narrow range; limited analytical signal |
| `Gender` | str | Male, Female | Binary categorical |
| `Degree` | str | B.Tech, B.Sc, BCA, MCA | 4-value categorical |
| `Branch` | str | CSE, IT, ECE, ME, Civil | 5-value categorical |
| `CGPA` | float | 4.50–9.80 | Only true continuous feature |
| `Internships` | int | 0–3 | Ordinal |
| `Projects` | int | 1–6 | Ordinal |
| `Coding_Skills` | int | 1–10 | Ordinal |
| `Communication_Skills` | int | 1–10 | Ordinal |
| `Aptitude_Test_Score` | int | 35–100 | Ordinal-integer |
| `Soft_Skills_Rating` | int | 1–10 | Ordinal |
| `Certifications` | int | 0–3 | Ordinal |
| `Backlogs` | int | 0–3 | Ordinal |
| `Placement_Status` | str | Placed, Not Placed | Target variable |

### Data Quality
- No missing values in either file (confirmed by `src/data_validation.py`).
- No duplicate Student_IDs within each file.
- No Student_ID overlap between train and test.
- All column values fall within documented ranges.

---

## 7. Data Preparation

Implemented in `src/data_preparation.py`. The original CSV files are never modified.

### Steps

**1. Load and tag splits**  
`train.csv` and `test.csv` are loaded separately. A `split` column (`"train"` / `"test"`) is added to each before combining into a single 50,000-row DataFrame.

**2. String cleaning**  
Leading and trailing whitespace is stripped from all string columns. Categorical values are confirmed to be consistently cased.

**3. Drop Student_ID**  
`Student_ID` is a surrogate key with no predictive or analytical value. It is dropped from the analysis-ready dataset.

**4. Derived columns**

| New Column | Formula | Purpose |
|---|---|---|
| `Placement_Binary` | 1 if Placed, 0 if Not Placed | Simplifies rate computation via `.mean()` |
| `CGPA_Band` | pd.cut with fixed bins | Enables categorical bar charts across CGPA ranges |
| `Skills_Composite` | mean of Coding, Communication, Soft Skills | Single-axis overall skill comparison |

**CGPA Band boundaries:**

| Band | CGPA Range |
|---|---|
| `< 6.0` | Below 6.0 |
| `6.0-6.9` | 6.0 to 6.99 |
| `7.0-7.9` | 7.0 to 7.99 |
| `8.0-8.9` | 8.0 to 8.99 |
| `9.0+` | 9.0 and above |

**5. Save output**  
The combined, enriched DataFrame is saved to `data/processed_data.csv` (18 columns, 50,000 rows).

---

## 8. Exploratory Data Analysis

All EDA is implemented in `src/eda.py` using Plotly. Every function accepts a DataFrame and returns a Plotly figure — there is no global state, and no charts are saved to disk. The Streamlit dashboard imports and renders these functions directly.

### Analyses Performed

**Placement Overview**
- Overall distribution: pie chart + bar chart of Placed vs Not Placed
- Key Performance Indicators (KPIs): total students, placed, not placed, placement rate, average CGPA

**Academic Performance**
- CGPA distribution histogram (overlaid by placement status)
- CGPA Band vs Placement Rate (grouped bar with rate line overlay)
- Backlogs vs Placement Rate
- Aptitude Test Score box plot by placement status
- Placement by Degree
- CGPA vs Aptitude scatter plot

**Demographic**
- Placement by Gender (grouped bar + rate)
- Age distribution by placement status (box plot)

**Skills**
- Coding Skills vs Placement Rate (bar chart per score level)
- Communication Skills vs Placement Rate
- Soft Skills Rating vs Placement Rate
- Skills Composite box plot by placement status

**Activity**
- Internships vs Placement Rate
- Projects vs Placement Rate
- Certifications vs Placement Rate

**Correlation and Multi-factor**
- Pearson correlation heatmap (all numeric features)
- Point-biserial correlation of each feature with Placement_Binary (top factors chart)

---

## 9. Key Findings

All values are computed from `data/processed_data.csv` by `src/insights.py`. All relationships are associations in this dataset; they do not establish causation.

### CGPA
Students in the `9.0+` CGPA band showed a placement rate of **73.8%** in this dataset, compared to **0.0%** for students in the `< 6.0` band. The mean CGPA of placed students was **7.65** vs **6.63** for unplaced students. A strong positive association exists between CGPA and placement rate.

### Backlogs
Students with **0 backlogs** had a placement rate of **59.2%**. Students with **1 backlog** had a rate of **33.1%**. Students with **2 or 3 backlogs** had placement rates of **0.0%**. Backlogs show a sharp negative association with placement outcomes.

### Internships
Placement rates increase consistently with internship count:
- 0 internships: **21.0%**
- 1 internship: **44.3%**
- 2 internships: **55.6%**
- 3 internships: **67.6%**

### Projects
Placement rates increase with project count:
- 1 project: **0.0%**
- 2 projects: **0.0%**
- 3 projects: **7.4%**
- 4 projects: **50.6%**
- 5 projects: **67.2%**
- 6 projects: **74.3%**

### Certifications
- 0 certifications: **0.0%** placed
- 1 certification: **1.8%** placed
- 2 certifications: **48.1%** placed
- 3 certifications: **67.9%** placed

### Coding Skills
Mean Coding_Skills score: placed **6.86** vs not placed **5.03**. Students scoring 4 or below: **0.0%** placement rate. Students scoring 8 or above: **61.1%** placement rate.

### Aptitude Test Score
Mean Aptitude_Test_Score: placed **76.6** vs not placed **65.3**. Aptitude performance shows one of the strongest associations with placement in the point-biserial correlation analysis.

### Communication and Soft Skills
Mean Communication_Skills: placed **6.16** vs not placed **5.13**. Soft_Skills_Rating showed near-identical means (placed: 5.50 vs not placed: 5.51), suggesting it has a weaker individual association with placement in this dataset.

### Branch
- CSE: **42.4%** placed (highest)
- IT: **42.1%**
- ECE: **36.3%**
- ME: **32.5%**
- Civil: **28.0%** placed (lowest)

### Degree
Placement rates are similar across degree types (35.8%–36.9%), suggesting degree type alone is a weak differentiator in this dataset.

### Gender
Male students: **36.3%** placed. Female students: **36.2%** placed. The difference is 0.1 percentage points — effectively negligible in this dataset.

### Multi-factor Pattern
Students with a CGPA of 8.0 or above and at least one internship (n = 6,478) had a combined placement rate of **73.9%**, nearly double the dataset average of **36.2%**. The mean Skills_Composite score was **6.17** for placed students vs **5.22** for unplaced students, reinforcing that a combination of academic performance, practical experience, and skills is associated with stronger placement outcomes.

---

## 10. Business and Institutional Insights

Twelve structured insight cards are generated by `src/insights.py`, each targeting a specific audience. Key institutional takeaways:

**For Students:**
- Maintain CGPA in the higher bands (7.0+ preferred), clear backlogs early, and pursue at least one internship.
- Improve coding proficiency and aptitude scores — these show the largest gaps between placed and unplaced groups.
- Aim for 3+ projects and 2+ certifications as practical experience indicators.

**For the Placement Cell:**
- Flag students with 2+ backlogs for early academic support.
- Schedule mock aptitude drives in the penultimate semester.
- Monitor branch-wise and gender-wise placement rates for persistent disparities.
- Design targeted preparation tracks for Civil and ME branch students.

**For Faculty and Departments:**
- Use CGPA-band breakdowns in mid-semester reviews to identify at-risk cohorts.
- Integrate project-based learning to ensure all students complete multiple projects before graduation.
- Incorporate coding-intensive lab sessions to raise the overall coding baseline.

**For the Institution:**
- Strengthen industry tie-ups to increase internship access, particularly for lower-placed branches.
- Design holistic preparation programmes addressing academics, practical experience, and soft skills together.
- Investigate structural gaps in curriculum-industry alignment for Civil and Mechanical branches.

---

## 11. Dashboard Design

The dashboard (`dashboard/app.py`) is built with Streamlit and uses Plotly for all interactive charts.

### Design Principles
- Single-file architecture (`app.py`) for simplicity and portability.
- All chart functions imported from `src/eda.py` — no duplicated logic.
- `@st.cache_data` on data loading to avoid re-reading CSV on every filter interaction.
- `st.spinner()` wrappers on all computationally heavy sections.
- Graceful handling of empty filter states (warning message rather than crash).
- Colour convention: Placed = green (`#2ecc71`), Not Placed = red (`#e74c3c`), applied consistently across all charts.

### Tab Structure

| Tab | Contents |
|---|---|
| Overview | KPI cards, placement distribution, CGPA Band, Branch, Gender charts |
| Academic Analysis | CGPA histogram, CGPA Band rate, Backlogs, Aptitude, Scatter, Degree |
| Skills Analysis | Score comparison table, Coding, Communication, Soft Skills, Aptitude, Composite |
| Activity Analysis | Internships, Projects, Certifications, Top factors, Correlation heatmap |
| Insights & Recommendations | 12 audience-filtered insight cards with observation, recommendation, and audience badges |

### Sidebar Filters
Dataset Split (All/Train/Test), Degree (multi-select), Branch (multi-select), Gender (multi-select). All filters are reactive and apply immediately to all tabs.

---

## 12. Technologies Used

| Technology | Role |
|---|---|
| Python 3.10+ | Core language |
| pandas 2.x | Data loading, cleaning, aggregation, groupby analysis |
| NumPy 1.x | Numeric array operations, correlation matrix computation |
| Plotly 5.x | All interactive charts (histogram, bar, scatter, box, heatmap, pie) |
| Streamlit 1.x | Web dashboard framework — tabs, sidebar, metrics, expanders |
| SciPy 1.x | Point-biserial correlation (`scipy.stats.pointbiserialr`) |

No machine learning frameworks (scikit-learn, TensorFlow, PyTorch) were used. This is intentional — the project is scoped to descriptive and diagnostic analytics only.

---

## 13. Project Workflow

```
train.csv + test.csv
        |
        v
[Sub-Task 1] src/data_validation.py
  -- Validates schema, dtypes, ranges, nulls, duplicates
        |
        v
[Sub-Task 2] src/data_preparation.py
  -- Cleans, enriches, combines -> data/processed_data.csv
        |
        +--------------------+
        |                    |
        v                    v
[Sub-Task 3]          [Sub-Task 4]
src/eda.py            src/insights.py
  -- Plotly chart        -- Structured insight
     functions             dictionaries
        |                    |
        +--------------------+
                 |
                 v
[Sub-Task 5] dashboard/app.py
  -- Streamlit dashboard integrating
     all EDA charts + insights
                 |
                 v
[Sub-Task 6] README.md + PROJECT_REPORT.md + requirements.txt
  -- Documentation and dependency management
```

Files never modified by any sub-task: `train.csv`, `test.csv`.

---

## 14. Limitations

1. **Synthetic dataset.** The data is synthetic. Patterns and rates may not reflect real institutional placement data.
2. **Class imbalance.** The dataset is ~36/64 imbalanced. All analyses use placement rates rather than counts to avoid misleading conclusions.
3. **Associations only.** All reported findings are associations in this dataset. Confounding factors not captured in the data (e.g., interview performance, location, socioeconomic background) are not accounted for.
4. **Missing real-world features.** The dataset does not include attendance records, extracurricular activities, or specific company information. These are important placement factors in practice.
5. **No temporal dimension.** All records are treated as a single cross-sectional dataset. Year-over-year trend analysis is not possible.
6. **Ordinal features treated as grouped.** Skill and activity columns (Coding_Skills, Internships, etc.) are ordinal integers. The analysis respects this by grouping rather than treating them as continuous.
7. **No predictive capability.** This system describes what happened in the data — it does not predict future placement outcomes for individual students.

---

## 15. Conclusion

This project demonstrates that a well-structured descriptive analytics pipeline can surface meaningful patterns from student placement data and communicate them effectively to multiple institutional stakeholders. The key finding is that no single factor determines placement outcomes in this dataset — rather, placement is associated with a combination of academic performance (CGPA, backlogs), practical experience (internships, projects, certifications), and skills (coding, aptitude, communication). Students with strong profiles across all three dimensions show placement rates nearly double the dataset average.

The Streamlit dashboard provides a practical decision-support tool that enables non-technical users to explore placement patterns, filter by demographic or programme group, and access data-grounded recommendations — all without requiring direct access to the underlying data files.

---

## 16. Future Scope

1. **Predictive analytics layer:** A machine learning classification model (logistic regression, random forest) could be added as a separate module to predict placement probability for individual students, complementing the descriptive analytics.
2. **Year-over-year tracking:** If historical placement data across multiple academic years is available, trend analysis could identify whether placement rates are improving or declining over time.
3. **Real institutional data integration:** Replacing the synthetic dataset with actual institutional records would make the system operationally useful for college placement offices.
4. **Automated report generation:** A PDF export feature for the dashboard could allow placement cells to generate standardised reports for institutional management.
5. **Company-specific analysis:** If recruiter and company data are available, analysis could be extended to identify which student profiles match specific recruiters' selection patterns.

---

## 17. Dataset Source

`train.csv` and `test.csv` are provided as part of the **IBM Virtual Internship Programme** dataset. Both files are included in this repository. The data is synthetic and intended for educational analytics purposes.

---

## 18. References and Acknowledgements

- **IBM Virtual Internship Programme** — for the project brief, dataset, and problem statement.
- **Streamlit documentation** (https://docs.streamlit.io) — for dashboard development reference.
- **Plotly Python documentation** (https://plotly.com/python) — for chart implementation reference.
- **pandas documentation** (https://pandas.pydata.org/docs) — for data manipulation reference.
- **SciPy documentation** (https://docs.scipy.org) — for point-biserial correlation reference.

All analytics implementation, code, dashboard design, insights, and documentation in this repository are original work produced as part of the IBM Virtual Internship submission.

---

*Report prepared for IBM Virtual Internship — Student Placement Analytics and Decision Support System*
