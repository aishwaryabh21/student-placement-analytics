"""
data_validation.py
------------------
Sub-Task 1: Data Validation and Profiling
Student Placement Analytics and Decision Support System

Loads train.csv and test.csv, profiles every column, checks for quality
issues, and prints a clear summary report.  The original CSV files are
never modified.
"""

import os
import pandas as pd

# ---------------------------------------------------------------------------
# Paths (relative to the project root, one level above this file)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(PROJECT_ROOT, "train.csv")
TEST_PATH  = os.path.join(PROJECT_ROOT, "test.csv")

# ---------------------------------------------------------------------------
# Expected values / ranges (from project_plan.md and AGENTS.md)
# ---------------------------------------------------------------------------

# Categorical: every value that is allowed in each column
EXPECTED_CATEGORICAL = {
    "Gender":           {"Male", "Female"},
    "Degree":           {"B.Tech", "B.Sc", "BCA", "MCA"},
    "Branch":           {"CSE", "IT", "ECE", "ME", "Civil"},
    "Placement_Status": {"Placed", "Not Placed"},
}

# Numeric: (min_allowed, max_allowed)  -- both boundaries are inclusive
EXPECTED_NUMERIC_RANGES = {
    "Age":                   (18,   24),
    "CGPA":                  (4.50, 9.80),   # 9.8 is valid
    "Internships":           (0,    3),
    "Projects":              (1,    6),
    "Coding_Skills":         (1,    10),
    "Communication_Skills":  (1,    10),
    "Aptitude_Test_Score":   (35,   100),
    "Soft_Skills_Rating":    (1,    10),
    "Certifications":        (0,    3),
    "Backlogs":              (0,    3),
}

# The 15 columns must appear in this exact order
EXPECTED_COLUMNS = [
    "Student_ID", "Age", "Gender", "Degree", "Branch", "CGPA",
    "Internships", "Projects", "Coding_Skills", "Communication_Skills",
    "Aptitude_Test_Score", "Soft_Skills_Rating", "Certifications",
    "Backlogs", "Placement_Status",
]


# ---------------------------------------------------------------------------
# Helper: pretty section header
# ---------------------------------------------------------------------------
def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# 1. Load files
# ---------------------------------------------------------------------------
def load_files() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train.csv and test.csv without modifying them."""
    print(f"\nLoading files from: {PROJECT_ROOT}")
    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)
    print(f"  train.csv loaded  ->  {train.shape[0]:,} rows  x  {train.shape[1]} columns")
    print(f"  test.csv  loaded  ->  {test.shape[0]:,} rows  x  {test.shape[1]} columns")
    return train, test


# ---------------------------------------------------------------------------
# 2. Basic shape / dtype / null report
# ---------------------------------------------------------------------------
def profile_basic(df: pd.DataFrame, name: str) -> list[str]:
    """
    Print shape, dtypes, null counts, and sample values.
    Returns a list of issue strings (empty if all clear).
    """
    section(f"Basic Profile -- {name}")
    issues = []

    # Shape
    print(f"\nShape : {df.shape[0]:,} rows  x  {df.shape[1]} columns")

    # Column names and dtypes
    print(f"\n{'Column':<28} {'Dtype':<12} {'Nulls':>6}  Sample values")
    print("-" * 70)
    for col in df.columns:
        nulls   = df[col].isnull().sum()
        samples = df[col].dropna().unique()[:3]
        sample_str = ", ".join(str(v) for v in samples)
        print(f"  {col:<26} {str(df[col].dtype):<12} {nulls:>6}  {sample_str}")
        if nulls > 0:
            issues.append(f"{name} | '{col}' has {nulls} null value(s)")

    return issues


# ---------------------------------------------------------------------------
# 3. Schema consistency check
# ---------------------------------------------------------------------------
def check_schema(train: pd.DataFrame, test: pd.DataFrame) -> list[str]:
    """Verify both files have identical column names in the same order."""
    section("Schema Consistency Check (train vs test)")
    issues = []

    train_cols = list(train.columns)
    test_cols  = list(test.columns)

    if train_cols == test_cols == EXPECTED_COLUMNS:
        print("\n  [OK]  Column names and order match the expected schema in both files.")
    else:
        if train_cols != EXPECTED_COLUMNS:
            issues.append(f"train.csv column order differs from expected schema")
            print(f"  [!!]  train.csv columns : {train_cols}")
            print(f"     Expected          : {EXPECTED_COLUMNS}")
        if test_cols != EXPECTED_COLUMNS:
            issues.append(f"test.csv column order differs from expected schema")
            print(f"  [!!]  test.csv  columns : {test_cols}")
            print(f"     Expected          : {EXPECTED_COLUMNS}")
        if train_cols != test_cols:
            issues.append("train.csv and test.csv have different column schemas")

    return issues


# ---------------------------------------------------------------------------
# 4. Categorical column validation
# ---------------------------------------------------------------------------
def validate_categoricals(df: pd.DataFrame, name: str) -> list[str]:
    """Check that categorical columns contain only expected values."""
    section(f"Categorical Column Validation -- {name}")
    issues = []

    for col, allowed in EXPECTED_CATEGORICAL.items():
        if col not in df.columns:
            issues.append(f"{name} | Column '{col}' is missing")
            print(f"  [!!]  '{col}' -- MISSING")
            continue

        value_counts = df[col].value_counts()
        found_values = set(df[col].dropna().unique())
        unexpected   = found_values - allowed

        print(f"\n  Column : {col}")
        print(f"  Allowed: {sorted(allowed)}")
        for val, count in value_counts.items():
            tag = "  <- UNEXPECTED" if val not in allowed else ""
            print(f"    '{val}': {count:,}{tag}")

        if unexpected:
            issues.append(
                f"{name} | '{col}' contains unexpected values: {unexpected}"
            )
        else:
            print(f"  [OK]  All values are within the expected set.")

    return issues


# ---------------------------------------------------------------------------
# 5. Numeric column range validation
# ---------------------------------------------------------------------------
def validate_numeric_ranges(df: pd.DataFrame, name: str) -> list[str]:
    """Check that numeric columns stay within documented ranges."""
    section(f"Numeric Column Range Validation -- {name}")
    issues = []

    print(f"\n  {'Column':<26} {'Min':>8} {'Max':>8} {'Mean':>8} {'Std':>8}  {'Out-of-range rows':>18}")
    print("  " + "-" * 80)

    for col, (lo, hi) in EXPECTED_NUMERIC_RANGES.items():
        if col not in df.columns:
            issues.append(f"{name} | Column '{col}' is missing")
            continue

        series     = pd.to_numeric(df[col], errors="coerce")
        null_count = series.isnull().sum()
        if null_count > 0:
            issues.append(f"{name} | '{col}' has {null_count} non-numeric / null value(s)")

        col_min  = series.min()
        col_max  = series.max()
        col_mean = series.mean()
        col_std  = series.std()

        # Count rows strictly outside the [lo, hi] range
        out_of_range = ((series < lo) | (series > hi)).sum()

        flag = "  [!!]  OUT OF RANGE" if out_of_range > 0 else ""

        print(
            f"  {col:<26} {col_min:>8.2f} {col_max:>8.2f} "
            f"{col_mean:>8.2f} {col_std:>8.2f}  {out_of_range:>18}{flag}"
        )

        if out_of_range > 0:
            issues.append(
                f"{name} | '{col}' has {out_of_range} row(s) outside "
                f"the allowed range [{lo}, {hi}]"
            )

    return issues


# ---------------------------------------------------------------------------
# 6. Duplicate Student_ID check (within a file)
# ---------------------------------------------------------------------------
def check_duplicate_ids(df: pd.DataFrame, name: str) -> list[str]:
    """Check for duplicate Student_ID values within a single file."""
    section(f"Duplicate Student_ID Check -- {name}")
    issues = []

    dupes = df[df.duplicated(subset=["Student_ID"], keep=False)]
    if dupes.empty:
        print(f"\n  [OK]  No duplicate Student_IDs found in {name}.")
    else:
        count = df["Student_ID"].duplicated().sum()
        print(f"\n  [!!]  {count} duplicate Student_ID(s) found in {name}:")
        print(dupes[["Student_ID"]].drop_duplicates().to_string(index=False))
        issues.append(f"{name} | {count} duplicate Student_ID(s) found")

    return issues


# ---------------------------------------------------------------------------
# 7. Student_ID overlap between train and test
# ---------------------------------------------------------------------------
def check_id_overlap(train: pd.DataFrame, test: pd.DataFrame) -> list[str]:
    """Check whether any Student_ID appears in both train.csv and test.csv."""
    section("Student_ID Overlap Check (train vs test)")
    issues = []

    train_ids  = set(train["Student_ID"])
    test_ids   = set(test["Student_ID"])
    overlap    = train_ids & test_ids

    if not overlap:
        print(f"\n  [OK]  No Student_ID overlap between train.csv and test.csv.")
    else:
        print(f"\n  [!!]  {len(overlap)} Student_ID(s) appear in both files: {sorted(overlap)[:10]} ...")
        issues.append(f"{len(overlap)} Student_ID(s) overlap between train.csv and test.csv")

    return issues


# ---------------------------------------------------------------------------
# 8. Blank row check
# ---------------------------------------------------------------------------
def check_blank_rows(df: pd.DataFrame, name: str) -> list[str]:
    """Check for rows where every column is null/empty."""
    section(f"Blank Row Check -- {name}")
    issues = []

    blank_rows = df[df.isnull().all(axis=1)]
    if blank_rows.empty:
        print(f"\n  [OK]  No completely blank rows found in {name}.")
    else:
        print(f"\n  [!!]  {len(blank_rows)} completely blank row(s) found in {name}.")
        issues.append(f"{name} | {len(blank_rows)} completely blank row(s)")

    return issues


# ---------------------------------------------------------------------------
# 9. Final summary
# ---------------------------------------------------------------------------
def print_summary(all_issues: list[str]) -> None:
    """Print a consolidated summary of all issues collected."""
    section("VALIDATION SUMMARY")

    if not all_issues:
        print("\n  [OK]  No issues found. Both files passed all validation checks.")
    else:
        print(f"\n  {len(all_issues)} issue(s) detected:\n")
        for i, issue in enumerate(all_issues, start=1):
            print(f"  [{i}] {issue}")

    print("\n" + "=" * 60)
    print("  Validation complete.")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    all_issues = []

    # Load
    train, test = load_files()

    # Schema
    all_issues += check_schema(train, test)

    # Basic profile (nulls, dtypes, sample values)
    all_issues += profile_basic(train, "train.csv")
    all_issues += profile_basic(test,  "test.csv")

    # Blank rows
    all_issues += check_blank_rows(train, "train.csv")
    all_issues += check_blank_rows(test,  "test.csv")

    # Categorical validation
    all_issues += validate_categoricals(train, "train.csv")
    all_issues += validate_categoricals(test,  "test.csv")

    # Numeric range validation
    all_issues += validate_numeric_ranges(train, "train.csv")
    all_issues += validate_numeric_ranges(test,  "test.csv")

    # Duplicate IDs
    all_issues += check_duplicate_ids(train, "train.csv")
    all_issues += check_duplicate_ids(test,  "test.csv")

    # ID overlap
    all_issues += check_id_overlap(train, test)

    # Summary
    print_summary(all_issues)


if __name__ == "__main__":
    main()


