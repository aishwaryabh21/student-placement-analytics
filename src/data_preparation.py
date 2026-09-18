"""
data_preparation.py
-------------------
Sub-Task 2: Data Preparation and Cleaning
Student Placement Analytics and Decision Support System

Loads train.csv and test.csv, cleans and enriches the data with derived
columns, and saves the combined analysis-ready dataset to
data/processed_data.csv.

The original CSV files are never modified.
"""

import os
import pandas as pd

# ---------------------------------------------------------------------------
# Paths (relative to the project root, one level above this file)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH   = os.path.join(PROJECT_ROOT, "train.csv")
TEST_PATH    = os.path.join(PROJECT_ROOT, "test.csv")
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")
OUTPUT_PATH  = os.path.join(DATA_DIR, "processed_data.csv")

# ---------------------------------------------------------------------------
# CGPA band definitions (fixed, interpretable ranges from project_plan.md)
# These labels will appear in charts exactly as written here.
#   "< 6.0"   -> CGPA below 6.0
#   "6.0-6.9" -> CGPA from 6.0 up to (but not including) 7.0
#   "7.0-7.9" -> CGPA from 7.0 up to (but not including) 8.0
#   "8.0-8.9" -> CGPA from 8.0 up to (but not including) 9.0
#   "9.0+"    -> CGPA from 9.0 up to the maximum 9.8
# ---------------------------------------------------------------------------
CGPA_BINS   = [0.0,  6.0,  7.0,  8.0,  9.0, 10.0]
CGPA_LABELS = ["< 6.0", "6.0-6.9", "7.0-7.9", "8.0-8.9", "9.0+"]


# ---------------------------------------------------------------------------
# Step 1: Load files and tag each row with its split
# ---------------------------------------------------------------------------
def load_and_tag() -> pd.DataFrame:
    """
    Load train.csv and test.csv into separate DataFrames.
    Add a 'split' column before combining so we can always tell
    which file each row came from.
    Returns the combined DataFrame.
    """
    print("Loading train.csv ...")
    train = pd.read_csv(TRAIN_PATH)
    train["split"] = "train"   # tag every training row

    print("Loading test.csv ...")
    test  = pd.read_csv(TEST_PATH)
    test["split"]  = "test"    # tag every test row

    print(f"  train rows: {len(train):,}")
    print(f"  test  rows: {len(test):,}")

    # Combine into one DataFrame and reset the index to be sequential
    combined = pd.concat([train, test], ignore_index=True)
    print(f"  combined   : {len(combined):,} rows")
    return combined


# ---------------------------------------------------------------------------
# Step 2: Clean string columns
# ---------------------------------------------------------------------------
def clean_strings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strip leading/trailing whitespace from every string (object) column.
    This is a defensive step — the data is already clean, but good practice.
    """
    for col in df.select_dtypes(include=["object", "str"]).columns:
        df[col] = df[col].str.strip()
    print("\nString columns stripped of leading/trailing whitespace.")
    return df


# ---------------------------------------------------------------------------
# Step 3: Standardize categorical casing
# ---------------------------------------------------------------------------
def standardize_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make sure the categorical values use the expected casing.
    For example: 'placed' -> 'Placed', 'b.tech' -> 'B.Tech'
    The data is already clean, but this step prevents silent casing bugs.
    """
    # Expected proper-cased mappings for each categorical column
    casing_map = {
        "Gender": {v.lower(): v for v in ["Male", "Female"]},
        "Degree": {v.lower(): v for v in ["B.Tech", "B.Sc", "BCA", "MCA"]},
        "Branch": {v.lower(): v for v in ["CSE", "IT", "ECE", "ME", "Civil"]},
        "Placement_Status": {v.lower(): v for v in ["Placed", "Not Placed"]},
    }

    for col, mapping in casing_map.items():
        if col in df.columns:
            # Map lowercase version of current values back to proper casing
            df[col] = df[col].str.strip().str.lower().map(mapping).fillna(df[col])

    print("Categorical columns standardized to consistent casing.")
    return df


# ---------------------------------------------------------------------------
# Step 4: Drop Student_ID
# ---------------------------------------------------------------------------
def drop_student_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove Student_ID from the analysis dataset.
    It is a surrogate key with no analytical value.
    The original CSV files still contain it unchanged.
    """
    df = df.drop(columns=["Student_ID"])
    print("Student_ID dropped (surrogate key, no analytical value).")
    return df


# ---------------------------------------------------------------------------
# Step 5: Add derived columns
# ---------------------------------------------------------------------------
def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add three derived columns that simplify downstream analysis:

    1. Placement_Binary  - integer version of the target (1=Placed, 0=Not Placed)
    2. CGPA_Band         - fixed interpretable CGPA range label
    3. Skills_Composite  - average of the three skill scores
    """

    # --- Placement_Binary ---------------------------------------------------
    # Map the string target to 0/1 so we can compute placement rates easily
    # using .mean()  (e.g.  df.groupby('Branch')['Placement_Binary'].mean() )
    df["Placement_Binary"] = df["Placement_Status"].map({"Placed": 1, "Not Placed": 0})

    # --- CGPA_Band ----------------------------------------------------------
    # Bin CGPA into fixed, meaningful ranges defined at the top of this file.
    # right=False means the left boundary is included: [6.0, 7.0) -> "6.0-6.9"
    df["CGPA_Band"] = pd.cut(
        df["CGPA"],
        bins=CGPA_BINS,
        labels=CGPA_LABELS,
        right=False,          # left-closed intervals: [lo, hi)
        include_lowest=True,  # include 4.5 in the first bin
    )

    # --- Skills_Composite ---------------------------------------------------
    # Simple average of three skill scores (each rated 1-10).
    # Rounded to 2 decimal places for readability.
    df["Skills_Composite"] = (
        df["Coding_Skills"]
        + df["Communication_Skills"]
        + df["Soft_Skills_Rating"]
    ) / 3
    df["Skills_Composite"] = df["Skills_Composite"].round(2)

    print("Derived columns added: Placement_Binary, CGPA_Band, Skills_Composite.")
    return df


# ---------------------------------------------------------------------------
# Step 6: Save to data/processed_data.csv
# ---------------------------------------------------------------------------
def save_processed(df: pd.DataFrame) -> None:
    """Create the data/ directory if needed and save the processed dataset."""
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nProcessed dataset saved to: {OUTPUT_PATH}")


# ---------------------------------------------------------------------------
# Step 7: Sanity-check summary
# ---------------------------------------------------------------------------
def print_summary(df: pd.DataFrame) -> None:
    """Print a clear summary so we can confirm the output is correct."""
    print("\n" + "=" * 60)
    print("  SANITY CHECK SUMMARY")
    print("=" * 60)

    # Shape
    print(f"\n  Final shape : {df.shape[0]:,} rows  x  {df.shape[1]} columns")

    # Column list
    print(f"\n  Columns ({len(df.columns)}):")
    for i, col in enumerate(df.columns, start=1):
        print(f"    {i:>2}. {col}")

    # Split counts
    print("\n  Rows per split:")
    for split_val, count in df["split"].value_counts().items():
        print(f"    {split_val:>6}: {count:,}")

    # Placement counts
    print("\n  Placement_Status counts:")
    for status, count in df["Placement_Status"].value_counts().items():
        pct = count / len(df) * 100
        print(f"    {status:<12}: {count:,}  ({pct:.1f}%)")

    # CGPA Band distribution
    print("\n  CGPA_Band distribution:")
    for band, count in df["CGPA_Band"].value_counts().sort_index().items():
        pct = count / len(df) * 100
        print(f"    {str(band):<10}: {count:,}  ({pct:.1f}%)")

    # Missing values
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("\n  Missing values : None")
    else:
        print("\n  Missing values:")
        for col, n in missing.items():
            print(f"    {col}: {n}")

    # Skills_Composite range
    sc = df["Skills_Composite"]
    print(f"\n  Skills_Composite : min={sc.min():.2f}  max={sc.max():.2f}  mean={sc.mean():.2f}")

    # Sample rows
    print("\n  Sample rows (first 3):")
    print(df.head(3).to_string(index=False))

    print("\n" + "=" * 60)
    print("  Data preparation complete.")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  Sub-Task 2: Data Preparation and Cleaning")
    print("=" * 60)

    # Load and tag
    df = load_and_tag()

    # Clean strings (whitespace)
    df = clean_strings(df)

    # Standardize categorical casing
    df = standardize_categoricals(df)

    # Drop Student_ID
    df = drop_student_id(df)

    # Add derived columns
    df = add_derived_columns(df)

    # Save
    save_processed(df)

    # Sanity check
    print_summary(df)


if __name__ == "__main__":
    main()
