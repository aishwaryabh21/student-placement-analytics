# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Project Overview

**Student Placement Analytics** — a pure data project. The repository contains only two CSV files:
- `train.csv` — 45,000 labelled student records (training set)
- `test.csv` — 5,000 labelled student records (test/evaluation set)

There is no source code, no build system, and no package manager. Any analytical code (Python, R, notebooks, etc.) must be created from scratch or brought in by the user.

## Dataset Schema

Both files share the same 15-column schema:

| Column | Type | Range / Values |
|---|---|---|
| `Student_ID` | int | unique identifier |
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
| `Placement_Status` | str | `Placed`, `Not Placed` ← **target** |

## Class Imbalance — Critical

The dataset is **imbalanced**: roughly 36% Placed vs 64% Not Placed (train: 16,312 / 28,688; test: 1,812 / 3,188). Any classifier must account for this — do not evaluate with raw accuracy alone; use F1, ROC-AUC, or balanced accuracy.

## No Build / Test / Lint Commands

There are no scripts, Makefiles, notebooks, or dependencies in this repository. Add any tooling you need.

## Naming Conventions (columns)

- Column names use `PascalCase` with underscores (`Soft_Skills_Rating`, `Aptitude_Test_Score`).
- Target column: `Placement_Status` (string, not 0/1).
- `Student_ID` is not a predictive feature — exclude from model inputs.
