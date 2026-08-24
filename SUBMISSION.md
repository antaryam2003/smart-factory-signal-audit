# Submission Form

**Project:** Manufacturing Process Health and Operational Efficiency Analysis in 6G-Enabled Smart Factories
**Programme:** Thales Group / Unified Mentor
**Dataset:** `Thales_Group_Manufacturing.csv` — 100,000 readings · 50 machines · 1 Jan – 10 Mar 2025

---

## Form fields

| Field | Value | Status |
|---|---|---|
| **GitHub Repository Link** | https://github.com/antaryam2003/smart-factory-signal-audit | ✅ Public |
| **Research paper link** | https://claude.ai/code/artifact/ec7edebe-7808-4632-a4ca-ebe7e9e601a4 | ⚠️ Private — share before submitting |
| **Deployed project link** | https://smart-factory-signal-audit.streamlit.app | ✅ Live |
| **Project Feedback video link** | *(to be recorded)* | ⬜ Pending |

> All fields must contain valid URLs starting with `https://`.

---

## Before submitting

- [ ] **Share the research paper artifact.** Open the link and use **Share** in the top right, otherwise reviewers get a 404. Public fallback if preferred: [`reports/research_paper.md`](reports/research_paper.md) in this repository.
- [ ] **Open the deployed app once** to confirm it renders. First load takes ~20–30 s while it validates 100,000 rows and fits the ablation models; it is cached afterwards.
- [ ] **Record the feedback video** covering experience and learnings.

---

## What each link contains

### GitHub repository
Full source for the diagnostic layer: validation pipeline (`src/data_loader.py`), the five KPIs
(`src/kpis.py`), the statistical test suite (`src/statistics_tests.py`), the feature-ablation study
(`src/modeling.py`), the Streamlit dashboard (`app.py`), and a reproducible EDA runner
(`analysis/run_eda.py`) that regenerates every figure and `reports/findings.json`.

### Research paper
Nine sections covering data validation, methodology, results, discussion, recommendations and
limitations — including the recovered label rule, the feature-ablation table and the six-check data
acceptance protocol. Also available as Markdown at [`reports/research_paper.md`](reports/research_paper.md).

### Deployed dashboard
Five live modules — Factory Health Overview, Machine Health Dashboard, Production & Quality Panel,
Efficiency Diagnostics, and Data Quality & Validation — with machine, operation-mode, date, hour and
efficiency-status filters. Deployed from `main` / `app.py` on Python 3.13.

### Executive summary
Not a form field, but included in the repository at
[`reports/executive_summary.md`](reports/executive_summary.md) for government and institutional
stakeholders, as required by the project brief's deliverables.

---

## Headline findings

| Finding | Evidence |
|---|---|
| The nine sensor channels are mutually independent | Max \|Pearson r\| across all 36 pairs = **0.0075** |
| Sensors carry no information about efficiency | Balanced accuracy **0.3333** on 3 classes = random guessing |
| `Efficiency_Status` is a deterministic arithmetic rule | Recovered exactly; reproduces **99.998 %** of rows |
| No machine is meaningfully different from any other | Health Index spread **1.27 / 100**; zero 3σ outliers |
| The file has structural defects | 1 March duplicated → **2,880** repeated timestamps |

The same model that scores balanced accuracy 0.3333 reports **77.8 % raw accuracy** purely from class
imbalance. A permutation test settles it: training on randomly shuffled labels gives 0.77824 versus
0.77822 on real labels — a difference of **−0.00003**.
