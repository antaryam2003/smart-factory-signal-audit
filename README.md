# Manufacturing Process Health & Operational Efficiency Analysis in 6G-Enabled Smart Factories

A diagnostic intelligence layer for industrial IIoT telemetry: data validation, machine health
scoring, production and quality diagnostics, and a live Streamlit dashboard — built for the
**Thales Group / Unified Mentor** project.

**Dataset:** 100,000 readings · 50 machines · 3 operation modes · 1 Jan – 10 Mar 2025

---

## Headline findings

Applying the diagnostic layer to the supplied telemetry produced an unambiguous verdict, backed
by five independent lines of statistical evidence:

| Finding | Evidence |
|---|---|
| The nine sensor channels are mutually independent | Max \|Pearson r\| across all 36 pairs = **0.0075** |
| Sensors carry **no** information about efficiency | Balanced accuracy **0.3333** on 3 classes = random guessing |
| The efficiency label is a deterministic arithmetic rule | Recovered exactly; reproduces **99.998%** of rows |
| No machine is meaningfully different from any other | Health Index spread **1.27 / 100**; zero 3σ outliers |
| The file has structural defects | 1 Mar duplicated → **2,880** repeated timestamps |

**The trap this creates:** the same model that scores balanced accuracy 0.333 (chance) reports
**77.8% raw accuracy** — a figure that would pass most project reviews. A permutation test settles
it: training on randomly shuffled labels gives 0.77824 versus 0.77822 on real labels, a difference
of **−0.00003**. The model has learned nothing.

Full reasoning, methodology and recommendations: [`reports/research_paper.md`](reports/research_paper.md).

---

## Quick start

```bash
git clone <repository-url>
cd Thales-group

pip install -r requirements.txt

python analysis/run_eda.py     # regenerate all figures + reports/findings.json
streamlit run app.py           # launch the dashboard at http://localhost:8501
```

Requires Python 3.9+. First dashboard load takes ~20 s (validating 100k rows and fitting the
ablation models); everything is cached afterwards.

---

## The dashboard

Five modules, all recomputing live against the active filter selection.

| Module | Contents |
|---|---|
| **Factory Health Overview** | Efficiency distribution, five KPIs vs fleet baseline, average sensor metrics, time-spent-near-limits |
| **Machine Health Dashboard** | Per-machine sensor trends, health scorecard vs 3σ control limits, health-index composition, downloadable scorecard |
| **Production & Quality** | Speed vs defect rate and density, error frequency by machine/mode/hour, machine comparison, bottleneck analysis |
| **Efficiency Diagnostics** | Recovered label rule, feature ablation, mode & shift comparison, interactive cross-metric explorer, full statistical suite |
| **Data Quality & Validation** | Every validation check, capture-block timeline, duplicate-window warning, sampling cadence diagnostics |

**User controls:** machine selector · operation mode filter · date range · hour-of-day slider ·
efficiency status filter · metric comparison toggles throughout.

---

## Key Performance Indicators

| KPI | Definition | Fleet value |
|---|---|---|
| **Machine Health Index** | `100 × (1 − mean stress)` over temperature, vibration and power. Stress is 0 inside the warning threshold, ramping linearly to 1 at the hard limit. | 87.50 / 100 |
| **Average Production Speed** | Mean output rate per machine | 275.9 units/hr |
| **Defect Density Score** | `defect rate × production speed ÷ 100` — defective *units per hour*, not a bare percentage | 13.80 units/hr |
| **Error Frequency Index** | Mean operational error rate | 7.50 % |
| **Efficiency Distribution** | High / Medium / Low spread | 2.99 / 19.19 / 77.83 % |

All thresholds live in [`src/config.py`](src/config.py) so the paper, the analysis pipeline and the
dashboard cannot drift apart. Replace them with manufacturer limits for a real deployment.

---

## Repository layout

```
├── app.py                          Streamlit dashboard (5 modules)
├── analysis/
│   └── run_eda.py                  Reproducible EDA → figures + findings.json
├── src/
│   ├── config.py                   Paths, sensor envelopes, thresholds, palette
│   ├── data_loader.py              Validation, cleaning, feature engineering
│   ├── kpis.py                     The five KPIs + control limits
│   ├── statistics_tests.py         KS, chi-square, ANOVA, binomial homogeneity
│   ├── modeling.py                 Ablation, mutual information, rule recovery
│   └── charts.py                   Plotly chart builders
├── data/
│   └── Thales_Group_Manufacturing.csv
├── reports/
│   ├── research_paper.md           Full paper
│   ├── executive_summary.md        For government stakeholders
│   ├── findings.json               Every statistic, machine-readable
│   ├── machine_scorecard.csv       Per-machine KPI table
│   └── figures/                    Nine publication figures
├── requirements.txt
└── .streamlit/config.toml
```

---

## Methodology

Follows the seven-stage protocol in the research paper:

1. **Data validation & preparation** — envelope checks, timeline reconstruction, duplicate resolution
2. **Machine-level sensor health** — per-machine profiles, threshold proximity, cross-mode stability
3. **Production performance diagnostics** — throughput trends, output consistency, 3σ control limits
4. **Quality & error analysis** — defect/sensor correlation, error spikes, bottleneck identification
5. **Efficiency distribution** — spread across machines, modes and shifts
6. **Cross-metric diagnostics** — temperature vs defects, vibration vs errors, power vs throughput
7. **Signal validation** — feature ablation, permutation testing, mutual information, rule recovery

Two deliberate choices shape every result:

- **Binned means over scatter plots.** 100,000 noisy points fill their bounding box whether or not a
  relationship exists. Binned conditional means with 95% intervals make a flat relationship
  visibly flat.
- **Balanced accuracy over raw accuracy.** With 77.8% of readings in one class, raw accuracy measures
  the class imbalance, not the model. Every result is reported against both trivial baselines.

---

## Figures

| | |
|---|---|
| `01_sensor_distributions.png` | All nine sensors uniform across their full range |
| `02_correlation_heatmap.png` | No sensor pair correlates beyond \|r\| = 0.01 |
| `03_efficiency_overview.png` | Efficiency split identical across modes and shifts |
| `04_label_rule.png` | The label is a rectangle rule on two columns |
| `05_machine_scorecard.png` | No machine breaches 3σ control limits |
| `06_cross_metric_diagnostics.png` | Every cross-metric relationship is flat |
| `07_feature_ablation.png` | Raw accuracy misleads; balanced accuracy exposes it |
| `08_temporal_patterns.png` | No trend, seasonality or shift effect |
| `09_health_components.png` | Health index distribution and stress composition |

---

## Deploying

The app runs on [Streamlit Community Cloud](https://share.streamlit.io) with no changes:

1. Push this repository to GitHub (the 9 MB dataset is well inside the 100 MB file limit).
2. On Streamlit Cloud, create an app pointing at `app.py` on the `main` branch.
3. Dependencies install automatically from `requirements.txt`.

The two model-based panels fit on a class-stratified 25,000-row sample to stay responsive on
free-tier hardware; `analysis/run_eda.py` uses all 99,972 rows and agrees to four decimal places.

---

## Reproducibility

Every figure and statistic in the paper regenerates from `python analysis/run_eda.py`. All
randomised procedures use `random_state=42`. Results are written to `reports/findings.json` in
machine-readable form so any claim in the prose can be traced to its source.

---

## Running on your own data

Nothing is hard-coded to the values in this file. Point `src/config.py` at telemetry with the same
schema and the entire pipeline — validation, KPIs, dashboard, statistical suite, ablation study —
runs unchanged. On instrumented data the tests become genuinely diagnostic: correlations should be
non-zero, distributions skewed, sensors should beat baseline, and machines should separate under
control limits. **The analysis is designed so that "the data is fine" is a result it can produce.**
