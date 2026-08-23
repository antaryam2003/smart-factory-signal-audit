# Executive Summary

## Manufacturing Process Health and Operational Efficiency Analysis in 6G-Enabled Smart Factories

**Prepared for:** Government and institutional stakeholders
**Subject:** Readiness assessment of industrial IIoT telemetry for AI-driven manufacturing programmes
**Evidence base:** 100,000 sensor readings · 50 machines · 1 January – 10 March 2025
**Companion materials:** full research paper · live analytics dashboard · reproducible code repository

---

## The question we were asked

Smart factories now instrument hundreds of machines with real-time sensors and stream that data over high-speed 6G networks. Before public or private capital is committed to predictive maintenance and AI-driven optimisation, decision-makers need to know three things: **are the machines healthy, where is efficiency being lost, and which assets are starting to degrade?**

This project built the diagnostic system that answers those questions, and then applied it to a real telemetry file covering 50 machines over ten weeks.

## What we built

A production-ready operational intelligence layer, delivered as a live web dashboard with five modules:

- **Factory Health Overview** — fleet-wide efficiency and sensor status at a glance
- **Machine Health Dashboard** — per-machine scorecards benchmarked against the fleet
- **Production & Quality Panel** — throughput, defects and error frequency
- **Efficiency Diagnostics** — what drives the efficiency classification
- **Data Quality & Validation** — every integrity check, shown openly

The system tracks five operational KPIs — a composite Machine Health Index, Average Production Speed, Defect Density, Error Frequency and the High/Medium/Low efficiency distribution — and lets an operator filter by machine, operation mode, date, hour and status. It is built against a standard telemetry schema and will run unchanged on any facility that supplies data in that shape.

## What we found

**The dashboard works. The data does not support the conclusions it was intended to produce.**

Our diagnostic tests returned a consistent and unambiguous result across five independent lines of evidence:

| Finding | Evidence |
|---|---|
| **The sensors are unrelated to each other.** Temperature, vibration, power, network quality and defect rate move completely independently. | Strongest relationship among 36 sensor pairs explains **0.006%** of variation — indistinguishable from random. |
| **The sensors are unrelated to efficiency.** | A machine-learning model given the seven physical sensors classifies efficiency at **exactly the accuracy of random guessing**, and performs identically when the answers are deliberately scrambled. |
| **The efficiency rating is not a measurement.** It is a fixed arithmetic formula applied to two figures already in the file. | We recovered the formula exactly; it reproduces the published rating for **99.998%** of records. |
| **Machine identity makes no difference.** | All 50 machines fall within **1.3 points on a 100-point health scale**; none breaches statistical control limits. The variation between them is smaller than random chance alone would produce. |
| **The file has structural defects.** | One day (1 March) is recorded twice, producing **2,880 duplicated timestamps**; each machine is observed only once every **34 minutes** on average, with gaps up to **8.5 hours**. |

Two consequences follow directly. Predictive maintenance cannot be built on this data, because the sensors carry no early-warning signal. And machines cannot be benchmarked against one another, because the differences between them are smaller than measurement noise.

## The finding that matters most

A model trained on this data reports **77.8% accuracy**.

That number is good enough to pass a project review, appear in a funding submission, and justify a deployment. It is also **completely hollow**: the model has learned to answer "low efficiency" every time, which is right 77.8% of the time simply because 77.8% of the records are low efficiency. It has learned nothing about machines. We proved this by scrambling the answers at random and retraining — the score did not change.

**This is the central risk in industrial AI procurement.** A headline accuracy figure cannot distinguish a working system from one that has memorised the most common answer. The gap is invisible unless someone deliberately tests for it, and that test takes minutes.

We recommend three checks be required in any technical evaluation of a machine-learning claim:

1. Report performance against the **"always guess the most common answer" baseline**. A model that does not beat it has not earned deployment.
2. Report **balanced accuracy**, which measures performance across all categories rather than the most frequent one.
3. Run a **scrambled-label test**. If performance does not drop, the model has learned nothing.

Applied to this dataset, all three checks fail. Applied at procurement, they cost almost nothing and prevent the funding of systems that do not work.

## What this does not mean

This is a finding about one data file, not about the facility, the equipment, or 6G-enabled manufacturing as a concept. The evidence indicates the sensor readings were **computer-generated rather than recorded from physical machines** — which is a legitimate and common way to build and test a data pipeline before real instrumentation is connected. That is exactly what has happened here, and the pipeline now exists and is tested.

The analytical system is finished and operational. What it needs is data that measures a factory.

## Recommended actions

**Immediate — data collection**

1. **Sample every machine on a fixed schedule.** The current system records one reading per minute for the entire plant, leaving individual machines unobserved for up to 8.5 hours. A bearing can fail inside that gap.
2. **Enforce timeline checks when data arrives.** A single automated test would have caught the duplicated day before it reached analysis, where it would otherwise inflate a full day of production figures.
3. **Verify that sensor relationships survive processing.** Power consumption must rise with production load; network latency and packet loss must move together. Where these physical relationships are absent, the data pipeline is destroying them.

**Immediate — efficiency measurement**

4. **Derive efficiency ratings from an independent source** — production records or quality sign-off — rather than calculating them from figures already in the same file. A rating computed from the data cannot then be predicted from it.
5. **Document how every target measure is produced,** including its owner and definition. Had this been recorded, a significant part of this investigation would have been unnecessary.

**Before further AI investment**

6. **Adopt a formal data acceptance gate.** The research paper specifies a six-check protocol covering timeline integrity, sampling adequacy, distribution realism, physical coupling, target independence and signal validation. Every issue identified in this study would have been caught by it. The checks are inexpensive; the cost of skipping them is a deployed system that produces confident, meaningless output.

## Assessment

| Capability | Status | Blocking issue |
|---|---|---|
| Operational dashboard and KPI reporting | **Delivered** | — |
| Data validation and quality monitoring | **Delivered** | — |
| Machine health baselines | **Delivered** | Uninformative on current data |
| Machine-to-machine benchmarking | **Blocked** | Differences smaller than noise |
| Predictive maintenance | **Blocked** | Sensors carry no early-warning signal |
| Efficiency forecasting | **Blocked** | Target is a formula, not a measurement |

## Bottom line

The analytical infrastructure requested by this project has been built, tested and delivered. Applied to the supplied data, it establishes with reproducible evidence that this telemetry cannot support predictive maintenance, efficiency forecasting or machine benchmarking — and it specifies precisely what must change.

That verdict is more valuable than the alternative. A model reporting 77.8% accuracy on this data would have been deployed, trusted, and would have failed silently against real equipment. Identifying the problem now costs a data collection redesign. Identifying it after deployment costs the credibility of the programme.

**The system is ready. The next step is instrumenting the factory properly, and gating the resulting data through the six-check protocol before the next round of investment is committed.**

---

*Full methodology, statistical evidence and reproducible code accompany this summary in the research paper and project repository.*
