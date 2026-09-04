# ⚓ Freight Intelligence for SAIL

Forecasting when to fix a bulk charter, and which vessel to send into which
berth, for coking coal imported to the east coast of India.

Smart India Hackathon 2026 · Problem statement **SIH26006** · Ministry of Steel

[![Python 3.14](https://img.shields.io/badge/python-3.14-3776AB?logo=python&logoColor=white)](https://www.python.org)
[![Tests 448](https://img.shields.io/badge/tests-448%20passing-2ea44f)](#testing)

---

SAIL imports roughly three-quarters of its coking coal by sea. Two decisions
drive the landed cost, and this repository addresses both.

**🕐 When to fix the charter.** The Baltic Capesize index ranges 92 to 4,438 in
our sample; a week's timing moves lakhs per parcel.

**🚢 Which ship, into which berth.** East coast draft limits bind. Paradip's coal
berth is 16.0 m and a laden Capesize draws 18.2 m — so it sails part-laden, and
how much cargo it can carry is a calculation, not a yes/no.

A third question sits under both — what could stop the ship this week. The
dashboard answers all three, in five views:

| View | Answers |
| :--- | :--- |
| **Charter timing** | Fix now or wait? Shows the forecast *and* what actually happened. |
| **Vessel & port** | Which class, into which berth, how much it lifts — and the cheapest fleet for a parcel. |
| **Port activity** | How busy every berth is this week, both ends of the lane. |
| **Risk** | What could go wrong in the next ten days — and which warnings have a measured effect behind them. |
| **Validation** | Every fold, including the ones we lose. |

## ⚙️ Quick start

```bash
git clone https://github.com/murthyroshan/Freight-Forecasting.git
cd Freight-Forecasting
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

No API keys. Every data source is free and keyless.

```bash
python -m src.fetch_data      # 29 parquet files  (~3 min first run)
python -m src.build_panel     # 1,682 rows × 17 features
python -m src.train_model     # walk-forward evaluation
python -m src.live_model      # the control experiment
python -m src.optimise        # vessel and berth selection
python -m src.ballast         # the empty return leg
python app.py                 # → http://127.0.0.1:5000
```

## 📊 Results

Capesize direction, five business days ahead. Expanding-window walk-forward,
eight folds, five-day purge gap between train and test.

| Model | RMSE | Skill vs no-change | Direction |
| :--- | ---: | ---: | ---: |
| assume no change | 0.2469 | — | — |
| momentum | 0.2342 | +5.1% | 62.8% |
| **ridge** | **0.2285** | **+7.5%** | **64.9%** |
| LightGBM | 0.2330 | +5.7% | 60.3% |

Beats the baseline in **6 of 8** folds. Conformal intervals reach **80.6%**
coverage against an 80% target. Direction is significant at **p < 0.0001**,
computed on the ~202 *independent* windows rather than the 1,010 overlapping
rows.

> **+7.5% is small, and it is the honest number.** Freight is close to a random
> walk. Any claim of 80–90% "accuracy" on a return series is measuring the wrong
> thing or leaking the target.

**Part-load capacity.** A Capesize loads **152,320 t of a possible 176,500 t**
for Paradip — 86% utilisation, 24,180 t left ashore every voyage. Given only the
berth's draft and no deadweight figure, the model implies 155,820 DWT against a
documented limit of ~155,000.

**Live port activity.** 11 ports from IMF PortWatch, current to **2026-08-28** —
the only current data in the system, since the Baltic history ends 2019-07-31.

**Fleet selection.** Voyage counts are whole numbers, so the cheapest mix is an
integer problem, not a division: `src/optimise.py` solves it by branch-and-bound
and `test_optimise.py` checks the answer against every enumerated fleet on
problems small enough to price by hand.

**The empty leg.** A ship with no return cargo sails in ballast, and the owner
prices that into the rate before you negotiate. Haldia ships out **2.5 kt/day**
against **44.8** in — a ratio of 0.06 that has never exceeded 0.2 in eight years,
so **94%** of arriving tonnage leaves empty. Paradip runs **1.46** and has been
above parity every year. Priced into the optimiser at 45% of a laden voyage,
Haldia gets **30% dearer** and Paradip does not move — which is why Paradip wins
even though its 16.0 m berth forces a Capesize to sail part-laden.

> **The costs are yours, not ours.** This repository has no verified freight,
> lighterage or haulage figures for this lane. They are required arguments with
> no defaults — the optimiser is exact over what you give it, and refuses to run
> rather than assume a rate. The same rule bounds deliverable (c): the *share* of
> ships leaving empty is measured, what an empty leg *costs* is not, and **idle
> time is not modelled at all** because PortWatch publishes arrivals but not
> departures. A berth with no arrivals feed is priced as `null`, never as zero —
> otherwise the optimiser would prefer exactly the berths nobody has data on.

## 🧭 How it is built

![Pipeline: Mendeley Baltic indices and Yahoo market data feed build_panel.py then train_model.py; IMF PortWatch feeds congestion.py; ports.py derives part-load capacity from physics; all three reach the Flask dashboard. A MILP selection stage is not yet built.](assets/pipeline.svg)

| Path | Purpose |
| :--- | :--- |
| `app.py` | Flask API. Serves artefacts, computes nothing. |
| `templates/dashboard.html` | Four-view dashboard; every number arrives via `fetch()`. |
| `src/fetch_data.py` | Baltic + Yahoo + PortWatch + Open-Meteo → `data/raw` |
| `src/build_panel.py` | Features and the target definition. |
| `src/train_model.py` | **Module A** — Capesize direction, walk-forward. |
| `src/live_model.py` | The BDRY control experiment. |
| `src/ports.py` | **Module B** — port constraints and part-load capacity. |
| `src/congestion.py` | **Module B** — live port activity. |
| `src/risk.py` | **Module B** — risk warnings, measured and otherwise. |
| `src/optimise.py` | **Module B** — which ships into which berths (MILP). |
| `src/ballast.py` | **Module B** — the empty return leg. |
| `tests/` | 448 checks. See [Testing](#testing). |

## ⚖️ The rule this repository runs on

> **No number ships unless a script here printed it.**

Every figure in the dashboard and in this file is read from a generated
artefact. Nothing is typed into the markup, and a missing artefact returns
`503` with instructions rather than a plausible substitute.

Two things follow from taking that seriously:

**We tried to break our own model.** The same method applied to BDRY — a traded
freight ETF over the same underlying — scores **−11.2%**, worse than assuming no
change. That is the correct answer for an arbitraged instrument, and it is why
the Baltic result is credible rather than suspicious.

**We disproved our own explanation.** The September–December dip in port arrivals
looked like cyclone season. Tested against Open-Meteo wind, it is not — those are
the *calmest* months of the year. Both experiments are written up in
[`docs/METHOD.md`](docs/METHOD.md).

<a id="testing"></a>

## 🧪 Testing

```bash
python -m tests.test_leakage     #   8 · features cannot see the future
python -m tests.test_ports       #  54 · part-load physics
python -m tests.test_congestion  #  50 · activity signal and its guards
python -m tests.test_risk        #  80 · thresholds, and evidence claims
python -m tests.test_optimise    #  56 · selection, against brute force
python -m tests.test_ballast     #  43 · the empty leg, and its limits
python -m tests.test_app         # 157 · every API claim, recomputed
```

`test_leakage.py` corrupts every raw input after a cut date, rebuilds the panel
and requires all earlier features to be bit-identical — catching a centred
window or a negative shift regardless of what it is called. `test_app.py` does
not trust the API: it recomputes each served claim from the stored artefacts.
`test_risk.py` injects a 118 km/h forecast to exercise the cyclone path without
waiting for a cyclone, and fails if any warning without a measured effect size
behind it is ever flagged as measured.

## 📋 Status

| # | Deliverable | State |
| :-- | :--- | :--- |
| a | Market entry timing | **Delivered** |
| b | Vessel-type optimisation under port constraints | **Delivered** |
| c | Idle / deadhead management | Deadhead **delivered**; idle not measurable |
| d | Risk warnings | **Delivered** |

## 🗂️ Data and licence

| Source | Contents | Coverage |
| :--- | :--- | :--- |
| Mendeley `10.17632/t76ckh2ygg` *(CC BY 4.0)* | Baltic Capesize, Panamax, Supramax, Handysize | 1,749 rows, 2012-08-01 → 2019-07-31 |
| IMF PortWatch | daily port calls and dry bulk tonnage, 11 ports | 2,797 rows each, → 2026-08-28 |
| Yahoo Finance | BDRY, Brent, copper, DXY, S&P 500, USD/INR, owners, miners | 2012 → current |
| Open-Meteo | rainfall, wind, gusts at all five discharge ports | 5,358 rows each, plus a live 10-day forecast |

Baltic index history is redistributed under **CC BY 4.0**. Baltic Exchange
indices themselves are proprietary and are **not** redistributed here.

Full methodology, the port physics and both falsification tests:
**[`docs/METHOD.md`](docs/METHOD.md)**
