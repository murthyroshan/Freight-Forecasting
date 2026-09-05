# ⚓ Freight Intelligence for SAIL

Forecasting when to fix a bulk charter, and which vessel to send into which
berth, for coking coal imported to the east coast of India.

Smart India Hackathon 2026 · Problem statement **SIH26006** · Ministry of Steel

[![Python 3.14](https://img.shields.io/badge/python-3.14-3776AB?logo=python&logoColor=white)](https://www.python.org)
[![Tests 569](https://img.shields.io/badge/tests-569%20passing-2ea44f)](#testing)

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
python -m src.build_panel     # 3,284 rows × 17 features
python -m src.train_model     # walk-forward evaluation
python -m src.forecast        # the forward call, 1 to 21 trading days
python -m src.licensed_model  # the licensed-years model for the replay
python -m src.procurement     # what the forecast was worth
python -m src.seasonal        # the month-by-month record
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
| assume no change | 0.2908 | — | — |
| momentum | 0.2777 | +4.5% | 61.5% |
| **ridge** | **0.2745** | **+5.6%** | **61.7%** |
| LightGBM | 0.2810 | +3.4% | 62.5% |

Beats the baseline in **5 of 8** folds. Conformal intervals reach **81.7%**
coverage against an 80% target — against **81.0%** for the same intervals without
volatility scaling, on the same folds. Direction is significant at
**p = 1.0e-05**, computed on the ~394 *independent* windows rather than the 1,971
overlapping rows.

> **Read the direction figure, not the skill figure.** Direction accuracy is
> robust: 61.7% overall, and 61.4–61.6% however you cut the sample. RMSE skill is
> not — twelve days in early 2020, when the index fell from 207 to about 1, carry
> most of it, and excluding them takes +5.6% down to **+2.0%**. Both numbers are
> real; only one of them is stable.

> **Nobody gets 70–80% on a five-day freight return.** A trivial momentum rule
> scores 78.1% on *next-day* direction and 57.8% at five days — the high figures
> quoted for freight are a one-day-horizon artefact of a smoothed broker survey.
> In the one BDI study using a return target, a stated benchmark and DM tests,
> sign accuracy ran 48–57%.

**Where the edge actually lives.** Sorted by how much the rate really moved:

| Actual 5-day move | Direction | Lift over the best constant guess |
| :--- | ---: | ---: |
| smallest quartile (0–6.5%) | 52.3% | −1.6 pp |
| 2nd quartile (6.5–13.3%) | 58.2% | +4.3 pp |
| 3rd quartile (13.3–23.5%) | 65.7% | +14.6 pp |
| **largest quartile (23.5%+)** | **70.4%** | **+17.8 pp** |

On the weeks that move, the model earns its keep. On the quiet ones it adds
nothing, and says so. The pattern holds in the recent period too, weaker:
**62.8%** on the largest quartile since 2022, against 50.9% on the smallest.

*Lift is direction accuracy minus the best constant guess on those same rows, so
it cannot be won by always saying "up".*

**And the model can tell you in advance which weeks those are.** The table above
sorts by how much the rate turned out to move — something you only learn
afterwards. This one sorts by the size of the prediction, which is known the
moment the model runs:

| Act only when… | Weeks | Direction | vs all weeks |
| :--- | ---: | ---: | ---: |
| always — a call every week | 1,721 | 61.8% | — |
| the call is in the strongest 75% | 1,320 | 65.4% | +3.6 pp |
| the call is in the strongest 50% | 894 | **69.6%** | +7.8 pp |
| **the call is in the strongest 25%** | 433 | **72.5%** | **+10.8 pp** |

A desk fixing a handful of cargoes a quarter does not need a call every week, and
this is the number it should be judged on. The threshold for "the strongest half"
is a quantile of *earlier* predictions only, so no week is ranked using anything
from after it — the first 250 scored weeks are ineligible for that reason, which
is why the *always* row reads 61.8% and not 61.7%. Each tier is significant
against the best constant call on its own weeks, counted in non-overlapping
windows and adjusted for testing three tiers. Tiering pure noise the same way
returns 49–53%, so the slicing is not what creates the lift; `tests/test_walkforward.py`
asserts that, and that truncating the history leaves every earlier decision
unchanged.

**What that is worth in freight, not in percentage points.** Direction accuracy
is a statistic; a steel producer buys freight. So: a cargo has to move, and the
desk either fixes today or holds one horizon and fixes then. Over **395
independent fixtures** (spaced a full horizon apart, so no market move is counted
twice):

| Policy | Saved, % of rate | Win rate | |
| :--- | ---: | ---: | :--- |
| always wait | **−2.88%** | 47.6% | no model at all |
| wait when momentum says fall | +1.52% | 57.1% | the naive rule |
| **wait when the model says fall** | **+1.91%** | 60.1% | this model |
| perfect foresight | +7.86% | 100% | the ceiling |

Every figure is a share **of the freight rate**. The Baltic series here is an
index in points, this repository has no sourced conversion to dollars, and
inventing one would put a fabricated number at the centre of the result — so
supply your own rate and multiply.

The control is the number that matters. Waiting every time *loses* 2.88%, because
the index rose across this period, so the policy had to earn its result against a
market that punished waiting. It beats fixing today (p = 0.011) and beats always
waiting (p = 0.00001), and captures 24% of what perfect foresight would have
taken. **It is not distinguishable from the momentum rule on money** (p = 0.32);
the model earns its place on RMSE and direction, and on this decision the two are
a tie. It also loses on 40% of the weeks it holds, worst single decision
−110.1%, and holding burns laycan, which is not priced here.

**Where it stops working.** One number over eight years invites exactly one
question, so here is the answer split by calendar year — the one boundary nobody
can accuse us of choosing:

| Year | Direction | Base rate | RMSE skill | Volatility of y | Strongest 50% |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 2019 | 69.5% | 51.0% | +10.5% | 30.2% | 81.5% |
| 2020 | 70.7% | 57.1% | +11.6% | 61.3% | 85.3% |
| 2021 | 66.8% | 58.3% | +7.4% | 19.8% | 72.7% |
| **2022** | **51.0%** | 54.4% | **−7.8%** | 32.4% | 54.9% |
| 2023 | 62.8% | 52.1% | +6.4% | 28.9% | 70.9% |
| 2024 | 61.9% | 52.7% | +1.7% | 18.6% | 74.8% |
| 2025 | 57.4% | 50.6% | +3.8% | 19.4% | 69.9% |
| **2026** | **56.0%** | 62.0% | **−18.9%** | 12.1% | 55.7% |

RMSE skill is negative in **2 of 9 years**, and direction fails to beat that
year's own base rate in the same two — a bad year here is bad on every measure at
once. The confidence rule does not rescue them either (2022 at 54.9%, 2026 at
55.7%): in a bad year the model is confidently wrong. The aggregate is carried by
2019–2021.

The pattern is not random. Volatility of the target fell from 34.7% across
2018–2021 to 12.1% in 2026, and RMSE skill is a variance-explained measure — in a
calm market there is little variance to explain and a handful of large misses
dominate what is left. Direction is scale-free and degrades far more gently.
That is the fragility this project documented *before* it measured it, which is
why it asks to be judged on direction.

**Part-load capacity.** A Capesize loads **152,320 t of a possible 176,500 t**
for Paradip — 86% utilisation, 24,180 t left ashore every voyage. Given only the
berth's draft and no deadweight figure, the model implies 155,820 DWT against a
documented limit of ~155,000.

**Live port activity.** 11 ports from IMF PortWatch, current to **2026-08-28** —
one day behind the freight forecast, which now also reaches the present.

**Fleet selection.** Voyage counts are whole numbers, so the cheapest mix is an
integer problem, not a division: `src/optimise.py` solves it by branch-and-bound
and `test_optimise.py` checks the answer against every enumerated fleet on
problems small enough to price by hand.

**The empty leg.** A ship with no return cargo sails in ballast, and the owner
prices that into the rate before you negotiate. Haldia ships out **2.5 kt/day**
against **44.8** in — a ratio of 0.06 that has never exceeded 0.2 in eight years,
so **94%** of arriving tonnage leaves empty. Paradip runs **1.46** and has been
above parity every year. Priced into the optimiser at 45% of a laden voyage,
Haldia gets roughly **30% dearer** and Paradip does not move at all. On the
dashboard's own default costs that is enough to make Paradip the cheapest berth
despite its 16.0 m draft forcing a Capesize to sail part-laden — the part-load
penalty is smaller than the empty-leg penalty. The exact figure moves with the
costs you enter, which is the point: the imbalance is measured, the money is
yours.

> **The costs are yours, not ours.** This repository has no verified freight,
> lighterage or haulage figures for this lane. They are required arguments with
> no defaults — the optimiser is exact over what you give it, and refuses to run
> rather than assume a rate. The same rule bounds deliverable (c): the *share* of
> ships leaving empty is measured, what an empty leg *costs* is not, and **idle
> time is not modelled at all** because PortWatch publishes arrivals but not
> departures. A berth with no arrivals feed is priced as `null`, never as zero —
> otherwise the optimiser would prefer exactly the berths nobody has data on.

## 🧭 How it is built

![Pipeline: Mendeley Baltic indices and Yahoo market data feed build_panel.py then train_model.py; IMF PortWatch feeds congestion.py and ballast.py; ports.py derives part-load capacity from physics; Open-Meteo history falsifies the seasonal story while its live forecast drives risk.py; ports, congestion and ballast feed the optimise.py fleet selection, and all of it reaches the Flask dashboard.](assets/pipeline.svg)

| Path | Purpose |
| :--- | :--- |
| `app.py` | Flask API. Serves artefacts, computes nothing. |
| `templates/dashboard.html` | Five-view dashboard; every number arrives via `fetch()`. |
| `src/fetch_data.py` | Baltic + Yahoo + PortWatch + Open-Meteo → `data/raw` |
| `src/build_panel.py` | Features and the target definition. |
| `src/train_model.py` | **Module A** — Capesize direction, walk-forward. |
| `src/procurement.py` | **Module A, part two** — what the forecast is worth, as a share of the freight rate. |
| `src/live_model.py` | The BDRY control experiment. |
| `src/ports.py` | **Module B** — port constraints and part-load capacity. |
| `src/congestion.py` | **Module B** — live port activity. |
| `src/risk.py` | **Module B** — risk warnings, measured and otherwise. |
| `src/optimise.py` | **Module B** — which ships into which berths (MILP). |
| `src/ballast.py` | **Module B** — the empty return leg. |
| `tests/` | 569 checks. See [Testing](#testing). |

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
python -m tests.test_leakage       #  31 · features, the splice, and the target
python -m tests.test_walkforward   #  76 · the purge gap and the headline claims
python -m tests.test_ports         #  54 · part-load physics
python -m tests.test_congestion    #  59 · activity signal and its guards
python -m tests.test_risk          #  86 · thresholds, and evidence claims
python -m tests.test_optimise      #  62 · selection, against brute force
python -m tests.test_ballast       #  43 · the empty leg, and its limits
python -m tests.test_app           # 158 · every API claim, recomputed
```

`test_walkforward.py` exists because a mutation test found that inverting the
purge gap — so the training window overlapped the test window — left every other
suite green. It asserts the fold boundaries directly, and checks coverage and
skill against their *targets* rather than only against what `metrics.json` says.

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
| East Money *(public mirror, no stated licence)* | extends the Baltic series past our licensed copy | 1,750 rows, 2019-08-01 → current |
| IMF PortWatch | daily port calls and dry bulk tonnage, 11 ports | 2,797 rows each, → 2026-08-28 |
| Yahoo Finance | BDRY, Brent, copper, DXY, S&P 500, USD/INR, owners, miners | 2012 → current |
| Open-Meteo | rainfall, wind, gusts at all five discharge ports | 5,358 rows each, plus a live 10-day forecast |

Baltic index history is redistributed under **CC BY 4.0**. Baltic Exchange
indices themselves are proprietary and are **not** redistributed here.
The post-2019 extension is fetched at runtime from a public mirror and never
committed; production would need a Baltic Exchange licence.

Full methodology, the port physics and both falsification tests:
**[`docs/METHOD.md`](docs/METHOD.md)**
