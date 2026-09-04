# Method

Everything the [README](../README.md) states in one line, argued in full.
Four parts: how the forecast is built and scored, the two experiments we ran
against our own conclusions, and the physics behind the port model.

---

## 1 · The forecast

### The target is a return, not a level

Capesize ranges **92 to 4,438** across the sample, a 53% coefficient of
variation. A model predicting the *level* scores impressively by answering
"about the same as yesterday" and is useless for a timing decision.

The target is therefore the **five-business-day forward log return**. That also
makes leakage structural rather than a thing anyone has to remember: the target
lies strictly in the future, so any feature built from data up to and including
*t* is legitimate by construction.

Log rather than simple return: Capesize bottomed at **92 on 2019-04-02**, after
the Vale Brumadinho dam collapse wiped out Brazilian iron ore exports. Those
prints are real, so they stay — but a simple return off a denominator of 92
gives +311%, skew +3.1 and kurtosis +25.6, and RMSE would then be decided by
about four days in 2019. Logs give skew +0.5 and kurtosis +3.2 over the
identical rows.

### Capesize specifically

SAIL lifts 150–180k t coking coal parcels from Hay Point, Gladstone and the US
east coast. That is Capesize work. It is also the only index whose returns track
BDRY closely enough for the live-proxy question to be worth asking at all —
weekly r = +0.671, against +0.383 for Panamax and +0.288 for Supramax.

A caution that generalises: **Supramax has the *highest* level correlation with
BDRY (+0.848) and the *lowest* return correlation (+0.101).** That is textbook
spurious correlation between two trending series, sitting in our own data.

### How it is scored

Expanding-window walk-forward, eight folds, with a **five-day purge gap** between
train and test. Without the gap the last training rows overlap the first test
row's target window and the model is scored on data it partly saw.

Baselines first, because "accuracy" on a return series is meaningless — you can
score 97% on a series that never moves much. What matters is beating what a desk
already has for free: assuming no change, and the single strongest feature
fitted by OLS.

| Model | RMSE | MAE | Skill vs no-change | Direction |
| :--- | ---: | ---: | ---: | ---: |
| assume no change | 0.2469 | 0.1807 | — | — |
| momentum (1 feature, OLS) | 0.2342 | 0.1696 | +5.1% | 62.8% |
| **ridge (17 features)** | **0.2285** | 0.1716 | **+7.5%** | **64.9%** |
| LightGBM | 0.2330 | 0.1708 | +5.7% | 60.3% |

Per fold, ridge beats the no-change baseline in **6 of 8**. The two losses are
2017-H1 and 2018-H2 and are shown in the dashboard rather than hidden. Its
largest edge is fold 7 — the Vale window, where the no-change RMSE blows out to
0.396 and ridge cuts it to 0.357. It helps most where it matters most.

### Why ridge beats the gradient-boosted model

Daily freight autocorrelation is ~0.99 and the five-day targets overlap, so
1,682 rows is really about **336 effective windows**, and the 1,010 rows the
model is actually scored on are about **202** — which is the figure the
significance test uses, and the one `models/metrics.json` records as
`n_effective`. A normally sized GBM
memorises that instantly. The feature count is capped at 17 for the same reason.

The significance test uses that smaller number too: 64.9% direction on ~202
independent windows gives p = 0.0003 — tested against the realised
up-rate of 51.7% rather than a coin, and Bonferroni-adjusted over the three
candidate models, since the model tested is the one chosen by this same score. Computed on the 1,010 overlapping rows it
would be inflated roughly fivefold.

### Intervals

Split-conformal residual quantiles, in the **locally weighted** variant — each
residual scaled by a volatility estimate known at *t*, then rescaled on the test
side. Plain split conformal gives **78.4%** coverage for a nominal 80%, because
freight volatility clusters and one global quantile is too narrow in stressed
regimes. The weighted form reaches **82.0%**, at the cost of a mean interval
36% wider (0.506 to 0.689). Both figures are produced by
`walk_forward(..., conformal='plain'|'local')` on the same folds with the same
finite-sample-corrected quantile, and both are written to `models/metrics.json`
— neither is a number typed into this file.

---

## 2 · Falsification test one — the traded proxy

Our Baltic history ends 2019-07-31; current assessments are a licensed feed we
may not redistribute. So: could the free, exchange-traded BDRY stand in?

| Series | What it is | Lag-1 autocorrelation | Our skill |
| :--- | :--- | ---: | ---: |
| Baltic Capesize | daily broker **survey** | **+0.624** | **+7.5%** |
| BDRY | liquid, arbitraged **ETF** | **+0.047** | **−11.2%** |

**It cannot, and that is the correct result.** The Baltic index is a survey whose
panellists anchor on the previous print, so it is autocorrelated and genuinely
forecastable. BDRY is arbitraged — if its returns were predictable at +0.62,
someone would have traded that away.

Scoring well on *both* would have meant we were fooling ourselves. This is the
experiment that would have caught us, and it is reported because we ran it.

**Consequence.** Production runs on SAIL's own Baltic licence (≈£2,000/yr plus
£595 setup — a rounding error at SAIL's scale). The licence restricts *our*
redistribution, not SAIL's internal use.

---

## 3 · Falsification test two — the seasonal dip

Arrivals at all five Indian discharge ports fall between September and December
— tested port by port, not on the average, so it is not an artefact of how the
mean is taken.

Bay of Bengal cyclone season overlaps that window, so we checked it against
Open-Meteo wind for the same ports and dates. **Two things are true, and they
point in opposite directions.**

### Cyclones absolutely do stop ships

On the day of and the day before a gust above 90 km/h:

| Port | Storm window | Normal | Change | p |
| :--- | ---: | ---: | ---: | ---: |
| Paradip | 0.36 /day | 2.91 | **−88%** | <0.0001 |
| Dhamra | 0.00 /day | 1.00 | **−100%** | <0.0001 |
| Haldia | 1.20 /day | 1.91 | −37% | 0.046 |

The data captures real storms at the right dates — Cyclone Fani (3 May 2019)
registers **144 km/h**, the highest reading in the series; Amphan 125, Yaas 101,
Dana 89. An event study shows arrivals collapsing at lag −1 and 0, then
rebounding *above* baseline the next day: ports clear vessels before the storm
and catch up after.

### But they cannot explain the season

Paradip has had **7 such days in 8 years — 5 in May, 2 in November.** September
to December contains 2 of them, affecting roughly 4 days out of 854 in that
window: **0.4%**, against an observed shortfall of about **15%** of arrivals.
Wrong by a factor of thirty.

Sep–Dec is in fact the *calmest* stretch of the year (mean gust 34 vs 42), and
monthly arrivals correlate **positively** with monthly wind at Paradip (+0.774)
and Dhamra (+0.794).

**Weather is a real short-horizon disruption signal and a poor seasonal one.**
Whatever drives the four-month dip is more likely a demand or stocking cycle,
which PortWatch cannot see.

Both halves are locked behind tests, so neither can be dropped in favour of the
tidier one.

---

## 4 · The port model

### Part-load capacity

The useful question is not *"does this ship fit"* but *"how much cargo can it
carry into this port"*.

Over the operating range a hull immerses almost linearly with weight. The
constant is **TPC**, tonnes per centimetre, estimated from waterplane area
(`Lwl × B × Cw`, with Cw = 0.85 for a full-form bulk carrier). This is an
estimate, not a hydrostatic table — but the values it produces sit inside the
published ranges for each class.

| Class | DWT | Laden draft | TPC | Cargo capacity |
| :--- | ---: | ---: | ---: | ---: |
| Panamax | 82,000 | 14.4 m | 62.5 t/cm | 79,500 t |
| Capesize | 180,000 | 18.2 m | 109.9 t/cm | 176,500 t |
| Newcastlemax | 208,000 | 18.5 m | 126.8 t/cm | 204,200 t |

**Maximum cargo by class and berth, tonnes:**

| Class | Gangavaram | Vizag | Dhamra | Paradip | Gopalpur | Haldia |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Panamax | 79,500 | 79,500 | 79,500 | 79,500 | 79,500 | 41,739 |
| Capesize | 176,500 | 175,400 | 174,301 | **152,320** | 135,834 | 68,435 |
| Newcastlemax | 204,200 | 199,129 | 197,861 | **cannot** | 153,493 | 75,814 |

Newcastlemax is rejected from Paradip outright: 50.0 m beam against a 46.0 m
limit, and no amount of part-loading makes a ship narrower.

**Validation.** The model is given Paradip's *draft* and nothing about
deadweight, yet its Capesize answer implies **155,820 DWT** against an
independently documented berth limit of ~155,000 — within 0.6%.

### Ranking on the right quantity

Options are ranked by how much of the **chartered** capacity a parcel fills, not
by how full each ship could be. The latter scores a Capesize and a Newcastlemax
equally — both load to 100% of their own capacity — while a 160,000 t parcel
leaves 44,200 t of the Newcastlemax empty. Freight is paid on the ship, so the
larger vessel is the worse fixture even though it fits.

### Water density

A ship floats deeper in less dense water. Haldia sits in the brackish Hooghly,
which costs a Capesize a further **0.28 m** of draft via the standard Dock Water
Allowance.

### What is reported as an upper bound

Where a berth's LOA, beam or channel limits are not verified, the tonnage is
labelled explicitly. A Capesize at Haldia computes to 68,435 t on draft alone,
but a 289 m ship cannot transit the Hooghly — so it is reported as
`uneconomic` at 39% utilisation with a caveat, never as a berthing plan. Only
Paradip and Sagar-Sandheads have verified geometry.

**Sagar-Sandheads is an anchorage in 40–50 m of water, not a berth.** Capesizes
anchor and floating cranes lighter into barges for Haldia. Modelling it as a
deep-draft port would invert the answer for the entire Haldia trade.

---

## 5 · Port activity

PortWatch publishes AIS-derived **arrivals and tonnage** — not waiting time,
queue length or berth occupancy. Nothing here is called congestion. A high
reading means the berth is under load, not that a ship will wait N days.

**Each port is scored against itself.** New Orleans averages 6.6 dry-bulk calls a
day, Gopalpur 0.31; a shared threshold would mark one permanently dead and the
other permanently swamped.

**Bands come from the percentile, not the ratio.** Measured here, 1.32× the
median sits at the 80th percentile at Paradip, 82nd at Haldia, 85th at Dhamra
and 86th at Visakhapatnam — a six-point spread, so one ratio threshold would
call the same load "busy" at one berth and "very busy" at another.

**Low-volume ports are refused a reading.** Gopalpur's baseline is 0.29 calls a
day, where one ship arriving or not swings a ratio by hundreds of percent. It is
flagged `unreliable` with the reason stated.

**Two tonnage figures per port.** Every Indian discharge port also loads: Paradip
averages **59 kt/day inbound against 87 kt/day outbound** over the full record —
60% leaves as
iron ore, competing for the same berths. Against Paradip's call counts, imports
alone correlate r = +0.620 and exports alone +0.751, but together **+0.898**. At
the load end the distinction is starker still: Hay Point and Newcastle record
*zero* dry-bulk imports.

---

## 6 · Why two modules and not one

PortWatch begins 2019-01-01. The Baltic series ends 2019-07-31. **211 days of
overlap** — nowhere near enough to learn congestion effects on rates.

So the rate model is statistical and the port model is deterministic. That split
is forced by the data, not chosen for convenience.
