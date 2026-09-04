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
| assume no change | 0.2908 | 0.1858 | — | — |
| momentum (1 feature, OLS) | 0.2777 | 0.1771 | +4.5% | 61.5% |
| **ridge (17 features)** | **0.2745** | 0.1786 | **+5.6%** | **61.7%** |
| LightGBM | 0.2810 | 0.1757 | +3.4% | 62.5% |

Per fold, ridge beats the no-change baseline in **5 of 8**. The three losses are
2021-H2, 2023-H2 and the most recent year, and all are shown in the dashboard
rather than hidden. Its largest edge is fold 1 — the COVID collapse, where the
no-change RMSE blows out to 0.555 and ridge cuts it to 0.494.

### The skill figure is fragile; the direction figure is not

This matters more than the headline. Twelve rows out of 1,971 — all between
January and May 2020, when the index fell from 207 to about 1 — carry most of
the measured RMSE skill:

| Sample | Skill | Direction |
| :--- | ---: | ---: |
| everything | +5.60% | 61.7% |
| excluding \|y\| > 2 (2 rows) | +4.38% | 61.7% |
| excluding \|y\| > 1 (12 rows) | **+2.03%** | 61.5% |
| excluding Jan–Jun 2020 entirely | +2.87% | 61.4% |
| only days with the index above 300 | +4.84% | 61.6% |

RMSE skill moves from +5.60% to +2.03%; **direction accuracy barely moves at
all.** Quote the direction figure. The skill figure is reported here because it
was measured, not because it is the one to lean on.

### Where the edge lives

Sorted by how much the rate actually moved over the five days:

| Actual move | n | Direction | Lift over the best constant guess |
| :--- | ---: | ---: | ---: |
| smallest quartile (0–6.5%) | 493 | 52.3% | −1.6 pp |
| 2nd quartile (6.5–13.3%) | 493 | 58.2% | +4.3 pp |
| 3rd quartile (13.3–23.5%) | 493 | 65.7% | +14.6 pp |
| **largest quartile (23.5%+)** | 493 | **70.4%** | **+17.8 pp** |

Lift is direction accuracy minus the best constant guess on those same rows, so
it cannot be won by always predicting "up".

The same shape holds on the recent period alone, at a lower level — since 2022
the largest quartile scores **62.8%** and the smallest **50.9%**. The model adds
nothing in flat weeks and a great deal in moving ones. A chartering desk only
acts when the move is large enough to matter, which is exactly where the edge
sits, and it is why the aggregate looks weak in a calm market: most weeks are
small-move weeks, and nobody can call those.

### Knowing in advance which weeks to trust

The table above has a flaw as a decision aid: it sorts on the realised move,
which nobody knows at the time. It says the edge is concentrated, but not how to
find it prospectively.

The model's own conviction does that. Sorting instead by |prediction| — available
the moment the model runs, needing no outcome:

| Act only when… | Weeks | Independent windows | Direction | vs all weeks | p |
| :--- | ---: | ---: | ---: | ---: | ---: |
| always | 1,721 | 344 | 61.8% | — | — |
| strongest 75% | 1,320 | 264 | 65.4% | +3.6 pp | <0.0001 |
| strongest 50% | 894 | 178 | **69.6%** | +7.8 pp | <0.0001 |
| **strongest 25%** | 433 | 86 | **72.5%** | **+10.8 pp** | 0.0001 |

Three things make this an honest table rather than a flattering one.

**The threshold is causal.** "The strongest half" is a quantile of predictions
from *strictly before* the week being judged, so appending later data can never
change a decision already taken. A quantile over the whole test period — the
obvious implementation — would need next year's predictions to rank this one, and
the whole table would be a look-ahead artefact. `tier_mask()` is factored out for
exactly the reason `folds()` is: so a test can assert the property directly.
`tests/test_walkforward.py` checks that scoring a truncated history returns
identical decisions for the rows the two runs share, and that multiplying the
last third of the predictions by fifty does not un-call a single earlier week.

**The baseline is the same rows.** The first 250 scored weeks cannot be ranked
— there is no history yet to place them against — so they are excluded from every
row of the table, the *always* row included. That is why it reads 61.8% rather
than the 61.7% headline. Comparing a tier against all 1,971 rows would credit the
tiering with the burn-in it merely dropped.

**The null is not a coin.** Each p-value tests that tier against the best
constant call on its own weeks, on non-overlapping windows rather than rows, and
is Bonferroni-adjusted for the three tiers. A tier that quietly selected
mostly-rising weeks would have to beat "always up" on them.

The remaining question is whether slicing by |prediction| would lift *any*
model, in which case this measures the slicing and not the model. It does not:
run through the same tiering, random predictions on random outcomes score
49–53%, and the suite asserts it.

What this changes in practice is the product. The honest claim is not "the model
is right 62% of the time"; it is "the model is right 62% of the time if you make
it answer every week, and 70% on the half of weeks it has something to say
about — and it tells you which half in advance." Charterers do not fix cargoes
weekly. They fix a few a quarter, and can wait for the weeks the model is
confident about.

### Why ridge beats the gradient-boosted model

Daily freight autocorrelation is ~0.99 and the five-day targets overlap, so
3,284 rows is really about **656 effective windows**, and the 1,971 rows the
model is actually scored on are about **394** — which is the figure the
significance test uses, and the one `models/metrics.json` records as
`n_effective`. A normally sized GBM
memorises that instantly. The feature count is capped at 17 for the same reason.

The significance test uses that smaller number too: 61.7% direction on ~394
independent windows gives p = 1.0e-05 — tested against the realised
up-rate of 51.0% rather than a coin, and Bonferroni-adjusted over the three
candidate models, since the model tested is the one chosen by this same score.
Computed on the 1,971 overlapping rows it would be inflated roughly fivefold.

### Intervals

Split-conformal residual quantiles, in the **locally weighted** variant — each
residual scaled by a volatility estimate known at *t*, then rescaled on the test
side. Plain split conformal gives **81.0%** coverage for a nominal 80%, because
freight volatility clusters and one global quantile is too narrow in stressed
regimes. The weighted form reaches **81.7%**, at the cost of a mean interval
20% wider (0.568 to 0.682). Both figures are produced by
`walk_forward(..., conformal='plain'|'local')` on the same folds with the same
finite-sample-corrected quantile, and both are written to `models/metrics.json`
— neither is a number typed into this file.

---

## 2 · Falsification test one — the traded proxy

Our *licensed* Baltic history ends 2019-07-31, and the Baltic Exchange charges
for current assessments. So: could the free, exchange-traded BDRY stand in?

(The series is now extended past 2019 from a public mirror of the same Baltic
indices — see §6 — but that mirror carries no licence, so the question below is
still the right one to have asked.)

| Series | What it is | Lag-1 autocorrelation | Our skill |
| :--- | :--- | ---: | ---: |
| Baltic Capesize | daily broker **survey** | **+0.624** | **+5.6%** |
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

## 6 · Extending the Baltic series past 2019

Our licensed Mendeley copy ends **2019-07-31**. Everything else in the system —
port calls, weather, market data — runs to the present, so the *label* was the
only thing stopping the model from forecasting today.

**The source.** East Money, a public financial portal, mirrors the Baltic
indices through a keyless JSON API. It was validated before use, not after:

| Series | Overlapping days | Byte-identical | Largest disagreement |
| :--- | ---: | ---: | ---: |
| Capesize | 1,749 | **99.5%** | 121 points |
| Panamax | 1,735 | **99.2%** | 46 points |
| Supramax | 1,731 | 42.9% | 123 points |

Supramax agrees on far fewer days, but the level correlation is **0.9991** and
the mean gap is 0.5 points on a series around 900 — early-year revision noise,
not a different index. It converges to 99% by 2019.

`fetch_data.py` therefore validates on **value, not on exact matching**: each
series must correlate at least 0.99 with the licensed copy and disagree by no
more than 5% at the 99th percentile, *and* reproduce the licensed value exactly
on the join day. Gating on the byte-identical rate instead would have left
Supramax three points above its floor — and since one failing series aborts the
whole fetch, a bad year of revisions would have silently killed the Capesize
extension, which is the series the model actually needs.

**The splice only appends.** No licensed value is ever overwritten. The largest
move within a week of the join is 5.2% against a 3.2% typical daily move, so no
seam was introduced. `tests/test_leakage.py` §7 asserts all of this.

**The index goes negative.** The Capesize basis is a timecharter equivalent, and
a TCE can fall below zero when the market collapses — it did for 44 sessions
between 2020-01-31 and 2020-05-14, bottoming at −372. A log return is undefined
there, so those days are dropped. It costs 12 rows of 3,296, and every
alternative target scored worse on the extended series:

| Target | Skill |
| :--- | ---: |
| **log(P₊₅/P) — kept** | **+5.5%** |
| (P₊₅−P)/P simple | −2.3% |
| log with a +1000 offset | −2.5% |
| (P₊₅−P)/vol₂₁ standardised | −21.9% |

Dropping rows from the middle leaves gaps in the panel, and the walk-forward
purges by position. A gap therefore makes the purge span *more* source days,
never fewer — measured at 5 to 75, so it is conservative at the gaps rather
than narrower. Asserted in `tests/test_leakage.py` §9.

**Licence.** The Baltic Exchange indices are proprietary and East Money states
no licence for its mirror. This repository fetches at runtime and redistributes
nothing — `data/` is gitignored and no index value is committed. A production
deployment would need a Baltic Exchange licence; the licence restricts
redistribution, not a licensee's own internal use.

---

## 7 · Why two modules and not one

PortWatch begins 2019-01-01. The Baltic series ends 2019-07-31. **211 days of
overlap** — nowhere near enough to learn congestion effects on rates.

So the rate model is statistical and the port model is deterministic. That split
is forced by the data, not chosen for convenience.
