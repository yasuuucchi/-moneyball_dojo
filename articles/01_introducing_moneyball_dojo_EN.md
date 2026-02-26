# A Japanese AI Engineer Built 9 Models to Beat MLB Sportsbooks — Here's the Data

## Introducing Moneyball Dojo: 9 machine learning models, every game, every day, full transparency

I have a confession: I watch more baseball than any reasonable person should.

Growing up in Tokyo, I was the kid who'd stay up until 3 AM watching MLB games on delayed broadcast — then get up for school and argue about NPB stats with friends who thought I was insane for caring about both leagues. In college, I studied machine learning. The two obsessions were bound to collide.

Last year, I stopped talking about it and started building. The result is Moneyball Dojo — an AI prediction system that runs 9 different models across every MLB game, every single day. Not a hobby project with gut feelings dressed up as data. An actual engineered system with logged predictions, calculated edges, and full public accountability.

Here's what it is, what it does, and why I'm making it public.

## The Problem with Most Prediction Services

Go look at any sports prediction account on X. You'll see the same pattern:

- Cherry-picked records ("12-3 last week!" — but silence about the 4-9 week before)
- Vague methodology ("our proprietary algorithm" — translation: coin flip)
- Single-market focus (moneyline picks only, ignoring the dozen other ways to find value)

The dirty secret of sports prediction is that most services are marketing operations first and analytical operations second. The predictions exist to sell subscriptions, not the other way around.

I'm an engineer. That offends me on a fundamental level. So I built something different.

## What Moneyball Dojo Actually Is

At its core, Moneyball Dojo is **9 specialized machine learning models** — each trained on 37 to 52 features using three full seasons of MLB data (2022–2024), then validated against the entire 2025 season using strict walk-forward methodology (no future information leaks into predictions).

Here's what the system covers — with real numbers from our 2025 walk-forward backtest:

| Model | What It Predicts | STRONG Picks | All Picks |
|-------|-----------------|-------------|-----------|
| **Moneyline** | Game winner | **58.7%** (550 games) | 52.9% |
| **Run Line** | Cover -1.5 spread | **67.3%** (1,482 games) | 64.5% |
| **First 5 Innings** | F5 winner | **59.9%** (1,096 games) | 54.5% |
| **NRFI** | No Run First Inning | **60.1%** (667 games) | 56.5% |
| **Over/Under** | Total runs | — | MAE 3.64 |
| **Pitcher Strikeouts** | K prop lines | — | In validation |
| **Batter Props** | Hits/HR/RBI lines | — | In validation |
| **Stolen Bases** | SB prop lines | — | In validation |
| **Pitcher Outs** | Recording outs | — | In validation |

Walk-forward means the model only sees data available before each game — the same information you'd have when placing a bet. No hindsight, no leakage, no inflated numbers.

The key insight: **don't bet on everything.** The edge lives in selectivity. STRONG picks across all four models hit well above the -110 breakeven rate of 52.4%.

## Edge-First, Not Pick-First

We don't just predict who wins. We calculate the **edge** — the gap between what our model says and what the sportsbook odds imply.

A team can be a likely winner and still be a terrible bet if the odds already reflect that. Value exists in the gap, not in the outcome.

Every pick is tiered by edge size. Here's how each tier performed on 2,425 moneyline predictions in the 2025 backtest:

- **STRONG** (8%+ edge): **58.7% hit rate.** 550 picks. Well above breakeven.
- **MODERATE** (4–8%): **53.0% hit rate.** 709 picks. Marginal edge.
- **LEAN** (<4%): **50.5% hit rate.** 750 picks. Essentially noise — published for transparency only.
- **PASS**: No edge. Not published.

The pattern is consistent across all models: bigger edge = higher accuracy. This isn't cherry-picking — it's the tier system doing exactly what it's designed to do. The disciplined play: **STRONG picks only.**

## What the Features Actually Look At

The models analyze 37 to 52 features per game, including:

- **Rolling performance** (last 15 games): win rate, run differential, momentum
- **Pitching matchups**: ERA, WHIP, K rates, pitch-level stats
- **Offensive splits**: BA, OBP, SLG with home/away breakdowns
- **Venue effects**: Park factors, first-inning scoring tendencies (for NRFI)
- **Pythagorean expectation**: Expected wins based on runs scored vs. allowed
- **Rest and travel**: Days off and distance traveled

Every feature is computed using only data available before the prediction date. The ensemble combines XGBoost, LightGBM, and Logistic Regression via soft voting, with Optuna-optimized hyperparameters.

## What 58.7% Actually Means for Your Wallet

Let's be straight about the math, because most prediction services won't be.

At standard -110 odds:
- **Breakeven**: 52.4% accuracy
- **58.7% (STRONG moneyline)**: ~6.0% ROI per bet
- Over 550 STRONG picks in a season, that's **+33 units** on flat $100 bets

That's real money. But it also means:

- **Bad weeks happen.** A 59% model will have losing weeks regularly.
- **Drawdowns are real.** You can go -10 units on a cold streak and still be on track.
- **Discipline is the edge.** Betting LEAN and MODERATE picks erases most of the profit. STRONG only.

We're not promising you'll get rich. We're showing you validated numbers and letting you decide.

## Why "Dojo"?

In Japanese martial arts, a dojo (道場) is a place of disciplined practice. You don't become a master through talent alone — you repeat the same technique thousands of times until it becomes instinct.

Machine learning works the same way. Obsessive data collection. Rigorous hypothesis testing. Incremental refinement. Accepting that you're never finished.

That philosophy is baked into this project. Every week we review what the models got wrong, retrain on new data, and ship updated predictions. You're not subscribing to static picks — you're joining a system that improves.

## Radical Transparency — For Real

Every prediction service claims transparency. Here's what ours actually looks like:

- **Every pick logged** in a public Google Sheet — before games start
- **Weekly performance reviews** with full win-loss records and ROI calculations (vig included)
- **Monthly model audits** — what's working, what's degrading, what we're changing
- **Honest miss analysis** — when we're wrong, we explain why

No silent methodology changes. No deleted tweets. No cherry-picked streaks.

## What to Expect

**Daily Digest** (every morning, before games)
- AI predictions across all MLB games
- Edge calculations for every pick across all 9 models
- One featured deep-dive matchup analysis
- Quick takes on every other game

**Weekly Performance Review** (Sundays)
- Full win-loss record across all markets
- ROI tracking for each model (after vig)
- Closing line value (CLV) analysis
- Honest commentary on misses and adjustments

During preseason, all content is free. After Opening Day on March 27, we'll introduce a premium tier with the full daily digest and all 9 model outputs. Free subscribers will continue to receive weekly summaries and selected picks.

## Who I Am

I'm a 32-year-old AI engineer based in Tokyo. I've spent my career building machine learning systems. Baseball — both NPB and MLB — has been a genuine obsession since childhood.

I'm not a professional gambler or a sports media personality. I'm an engineer who got frustrated that most prediction services were bad at the one thing they were supposed to do: predict. So I built a better system and decided to make it public.

Moneyball Dojo is the project where engineering rigor meets baseball obsession. You'll see the engine running in real time. Every win, every loss, every adjustment — all in the open.

Welcome to the Dojo. Let's get to work.

---

*Moneyball Dojo launches with the 2026 MLB season on March 27. Subscribe now to get preseason content and be ready for Opening Day.*
