# A Japanese AI Engineer Built 9 Models to Beat MLB Sportsbooks — Here's the Honest Version

## Introducing Moneyball Dojo: 9 machine learning models, every game, every day, full transparency — including when we screw up

I have a confession: I watch more baseball than any reasonable person should.

Growing up in Tokyo, I was the kid who'd stay up until 3 AM watching MLB games on delayed broadcast — then get up for school and argue about NPB stats with friends who thought I was insane for caring about both leagues. In college, I studied machine learning. The two obsessions were bound to collide.

Last year, I stopped talking about it and started building. The result is Moneyball Dojo — an AI prediction system that runs 9 different models across every MLB game, every single day. Not a hobby project with gut feelings dressed up as data. An actual engineered system with logged predictions, calculated edges, and full public accountability.

Here's what it is, what it does, and why I'm starting with a correction.

## First Things First: We Found a Bug and We're Telling You

Before we get into the system, I owe you this.

Our initial backtesting showed impressive numbers — 64.7% overall accuracy, 72.9% on high-confidence picks. We published those numbers. Then we did what engineers are supposed to do: we hired a third-party review.

What they found: **data leakage in our feature pipeline.** Specifically, some of our team performance metrics were calculated using full-season data instead of only data available before each game. That means the model was peeking at future results — not intentionally, but the effect was the same. Inflated accuracy.

After fixing the leakage, our walk-forward validation shows:

- **Moneyline accuracy: ~59–60%**
- **Consistent across 2024 and 2025 test windows**

That's roughly 5 percentage points lower than we originally claimed. It's still profitable territory after vig at -110 (breakeven is 52.4%), but it's not the headline-grabbing number we published before.

We could have quietly updated the numbers and hoped nobody noticed. Instead, we're leading with the correction. If this project is about transparency, it starts here.

## The Problem with Most Prediction Services

Go look at any sports prediction account on X. You'll see the same pattern:

- Cherry-picked records ("12-3 last week!" — but silence about the 4-9 week before)
- Vague methodology ("our proprietary algorithm" — translation: coin flip)
- Single-market focus (moneyline picks only, ignoring the dozen other ways to find value)

The dirty secret of sports prediction is that most services are marketing operations first and analytical operations second. The predictions exist to sell subscriptions, not the other way around.

I'm an engineer. That offends me on a fundamental level. So I built something different — and when I found a flaw, I fixed it and told you about it.

## What Moneyball Dojo Actually Is

At its core, Moneyball Dojo is **9 specialized machine learning models** — each trained on 37 to 52 features using three full seasons of MLB data (2022–2024), then validated against held-out 2025 data using walk-forward methodology.

Here's what the system covers:

| Model | What It Predicts | Features |
|-------|-----------------|----------|
| **Moneyline** | Game winner | 37 |
| **Run Line** | Cover -1.5 spread | 52 |
| **First 5 Innings** | F5 winner | 48 |
| **NRFI** | No Run First Inning | 45 |
| **Over/Under** | Total runs | 52 |
| **Pitcher Strikeouts** | K prop lines | — |
| **Batter Props** | Hits/HR/RBI lines | — |
| **Stolen Bases** | SB prop lines | — |
| **Pitcher Outs** | Recording outs | — |

Walk-forward backtest accuracy for the moneyline model: **59–60%** across two independent test windows (2024 and 2025). We'll publish full calibration data and ROI tracking from day one of the 2026 season — no more backtests, only real results.

Most prediction services give you one type of pick. We give you nine angles on every game. Find your edge in moneyline, run line, first five, NRFI, totals, or player props — all from the same system, all tracked transparently.

## Edge-First, Not Pick-First

We don't just predict who wins. We calculate the **edge** — the gap between what our model says and what the sportsbook odds imply.

A team can be a likely winner and still be a terrible bet if the odds already reflect that. Value exists in the gap, not in the outcome.

Every pick is tiered by edge size:

- **STRONG** (8%+ edge): High conviction. Model and market sharply disagree.
- **MODERATE** (4–8%): Real value. Our bread-and-butter picks.
- **LEAN** (<4%): Signal exists but thin. Published for transparency.
- **PASS**: No edge found. The most disciplined call is sometimes no call.

An important note: these tiers are based on edge magnitude, not calibrated win probability. We're building calibration analysis now and will publish the data showing whether STRONG picks actually convert at a higher rate. Until that data exists from live predictions, we're not making accuracy claims by tier.

## What the Features Actually Look At

The models analyze 37 to 52 features per game, including:

- **Rolling performance** (last 15 games): win rate, run differential, momentum — computed using only games played before the prediction date
- **Pitching matchups**: ERA, WHIP, K rates, pitch-level stats
- **Offensive splits**: BA, OBP, SLG with home/away breakdowns
- **Venue effects**: Park factors with Bayesian shrinkage for small-sample venues
- **Pythagorean expectation**: Expected wins based on runs scored vs. allowed
- **Rest and travel**: Days off and distance traveled (computed incrementally, no leakage)

The ensemble combines XGBoost, LightGBM, and Logistic Regression via soft voting, with Optuna-optimized hyperparameters validated through cross-validation on training data only.

## What 59–60% Actually Means for Your Wallet

Let's be straight about the math, because most prediction services won't be.

At standard -110 odds:
- **Breakeven**: 52.4% accuracy
- **59% accuracy**: ~3.6% ROI per bet
- **60% accuracy**: ~4.5% ROI per bet

That sounds small. Over a full season with disciplined flat-unit betting, it compounds. But it also means:

- **Bad weeks happen.** A 60% model will have losing weeks regularly.
- **Drawdowns are real.** You can go -10 units on a cold streak and still be on track.
- **Bankroll management matters.** We'll publish guidance, but flat 1-unit bets on STRONG picks only is the conservative starting point.

We're not promising you'll get rich. We're saying the model has a measurable edge, and we'll prove it with live tracked results or admit when it's not working.

## Why "Dojo"?

In Japanese martial arts, a dojo is a place of disciplined practice. You don't become a master through talent alone — you repeat the same technique thousands of times until it becomes instinct.

Machine learning works the same way. Obsessive data collection. Rigorous hypothesis testing. Incremental refinement. Accepting that you're never finished — and telling the truth when you find a mistake.

## Radical Transparency — Starting with Our Own Mistakes

Every prediction service claims transparency. Here's what ours actually looks like:

- **Every pick logged** in a public Google Sheet — before games start
- **Weekly performance reviews** with full win-loss records and ROI calculations (vig included)
- **Monthly model audits** — what's working, what's degrading, what we're changing
- **Honest miss analysis** — when we're wrong, we explain why
- **Corrections published immediately** — like this one

No silent methodology changes. No deleted tweets. No cherry-picked streaks. And when we find data leakage in our own pipeline, we tell you before we tell anyone else.

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

I'm not a professional gambler or a sports media personality. I'm an engineer who got frustrated that most prediction services were bad at the one thing they were supposed to do: predict. So I built a system, found a bug in it, fixed it, and decided to make both the system and the mistake public.

Moneyball Dojo is the project where engineering rigor meets baseball obsession. You'll see the engine running in real time. Every win, every loss, every adjustment, every correction — all in the open.

Welcome to the Dojo. Let's get to work.

---

*Moneyball Dojo launches with the 2026 MLB season on March 27. Subscribe now to get preseason content and be ready for Opening Day.*
