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

At its core, Moneyball Dojo is **9 specialized XGBoost models** — each trained on 37 to 52 features using three full seasons of MLB data (2022–2024), then validated against the entire 2025 season.

Here's what the system covers:

| Model | What It Predicts | Features | 2025 Backtest |
|-------|-----------------|----------|---------------|
| **Moneyline** | Game winner | 37 | 64.7% overall · 72.9% on STRONG picks |
| **Run Line** | Cover -1.5 spread | 52 | 66.3% overall · 73.4% on STRONG picks |
| **First 5 Innings** | F5 winner | 48 | 64.6% overall · 69.5% on STRONG picks |
| **NRFI** | No Run First Inning | 45 | 62.1% overall · 68.9% on STRONG picks |
| **Over/Under** | Total runs | 52 | MAE: 3.09 runs |
| **Pitcher Strikeouts** | K prop lines | — | In validation |
| **Batter Props** | Hits/HR/RBI lines | — | In validation |
| **Stolen Bases** | SB prop lines | — | In validation |
| **Pitcher Outs** | Recording outs | — | In validation |

That's not a typo — **72.9% accuracy on our highest-confidence moneyline picks** across 1,317 games in 2025. Run line STRONG picks hit at 73.4% over 1,303 games.

Most prediction services give you one type of pick. We give you nine angles on every game. Find your edge in moneyline, run line, first five, NRFI, totals, or player props — all from the same system, all tracked transparently.

## Edge-First, Not Pick-First

We don't just predict who wins. We calculate the **edge** — the gap between what our model says and what the sportsbook odds imply.

A team can be a likely winner and still be a terrible bet if the odds already reflect that. Value exists in the gap, not in the outcome.

Every pick is tiered by edge size:

- **STRONG** (8%+ edge): High conviction. Model and market sharply disagree.
- **MODERATE** (4–8%): Real value. Our bread-and-butter picks.
- **LEAN** (<4%): Signal exists but thin. Published for transparency.
- **PASS**: No edge found. The most disciplined call is sometimes no call.

This is where the numbers matter. Our STRONG picks don't just hit more often — they hit at **72.9%** because the model is identifying games where the market is genuinely mispriced.

## What the Features Actually Look At

The models analyze 37 to 52 features per game, including:

- **Rolling performance** (last 15 games): win rate, run differential, momentum
- **Pitching matchups**: ERA, WHIP, K rates, pitch-level stats
- **Offensive splits**: BA, OBP, SLG with home/away breakdowns
- **Venue effects**: Park factors, first-inning scoring tendencies (for NRFI)
- **Pythagorean expectation**: Expected wins based on runs scored vs. allowed
- **Cross-features**: Engineered variables capturing matchup-specific dynamics

The single most predictive feature? **Rolling win percentage differential** — the gap in recent form between the two teams. It accounts for 12.5% of model importance. Recent momentum beats season-long averages.

## Why "Dojo"?

In Japanese martial arts, a dojo (道場) is a place of disciplined practice. You don't become a master through talent alone — you repeat the same technique thousands of times until it becomes instinct.

Machine learning works the same way. Obsessive data collection. Rigorous hypothesis testing. Incremental refinement. Accepting that you're never finished.

That philosophy is baked into this project. Every week we review what the models got wrong, retrain on new data, and ship updated predictions. You're not subscribing to static picks — you're joining a system that improves.

## Radical Transparency — For Real

Every prediction service claims transparency. Here's what ours actually looks like:

- **Every pick logged** in a public Google Sheet — before games start
- **Weekly performance reviews** with full win-loss records and ROI calculations
- **Monthly model audits** — what's working, what's degrading, what we're changing
- **Honest miss analysis** — when we're wrong, we explain why

No silent methodology changes. No deleted tweets. No cherry-picked streaks.

When the model had a rough stretch in July–August 2025 (backtest accuracy dipped to 60%), we didn't hide it. We analyzed it: mid-season fatigue patterns, roster changes from the trade deadline, and bullpen usage shifts all contributed. The model recovered to 69.3% in September. That kind of honest cycle — dip, diagnose, improve — is the whole point.

## What to Expect

**Daily Digest** (every morning, before games)
- AI predictions across all MLB games
- Edge calculations for every pick across all 9 models
- One featured deep-dive matchup analysis
- Quick takes on every other game

**Weekly Performance Review** (Sundays)
- Full win-loss record across all markets
- ROI tracking for each model
- Honest commentary on misses and adjustments

During preseason, all content is free. After Opening Day on March 27, we'll introduce a premium tier with the full daily digest and all 9 model outputs. Free subscribers will continue to receive weekly summaries and selected picks.

## Who I Am

I'm a 32-year-old AI engineer based in Tokyo. I've spent my career building machine learning systems. Baseball — both NPB and MLB — has been a genuine obsession since childhood.

I'm not a professional gambler or a sports media personality. I'm an engineer who got frustrated that most prediction services were bad at the one thing they were supposed to do: predict. So I built a better system and decided to make it public.

Moneyball Dojo is the project where engineering rigor meets baseball obsession. You'll see the engine running in real time. Every win, every loss, every adjustment — all in the open.

Welcome to the Dojo. Let's get to work.

---

*Moneyball Dojo launches with the 2026 MLB season on March 27. Subscribe now to get preseason content and be ready for Opening Day.*
