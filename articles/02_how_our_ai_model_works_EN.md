# How Our AI Model Works — A Transparent Look at the Engine Behind the Predictions

When I started building Moneyball Dojo, I knew that credibility would come from one place: radical transparency. In Japan, we have a concept called *shokunin* — the craftsman's obsession with mastery. That's what I'm chasing here. Not perfect predictions, but an honest, relentless commitment to showing you exactly how the sausage is made.

So today, I want to walk you through the engine that powers every prediction you'll see from us. Not the glossy version. The real one — with all the limitations, all the reasoning, all the humility that comes with building machine learning models for sports.

## Why XGBoost? The Boring Answer (But Important)

At the heart of Moneyball Dojo is an algorithm called **XGBoost**. I can already hear you thinking: "That sounds like a robot punching something." Fair enough. Let me explain it simply.

Imagine you're a scout trying to predict whether a baseball team will win today. You don't just look at one stat — you weigh dozens. You notice that when a team's offense is strong AND they have a great defense AND the pitcher has a low ERA, they tend to win more often. You adjust those weightings based on experience. You compound the learnings. That's fundamentally what XGBoost does, except it processes data at machine speed and finds patterns human scouts might miss.

XGBoost stands for "Extreme Gradient Boosting." Here's the practical version: it takes weak predictions and layers them on top of each other, each layer learning from the mistakes of the previous one. After thousands of iterations, you get a composite model that's surprisingly accurate.

Why did I choose it? Three reasons:

1. **Speed and accuracy**: XGBoost consistently outperforms other algorithms on structured data (like baseball stats). It's what powers prediction systems at companies like Uber and PayPal.

2. **Interpretability**: Unlike deep learning black boxes, XGBoost tells us *which features matter most*. In baseball prediction, knowing why the model made a call is crucial.

3. **Reliability**: It's battle-tested. I'd rather use a proven tool that I understand deeply than chase cutting-edge models that might break in production.

The dojo principle applies here too — master one weapon thoroughly before reaching for another.

## The 18 Features: Our Scouting Report

Every prediction needs input data. I call these **features** — the 18 statistics that feed into our model. Think of them as the scouting report we run before every prediction.

**Offensive Metrics:**
- **BA** (Batting Average): How often hitters get hits
- **OBP** (On-Base Percentage): How often they reach base (including walks)
- **SLG** (Slugging Percentage): How much power they have

**Pitching Metrics:**
- **ERA** (Earned Run Average): Runs allowed per nine innings
- **WHIP** (Walks + Hits per Innings Pitched): Opposing hitters faced per inning pitched

We also track **Home and Away variants** of these — because ballparks matter, schedules matter, and travel exhausts players.

**Composite Strength Scores** (These are my favorites):
- **Offensive_Strength**: A normalized score combining BA, OBP, and SLG into one "how scary is this offense" metric
- **Defensive_Strength**: A composite of team fielding quality and pitching depth

**The Differentials** (Where the edge hides):
- **BA_Diff, OBP_Diff, SLG_Diff**: Your team's offense versus the opponent's
- **ERA_Diff, WHIP_Diff**: Your pitching versus theirs

Here's the thing: the differentials are where we hunt for edges. In betting, margins matter. A team's raw stats mean less than how they stack up against the specific opponent they're facing today.

## Feature Importance: What Actually Matters

After training on thousands of games, the model tells us which features moved the needle most. This is my favorite part — it's where intuition meets data.

**Top Feature Importance Rankings:**

1. **Defensive_Strength** (6.8%): Your defense and pitching is the strongest single predictor. This makes sense — baseball is fundamentally a game where preventing runs is often harder than scoring them.

2. **BA_Diff** (6.5%): How much better (or worse) your batters are compared to the opponent. Context matters enormously.

3. **Away_WHIP** (6.4%): How pitchers perform specifically in away games. Travel and crowd impact are real.

4. **WHIP_Diff** (6.3%): Your pitching efficiency versus the opponent's. Walks and hits given up drive swing decisions at the plate.

What this tells me is that **pitching dominates prediction more than offense**. That's both surprising and not — it aligns with Bill James's early work showing that prevention matters more than production. But Away_WHIP in the top 4? That's the model telling me travel and environment matter in ways raw stats alone don't capture.

No feature dominates more than ~7%. That's actually healthy. It means the prediction isn't resting on a single point of failure.

## How We Calculate the Edge

Here's where it gets real. A prediction is only valuable if there's an *edge* — a gap between what you believe will happen and what the market believes.

Our edge calculation is simple:

```
Edge = Our Model Probability - Implied Probability from Market Odds
```

Let's say we think Team A has a 55% chance to win. The sportsbook has them at -110 (about 52.4%). Our edge: 2.6 percentage points. That's playable.

But here's the catch: a 55% model prediction isn't that confident in absolute terms. Over thousands of games, being right 55% of the time isn't enough to overcome juice (the vigorish sportsbooks charge). You need a bigger edge.

That's why we have a confidence tier system.

## The Confidence Tiers: STRONG, MODERATE, LEAN, PASS

Not all predictions are created equal. Some days, the model is screaming. Other days, it's uncertain. We tier everything:

**STRONG (8%+ Edge):**
The model is convinced. Offensive_Strength is dominant, the matchup heavily favors one side, everything lines up. These are the predictions we're most confident in. We expect these to win at a significantly higher rate.

**MODERATE (4-8% Edge):**
There's a real lean here. The model sees value, but it's not unanimous. These are our bread-and-butter predictions. More conservative than STRONG, but still actionable.

**LEAN (<4% Edge):**
The model sees something, but barely. We publish them because transparency means showing everything — even the close calls. If you bet these, you're trusting the model's edge math more than the confidence judgment.

**PASS:**
When the model prediction sits within a few percentage points of 50%, we pass. There's no edge. The market has priced it correctly, or we don't have enough signal. Sometimes the most disciplined call is saying "I don't know."

This is core to our philosophy. Every prediction comes with its confidence tier. You know exactly what we think about what we think.

## The Honest Reckoning: 53% Accuracy (And Why That Matters)

Here's where I'm radically transparent: Our synthetic testing shows approximately **53% accuracy**. That's better than a coin flip, worse than a championship prediction system needs to be.

Why am I telling you this? Because I want to set proper expectations. We're not launching with a 65% prediction system (those mostly don't exist, despite what marketing materials claim). We're launching with an honest, sound model that has room to improve.

And improvement is coming. Right now, we're testing on synthetic data — games from previous seasons played against historical odds. Once the live season starts, we'll have real money movements, real player health reports, real weather data from actual game days. That will sharpen everything.

Think of it like a martial artist training against a dummy before facing real opponents. The dummy work prepares you, but the real match teaches you everything you're missing.

## Why Transparency Is Non-Negotiable

Here's what a lot of prediction services do: they publish winners and quietly forget about losers. They adjust their methodology silently. They change goalposts. 

That's not us.

Every prediction we make gets logged. Every edge calculation gets recorded. At the end of the season, we'll publish a comprehensive breakdown: How many STRONG predictions hit? How many MODERATE? What was our actual accuracy versus the model's confidence? Where did we miss?

That's uncomfortable sometimes. Predictions fail. Models are wrong. But that's also the only way to build real trust — and the only way to actually improve.

In Japan, there's a martial arts saying: *The dojo is where you fail safely, so you can succeed in public.* We're treating this newsletter like a public dojo. You're watching us train, test, fail, learn, and refine. 

## The Training Never Stops

This is the part that excites me most. The model isn't static. Every season teaches us something new:

- New player trades change team dynamics
- Ballpark effects shift as grass ages or climate changes
- Coaching philosophies evolve
- Injuries create unexpected edges

We'll retrain the model as the season progresses. We'll add features if they prove predictive. We'll remove ones that were just noise. We'll be ruthless about it.

This is where XGBoost's interpretability shines. We can see exactly how the model changes its thinking as new data comes in. We can learn with it, not just trust it.

## The Bottom Line

Moneyball Dojo is an AI system built on three principles:

1. **Sound methodology**: We use battle-tested algorithms with transparent reasoning
2. **Honest assessment**: 53% accuracy today, committed to improvement, losses logged publicly
3. **Relentless discipline**: Every prediction comes with a confidence tier; every assumption gets questioned

The model isn't perfect. But it's *honest*, it's *auditable*, and it's built to improve.

This is baseball's version of the dojo — a disciplined training ground where data meets intuition, where failure is feedback, and where rigor is the foundation of insight.

**If you believe in radical transparency in prediction, if you want to see the machine working — not just its outputs — then Moneyball Dojo is for you.**

Subscribe to see this model live over the 2026 season. Watch us predict. Watch us miss. Watch us learn. And most importantly: watch us show our work every single step.

The edge is in the details. And we're showing you all of them.

---

*Next week: The 2026 Preseason Hot Takes — and why our model disagrees with the consensus.*

*Questions about the model? Send them our way. Transparency means answering what you're curious about.*
