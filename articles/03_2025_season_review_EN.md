# The Numbers Behind the 2025 Season — And What They Mean for 2026

## What Recent MLB Data Tells Us About Where the Game Is Going

I've spent the last few months buried in Statcast data, and I have to tell you: baseball is changing faster than most people realize.

If you've been following Moneyball Dojo, you know our approach. We don't trust narratives. We trust numbers. And the numbers from recent seasons — particularly 2022 through 2025 — tell a story that should change how you think about the 2026 season.

Let me walk you through what the data says.

## The Pitch Clock Changed Everything (And We're Still Catching Up)

When MLB introduced the pitch clock in 2023, the immediate impact was obvious: games got shorter. Average game time dropped from 3 hours 4 minutes in 2022 to 2 hours 40 minutes. But the second-order effects are what matter for prediction.

Pitchers are making quicker decisions. Batters have less time to reset between pitches. The pace of play has altered how fatigue accumulates — particularly in later innings, where we've seen a measurable increase in pitcher efficiency drops after the 6th inning compared to pre-clock data.

For our model, this means inning-by-inning performance decay is a more important feature than it used to be. We're incorporating pitch clock era adjustments into how we weight late-game performance.

## The Stolen Base Renaissance

The shift ban and larger bases in 2023 triggered a stolen base explosion that has continued. Stolen base attempts rose roughly 30% from pre-rule levels, and success rates climbed with them. Baserunning is back as a meaningful offensive weapon.

What does this mean for prediction? Teams with elite baserunning now have a tangible advantage that wasn't captured well in traditional models. Speed on the bases creates pressure on pitchers, leads to defensive errors, and manufactures runs in ways that raw power numbers don't reflect.

Our 2026 model will incorporate baserunning metrics more aggressively. This is one of the areas where we expect to find market inefficiency — bookmakers are historically slow to price in baserunning value.

## WHIP Dominates — Confirmed by Our Model

When we trained our XGBoost model on 2022-2024 data, one result stood out: **pitching metrics dominate prediction accuracy**. Defensive_Strength was our top feature at 6.8%, and WHIP-related features (Away_WHIP at 6.4%, WHIP_Diff at 6.3%) consistently ranked in the top 4.

This aligns with what the raw data tells us. The correlation between team WHIP and winning percentage has been remarkably stable across recent seasons — stronger, in fact, than the correlation between OPS and winning percentage. Teams that prevent baserunners win more games. It sounds obvious, but the market still tends to overvalue offensive fireworks.

The lesson for 2026: when evaluating matchups, look at the pitching staff first. A team with a 1.20 WHIP facing a team with a 1.35 WHIP has a structural advantage that our model weighs heavily.

## The Surprise Factor: Why Projections Miss

Every season produces surprise teams. Teams that were projected to finish last end up in the wild card race. Preseason favorites collapse. Why?

Looking at the data, three factors consistently explain the gap between projections and reality:

**1. Bullpen Volatility.** Starting pitcher performance is relatively stable year-to-year. Bullpen performance is not. Relief pitchers are volatile by nature — small samples, high-leverage situations, and injury risk mean that a team's bullpen can swing from top-5 to bottom-10 within a single month. Our model accounts for this by weighting recent bullpen performance more heavily than season-long averages.

**2. Rookie Breakouts.** Projection systems struggle with rookies because there's limited MLB data. A 22-year-old with 50 plate appearances doesn't give models much to work with. When a rookie breaks out — think of how certain players have changed team trajectories in recent years — the models that adapt fastest gain the biggest edge.

**3. Schedule Effects.** This one is underrated. Teams playing many games against weak opponents early in the season can build records that inflate their perceived strength. Our model tries to adjust for strength of schedule, but it's an imperfect science.

## What 2022-2024 Data Teaches Us About Edges

Here's where it gets practical. After backtesting our model across three seasons of data, we've identified several patterns in where market edges tend to exist:

**Underdogs in pitching matchups.** When an underdog has a clear pitching advantage (lower ERA, lower WHIP) but the market is pricing the game based on overall team reputation, our model finds consistent value. The market tends to overweight a team's brand and underweight the specific pitching matchup.

**Division rivalry adjustments.** Games between division rivals tend to be closer than the market expects. Familiarity breeds parity. Our model shows that big favorites in division games hit at a lower rate than the lines suggest.

**Travel disadvantage.** Away_WHIP being in our top-4 features isn't a coincidence. Teams on the road, especially on the second or third game of an away series, show measurable performance drops. West Coast teams traveling East (and dealing with early start times relative to their body clocks) are a specific spot where we've found edges.

## Building the 2026 Model: What's New

Based on everything we've learned, here's what we're incorporating for the 2026 season:

**Enhanced pitching features.** We're adding pitch mix data and expected pitch effectiveness metrics alongside our existing ERA and WHIP features. Pitchers who rely heavily on one pitch type may be more vulnerable against lineups that have seen them before.

**Baserunning composite.** A new feature combining stolen base success rate, extra-base taking, and baserunning runs above average. This captures the speed dimension that our 2024 model underweighted.

**Fatigue indicators.** Using day-of-week and games-in-stretch data to model fatigue effects. Teams in the middle of a 10-game homestand behave differently than teams at the start of a road trip.

**Market odds integration.** Rather than just comparing our predictions to market odds after the fact, we're building market implied probability directly into our feature set. The market itself contains information — and sometimes, knowing where the market is confident tells us where to look for disagreement.

## Looking Ahead: 2026 Division Previews

Next week, we'll publish our first division-by-division preview for the 2026 season. We'll share what our model thinks about each division's competitive balance, which teams are likely over-valued by the market, and where the smart money should be paying attention.

I'll say this much as a teaser: one AL division has a clear competitive imbalance that the market hasn't fully priced in. And one NL team that everyone is sleeping on has the pitching infrastructure to make a serious run.

More next week.

## The Data Never Lies (But It Does Whisper)

Here's the thing about data-driven prediction: the signal is often quiet. Baseball isn't a game of dramatic certainties. It's a game of small edges accumulated over 162 games. A 2% advantage in prediction accuracy doesn't feel like much on any given Tuesday night. But across a full season? That 2% is the difference between a profitable model and a coin flip.

That's what Moneyball Dojo is built for. Not loud claims. Not guaranteed picks. Just disciplined, transparent, data-driven analysis that gives you a genuine edge — one game at a time.

The 2026 season starts on March 27. We'll be ready.

---

*Subscribe to Moneyball Dojo for free to get daily predictions when the season starts. Follow along as we test our model in real time, with full transparency on wins and losses.*

*Next up: 2026 Division Preview — Our AI's Bold Predictions for Every Division.*
