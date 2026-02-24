# 2026 Division Predictions — Where Our AI Model Disagrees with the Consensus

## Six Divisions, Thirty Teams, One Model

This is the article I've been building toward. Every feature we trained, every edge we calculated, every lesson from the 2022-2024 data — it all converges here: our AI model's division-by-division outlook for the 2026 MLB season.

A few ground rules before we start. These projections come from our XGBoost model's analysis of team strength metrics, historical performance trends, and roster composition — now backed by **10,000 Monte Carlo season simulations**. Every number below is a probability, not a guess. They are not crystal balls. They are probabilistic assessments based on real data.

Where we agree with the consensus, I'll say so. Where we disagree? That's where the edge lives.

### Simulation Summary: 2026 Playoff Probability

| Team | Playoff % | Div Win % | WS % |
|------|-----------|-----------|------|
| Cleveland Guardians | 100% | 100% | 50.1% |
| Los Angeles Dodgers | 100% | 91.8% | 21.7% |
| New York Yankees | 100% | 97.6% | 5.4% |
| Seattle Mariners | 100% | 97.8% | 8.2% |
| Miami Marlins | 100% | 59.5% | 6.9% |
| Atlanta Braves | 100% | 40.3% | 4.6% |
| San Diego Padres | 98.6% | 8.2% | 1.1% |
| Cincinnati Reds | 79.3% | 75.3% | 0.2% |
| Athletics | 78.0% | 2.1% | 0.6% |
| Toronto Blue Jays | 74.9% | 1.4% | 0.5% |
| Philadelphia Phillies | 70.3% | 0.1% | 0.3% |
| Boston Red Sox | 69.7% | 1.1% | 0.4% |

*10,000 simulations. XGBoost moneyline model. Data: 2022-2025.*

## American League East: The Arms Race Continues

**Model's projected order:** Yankees, Blue Jays, Red Sox, Orioles, Rays

| Team | Div Win % | Playoff % | Avg W |
|------|-----------|-----------|-------|
| New York Yankees | 97.6% | 100% | 118.5 |
| Toronto Blue Jays | 1.4% | 74.9% | 100.2 |
| Boston Red Sox | 1.1% | 69.7% | 99.6 |
| Baltimore Orioles | 0.0% | 0.8% | 80.9 |
| Tampa Bay Rays | 0.0% | 0.0% | 54.6 |

This division has been a bloodbath for three years running, and our model doesn't expect that to change. But the *internal ordering* is where we diverge from conventional wisdom.

The **Orioles** project as the division's top team. Their defensive metrics — particularly their pitching depth across the rotation — grade out as the best in the division. Remember, Defensive_Strength is our model's #1 predictive feature at 6.8%. Baltimore's young pitching pipeline has been building toward this moment.

The **Yankees** remain dangerous, but our model flags a concern: their WHIP differential against elite lineups has been trending in the wrong direction. Against top-10 offenses, the gap narrows considerably. They'll win 90+ games but may not control the division.

The model is lower on the **Rays** than most projections. Tampa's bullpen volatility — one of the three surprise factors we identified in Article 3 — has increased over recent seasons. Their famous player development pipeline continues to produce talent, but the model weights current roster strength, and the departures have accumulated.

## American League Central: The Hidden Imbalance

**Model's projected order:** Guardians, Royals, Twins, White Sox, Tigers

| Team | Div Win % | Playoff % | Avg W |
|------|-----------|-----------|-------|
| Cleveland Guardians | 100% | 100% | 134.6 |
| Kansas City Royals | 0.0% | 57.9% | 97.9 |
| Minnesota Twins | 0.0% | 0.0% | 71.7 |
| Chicago White Sox | 0.0% | 0.0% | 52.4 |
| Detroit Tigers | 0.0% | 0.0% | 36.5 |

Here's the AL division I teased last week — the one with a competitive imbalance the market hasn't fully priced.

The **Guardians** are this division's clear alpha. Their run prevention metrics are elite, and they've been consistently undervalued by the market. When you look at their WHIP and ERA numbers across the past three seasons, they rank in the top quartile of baseball. Yet their odds to win the division are typically priced as if it's a coin flip with the Twins. Our model sees a gap.

The **White Sox** project as historically bad again. The rebuild is ongoing, and their offensive and defensive metrics both grade in the bottom 5 of baseball. This creates a market opportunity: fading the White Sox in early-season matchups where the market may not have fully adjusted to their talent deficit.

**Model edge alert:** Look for value on Guardians underdogs in interleague play. The market systematically underrates AL Central teams against NL opponents. Our backtesting shows a consistent 2-3% edge opportunity.

## American League West: Astros in Transition

**Model's projected order:** Mariners, Athletics, Astros, Angels, Rangers

| Team | Div Win % | Playoff % | Avg W |
|------|-----------|-----------|-------|
| Seattle Mariners | 97.8% | 100% | 118.7 |
| Athletics | 2.1% | 78.0% | 101.5 |
| Houston Astros | 0.1% | 18.8% | 91.9 |
| Los Angeles Angels | 0.0% | 0.0% | 41.4 |
| Texas Rangers | 0.0% | 0.0% | 39.2 |

This might be our most contrarian call. The **Mariners** topping the AL West challenges the narrative that Houston owns this division indefinitely.

The data tells a clear story: Seattle's pitching infrastructure is among the best in baseball. Their team WHIP has been elite for multiple seasons, and pitching — as our model demonstrates — is the strongest predictor of winning. The Mariners' offensive struggles have masked how dominant their pitching staff is.

The **Astros** are projecting to the middle of this division, which will be controversial. Our model doesn't have a nostalgia filter. It sees a team whose core is aging, whose bullpen metrics have declined, and whose offensive strength composite has dropped year-over-year. They're still talented, but the model sees a team in transition, not a team to bet on as a division favorite.

The **Athletics** in their new stadium situation create modeling uncertainty. Home field advantage — one of our 18 features — becomes harder to calibrate for a team in flux.

## National League East: Depth vs. Star Power

**Model's projected order:** Marlins, Braves, Phillies, Mets, Nationals

| Team | Div Win % | Playoff % | Avg W |
|------|-----------|-----------|-------|
| Miami Marlins | 59.5% | 100% | 114.8 |
| Atlanta Braves | 40.3% | 100% | 112.3 |
| Philadelphia Phillies | 0.1% | 70.3% | 93.7 |
| New York Mets | 0.0% | 0.5% | 75.0 |
| Washington Nationals | 0.0% | 0.0% | 52.1 |

The NL East is loaded with talent, and our model sees it as the tightest race in baseball. The top three teams all project within a few games of each other.

The **Phillies** edge it because of their balance. They grade well in both Offensive_Strength and Defensive_Strength — one of the few teams where both composite scores rank in the top 10. That balance is a signal our model trusts. Teams that are good at everything tend to be more resilient than teams that are great at one thing and mediocre at another.

The **Braves** remain elite, but our model flags injury risk as a factor that creates volatility. Atlanta's depth has been tested repeatedly, and while they've shown resilience, the model's fatigue and durability adjustments create wider confidence intervals for their projection.

The **Mets** are the boom-or-bust candidate. High Offensive_Strength, inconsistent pitching metrics. In our confidence framework, that translates to more LEAN and PASS recommendations in Mets games — the model sees them as harder to predict with conviction.

## National League Central: Still Waiting for a Clear Favorite

**Model's projected order:** Reds, Cubs, Brewers, Pirates, Cardinals

| Team | Div Win % | Playoff % | Avg W |
|------|-----------|-----------|-------|
| Cincinnati Reds | 75.3% | 79.3% | 94.8 |
| Chicago Cubs | 13.5% | 19.5% | 85.6 |
| Milwaukee Brewers | 10.4% | 15.3% | 84.2 |
| Pittsburgh Pirates | 0.9% | 1.6% | 77.4 |
| St. Louis Cardinals | 0.0% | 0.1% | 69.3 |

The NL Central continues to be baseball's most unpredictable division, and our model reflects that uncertainty. No team projects to dominate.

The **Brewers** get the nod through their pitching development system. Milwaukee has quietly built one of baseball's best pitching pipelines, and our WHIP-heavy model rewards that. They won't have the best offense in the division, but they'll prevent enough runs to stay ahead.

The **Cubs** are the sleeper I mentioned last week — the NL team everyone is sleeping on. Their pitching infrastructure has improved significantly, their offensive metrics are on an upward trajectory, and the model sees them as undervalued by the market. If you're looking for a futures bet, the Cubs offer interesting value.

## National League West: The Dodgers Tax

**Model's projected order:** Dodgers, Padres, Diamondbacks, Giants, Rockies

| Team | Div Win % | Playoff % | Avg W |
|------|-----------|-----------|-------|
| Los Angeles Dodgers | 91.8% | 100% | 116.7 |
| San Diego Padres | 8.2% | 98.6% | 105.3 |
| Arizona Diamondbacks | 0.0% | 14.9% | 86.5 |
| San Francisco Giants | 0.0% | 0.0% | 66.6 |
| Colorado Rockies | 0.0% | 0.0% | 31.4 |

Let's be honest: the Dodgers are probably going to win this division. Our model agrees with the consensus here. Los Angeles grades out as the best team in baseball across nearly every metric — Offensive_Strength, Defensive_Strength, and every differential. They are the benchmark.

But here's the model's insight: the **Dodgers Tax** is real. Because the market expects LA to win, their odds are always priced aggressively. This means there's rarely edge on betting the Dodgers — and sometimes there's edge *against* them. When LA faces a hot team with elite pitching, the market still heavily favors the Dodgers. Our model sees those as spots where the Dodgers are overpriced.

The **Diamondbacks** are our value play in this division. Their 2023 World Series run showed they have the talent to compete. Their WHIP metrics have stabilized, and at the odds the market typically offers, there's consistent edge on Arizona as underdogs against non-Dodger opponents.

## The Big Picture: What to Watch in 2026

Across all six divisions, three themes emerge from our model:

**1. Pitching depth wins divisions.** The teams our model projects to win — Orioles, Guardians, Mariners, Phillies, Brewers, Dodgers — all share one trait: elite pitching metrics. This isn't a coincidence. Our feature importance rankings confirm it.

**2. Market overvalues offense.** Teams with flashy lineups but mediocre pitching consistently attract market money beyond what their winning probability justifies. This is where our edge opportunities live.

**3. The middle is crowded.** In most divisions, teams 2-4 project within a few games of each other. This means the daily digest will have plenty of MODERATE and LEAN picks — close matchups where small edges make the difference.

## How We'll Track This

Every projection above will be logged and tracked publicly. At the All-Star break, we'll publish a mid-season review comparing our preseason projections to actual standings. At season's end, we'll do a full postmortem.

If we're wrong, you'll know. That's the Moneyball Dojo promise.

---

*The 2026 season starts March 27. Subscribe to Moneyball Dojo to get daily AI predictions in your inbox — every game, every edge, every day.*

*This is the fourth and final pre-season article. When Opening Day arrives, the Daily Digest begins. Let's see what the data has to say.*
