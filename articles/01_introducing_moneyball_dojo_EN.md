# Why a Japanese AI Engineer Is Building MLB Predictions That America Can't

## Introducing Moneyball Dojo — where Japanese baseball intelligence meets machine learning

I grew up watching two baseball worlds at once.

In Tokyo, I'd watch NPB games obsessively — studying how Japanese pitchers sequenced their pitches differently from Americans, how NPB managers made bullpen decisions based on matchup tendencies that MLB skippers wouldn't consider until years later. And then I'd stay up late watching MLB on the delayed broadcast, trying to understand why the same player who dominated in Japan sometimes struggled in America — and why others, like Ichiro, made the jump look effortless.

That dual perspective is something most American baseball analysts simply don't have. And it's the foundation of what I'm building here.

## The Gap Nobody Talks About

Every year, more Japanese players cross the Pacific. Ohtani reshaped what we thought a baseball player could be. Yoshida, Imanaga, Yamamoto, Suzuki — the pipeline is accelerating. And every year, American projection systems get these players wrong.

Why? Because they're projecting NPB-to-MLB transitions using only MLB-side data. They don't have the context of what a pitcher's pitch mix looked like in NPB, how a hitter's approach changed across seasons in the Pacific League, or what the coaching philosophies were that shaped these players before they arrived.

I do. I've watched these players for years before they became MLB headlines. That's not a boast — it's a structural informational advantage that comes from being Japanese and obsessively following both leagues.

Moneyball Dojo exists to exploit that gap.

## What We're Actually Building

At its core, Moneyball Dojo is an AI prediction system. We use XGBoost — a machine learning algorithm trained on MLB data — to predict every game, every day. The model analyzes 18 features: offensive metrics (BA, OBP, SLG), pitching metrics (ERA, WHIP), composite strength scores, and the differentials between opposing teams.

But here's what makes us different from every other prediction bot:

**The NPB Intelligence Layer.** When a Japanese player is involved in a matchup, our analysis goes deeper than the model alone. We layer in NPB context — how this pitcher performed against similar batting lineups in Japan, how this hitter's swing mechanics translate to American pitching styles, what their adjustment curve looked like historically. This is qualitative edge that no algorithm captures on its own.

**Edge-First Philosophy.** We don't just predict winners. We calculate the *edge* — the gap between what our model thinks will happen and what the sportsbook odds imply. A prediction is only valuable when there's a meaningful gap. We tier every pick:

- **STRONG** (8%+ edge): The model is screaming. Everything lines up.
- **MODERATE** (4-8%): Real value. Our bread-and-butter picks.
- **LEAN** (<4%): Signal exists but weak. Published for transparency.
- **PASS**: No edge. The most disciplined call is sometimes "I don't know."

**Radical Transparency.** Every prediction gets logged. Every win. Every loss. Weekly performance reviews with full ROI tracking. When we're wrong — and we will be wrong — you'll see it. No hiding, no silent methodology changes, no cherry-picked records.

## Why "Dojo"?

In Japanese martial arts, a dojo (道場) is a place of disciplined practice. You don't become a master through talent alone — you train the same technique thousands of times until it becomes instinct. The dojo is where that repetition happens.

Machine learning works the same way. You don't build a great model through luck. You gather data obsessively, test hypotheses rigorously, refine incrementally, and accept that you're never finished. The goal isn't perfection — it's steady, disciplined improvement.

That's our philosophy. You're not subscribing to a prediction service. You're joining a training ground where we improve together.

## What to Expect

**Daily Digest** (every morning before games)
- AI predictions for every MLB game
- Win probability and market edge for each pick
- One featured deep-dive analysis
- NPB crossover insights when Japanese players are in key matchups

**Weekly Performance Review** (Sundays)
- Full win-loss record and ROI tracking
- What the data taught us that week
- Honest commentary on our misses

**NPB-to-MLB Intelligence** (special coverage)
- Deep analysis of Japanese players in the majors
- Pre-scouting reports on NPB players likely to make the jump
- Context that American analysts miss

During the preseason, all content is free. Once the regular season begins on March 27, we'll introduce a premium tier with the full daily digest and NPB intelligence reports. Free subscribers will still receive weekly summaries and selected picks.

## The Honest Truth

Our model currently shows approximately 53% accuracy on backtested data. That's modest — and I'm telling you upfront because I respect your intelligence. Most prediction services claim 60-70% accuracy with zero proof. We're starting with honest numbers and a commitment to improve.

53% doesn't sound exciting, but here's what matters: consistent edge over the market is what generates returns over a full season. We're not trying to go 10-0 in a week. We're trying to find the 2-3% of edge that compounds across 2,430 games.

And with real-time data from the live season — actual injury reports, weather conditions, and market movements — the model will sharpen significantly beyond what backtesting shows.

## A Note on Who I Am

I'm a 32-year-old AI engineer based in Tokyo. I've spent my career building machine learning systems. Baseball has been my obsession since childhood — both NPB and MLB. Moneyball Dojo is the project where those two worlds finally converge.

I'm not a professional gambler. I'm not a sports media personality. I'm an engineer who got frustrated that nobody was bringing Japanese baseball intelligence into the American prediction market. So I built it myself.

This newsletter is the public version of that work. You'll see the engine running in real time. You'll see the wins and the losses. And you'll get a perspective on baseball that you literally cannot get from anyone based in America.

Welcome to the Dojo. Let's train together.

---

*Moneyball Dojo launches with the 2026 MLB season on March 27. Subscribe now to get preseason content and be ready for Opening Day.*
