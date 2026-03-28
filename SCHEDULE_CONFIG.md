# Moneyball Dojo - /schedule Setup Guide

## 1. Cloud Prediction Pipeline (/schedule)

Runs on Anthropic cloud infrastructure. Machine does NOT need to be on.

### Setup

```
/schedule
```

Then configure:

- **Name**: `Moneyball Daily Predictions`
- **Schedule**: Daily at 10:00 AM ET (before first pitch at ~1:00 PM ET)
- **Repository**: yasuuucchi/-moneyball_dojo (main branch)
- **Environment variables**:
  - `ANTHROPIC_API_KEY` = your Claude API key
  - `GOOGLE_SHEETS_SPREADSHEET_ID` = your Sheets ID
  - `GOOGLE_SHEETS_CREDENTIALS_JSON` = service account JSON (if using Sheets)

### Prompt

```
Run the Moneyball Dojo daily prediction pipeline.

1. Install dependencies: pip3 install -r requirements.txt
2. Run: python3 run_daily.py
3. Verify output files exist in output/latest/:
   - POST_TO_SUBSTACK.md (English digest)
   - POST_TO_NOTE.md (Japanese digest)
   - POST_TO_X.txt (Twitter morning post)
   - predictions.csv
   - predictions.json
4. Commit output files and push to main branch.
5. Report: number of games, number of STRONG picks, any errors encountered.

If run_daily.py fails, check:
- Is statsapi installed? (pip3 install MLB-StatsAPI)
- Are there games today? (offseason = no games)
- API rate limits? (retry after 30 seconds)
```

### Verify

After creating, test with:
```
/schedule run Moneyball Daily Predictions
```

---

## 2. Desktop Scheduled Task (Computer Use - Auto-Poster)

Runs on your local machine via Desktop app. Machine must be on.

### Prerequisites

1. Claude Desktop app installed
2. Computer Use enabled (Settings > Computer Use > Enable)
3. Screen recording + keyboard permissions granted
4. Environment variables set:
   - `SUBSTACK_EMAIL`, `SUBSTACK_PASSWORD`
   - `NOTE_EMAIL`, `NOTE_PASSWORD`
   - `X_EMAIL`, `X_PASSWORD`
   - `ANTHROPIC_API_KEY`

### Setup in Desktop App

1. Open Desktop app > Schedule tab
2. Create new local task:
   - **Name**: `Moneyball Auto-Post`
   - **Schedule**: Daily at 10:30 AM ET (30 min after prediction pipeline)
   - **Prompt**:
   ```
   Run the auto-poster to publish today's predictions.

   1. First, pull latest from git: git pull origin main
   2. Check that output/latest/ has today's content
   3. Run: python3 post_to_platforms.py
   4. Report results for each platform (Substack, note.com, X)
   ```

### Manual Trigger via Dispatch

From your phone (Claude Code iOS/Android):
```
Post today's predictions to all platforms.
Run: python3 post_to_platforms.py
```

Or for a single platform:
```
Post to Substack only: python3 post_to_platforms.py --platform substack
```

---

## 3. X Midday & Evening Posts

Additional X posts at different times:

### Midday Update (Desktop scheduled task)

- **Name**: `Moneyball X Midday`
- **Schedule**: Daily at 1:00 PM ET (first pitch time)
- **Prompt**:
```
Post the midday X update for Moneyball Dojo.
Read output/latest/POST_TO_X_MIDDAY.txt and post it to X (@MoneyballDojo).
Use Computer Use to open X, log in, and post the content.
```

### Evening Recap (Desktop scheduled task)

- **Name**: `Moneyball X Evening`
- **Schedule**: Daily at 10:30 PM ET (after most games end)
- **Prompt**:
```
Post the evening X recap for Moneyball Dojo.

1. Run: python3 update_daily_results.py (fetch today's results)
2. Read output/latest/POST_TO_X_EVENING.txt and post it to X (@MoneyballDojo).
3. Use Computer Use to open X, log in, and post the content.
```

---

## Full Daily Timeline

| Time (ET) | Task | Infrastructure | Action |
|-----------|------|---------------|--------|
| 10:00 AM | Prediction Pipeline | Cloud (/schedule) | run_daily.py generates predictions |
| 10:30 AM | Auto-Post | Desktop (local) | Substack + note.com + X morning post |
| 1:00 PM | X Midday | Desktop (local) | First-pitch hype post |
| 10:30 PM | Results + X Evening | Desktop (local) | Update results + evening recap |

---

## Troubleshooting

### /schedule not running
- Check: https://claude.ai/code/scheduled
- Verify environment variables are set
- Check logs for errors

### Computer Use failing
- Ensure Desktop app is open and unlocked
- Check screen resolution (1280x800 recommended)
- Ensure browser is installed (Firefox or Chrome)
- Check that credentials environment variables are set

### Content not generated
- Run `python3 run_daily.py` manually to debug
- Check `output/latest/` directory for files
- Check if games are scheduled today (offseason = no content)
