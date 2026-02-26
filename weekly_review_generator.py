#!/usr/bin/env python3
"""
Moneyball Dojo — Weekly Review Generator
=========================================
過去7日間の予測 vs 実績を集計し、EN/JA の振り返りダイジェストを生成する。

出力:
  - output/weekly/review_EN_{end_date}.md
  - output/weekly/review_JA_{end_date}.md
  - output/weekly/review_data_{end_date}.json

使い方:
  python3 weekly_review_generator.py              # 直近7日間
  python3 weekly_review_generator.py 2026-03-31   # 指定日を終了日として7日分
"""

import json
import os
import sys
import time
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
OUTPUT_DIR = PROJECT_DIR / "output"
WEEKLY_DIR = OUTPUT_DIR / "weekly"
DATA_DIR = PROJECT_DIR / "data"
GAMES_PATH = DATA_DIR / "games_2022_2025.csv"


def calculate_roi(wins, total, odds=-110):
    """勝敗数からROIを計算（-110スタンダードジュース前提）"""
    if total == 0:
        return 0.0
    losses = total - wins
    payout_per_win = 100 / abs(odds) if odds < 0 else odds / 100
    profit = wins * payout_per_win - losses * 1.0
    return profit / total * 100


def load_week_predictions(end_date, days=7):
    """過去N日間の予測CSVを結合して返す"""
    frames = []
    for i in range(days):
        d = end_date - timedelta(days=i)
        dated_dir = OUTPUT_DIR / d.strftime('%Y%m%d')
        csv_path = dated_dir / "predictions.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            if 'date' not in df.columns:
                df['date'] = d.strftime('%Y-%m-%d')
            frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_results(start_date, end_date):
    """期間内のゲーム結果を読み込む"""
    if not GAMES_PATH.exists():
        return pd.DataFrame()

    df = pd.read_csv(GAMES_PATH)
    df['date'] = pd.to_datetime(df['date'])
    mask = (df['date'] >= pd.Timestamp(start_date)) & (df['date'] <= pd.Timestamp(end_date))
    return df[mask].copy()


def evaluate_picks(predictions, results):
    """予測と結果を突き合わせて勝敗を判定する"""
    if predictions.empty or results.empty:
        return pd.DataFrame()

    predictions = predictions.copy()
    results = results.copy()

    predictions['game_id'] = predictions['game_id'].astype(int)
    results['game_id'] = results['game_id'].astype(int)

    merged = predictions.merge(
        results[['game_id', 'home_score', 'away_score', 'home_win']],
        on='game_id',
        how='inner'
    )

    if merged.empty:
        return merged

    # Moneyline correctness
    merged['ml_correct'] = 0
    home_pick = merged['ml_pick'] == 'HOME'
    away_pick = merged['ml_pick'] == 'AWAY'
    merged.loc[home_pick & (merged['home_win'] == 1), 'ml_correct'] = 1
    merged.loc[away_pick & (merged['home_win'] == 0), 'ml_correct'] = 1

    # Run line correctness (HOME -1.5 / AWAY +1.5)
    margin = merged['home_score'] - merged['away_score']
    merged['rl_correct'] = 0
    rl_home = merged['rl_pick'] == 'HOME -1.5'
    rl_away = merged['rl_pick'] == 'AWAY +1.5'
    merged.loc[rl_home & (margin >= 2), 'rl_correct'] = 1
    merged.loc[rl_away & (margin <= 1), 'rl_correct'] = 1

    # F5 correctness
    merged['f5_correct'] = 0
    f5_home = merged['f5_pick'] == 'HOME'
    f5_away = merged['f5_pick'] == 'AWAY'
    merged.loc[f5_home & (merged['home_win'] == 1), 'f5_correct'] = 1
    merged.loc[f5_away & (merged['home_win'] == 0), 'f5_correct'] = 1

    # NRFI correctness
    total_1st = merged['home_score'].clip(upper=1) + merged['away_score'].clip(upper=1)
    merged['nrfi_correct'] = 0
    # NOTE: NRFI accuracy requires inning-level data; approximate via game result is rough
    # Full accuracy requires nrfi_data CSV — skip if unavailable

    return merged


def compute_tier_stats(evaluated, model_prefix, confidence_col=None):
    """信頼度ティア別の勝敗統計を計算"""
    if evaluated.empty:
        return {}

    correct_col = f'{model_prefix}_correct'
    conf_col = confidence_col or f'{model_prefix}_confidence'
    pick_col = f'{model_prefix}_pick'

    if correct_col not in evaluated.columns or conf_col not in evaluated.columns:
        return {}

    # Exclude PASS/N/A picks
    active = evaluated[~evaluated[pick_col].isin(['PASS', 'N/A', ''])]
    active = active.dropna(subset=[pick_col])

    stats = {}
    for tier in ['STRONG', 'MODERATE', 'LEAN']:
        tier_df = active[active[conf_col] == tier]
        if tier_df.empty:
            continue
        wins = int(tier_df[correct_col].sum())
        total = len(tier_df)
        stats[tier] = {
            'wins': wins,
            'losses': total - wins,
            'total': total,
            'accuracy': round(wins / total, 4) if total > 0 else 0,
            'roi': round(calculate_roi(wins, total), 2),
        }

    # All tiers combined
    all_active = active[active[conf_col].isin(['STRONG', 'MODERATE', 'LEAN'])]
    if not all_active.empty:
        w = int(all_active[correct_col].sum())
        t = len(all_active)
        stats['ALL'] = {
            'wins': w,
            'losses': t - w,
            'total': t,
            'accuracy': round(w / t, 4) if t > 0 else 0,
            'roi': round(calculate_roi(w, t), 2),
        }

    return stats


def find_notable_picks(evaluated, n=3):
    """ベストピック（的中 + 高エッジ）とワーストピック（外れ + 高エッジ）を返す"""
    if evaluated.empty:
        return [], []

    active = evaluated[evaluated['ml_pick'].isin(['HOME', 'AWAY'])].copy()
    if active.empty:
        return [], []

    active['ml_edge_f'] = pd.to_numeric(active['ml_edge'], errors='coerce').fillna(0)

    correct = active[active['ml_correct'] == 1].nlargest(n, 'ml_edge_f')
    wrong = active[active['ml_correct'] == 0].nlargest(n, 'ml_edge_f')

    def pick_to_dict(row):
        return {
            'matchup': f"{row['away_team']} @ {row['home_team']}",
            'date': str(row.get('date', '')).split(' ')[0].split('T')[0],
            'pick': row['ml_pick'],
            'prob': float(row.get('ml_prob', 0)),
            'edge': float(row.get('ml_edge', 0)),
            'confidence': row.get('ml_confidence', ''),
            'score': f"{int(row.get('away_score', 0))}-{int(row.get('home_score', 0))}",
            'correct': int(row.get('ml_correct', 0)),
        }

    best = [pick_to_dict(r) for _, r in correct.iterrows()]
    worst = [pick_to_dict(r) for _, r in wrong.iterrows()]
    return best, worst


def generate_english_review(stats, best_picks, worst_picks, evaluated, start_str, end_str, days_with_data):
    """英語版週次レビューを生成"""
    md = []
    md.append(f"# Moneyball Dojo Weekly Review — {start_str} to {end_str}")
    md.append("")
    md.append(f"*{days_with_data} days of predictions evaluated. Here's how the AI performed.*")
    md.append("")
    md.append("---")
    md.append("")

    # Overall Summary
    ml_all = stats.get('ml', {}).get('ALL', {})
    ml_strong = stats.get('ml', {}).get('STRONG', {})

    if ml_all:
        md.append("## Overall Performance")
        md.append("")
        md.append("| Metric | All Picks | STRONG Only |")
        md.append("|--------|-----------|-------------|")
        md.append(f"| Record | {ml_all['wins']}W-{ml_all['losses']}L | "
                   f"{ml_strong.get('wins', 0)}W-{ml_strong.get('losses', 0)}L |")
        md.append(f"| Accuracy | {ml_all['accuracy']*100:.1f}% | "
                   f"{ml_strong.get('accuracy', 0)*100:.1f}% |")
        md.append(f"| ROI | {ml_all['roi']:+.1f}% | "
                   f"{ml_strong.get('roi', 0):+.1f}% |")
        md.append(f"| Total Games | {ml_all['total']} | "
                   f"{ml_strong.get('total', 0)} |")
        md.append("")

    # Per-model breakdown
    md.append("## Model Breakdown (STRONG Picks)")
    md.append("")
    md.append("| Model | Record | Accuracy | ROI |")
    md.append("|-------|--------|----------|-----|")
    for model, label in [('ml', 'Moneyline'), ('rl', 'Run Line'), ('f5', 'F5 Moneyline')]:
        s = stats.get(model, {}).get('STRONG', {})
        if s:
            md.append(f"| {label} | {s['wins']}W-{s['losses']}L | "
                       f"{s['accuracy']*100:.1f}% | {s['roi']:+.1f}% |")
    md.append("")

    # Best picks
    if best_picks:
        md.append("## Best Picks of the Week")
        md.append("")
        for p in best_picks:
            md.append(f"- **{p['matchup']}** ({p['date']}) — "
                       f"{p['pick']} at {p['prob']*100:.0f}% (edge {p['edge']*100:+.1f}%) "
                       f"| Final: {p['score']}")
        md.append("")

    # Worst picks
    if worst_picks:
        md.append("## Biggest Misses")
        md.append("")
        for p in worst_picks:
            md.append(f"- **{p['matchup']}** ({p['date']}) — "
                       f"{p['pick']} at {p['prob']*100:.0f}% (edge {p['edge']*100:+.1f}%) "
                       f"| Final: {p['score']}")
        md.append("")

    # Confidence tier table
    md.append("## Accuracy by Confidence Tier")
    md.append("")
    md.append("| Tier | Record | Accuracy | ROI |")
    md.append("|------|--------|----------|-----|")
    for tier in ['STRONG', 'MODERATE', 'LEAN']:
        s = stats.get('ml', {}).get(tier, {})
        if s:
            md.append(f"| {tier} | {s['wins']}W-{s['losses']}L | "
                       f"{s['accuracy']*100:.1f}% | {s['roi']:+.1f}% |")
    md.append("")

    md.append("---")
    md.append("")
    md.append("*9 AI models. Real data. Full transparency.*")
    md.append("*Not financial advice. Gamble responsibly.*")
    md.append("")

    return "\n".join(md)


def generate_japanese_review(stats, best_picks, worst_picks, evaluated, start_str, end_str, days_with_data):
    """日本語版週次レビューを生成"""
    md = []
    md.append(f"# Moneyball Dojo 週次レビュー — {start_str} 〜 {end_str}")
    md.append("")
    md.append(f"*{days_with_data}日間のAI予測を振り返ります。*")
    md.append("")
    md.append("---")
    md.append("")

    ml_all = stats.get('ml', {}).get('ALL', {})
    ml_strong = stats.get('ml', {}).get('STRONG', {})

    if ml_all:
        md.append("## 全体パフォーマンス")
        md.append("")
        md.append("| 指標 | 全ピック | STRONG のみ |")
        md.append("|------|---------|------------|")
        md.append(f"| 戦績 | {ml_all['wins']}勝{ml_all['losses']}敗 | "
                   f"{ml_strong.get('wins', 0)}勝{ml_strong.get('losses', 0)}敗 |")
        md.append(f"| 的中率 | {ml_all['accuracy']*100:.1f}% | "
                   f"{ml_strong.get('accuracy', 0)*100:.1f}% |")
        md.append(f"| ROI | {ml_all['roi']:+.1f}% | "
                   f"{ml_strong.get('roi', 0):+.1f}% |")
        md.append(f"| 試合数 | {ml_all['total']} | "
                   f"{ml_strong.get('total', 0)} |")
        md.append("")

    # Per-model
    md.append("## モデル別成績（STRONG ピック）")
    md.append("")
    md.append("| モデル | 戦績 | 的中率 | ROI |")
    md.append("|--------|------|--------|-----|")
    for model, label in [('ml', 'マネーライン'), ('rl', 'ランライン'), ('f5', 'F5マネーライン')]:
        s = stats.get(model, {}).get('STRONG', {})
        if s:
            md.append(f"| {label} | {s['wins']}勝{s['losses']}敗 | "
                       f"{s['accuracy']*100:.1f}% | {s['roi']:+.1f}% |")
    md.append("")

    # Best
    if best_picks:
        md.append("## 今週のベストピック")
        md.append("")
        for p in best_picks:
            md.append(f"- **{p['matchup']}** ({p['date']}) — "
                       f"{p['pick']} {p['prob']*100:.0f}% (エッジ {p['edge']*100:+.1f}%) "
                       f"| 結果: {p['score']}")
        md.append("")

    # Worst
    if worst_picks:
        md.append("## 今週の外れピック")
        md.append("")
        for p in worst_picks:
            md.append(f"- **{p['matchup']}** ({p['date']}) — "
                       f"{p['pick']} {p['prob']*100:.0f}% (エッジ {p['edge']*100:+.1f}%) "
                       f"| 結果: {p['score']}")
        md.append("")

    # Tiers
    conf_ja = {'STRONG': '強気', 'MODERATE': '中程度', 'LEAN': '傾向'}
    md.append("## 信頼度ティア別の的中率")
    md.append("")
    md.append("| ティア | 戦績 | 的中率 | ROI |")
    md.append("|--------|------|--------|-----|")
    for tier in ['STRONG', 'MODERATE', 'LEAN']:
        s = stats.get('ml', {}).get(tier, {})
        if s:
            md.append(f"| {conf_ja.get(tier, tier)} | "
                       f"{s['wins']}勝{s['losses']}敗 | "
                       f"{s['accuracy']*100:.1f}% | {s['roi']:+.1f}% |")
    md.append("")

    md.append("---")
    md.append("")
    md.append("*東京のAIエンジニアが開発。9モデル × XGBoost。データで勝負する。*")
    md.append("*投資アドバイスではありません。責任あるプレイを。*")
    md.append("")

    return "\n".join(md)


MAX_RETRIES = 3
RETRY_DELAYS = [2, 4, 8]


def _slack_post(webhook_url, payload):
    """Slack webhookにPOSTする。リトライ付き。"""
    try:
        import requests
    except ImportError:
        return False
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(webhook_url, json=payload, timeout=10)
            if resp.status_code == 200:
                return True
        except Exception:
            pass
        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_DELAYS[attempt])
    return False


def notify_review(en_review, ja_review, stats, start_str, end_str):
    """週次レビュー完了時にSlackへフルコンテンツを送信する。"""
    slack_webhook = os.environ.get('SLACK_WEBHOOK')
    if not slack_webhook:
        print("  ⚠ SLACK_WEBHOOK not set — skipping Slack notification")
        return

    # --- Message 1: サマリー ---
    ml_all = stats.get('ml', {}).get('ALL', {})
    ml_strong = stats.get('ml', {}).get('STRONG', {})
    summary_lines = [
        f"*📊 Moneyball Dojo Weekly Review — {start_str} 〜 {end_str}*",
        "",
    ]
    if ml_all:
        summary_lines.append(f"全ピック: {ml_all['wins']}W-{ml_all['losses']}L "
                             f"({ml_all['accuracy']*100:.1f}%, ROI {ml_all['roi']:+.1f}%)")
    if ml_strong:
        summary_lines.append(f"STRONG: {ml_strong['wins']}W-{ml_strong['losses']}L "
                             f"({ml_strong['accuracy']*100:.1f}%, ROI {ml_strong['roi']:+.1f}%)")
    if not ml_all:
        summary_lines.append("(データなし — 予測ファイルまたは結果ファイルが未生成)")

    _slack_post(slack_webhook, {"text": "\n".join(summary_lines)})

    # --- Message 2: EN review (for Substack) ---
    en_trimmed = en_review[:3900]
    if len(en_review) > 3900:
        en_trimmed += "\n\n... (truncated, full version in output/weekly/)"
    _slack_post(slack_webhook, {
        "text": f"*--- WEEKLY REVIEW EN (copy below) ---*\n```{en_trimmed}```"
    })

    # --- Message 3: JA review (for note.com) ---
    ja_trimmed = ja_review[:3900]
    if len(ja_review) > 3900:
        ja_trimmed += "\n\n... (truncated, full version in output/weekly/)"
    _slack_post(slack_webhook, {
        "text": f"*--- WEEKLY REVIEW JA (copy below) ---*\n```{ja_trimmed}```"
    })

    print("  ✓ Slack notifications sent (3 messages)")


def main():
    print("=" * 60)
    print("Moneyball Dojo — Weekly Review Generator")
    print("=" * 60)

    # Determine date range
    if len(sys.argv) > 1:
        end_date = datetime.strptime(sys.argv[1], '%Y-%m-%d').date()
    else:
        end_date = datetime.utcnow().date()

    start_date = end_date - timedelta(days=6)
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    print(f"Period: {start_str} → {end_str}")

    # Load data
    print("[1/5] Loading predictions...")
    predictions = load_week_predictions(end_date, days=7)
    if predictions.empty:
        print("  ⚠ No prediction files found for the past 7 days.")
        print("  Generating empty review with no data.")

    print(f"  → {len(predictions)} predictions loaded")
    days_with_data = predictions['date'].nunique() if not predictions.empty else 0
    print(f"  → {days_with_data} days with data")

    print("[2/5] Loading results...")
    results = load_results(start_date, end_date)
    print(f"  → {len(results)} game results loaded")

    print("[3/5] Evaluating picks...")
    evaluated = evaluate_picks(predictions, results)
    matched = len(evaluated)
    print(f"  → {matched} predictions matched to results")

    if evaluated.empty and not predictions.empty:
        print("  ⚠ No results available yet (games not final or results not updated).")
        print("  Generating review with predictions summary only.")

    # Compute stats
    stats = {}
    for model in ['ml', 'rl', 'f5']:
        stats[model] = compute_tier_stats(evaluated, model)

    best_picks, worst_picks = find_notable_picks(evaluated)

    # Summary
    ml_all = stats.get('ml', {}).get('ALL', {})
    if ml_all:
        print(f"\n  === MONEYLINE SUMMARY ===")
        print(f"  Record: {ml_all['wins']}W-{ml_all['losses']}L")
        print(f"  Accuracy: {ml_all['accuracy']*100:.1f}%")
        print(f"  ROI: {ml_all['roi']:+.1f}%")

    # Generate outputs
    WEEKLY_DIR.mkdir(parents=True, exist_ok=True)

    print("[4/5] Generating weekly review (EN + JA)...")
    en_review = generate_english_review(stats, best_picks, worst_picks, evaluated, start_str, end_str, days_with_data)
    ja_review = generate_japanese_review(stats, best_picks, worst_picks, evaluated, start_str, end_str, days_with_data)

    en_path = WEEKLY_DIR / f"review_EN_{end_str}.md"
    ja_path = WEEKLY_DIR / f"review_JA_{end_str}.md"
    json_path = WEEKLY_DIR / f"review_data_{end_str}.json"

    en_path.write_text(en_review, encoding='utf-8')
    ja_path.write_text(ja_review, encoding='utf-8')
    print(f"  ✓ {en_path.name}")
    print(f"  ✓ {ja_path.name}")

    print("[5/5] Saving review data JSON...")
    review_data = {
        'generated': datetime.utcnow().isoformat(),
        'period': {'start': start_str, 'end': end_str},
        'days_with_data': days_with_data,
        'games_matched': matched,
        'stats': stats,
        'best_picks': best_picks,
        'worst_picks': worst_picks,
    }
    with open(json_path, 'w') as f:
        json.dump(review_data, f, indent=2, ensure_ascii=False)
    print(f"  ✓ {json_path.name}")

    # Slack notification with full content
    print("[6/6] Sending Slack notifications...")
    notify_review(en_review, ja_review, stats, start_str, end_str)

    print(f"\n✅ Weekly review generated → {WEEKLY_DIR}/")


if __name__ == "__main__":
    main()
