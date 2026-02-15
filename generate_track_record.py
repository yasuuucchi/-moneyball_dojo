#!/usr/bin/env python3
"""
Moneyball Dojo — Track Record Generator
========================================
バックテスト結果からSTRONGピック限定の公開用実績レポートを生成する。

出力:
  - output/track_record_EN.md  (Substack / 公開ページ用)
  - output/track_record_JA.md  (note.com 用)
  - output/track_record_data.json (データ連携用)

使い方:
  python3 generate_track_record.py
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from collections import defaultdict

PROJECT_DIR = Path(__file__).parent
BACKTEST_DIR = PROJECT_DIR / "output" / "backtest_2025"
OUTPUT_DIR = PROJECT_DIR / "output"


def load_backtest_data():
    """バックテストデータを読み込む"""
    summary_path = BACKTEST_DIR / "backtest_summary.json"
    csv_path = BACKTEST_DIR / "backtest_per_game.csv"

    with open(summary_path, 'r') as f:
        summary = json.load(f)

    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.to_period('M')

    return summary, df


def calculate_roi(wins, total, odds=-110):
    """勝敗数からROIを計算（-110スタンダードジュース前提）"""
    if total == 0:
        return 0.0
    losses = total - wins
    # At -110: risk 1 unit, profit = 100/110 = 0.909 per win
    payout_per_win = 100 / abs(odds) if odds < 0 else odds / 100
    profit = wins * payout_per_win - losses * 1.0
    return profit / total * 100  # ROI as percentage


def compute_strong_stats(df, summary):
    """全モデルのSTRONGピック実績を計算"""
    stats = {}

    # Classifier models
    for model_name in ['Moneyline', 'Run Line', 'F5 Moneyline']:
        model_df = df[(df['model'] == model_name) & (df['confidence'] == 'STRONG')]
        if len(model_df) == 0:
            continue

        wins = int(model_df['correct'].sum())
        total = len(model_df)
        accuracy = wins / total

        # Monthly breakdown
        monthly = []
        for month, grp in model_df.groupby('month'):
            m_wins = int(grp['correct'].sum())
            m_total = len(grp)
            monthly.append({
                'month': str(month),
                'wins': m_wins,
                'total': m_total,
                'accuracy': m_wins / m_total,
                'roi': calculate_roi(m_wins, m_total),
            })

        stats[model_name] = {
            'wins': wins,
            'total': total,
            'accuracy': accuracy,
            'roi': calculate_roi(wins, total),
            'monthly': monthly,
        }

    # Over/Under (uses correct_8_5 field)
    ou_df = df[(df['model'] == 'Over/Under') & (df['confidence'] == 'STRONG')]
    if len(ou_df) > 0:
        wins = int(ou_df['correct_8_5'].sum())
        total = len(ou_df)
        accuracy = wins / total

        monthly = []
        for month, grp in ou_df.groupby('month'):
            m_wins = int(grp['correct_8_5'].sum())
            m_total = len(grp)
            monthly.append({
                'month': str(month),
                'wins': m_wins,
                'total': m_total,
                'accuracy': m_wins / m_total,
                'roi': calculate_roi(m_wins, m_total),
            })

        stats['Over/Under'] = {
            'wins': wins,
            'total': total,
            'accuracy': accuracy,
            'roi': calculate_roi(wins, total),
            'monthly': monthly,
        }

    return stats


def generate_english_track_record(stats, summary):
    """英語版トラックレコードページを生成"""
    md = []
    md.append("# Moneyball Dojo — Verified Track Record")
    md.append("")
    md.append("## 2025 Season Results (STRONG Picks Only)")
    md.append("")
    md.append("All results below are from our **walk-forward backtest** on the full 2025 MLB season.")
    md.append("Models were trained on 2022-2024 data only. No lookahead bias. No cherry-picking.")
    md.append("")
    md.append("---")
    md.append("")

    # Summary table
    md.append("## Performance Summary")
    md.append("")
    md.append("| Market | Record | Win Rate | ROI | Games |")
    md.append("|--------|--------|----------|-----|-------|")

    display_order = ['Moneyline', 'Run Line', 'Over/Under', 'F5 Moneyline']
    for name in display_order:
        s = stats.get(name)
        if not s:
            continue
        record = f"{s['wins']}W - {s['total'] - s['wins']}L"
        md.append(f"| {name} | {record} | **{s['accuracy']:.1%}** | "
                  f"**{s['roi']:+.1f}%** | {s['total']} |")

    md.append("")

    # Combined stats
    total_wins = sum(s['wins'] for s in stats.values())
    total_games = sum(s['total'] for s in stats.values())
    combined_acc = total_wins / total_games if total_games > 0 else 0
    combined_roi = calculate_roi(total_wins, total_games)
    md.append(f"> **Combined STRONG picks: {total_wins}W - {total_games - total_wins}L "
              f"({combined_acc:.1%}) | ROI: {combined_roi:+.1f}%**")
    md.append("")
    md.append("---")
    md.append("")

    # Detailed breakdown per model
    for name in display_order:
        s = stats.get(name)
        if not s:
            continue

        md.append(f"## {name} — STRONG Picks Detail")
        md.append("")

        # Full season summary from backtest_summary.json
        model_summary = summary.get('models', {}).get(name, {})
        total_all = model_summary.get('total_games', 0)

        if name == 'Over/Under':
            mae = model_summary.get('mae', 0)
            md.append(f"- **Overall MAE:** {mae:.3f} runs on {total_all} games")
        else:
            overall_acc = model_summary.get('overall_accuracy', 0)
            auc = model_summary.get('auc', 0)
            md.append(f"- **Overall model accuracy (all tiers):** {overall_acc:.1%} on {total_all} games")
            if auc:
                md.append(f"- **AUC-ROC:** {auc:.4f}")
        md.append(f"- **STRONG picks accuracy:** {s['accuracy']:.1%} ({s['wins']}/{s['total']})")
        md.append(f"- **STRONG picks ROI:** {s['roi']:+.1f}% (at standard -110 juice)")
        md.append("")

        # Monthly trend
        md.append("### Monthly Trend")
        md.append("")
        md.append("| Month | Record | Win Rate | ROI |")
        md.append("|-------|--------|----------|-----|")
        for m in s['monthly']:
            losses = m['total'] - m['wins']
            md.append(f"| {m['month']} | {m['wins']}W-{losses}L | "
                      f"{m['accuracy']:.1%} | {m['roi']:+.1f}% |")
        md.append("")

    md.append("---")
    md.append("")

    # Methodology
    md.append("## Methodology")
    md.append("")
    md.append("- **Algorithm:** XGBoost (gradient boosted decision trees)")
    md.append("- **Training data:** 2022-2024 MLB seasons (7,283 games)")
    md.append("- **Test data:** 2025 full season (2,426 games)")
    md.append("- **Walk-forward validation:** Rolling stats use only pre-game data")
    md.append("- **No data leakage:** 2025 data was never used in training")
    md.append("- **Confidence tiers:** STRONG picks require model probability "
              "significantly away from 50%")
    md.append("- **ROI calculation:** Assumes standard -110 American odds (1.909 decimal)")
    md.append("")

    # What is a STRONG pick
    md.append("### What Makes a STRONG Pick?")
    md.append("")
    md.append("Our AI model assigns confidence tiers based on the strength of its signal:")
    md.append("")
    md.append("| Tier | Criteria | 2025 ML Accuracy |")
    md.append("|------|----------|-----------------|")

    ml_tiers = summary.get('models', {}).get('Moneyline', {}).get('tiers', {})
    for tier_name, criteria in [('STRONG', 'High conviction signal'),
                                 ('MODERATE', 'Moderate conviction'),
                                 ('LEAN', 'Slight edge detected'),
                                 ('PASS', 'No actionable edge')]:
        tier_data = ml_tiers.get(tier_name, {})
        acc = tier_data.get('accuracy', 0)
        total = tier_data.get('total', 0)
        md.append(f"| {tier_name} | {criteria} | {acc:.1%} ({total} games) |")

    md.append("")
    md.append("We only publish STRONG picks — maximizing quality over quantity.")
    md.append("")
    md.append("---")
    md.append("")
    md.append(f"*Last updated: {datetime.now().strftime('%Y-%m-%d')}*")
    md.append("*Past performance does not guarantee future results. Not financial advice.*")

    return "\n".join(md)


def generate_japanese_track_record(stats, summary):
    """日本語版トラックレコードページを生成"""
    md = []
    md.append("# Moneyball Dojo — 検証済み実績レポート")
    md.append("")
    md.append("## 2025シーズン実績（STRONGピック限定）")
    md.append("")
    md.append("以下は、2025年MLBシーズン全試合に対する**ウォークフォワード・バックテスト**の結果です。")
    md.append("モデルは2022-2024年のデータのみで訓練。先読みバイアスなし。チェリーピッキングなし。")
    md.append("")
    md.append("---")
    md.append("")

    # Summary table
    md.append("## パフォーマンスサマリー")
    md.append("")
    md.append("| マーケット | 戦績 | 勝率 | ROI | 試合数 |")
    md.append("|-----------|------|------|-----|--------|")

    display_order = ['Moneyline', 'Run Line', 'Over/Under', 'F5 Moneyline']
    name_ja = {
        'Moneyline': 'マネーライン',
        'Run Line': 'ランライン (-1.5)',
        'Over/Under': 'オーバー/アンダー',
        'F5 Moneyline': '前半5回 ML',
    }
    for name in display_order:
        s = stats.get(name)
        if not s:
            continue
        record = f"{s['wins']}勝 {s['total'] - s['wins']}敗"
        md.append(f"| {name_ja.get(name, name)} | {record} | **{s['accuracy']:.1%}** | "
                  f"**{s['roi']:+.1f}%** | {s['total']} |")

    md.append("")

    # Combined stats
    total_wins = sum(s['wins'] for s in stats.values())
    total_games = sum(s['total'] for s in stats.values())
    combined_acc = total_wins / total_games if total_games > 0 else 0
    combined_roi = calculate_roi(total_wins, total_games)
    md.append(f"> **STRONG全体: {total_wins}勝 {total_games - total_wins}敗 "
              f"({combined_acc:.1%}) | ROI: {combined_roi:+.1f}%**")
    md.append("")
    md.append("---")
    md.append("")

    # Detailed breakdown per model
    for name in display_order:
        s = stats.get(name)
        if not s:
            continue

        md.append(f"## {name_ja.get(name, name)} — STRONG詳細")
        md.append("")

        model_summary = summary.get('models', {}).get(name, {})
        total_all = model_summary.get('total_games', 0)

        if name == 'Over/Under':
            mae = model_summary.get('mae', 0)
            md.append(f"- **全体MAE:** {mae:.3f}点（{total_all}試合）")
        else:
            overall_acc = model_summary.get('overall_accuracy', 0)
            auc = model_summary.get('auc', 0)
            md.append(f"- **全体精度（全ティア）:** {overall_acc:.1%}（{total_all}試合）")
            if auc:
                md.append(f"- **AUC-ROC:** {auc:.4f}")
        md.append(f"- **STRONGピック精度:** {s['accuracy']:.1%}（{s['wins']}/{s['total']}）")
        md.append(f"- **STRONG ROI:** {s['roi']:+.1f}%（標準-110ジュース前提）")
        md.append("")

        # Monthly trend
        md.append("### 月次推移")
        md.append("")
        md.append("| 月 | 戦績 | 勝率 | ROI |")
        md.append("|-----|------|------|-----|")
        for m in s['monthly']:
            losses = m['total'] - m['wins']
            md.append(f"| {m['month']} | {m['wins']}勝{losses}敗 | "
                      f"{m['accuracy']:.1%} | {m['roi']:+.1f}% |")
        md.append("")

    md.append("---")
    md.append("")

    # Methodology
    md.append("## 方法論")
    md.append("")
    md.append("- **アルゴリズム:** XGBoost（勾配ブースティング決定木）")
    md.append("- **訓練データ:** 2022-2024年MLBシーズン（7,283試合）")
    md.append("- **テストデータ:** 2025年フルシーズン（2,426試合）")
    md.append("- **ウォークフォワード検証:** ローリングスタッツは試合前データのみ使用")
    md.append("- **データリーケージなし:** 2025年データは訓練に一切使用せず")
    md.append("- **信頼度ティア:** STRONGは高確信度シグナルのみ")
    md.append("- **ROI計算:** 標準-110アメリカンオッズ前提")
    md.append("")
    md.append("---")
    md.append("")
    md.append(f"*最終更新: {datetime.now().strftime('%Y-%m-%d')}*")
    md.append("*過去の成績は将来の結果を保証するものではありません。投資アドバイスではありません。*")

    return "\n".join(md)


def generate_track_record_data(stats, summary):
    """データ連携用JSON（ダイジェスト埋め込み用）"""
    data = {
        'generated': datetime.now().isoformat(),
        'season': 2025,
        'strong_picks': {},
    }

    for name, s in stats.items():
        data['strong_picks'][name] = {
            'wins': s['wins'],
            'losses': s['total'] - s['wins'],
            'total': s['total'],
            'accuracy': round(s['accuracy'], 4),
            'roi': round(s['roi'], 1),
        }

    total_wins = sum(s['wins'] for s in stats.values())
    total_games = sum(s['total'] for s in stats.values())
    data['combined'] = {
        'wins': total_wins,
        'losses': total_games - total_wins,
        'total': total_games,
        'accuracy': round(total_wins / total_games, 4) if total_games > 0 else 0,
        'roi': round(calculate_roi(total_wins, total_games), 1),
    }

    return data


def main():
    print("=" * 60)
    print("MONEYBALL DOJO — TRACK RECORD GENERATOR")
    print("=" * 60)
    print()

    summary, df = load_backtest_data()
    print(f"Loaded {len(df)} backtest records")

    stats = compute_strong_stats(df, summary)
    print(f"Computed STRONG stats for {len(stats)} models")
    print()

    for name, s in stats.items():
        print(f"  {name}: {s['accuracy']:.1%} ({s['wins']}/{s['total']}) ROI: {s['roi']:+.1f}%")
    print()

    # Generate English report
    en_report = generate_english_track_record(stats, summary)
    en_path = OUTPUT_DIR / "track_record_EN.md"
    en_path.write_text(en_report, encoding='utf-8')
    print(f"English track record: {en_path}")

    # Generate Japanese report
    ja_report = generate_japanese_track_record(stats, summary)
    ja_path = OUTPUT_DIR / "track_record_JA.md"
    ja_path.write_text(ja_report, encoding='utf-8')
    print(f"Japanese track record: {ja_path}")

    # Generate data JSON
    data = generate_track_record_data(stats, summary)
    data_path = OUTPUT_DIR / "track_record_data.json"
    data_path.write_text(json.dumps(data, indent=2), encoding='utf-8')
    print(f"Track record data: {data_path}")

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == '__main__':
    main()
