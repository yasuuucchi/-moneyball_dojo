"""
Moneyball Dojo Daily Digest Generator
=====================================

Generates daily digest articles covering all games for the day, with both
English (Substack) and Japanese (note.com) versions.

The digest consolidates predictions from the AI model into one digestible
article per platform, featuring a summary table of all picks, detailed
analysis of the day's best matchup, quick takes on other games, and
performance recaps.

Key Features:
- Single unified daily digest (not 10-15 individual articles)
- Bilingual support (English & Japanese)
- Edge calculations for strategic picks
- Japanese AI Engineer persona integration
- Claude API prompt templates included
- Production-ready markdown output
"""

from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import json


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class GamePrediction:
    """A single game's prediction data"""
    game_id: str
    date: str  # YYYY-MM-DD
    time: str  # HH:MM EST
    home_team: str
    away_team: str
    league: str  # MLB, NFL, etc.
    
    # Model prediction
    win_prob: float  # 0.0-1.0
    confidence_tier: str  # STRONG, MODERATE, WEAK
    pick: str  # HOME, AWAY, OVER, UNDER, PASS
    edge: float  # Model prob - implied prob (%)
    
    # Context
    reasoning: str  # Brief explanation
    featured: bool = False  # Is this the featured matchup?


@dataclass
class DayResults:
    """Yesterday's betting results"""
    date: str  # YYYY-MM-DD
    record: str  # e.g., "3-1" (wins-losses)
    roi: float  # Daily ROI as %
    cumulative_roi: float  # Season ROI as %
    total_picks_season: int


@dataclass
class SeasonStats:
    """Cumulative season statistics"""
    wins: int
    losses: int
    pushes: int
    win_rate: float  # 0.0-1.0
    roi: float  # %
    clv: float  # Average customer lifetime value ($)


# =============================================================================
# Helper Functions
# =============================================================================

def calculate_implied_probability(american_odds: int) -> float:
    """Convert American odds to implied probability"""
    if american_odds < 0:
        return abs(american_odds) / (abs(american_odds) + 100)
    else:
        return 100 / (american_odds + 100)


def format_percentage(value: float) -> str:
    """Format a decimal as percentage string"""
    return f"{value * 100:.1f}%" if isinstance(value, float) and 0 <= value <= 1 else f"{value:.1f}%"


def format_edge(edge: float) -> str:
    """Format edge value with color/emoji hint"""
    if edge > 0.05:
        return f"+{edge*100:.1f}% (strong edge)"
    elif edge > 0.02:
        return f"+{edge*100:.1f}% (positive edge)"
    elif edge >= 0:
        return f"{edge*100:.1f}% (neutral)"
    else:
        return f"{edge*100:.1f}% (negative)"


def get_emoji_for_tier(tier: str) -> str:
    """Get emoji for confidence tier"""
    return {
        "STRONG": "🔥",
        "MODERATE": "👍",
        "WEAK": "⚠️",
    }.get(tier, "🎯")


def sort_by_edge(predictions: List[GamePrediction]) -> List[GamePrediction]:
    """Sort predictions by edge descending, with PASS moves to bottom"""
    non_pass = [p for p in predictions if p.pick != "PASS"]
    pass_picks = [p for p in predictions if p.pick == "PASS"]
    return sorted(non_pass, key=lambda x: x.edge, reverse=True) + pass_picks


# =============================================================================
# English Digest Generator
# =============================================================================

def generate_english_digest(
    predictions: List[GamePrediction],
    results: DayResults,
    season_stats: SeasonStats,
    model_version: str = "v2.1",
) -> str:
    """
    Generate English (Substack) daily digest.
    
    Args:
        predictions: List of GamePrediction objects for the day
        results: Yesterday's results
        season_stats: Cumulative season statistics
        model_version: Current model version
    
    Returns:
        Markdown string ready for Substack
    """
    
    # Find featured matchup (best edge among non-PASS picks)
    non_pass = [p for p in predictions if p.pick != "PASS"]
    featured = max(non_pass, key=lambda x: x.edge) if non_pass else None
    
    # Sort predictions by edge
    sorted_preds = sort_by_edge(predictions)
    
    # Build markdown
    md = []
    md.append(f"# Moneyball Dojo Daily Digest - {results.date}")
    md.append("")
    md.append("## Today's AI Model Picks")
    md.append("")
    
    # Games Table
    md.append("| League | Matchup | Pick | Win Prob | Edge | Confidence |")
    md.append("|--------|---------|------|----------|------|------------|")
    
    for pred in sorted_preds:
        matchup = f"{pred.away_team} @ {pred.home_team}"
        pick_display = f"**{pred.pick}**" if pred.pick != "PASS" else "PASS"
        edge_display = f"{format_edge(pred.edge)}" if pred.edge >= 0 else f"{pred.edge*100:.1f}%"
        emoji = get_emoji_for_tier(pred.confidence_tier)
        
        md.append(
            f"| {pred.league} | {matchup} | {pick_display} | "
            f"{format_percentage(pred.win_prob)} | {edge_display} | "
            f"{emoji} {pred.confidence_tier} |"
        )
    
    md.append("")
    
    # Featured Matchup Deep Dive
    if featured:
        md.append(f"## Featured Matchup: {featured.away_team} @ {featured.home_team}")
        md.append("")
        md.append(f"**Model Confidence:** {featured.confidence_tier}")
        md.append(f"**Pick:** {featured.pick}")
        md.append(f"**Win Probability:** {format_percentage(featured.win_prob)}")
        md.append(f"**Edge vs Market:** {format_edge(featured.edge)}")
        md.append("")
        md.append("### Analysis")
        md.append("")
        md.append(featured.reasoning)
        md.append("")
        md.append(
            "The model identifies strong value in this matchup with a significant positive edge. "
            "This aligns with our updated feature set that factors in recent form momentum, "
            "weather conditions, and historical head-to-head data. Monitor pregame developments."
        )
        md.append("")
    
    # Quick Takes on other games
    quick_picks = [p for p in sorted_preds if p != featured and p.pick != "PASS"][:3]
    
    if quick_picks:
        md.append("## Quick Takes")
        md.append("")
        
        for pred in quick_picks:
            md.append(f"**{pred.away_team} @ {pred.home_team}** (PICK: {pred.pick})")
            md.append(f"{pred.reasoning[:150]}...")
            md.append("")
    
    # Yesterday's Results
    md.append("## Yesterday's Results")
    md.append("")
    md.append(f"Record: **{results.record}**")
    md.append(f"Daily ROI: **{results.roi:+.2f}%**")
    md.append(f"Cumulative ROI: **{results.cumulative_roi:+.2f}%**")
    md.append("")
    
    # Season Record
    total_games = season_stats.wins + season_stats.losses + season_stats.pushes
    md.append("## Season Record")
    md.append("")
    md.append(f"**{season_stats.wins}W - {season_stats.losses}L - {season_stats.pushes}P**")
    md.append(f"Win Rate: {format_percentage(season_stats.win_rate)}")
    md.append(f"ROI: {season_stats.roi:+.2f}%")
    md.append("")
    
    # Model info
    md.append("---")
    md.append("")
    md.append(f"*Generated by Moneyball Dojo AI Model ({model_version})*")
    md.append("*Predictions are not financial advice. Gamble responsibly.*")
    
    return "\n".join(md)


# =============================================================================
# Japanese Digest Generator
# =============================================================================

def generate_japanese_digest(
    predictions: List[GamePrediction],
    results: DayResults,
    season_stats: SeasonStats,
    model_version: str = "v2.1",
) -> str:
    """
    Generate Japanese (note.com) daily digest with 日本人エンジニア persona.
    
    Args:
        predictions: List of GamePrediction objects for the day
        results: Yesterday's results
        season_stats: Cumulative season statistics
        model_version: Current model version
    
    Returns:
        Markdown string ready for note.com
    """
    
    # Find featured matchup
    non_pass = [p for p in predictions if p.pick != "PASS"]
    featured = max(non_pass, key=lambda x: x.edge) if non_pass else None
    
    # Sort predictions by edge
    sorted_preds = sort_by_edge(predictions)
    
    # Build markdown
    md = []
    md.append(f"# Moneyball Dojo デイリーダイジェスト - {results.date}")
    md.append("")
    md.append("こんにちは。日本人エンジニアのAIモデルです。")
    md.append("")
    md.append("本日の全試合をデータドリブンで分析しました。")
    md.append("")
    md.append("## 本日のAIモデル予測")
    md.append("")
    
    # Games Table
    md.append("| リーグ | 対戦 | 予測 | 勝率 | エッジ | 信頼度 |")
    md.append("|--------|------|------|------|--------|--------|")
    
    for pred in sorted_preds:
        matchup = f"{pred.away_team} @ {pred.home_team}"
        pick_display = f"**{pred.pick}**" if pred.pick != "PASS" else "見送り"
        edge_display = f"{format_edge(pred.edge)}" if pred.edge >= 0 else f"{pred.edge*100:.1f}%"
        emoji = get_emoji_for_tier(pred.confidence_tier)
        confidence_ja = {
            "STRONG": "強気",
            "MODERATE": "中程度",
            "WEAK": "弱気",
        }.get(pred.confidence_tier, "通常")
        
        md.append(
            f"| {pred.league} | {matchup} | {pick_display} | "
            f"{format_percentage(pred.win_prob)} | {edge_display} | "
            f"{emoji} {confidence_ja} |"
        )
    
    md.append("")
    
    # Featured Matchup
    if featured:
        md.append(f"## 注目の一戦: {featured.away_team} @ {featured.home_team}")
        md.append("")
        md.append(f"**予測:** {featured.pick}")
        md.append(f"**勝率:** {format_percentage(featured.win_prob)}")
        md.append(f"**エッジ:** {format_edge(featured.edge)}")
        md.append("")
        md.append("### 分析")
        md.append("")
        md.append(featured.reasoning)
        md.append("")
        md.append(
            "このマッチアップはマーケット評価に対して有意義なエッジを持っています。"
            "最新のフィーチャーセット（直近フォーム、天候条件、対戦履歴）を統合した"
            "当社のAIモデルが検出した強力なシグナルです。"
        )
        md.append("")
    
    # Quick Takes
    quick_picks = [p for p in sorted_preds if p != featured and p.pick != "PASS"][:3]
    
    if quick_picks:
        md.append("## その他の試合 - クイック分析")
        md.append("")
        
        for pred in quick_picks:
            md.append(f"**{pred.away_team} @ {pred.home_team}** (予測: {pred.pick})")
            md.append(f"{pred.reasoning[:100]}...")
            md.append("")
    
    # Yesterday's Results
    md.append("## 昨日の成績")
    md.append("")
    md.append(f"勝敗: **{results.record}**")
    md.append(f"日次ROI: **{results.roi:+.2f}%**")
    md.append("")
    
    # Season Stats
    md.append("## シーズン成績")
    md.append("")
    md.append(f"**{season_stats.wins}勝 - {season_stats.losses}敗 - {season_stats.pushes}プッシュ**")
    md.append(f"勝率: {format_percentage(season_stats.win_rate)}")
    md.append(f"ROI: {season_stats.roi:+.2f}%")
    md.append("")
    
    # Footer
    md.append("---")
    md.append("")
    md.append(f"*Moneyball Dojo AI Model ({model_version}) による分析*")
    md.append("*当該予測は投資アドバイスではありません。責任あるプレイをお願いします。*")
    
    return "\n".join(md)


# =============================================================================
# Claude API Prompt Templates
# =============================================================================

def get_english_enhancement_prompt(digest_md: str) -> str:
    """
    Claude API prompt to enhance the English digest with narrative flair.
    
    Returns:
        System prompt for Claude API
    """
    return f"""You are a sports analytics writer for a premium sports betting newsletter.
Your task is to enhance the following daily digest with engaging narrative writing
while maintaining analytical rigor. Keep the tables unchanged but improve the prose.

Focus on:
- Clear explanation of why each pick has value
- Contextualization for non-expert readers
- Compelling storytelling around the data
- Maintaining professional credibility

Here is the digest to enhance:

{digest_md}

Please provide an enhanced version maintaining the same structure but with improved prose."""


def get_japanese_enhancement_prompt(digest_md: str) -> str:
    """
    Claude API prompt to enhance the Japanese digest.
    
    Maintains the Japanese AI Engineer persona while improving readability.
    
    Returns:
        System prompt for Claude API
    """
    return f"""あなたは日本の一流スポーツアナリティクス執筆者です。
以下のダイジェストを磨きのかかった文章に改善してください。

要件:
- テーブルは変更しない
- 日本人エンジニアのキャラクターを保持
- わかりやすくかつ専門的な説明
- データドリブンなストーリーテリング

ダイジェスト:

{digest_md}

改善版をお願いします。"""


def get_model_version_prompt() -> str:
    """
    Claude API prompt for generating model version summaries.
    
    Returns:
        System prompt for Claude API
    """
    return """You are documenting a sports prediction AI model upgrade.
Summarize the key improvements in 2-3 sentences for newsletter subscribers.
Focus on impact (better win rate, higher edge detection, etc) not technical details."""


# =============================================================================
# Sample Data Generator
# =============================================================================

def generate_sample_predictions() -> Tuple[List[GamePrediction], DayResults, SeasonStats]:
    """Generate realistic sample data for demonstration"""
    
    predictions = [
        GamePrediction(
            game_id="MLB_2024_01_20_NYY_BOS",
            date="2024-01-20",
            time="19:10 EST",
            home_team="BOS",
            away_team="NYY",
            league="MLB",
            win_prob=0.568,
            confidence_tier="STRONG",
            pick="AWAY",
            edge=0.044,
            reasoning="Yankees' Cole vs Red Sox rotation mismatch. Strong recent offensive metrics plus favorable bullpen setup.",
            featured=True,
        ),
        GamePrediction(
            game_id="MLB_2024_01_20_LAD_SD",
            date="2024-01-20",
            time="19:40 EST",
            home_team="SD",
            away_team="LAD",
            league="MLB",
            win_prob=0.625,
            confidence_tier="MODERATE",
            pick="AWAY",
            edge=0.089,
            reasoning="Dodgers' superior pitching depth and recent form advantage over Padres rotation.",
        ),
        GamePrediction(
            game_id="NFL_2024_01_20_KC_BUF",
            date="2024-01-20",
            time="13:00 EST",
            home_team="KC",
            away_team="BUF",
            league="NFL",
            win_prob=0.520,
            confidence_tier="WEAK",
            pick="PASS",
            edge=-0.002,
            reasoning="Tight matchup with minimal edge. Better opportunities elsewhere.",
        ),
        GamePrediction(
            game_id="MLB_2024_01_20_ATL_MIA",
            date="2024-01-20",
            time="19:35 EST",
            home_team="MIA",
            away_team="ATL",
            league="MLB",
            win_prob=0.545,
            confidence_tier="MODERATE",
            pick="AWAY",
            edge=0.015,
            reasoning="Atlanta's lineup gets favorable matchup vs Miami's bullpen limitations.",
        ),
    ]
    
    results = DayResults(
        date="2024-01-19",
        record="3-1",
        roi=2.35,
        cumulative_roi=8.5,
        total_picks_season=47,
    )
    
    season_stats = SeasonStats(
        wins=32,
        losses=14,
        pushes=1,
        win_rate=0.685,
        roi=8.5,
        clv=245.50,
    )
    
    return predictions, results, season_stats


# =============================================================================
# Demo / Main
# =============================================================================

def print_sample_english_digest():
    """Print sample English digest"""
    predictions, results, season_stats = generate_sample_predictions()
    
    digest = generate_english_digest(
        predictions=predictions,
        results=results,
        season_stats=season_stats,
    )
    
    print("\n" + "="*80)
    print("ENGLISH DIGEST SAMPLE (Substack)")
    print("="*80 + "\n")
    print(digest)
    print("\n" + "="*80 + "\n")


def print_sample_japanese_digest():
    """Print sample Japanese digest"""
    predictions, results, season_stats = generate_sample_predictions()
    
    digest = generate_japanese_digest(
        predictions=predictions,
        results=results,
        season_stats=season_stats,
    )
    
    print("\n" + "="*80)
    print("JAPANESE DIGEST SAMPLE (note.com)")
    print("="*80 + "\n")
    print(digest)
    print("\n" + "="*80 + "\n")


def print_claude_prompt_examples():
    """Print Claude API prompt templates"""
    predictions, results, season_stats = generate_sample_predictions()
    
    digest = generate_english_digest(predictions, results, season_stats)
    
    print("\n" + "="*80)
    print("CLAUDE API PROMPT EXAMPLES")
    print("="*80 + "\n")
    
    print("ENGLISH ENHANCEMENT PROMPT:")
    print("-" * 80)
    print(get_english_enhancement_prompt(digest[:500] + "..."))
    
    print("\n\nJAPANESE ENHANCEMENT PROMPT:")
    print("-" * 80)
    print(get_japanese_enhancement_prompt(digest[:500] + "..."))
    
    print("\n\nMODEL VERSION SUMMARY PROMPT:")
    print("-" * 80)
    print(get_model_version_prompt())
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    print_sample_english_digest()
    print_sample_japanese_digest()
    print_claude_prompt_examples()
    
    # Show integration with sheets_schema
    print("\n" + "="*80)
    print("INTEGRATION WITH SHEETS SCHEMA")
    print("="*80 + "\n")
    print("""
WORKFLOW:
1. Prediction Pipeline → sheets_schema.predictions sheet
2. Daily Digest Generator reads from predictions sheet
3. Generates bilingual digests (English + Japanese)
4. Posts to Substack and note.com
5. Records articles in sheets_schema.articles sheet

KEY FIELDS USED FROM SHEETS:
- predictions.game_id → used to join with games sheet for matchup info
- predictions.win_prob → displayed in table and analysis
- predictions.edge → primary sort/ranking metric
- predictions.confidence_tier → color-coded tier display
- predictions.pick → core recommendation
- predictions.reasoning → quoted in analysis sections
- predictions.model_version → attribution in footer
- predictions.created_at → ensures latest model version only

EXAMPLE PYTHON INTEGRATION:
```python
from sheets_schema_v2 import PredictionRow
from daily_digest_generator import generate_english_digest, GamePrediction

# Read from Google Sheets API
predictions_raw = sheets_client.get_range('predictions')

# Convert to GamePrediction objects
predictions = [
    GamePrediction(
        game_id=row['game_id'],
        win_prob=float(row['win_prob']),
        edge=float(row['edge']),
        # ... other fields
    )
    for row in predictions_raw
]

# Generate digest
digest_md = generate_english_digest(predictions, results, season_stats)

# Post to Substack (API call)
substack_client.publish(digest_md)
```
""")
    print("="*80 + "\n")
