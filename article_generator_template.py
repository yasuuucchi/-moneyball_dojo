"""
Article Generator for Moneyball Dojo (Production)
==================================================
Anthropic API を使用して predictions.csv から EN/JA Markdown 記事を自動生成する。

使い方:
    # run_daily.py から呼び出される
    from article_generator_template import ArticleGenerator
    generator = ArticleGenerator()
    en_article, ja_article = generator.generate_daily_digest(predictions, target_date)

    # 単体テスト
    python article_generator_template.py
"""

import os
import json
import time
import logging
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class GameData:
    """Game prediction data for article generation."""
    game_id: str
    date: str
    away_team: str
    home_team: str
    away_pitcher: str
    home_pitcher: str
    model_probability: float
    confidence_tier: str
    pick: str
    away_team_name: str = ''
    home_team_name: str = ''
    away_runs_per_game: float = 0.0
    home_runs_per_game: float = 0.0
    away_era: float = 0.0
    home_era: float = 0.0


TEAM_NAMES = {
    'New York Yankees': 'NYY', 'Boston Red Sox': 'BOS',
    'Tampa Bay Rays': 'TB', 'Baltimore Orioles': 'BAL',
    'Toronto Blue Jays': 'TOR', 'New York Mets': 'NYM',
    'Atlanta Braves': 'ATL', 'Washington Nationals': 'WSH',
    'Philadelphia Phillies': 'PHI', 'Miami Marlins': 'MIA',
    'Los Angeles Dodgers': 'LAD', 'San Diego Padres': 'SD',
    'San Francisco Giants': 'SF', 'Arizona Diamondbacks': 'ARI',
    'Colorado Rockies': 'COL', 'Milwaukee Brewers': 'MIL',
    'Chicago Cubs': 'CHC', 'St. Louis Cardinals': 'STL',
    'Pittsburgh Pirates': 'PIT', 'Cincinnati Reds': 'CIN',
    'Houston Astros': 'HOU', 'Los Angeles Angels': 'LAA',
    'Oakland Athletics': 'OAK', 'Seattle Mariners': 'SEA',
    'Texas Rangers': 'TEX', 'Kansas City Royals': 'KC',
    'Minnesota Twins': 'MIN', 'Chicago White Sox': 'CWS',
    'Detroit Tigers': 'DET', 'Cleveland Guardians': 'CLE',
}
# Reverse lookup
TEAM_CODES = {v: k for k, v in TEAM_NAMES.items()}


# ============================================================================
# Article Generator
# ============================================================================

class ArticleGenerator:
    """
    Anthropic API を使用して Moneyball Dojo の日次ダイジェスト記事を生成する。
    API キーが未設定の場合はテンプレートベースのフォールバックを使用。
    """

    SYSTEM_PROMPT = """You are the Moneyball Dojo author — a data-driven baseball analyst and AI engineer based in Tokyo.

Your expertise:
- Advanced MLB analytics (sabermetrics, WAR, FIP, xwOBA, BABIP)
- Predictive modeling with XGBoost (9 models covering all major betting markets)
- Edge calculation: model probability vs market implied probability
- Japanese and American baseball (NPB & MLB)

Writing style:
- Data-first: every claim backed by a number
- Concise but insightful — no filler
- Confident on STRONG picks, transparent about uncertainty
- Witty one-liners welcome, but substance over style
- Markdown format with tables

Track record: Backtested 2025 season — 64.7% moneyline accuracy, +72.9% ROI on STRONG picks."""

    ENGLISH_DIGEST_PROMPT = """Write a Substack Daily Digest article in English based on this prediction data.

## Format Requirements
- Title: "Moneyball Dojo Daily Digest — {date}"
- Open with: "{game_count} games analyzed across {model_count} AI models. Here's what the numbers say."
- Full prediction table (Matchup | Pick | Win Prob | Edge | Confidence)
- "Featured Pick" section for the highest-edge game (3-4 paragraphs of analysis)
- "Quick Takes" for other STRONG/MODERATE picks (1-2 sentences each)
- Over/Under, Run Line, NRFI sections if data available
- Pitcher K props and Batter props tables if data available
- Footer: "Built by a Japanese AI engineer in Tokyo. {model_count} models × XGBoost. Data over gut feelings."
- Disclaimer: "Not financial advice. Gamble responsibly."

## Prediction Data (CSV)
{csv_data}

## Additional Context
{context}

Output ONLY the Markdown article — no preamble or explanation."""

    JAPANESE_DIGEST_PROMPT = """以下の予測データから、note.com 用の日本語 Daily Digest 記事を書いてください。

## フォーマット要件
- タイトル: "Moneyball Dojo デイリーダイジェスト — {date}"
- 冒頭: "こんにちは、Moneyball Dojoです。本日は{game_count}試合を{model_count}つのAIモデルで分析しました。"
- 全試合の予測テーブル（対戦 | 予測 | 勝率 | エッジ | 信頼度）
- 信頼度の日本語表記: STRONG→🔥 強気, MODERATE→👍 中程度, LEAN→→ 傾向, PASS→⏸ 見送り
- 「注目の一戦」セクション（エッジ最大の試合を3-4段落で深掘り）
- 「クイックテイク」でその他のSTRONG/MODERATEピックを紹介
- Over/Under、ランライン、NRFIセクション（データがあれば）
- 投手K予測、打者プロップス（データがあれば）
- フッター: "東京のAIエンジニアが開発。{model_count}モデル × XGBoost。データで勝負する。"
- 免責: "投資アドバイスではありません。責任あるプレイを。"

## 予測データ（CSV）
{csv_data}

## 追加コンテキスト
{context}

Markdown記事のみ出力してください。前置きや説明は不要です。"""

    MAX_RETRIES = 3
    RETRY_DELAYS = [2, 4, 8]  # seconds

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY') or os.environ.get('CLAUDE_API_KEY')
        self.client = None
        self.model = "claude-haiku-4-5-20251001"

        if self.api_key:
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
                logger.info("Anthropic client initialized successfully")
            except ImportError:
                logger.warning("anthropic package not installed — falling back to template mode")
            except Exception as e:
                logger.warning(f"Failed to initialize Anthropic client: {e}")

    @property
    def api_available(self) -> bool:
        return self.client is not None

    # ------------------------------------------------------------------
    # Core API call with retry
    # ------------------------------------------------------------------

    def _call_api(self, system: str, user_prompt: str) -> str:
        """Call Anthropic API with exponential backoff retry."""
        if not self.client:
            raise RuntimeError("Anthropic client not initialized")

        last_error = None
        for attempt in range(self.MAX_RETRIES):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    system=system,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                return response.content[0].text
            except Exception as e:
                last_error = e
                if attempt < self.MAX_RETRIES - 1:
                    delay = self.RETRY_DELAYS[attempt]
                    logger.warning(f"API call failed (attempt {attempt+1}): {e}. Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    logger.error(f"API call failed after {self.MAX_RETRIES} attempts: {e}")

        raise last_error

    # ------------------------------------------------------------------
    # CSV preparation
    # ------------------------------------------------------------------

    @staticmethod
    def _predictions_to_csv_string(predictions: list) -> str:
        """Convert predictions list to CSV string for prompt injection."""
        if not predictions:
            return "No predictions available."

        import io
        import csv

        # Select key columns
        key_cols = [
            'game_id', 'date', 'home_team', 'away_team',
            'home_pitcher', 'away_pitcher',
            'ml_prob', 'ml_pick', 'ml_edge', 'ml_confidence',
            'ou_predicted_total', 'rl_pick', 'rl_confidence',
            'f5_pick', 'f5_confidence',
            'nrfi_pick', 'nrfi_prob', 'nrfi_confidence',
        ]

        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        available_cols = [c for c in key_cols if c in predictions[0]]
        writer.writerow(available_cols)

        # Rows
        for p in predictions:
            row = []
            for col in available_cols:
                val = p.get(col, '')
                if isinstance(val, (list, dict)):
                    val = json.dumps(val, ensure_ascii=False)
                row.append(val)
            writer.writerow(row)

        return output.getvalue()

    @staticmethod
    def _build_context(predictions: list, model_count: int) -> str:
        """Build additional context string for the prompt."""
        if not predictions:
            return ""

        strong = sum(1 for p in predictions if p.get('ml_confidence') == 'STRONG')
        moderate = sum(1 for p in predictions if p.get('ml_confidence') == 'MODERATE')

        lines = [
            f"Models loaded: {model_count}",
            f"Total games: {len(predictions)}",
            f"STRONG picks: {strong}",
            f"MODERATE picks: {moderate}",
        ]

        # Best pick
        best = max(predictions, key=lambda p: abs(p.get('ml_edge', 0)))
        lines.append(f"Highest edge game: {best.get('away_team')} @ {best.get('home_team')} "
                      f"(edge: {best.get('ml_edge', 0)*100:+.1f}%, pick: {best.get('ml_pick')})")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Public API: generate daily digest
    # ------------------------------------------------------------------

    def generate_daily_digest(
        self,
        predictions: list,
        target_date: str,
        model_count: int = 9,
    ) -> Tuple[str, str]:
        """
        Generate both EN and JA daily digest articles.

        Args:
            predictions: List of prediction dicts from run_daily.py
            target_date: YYYY-MM-DD
            model_count: Number of models used

        Returns:
            Tuple of (english_article, japanese_article)
        """
        csv_data = self._predictions_to_csv_string(predictions)
        context = self._build_context(predictions, model_count)

        if self.api_available:
            logger.info("Generating articles via Anthropic API...")
            en_article = self._generate_en_via_api(
                csv_data, context, target_date, len(predictions), model_count
            )
            ja_article = self._generate_ja_via_api(
                csv_data, context, target_date, len(predictions), model_count
            )
        else:
            logger.info("API unavailable — using template fallback")
            en_article = None
            ja_article = None

        return en_article, ja_article

    def _generate_en_via_api(self, csv_data, context, date, game_count, model_count) -> str:
        prompt = self.ENGLISH_DIGEST_PROMPT.format(
            date=date,
            game_count=game_count,
            model_count=model_count,
            csv_data=csv_data,
            context=context,
        )
        return self._call_api(self.SYSTEM_PROMPT, prompt)

    def _generate_ja_via_api(self, csv_data, context, date, game_count, model_count) -> str:
        prompt = self.JAPANESE_DIGEST_PROMPT.format(
            date=date,
            game_count=game_count,
            model_count=model_count,
            csv_data=csv_data,
            context=context,
        )
        return self._call_api(self.SYSTEM_PROMPT, prompt)

    # ------------------------------------------------------------------
    # Single game article generation (for per-game deep dives)
    # ------------------------------------------------------------------

    def generate_english_article(self, game_data: Dict) -> str:
        """Generate English article for a single game."""
        game = self._to_game_data(game_data)
        if self.api_available:
            prompt = self._format_single_game_prompt_en(game)
            return self._call_api(self.SYSTEM_PROMPT, prompt)
        return self._demo_article_english(game)

    def generate_japanese_article(self, game_data: Dict) -> str:
        """Generate Japanese article for a single game."""
        game = self._to_game_data(game_data)
        if self.api_available:
            prompt = self._format_single_game_prompt_ja(game)
            return self._call_api(self.SYSTEM_PROMPT, prompt)
        return self._demo_article_japanese(game)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_game_data(d: Dict) -> GameData:
        return GameData(
            game_id=d.get('game_id', ''),
            date=d.get('date', ''),
            away_team=d.get('away_team', ''),
            home_team=d.get('home_team', ''),
            away_pitcher=d.get('away_pitcher', 'TBA'),
            home_pitcher=d.get('home_pitcher', 'TBA'),
            model_probability=d.get('ml_prob', d.get('model_probability', 0.5)),
            confidence_tier=d.get('ml_confidence', d.get('confidence_tier', 'N/A')),
            pick=d.get('ml_pick', d.get('pick', '')),
            away_team_name=TEAM_CODES.get(d.get('away_team', ''), d.get('away_team', '')),
            home_team_name=TEAM_CODES.get(d.get('home_team', ''), d.get('home_team', '')),
            away_era=d.get('away_pitcher_era', 4.0),
            home_era=d.get('home_pitcher_era', 4.0),
        )

    def _format_single_game_prompt_en(self, game: GameData) -> str:
        return f"""Write a 800-1200 word Substack article analyzing this MLB game.

Game: {game.away_team_name} ({game.away_team}) @ {game.home_team_name} ({game.home_team})
Date: {game.date}
Away Pitcher: {game.away_pitcher} (ERA: {game.away_era:.2f})
Home Pitcher: {game.home_pitcher} (ERA: {game.home_era:.2f})
Model Pick: {game.pick}
Win Probability: {game.model_probability:.1%}
Confidence: {game.confidence_tier}

Structure: Hook → Pitcher Analysis → Offensive Matchups → The Pick → Bottom Line
Include specific stats, league average comparisons, and honest uncertainty assessment."""

    def _format_single_game_prompt_ja(self, game: GameData) -> str:
        return f"""以下のMLB試合を分析するnote.com記事を2000-3000文字で書いてください。

試合: {game.away_team_name} ({game.away_team}) @ {game.home_team_name} ({game.home_team})
日付: {game.date}
アウェイ先発: {game.away_pitcher} (ERA: {game.away_era:.2f})
ホーム先発: {game.home_pitcher} (ERA: {game.home_era:.2f})
モデル予測: {game.pick}
勝率: {game.model_probability:.1%}
信頼度: {game.confidence_tier}

構成: 導入 → 先発投手分析 → 打線マッチアップ → 予測と推奨 → まとめ
具体的な統計、リーグ平均との比較、不確実性の誠実な評価を含めてください。"""

    @staticmethod
    def _demo_article_english(game: GameData) -> str:
        return f"""# {game.away_team} @ {game.home_team} - {game.date}

## The Matchup

{game.away_team_name} visits {game.home_team_name} featuring {game.away_pitcher} (ERA: {game.away_era:.2f}) vs {game.home_pitcher} (ERA: {game.home_era:.2f}).

## The Pick

**{game.pick}** at {game.model_probability:.1%} probability ({game.confidence_tier} confidence).

---
*Built by a Japanese AI engineer in Tokyo. Not financial advice.*
"""

    @staticmethod
    def _demo_article_japanese(game: GameData) -> str:
        return f"""# {game.away_team} @ {game.home_team} - {game.date}

## 試合概要

{game.away_team_name}が{game.home_team_name}を訪問。{game.away_pitcher}（ERA: {game.away_era:.2f}）vs {game.home_pitcher}（ERA: {game.home_era:.2f}）。

## 予測

**{game.pick}** — 勝率 {game.model_probability:.1%}（{game.confidence_tier}）

---
*東京のAIエンジニアが開発。投資アドバイスではありません。*
"""


def main():
    """Demo / self-test."""
    print("=" * 70)
    print("MONEYBALL DOJO ARTICLE GENERATOR")
    print("=" * 70)

    generator = ArticleGenerator()
    print(f"API available: {generator.api_available}")

    sample = {
        'game_id': 'test_001', 'date': '2026-03-27',
        'away_team': 'NYY', 'home_team': 'BOS',
        'away_pitcher': 'Gerrit Cole', 'home_pitcher': 'Brayan Bello',
        'ml_prob': 0.62, 'ml_confidence': 'STRONG', 'ml_pick': 'AWAY',
        'away_pitcher_era': 3.45, 'home_pitcher_era': 4.12,
    }

    en = generator.generate_english_article(sample)
    print("\n--- English ---")
    print(en[:500])

    ja = generator.generate_japanese_article(sample)
    print("\n--- Japanese ---")
    print(ja[:500])


if __name__ == '__main__':
    main()
