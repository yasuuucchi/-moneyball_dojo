"""
Article Generator Template for Moneyball Dojo
==============================================
Templates and utilities for generating game analysis articles using Claude API.

This module provides:
1. System prompts defining the Moneyball Dojo voice and style
2. Article generation prompts for both English and Japanese
3. Example functions showing how to call Claude API with prediction data
4. Utility functions for formatting game data for article context

Usage:
    from article_generator_template import ArticleGenerator

    # Initialize generator
    generator = ArticleGenerator(api_key='your-api-key')

    # Generate English article
    english_article = generator.generate_english_article(
        game_data={
            'away_team': 'NYY',
            'home_team': 'BOS',
            'away_pitcher': 'Gerrit Cole',
            'home_pitcher': 'Garrett Whitlock',
            'model_probability': 0.62,
            'confidence_tier': 'HIGH',
            'pick': 'HOME'
        }
    )

    # Generate Japanese article
    japanese_article = generator.generate_japanese_article(game_data)
"""

from typing import Dict, Optional, List
from dataclasses import dataclass
from datetime import datetime


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


class ArticleGenerator:
    """
    Generates game analysis articles using Claude API.
    Supports both English (Substack) and Japanese (note.com) versions.
    """

    # ========================================================================
    # SYSTEM PROMPTS - Define the Moneyball Dojo voice and expertise
    # ========================================================================

    MONEYBALL_SYSTEM_PROMPT = """You are the Moneyball Dojo author - a data-driven baseball analyst with deep expertise in:
- Advanced MLB analytics (sabermetrics, WAR, FIP, xwOBA, BABIP)
- Predictive modeling and probability assessment
- Betting line analysis and value identification
- Team tendencies, matchup dynamics, and contextual factors
- The intersection of baseball science and practical decision-making

Your writing style is:
- Clear and analytical but accessible to informed sports fans
- Evidence-based: every claim supported by data or precedent
- Actionable: readers understand the "why" behind predictions
- Balanced: acknowledging uncertainty and counterarguments
- Engaging: conversational tone despite technical content

When analyzing games, consider:
1. Pitcher matchups (ERA, WHIP, recent form)
2. Team offensive production (AVG, OBP, SLG vs. specific pitcher types)
3. Bullpen strength and usage patterns
4. Home/away splits and travel context
5. Recent performance trends
6. Lineup matchups and injuries
7. Betting line movement as a signal of sharp money

You write for readers who understand baseball deeply and appreciate rigorous analysis.
Assume knowledge of baseball fundamentals but explain advanced metrics as needed."""

    # ========================================================================
    # ENGLISH ARTICLE PROMPT - For Substack publication
    # ========================================================================

    ENGLISH_ARTICLE_TEMPLATE = """Given this game prediction data, write a compelling Substack article analyzing the matchup.

GAME DATA:
- Away Team: {away_team} ({away_team_name})
- Home Team: {home_team} ({home_team_name})
- Date: {date}
- Away Pitcher: {away_pitcher} (ERA: {away_era})
- Home Pitcher: {home_pitcher} (ERA: {home_era})
- Model Pick: {pick}
- Predicted Probability: {probability:.1%}
- Confidence Tier: {confidence_tier}

ARTICLE STRUCTURE:
1. Opening Hook (2-3 sentences): Why this game matters or what's interesting about it
2. Key Metrics Breakdown (3-4 paragraphs):
   - Pitcher matchup analysis with specific stats
   - Team offensive profiles against similar pitcher types
   - Key lineup considerations or injuries
3. Betting Line Analysis (2 paragraphs):
   - Current line and what it implies about sharp vs. public opinion
   - Value assessment relative to our model probability
4. The Pick (1-2 paragraphs):
   - Clear statement of recommendation
   - Primary reasons supporting the pick
   - Confidence level and caveats
5. Closing Thought (1-2 sentences): Risk/reward or bigger picture takeaway

WRITING GUIDELINES:
- Use specific stats and player names to add credibility
- Include comparative context (league average, historical precedent)
- Be honest about uncertainty
- Assume readers understand baseball and want depth
- Target length: 800-1200 words
- Include 1-2 specific betting line references if relevant
- Avoid generic statements - every claim should be supported

Write in first person as the Moneyball Dojo analyst. This is a long-form analysis piece,
not a quick take. Readers expect rigor and evidence."""

    # ========================================================================
    # JAPANESE ARTICLE PROMPT - For note.com publication
    # ========================================================================

    JAPANESE_ARTICLE_TEMPLATE = """与えられたゲーム予測データを基に、note.comの記事を書いてください。

ゲームデータ:
- アウェイチーム: {away_team} ({away_team_name})
- ホームチーム: {home_team} ({home_team_name})
- 日付: {date}
- アウェイ先発: {away_pitcher} (ERA: {away_era})
- ホーム先発: {home_pitcher} (ERA: {home_era})
- モデル予測: {pick}
- 勝利確率: {probability:.1%}
- 信頼度: {confidence_tier}

記事構成:
1. 導入部 (3-4段落):
   - このゲームがなぜ重要か、何が興味深いかの説明
   - 日本のファンにとって関連性のある文脈
2. 先発投手対決分析 (3-4段落):
   - 両先発投手の詳細な統計分析
   - 直近のフォーム、対戦相手との相性
   - ERA、WHIP、ストライクアウト率などの具体的なメトリクス
3. 打線分析 (3段落):
   - 各チームの得点力、長打率などの攻撃指標
   - 相手投手タイプに対する成績
   - 主要選手の調子やケガの情報
4. データ駆動型の予測 (2-3段落):
   - モデルの推奨ピック
   - 勝利確率とその根拠
   - 信頼度と不確実性の説明
5. 結論 (1-2段落):
   - 日本野球ファンへのメッセージ
   - より大きな文脈での意味

執筆ガイドライン:
- 具体的な統計数値と選手名を使用して信頼性を高める
- リーグ平均との比較や歴史的背景を含める
- データに基づく主張のみを述べる
- 不確実性について誠実に述べる
- 日本のMLBファンは深い分析を理解し、評価する
- 目標文字数: 2000-3000文字
- セイバーメトリクスの日本語説明を含める必要に応じて

マネーボールドージョのアナリストとして一人称で執筆してください。
日本のMLBファンに対する深い分析記事です。"""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the article generator.

        Args:
            api_key: Anthropic API key (optional, can be set via environment variable)
        """
        self.api_key = api_key
        self.model = "claude-opus-4-6"  # Use Claude Opus for article generation

    def format_game_data(self, game_dict: Dict) -> GameData:
        """
        Convert dictionary to structured GameData object.

        Args:
            game_dict: Dictionary with game information

        Returns:
            GameData object
        """
        return GameData(
            game_id=game_dict.get('game_id', ''),
            date=game_dict.get('date', ''),
            away_team=game_dict.get('away_team', ''),
            home_team=game_dict.get('home_team', ''),
            away_pitcher=game_dict.get('away_pitcher', 'TBA'),
            home_pitcher=game_dict.get('home_pitcher', 'TBA'),
            model_probability=game_dict.get('model_probability', 0.5),
            confidence_tier=game_dict.get('confidence_tier', 'MEDIUM'),
            pick=game_dict.get('pick', 'HOME'),
            away_team_name=game_dict.get('away_team_name', self._get_team_name(game_dict.get('away_team'))),
            home_team_name=game_dict.get('home_team_name', self._get_team_name(game_dict.get('home_team'))),
            away_runs_per_game=game_dict.get('away_runs_per_game', 4.2),
            home_runs_per_game=game_dict.get('home_runs_per_game', 4.3),
            away_era=game_dict.get('away_pitcher_era', 4.0),
            home_era=game_dict.get('home_pitcher_era', 4.0),
        )

    @staticmethod
    def _get_team_name(team_code: str) -> str:
        """Get full team name from abbreviation."""
        team_names = {
            'NYY': 'New York Yankees',
            'BOS': 'Boston Red Sox',
            'TB': 'Tampa Bay Rays',
            'BAL': 'Baltimore Orioles',
            'TOR': 'Toronto Blue Jays',
            'NYM': 'New York Mets',
            'ATL': 'Atlanta Braves',
            'WSH': 'Washington Nationals',
            'PHI': 'Philadelphia Phillies',
            'MIA': 'Miami Marlins',
            'LAD': 'Los Angeles Dodgers',
            'SD': 'San Diego Padres',
            'SF': 'San Francisco Giants',
            'ARI': 'Arizona Diamondbacks',
            'COL': 'Colorado Rockies',
            'MIL': 'Milwaukee Brewers',
            'CHC': 'Chicago Cubs',
            'STL': 'St. Louis Cardinals',
            'PIT': 'Pittsburgh Pirates',
            'CIN': 'Cincinnati Reds',
            'HOU': 'Houston Astros',
            'LAA': 'Los Angeles Angels',
            'OAK': 'Oakland Athletics',
            'SEA': 'Seattle Mariners',
            'TEX': 'Texas Rangers',
            'KC': 'Kansas City Royals',
            'MIN': 'Minnesota Twins',
            'CWS': 'Chicago White Sox',
            'DET': 'Detroit Tigers',
        }
        return team_names.get(team_code, team_code)

    def generate_english_article(self, game_data: Dict) -> str:
        """
        Generate English article for Substack publication.

        Args:
            game_data: Dictionary with game prediction data

        Returns:
            Generated article text (English)

        Example:
            article = generator.generate_english_article({
                'away_team': 'NYY',
                'home_team': 'BOS',
                'away_pitcher': 'Gerrit Cole',
                'home_pitcher': 'Garrett Whitlock',
                'model_probability': 0.62,
                'confidence_tier': 'HIGH',
                'pick': 'HOME',
                'date': '2024-06-15'
            })
        """
        game = self.format_game_data(game_data)

        # Format the prompt with game data
        user_prompt = self.ENGLISH_ARTICLE_TEMPLATE.format(
            away_team=game.away_team,
            away_team_name=game.away_team_name,
            home_team=game.home_team,
            home_team_name=game.home_team_name,
            date=game.date,
            away_pitcher=game.away_pitcher,
            home_pitcher=game.home_pitcher,
            away_era=f"{game.away_era:.2f}",
            home_era=f"{game.home_era:.2f}",
            pick=game.pick,
            probability=game.model_probability,
            confidence_tier=game.confidence_tier,
        )

        # In production, this would call Claude API:
        # response = client.messages.create(
        #     model=self.model,
        #     max_tokens=2000,
        #     system=self.MONEYBALL_SYSTEM_PROMPT,
        #     messages=[{"role": "user", "content": user_prompt}]
        # )
        # return response.content[0].text

        # For demo, return template with placeholders
        return self._generate_demo_article_english(game)

    def generate_japanese_article(self, game_data: Dict) -> str:
        """
        Generate Japanese article for note.com publication.

        Args:
            game_data: Dictionary with game prediction data

        Returns:
            Generated article text (Japanese)
        """
        game = self.format_game_data(game_data)

        # Format the prompt with game data
        user_prompt = self.JAPANESE_ARTICLE_TEMPLATE.format(
            away_team=game.away_team,
            away_team_name=game.away_team_name,
            home_team=game.home_team,
            home_team_name=game.home_team_name,
            date=game.date,
            away_pitcher=game.away_pitcher,
            home_pitcher=game.home_pitcher,
            away_era=f"{game.away_era:.2f}",
            home_era=f"{game.home_era:.2f}",
            pick=game.pick,
            probability=game.model_probability,
            confidence_tier=game.confidence_tier,
        )

        # In production, this would call Claude API
        # For demo, return template
        return self._generate_demo_article_japanese(game)

    @staticmethod
    def _generate_demo_article_english(game: GameData) -> str:
        """Generate demo English article."""
        return f"""# {game.away_team} @ {game.home_team} - {game.date}

## The Matchup

{game.away_team_name} travels to face {game.home_team_name} in what promises to be a compelling
pitching duel. On the surface, this looks like a classic pitcher-focused game, with both
{game.away_pitcher} ({game.away_era:.2f} ERA) and {game.home_pitcher} ({game.home_era:.2f} ERA)
taking the mound at the top of their respective rotations.

## Pitcher Analysis

{game.away_pitcher} has been outstanding for {game.away_team}, posting an ERA well below league
average. His recent form suggests continued excellence, particularly against the type of lineup
{game.home_team} fields. {game.home_pitcher} has shown resilience despite some recent struggles,
with underlying metrics suggesting regression toward the mean.

## Offensive Matchups

{game.home_team} averages {game.home_runs_per_game:.1f} runs per game at home, a figure that aligns
with their season-long output. The {game.away_team} are averaging {game.away_runs_per_game:.1f} runs
on the road, suggesting they may struggle against {game.home_pitcher}'s particular skillset.

## The Pick

Based on our model's assessment, the **{game.pick} TEAM** offers the better value at current lines.
Our model assigns a {game.model_probability:.1%} probability to this outcome, suggesting potential
value if the line reflects lower probability. This is a **{game.confidence_tier}** confidence
selection, reflecting both our conviction in the pitcher matchup advantage and some residual
uncertainty in the offensive projections.

## Bottom Line

This game exemplifies the kind of matchup where detailed pitching analysis generates an edge.
The sharper approach is to focus on what the pitchers do well and construct your thesis around
that advantage rather than trying to predict exact run totals.
"""

    @staticmethod
    def _generate_demo_article_japanese(game: GameData) -> str:
        """Generate demo Japanese article."""
        return f"""# {game.away_team} @ {game.home_team} - {game.date}

## 試合概要

{game.away_team_name}が{game.home_team_name}を訪問し、投手戦になることが予想されます。
{game.away_pitcher}（ERA: {game.away_era:.2f}）と{game.home_pitcher}（ERA: {game.home_era:.2f}）
という両チームの主力先発投手による対決となります。

## 先発投手分析

{game.away_pitcher}は{game.away_team}で優れた成績を上げており、リーグ平均を大きく下回るERAを記録しています。
最近の好調ぶりから、{game.home_team}の打線に対しても優位性を保つことが期待できます。

一方、{game.home_pitcher}は最近やや苦戦していますが、基礎的なメトリクスから見ると、
より良い成績へ収束する可能性があります。

## 打線マッチアップ

{game.home_team}はホームで平均{game.home_runs_per_game:.1f}点を獲得しており、
シーズン全体の成績と一致しています。一方、{game.away_team}はロード平均{game.away_runs_per_game:.1f}点と
やや打力が落ちる傾向が見られます。

## 予測と推奨

モデルの分析では、**{game.pick}チーム**が有利とされています。
当モデルはこの試合結果の確率を{game.model_probability:.1%}と評価しており、
現在のオッズがこれより低い確率を示唆していれば、バリューがあると考えられます。

この予測は**{game.confidence_tier}**信頼度レベルでの評価です。

## まとめ

このような投手戦中心のマッチアップでは、詳細な投手分析が優位性を生み出します。
複雑なトータルスコア予測よりも、各投手の得意分野に焦点を当てた分析が効果的です。
"""

    def create_api_call_example(self, game_data: Dict) -> Dict:
        """
        Create example of how to call Claude API for article generation.

        Args:
            game_data: Game prediction data

        Returns:
            Dictionary showing API call structure
        """
        game = self.format_game_data(game_data)

        return {
            "model": self.model,
            "max_tokens": 2000,
            "system": self.MONEYBALL_SYSTEM_PROMPT,
            "messages": [
                {
                    "role": "user",
                    "content": self.ENGLISH_ARTICLE_TEMPLATE.format(
                        away_team=game.away_team,
                        away_team_name=game.away_team_name,
                        home_team=game.home_team,
                        home_team_name=game.home_team_name,
                        date=game.date,
                        away_pitcher=game.away_pitcher,
                        home_pitcher=game.home_pitcher,
                        away_era=f"{game.away_era:.2f}",
                        home_era=f"{game.home_era:.2f}",
                        pick=game.pick,
                        probability=game.model_probability,
                        confidence_tier=game.confidence_tier,
                    )
                }
            ]
        }


def main():
    """Demonstrate article generation templates."""
    print("=" * 80)
    print("MONEYBALL DOJO ARTICLE GENERATOR")
    print("=" * 80)

    generator = ArticleGenerator()

    # Sample game data
    game_data = {
        'game_id': 'mlb_20240615_001',
        'date': '2024-06-15',
        'away_team': 'NYY',
        'home_team': 'BOS',
        'away_pitcher': 'Gerrit Cole',
        'home_pitcher': 'Garrett Whitlock',
        'model_probability': 0.62,
        'confidence_tier': 'HIGH',
        'pick': 'HOME',
        'away_pitcher_era': 3.45,
        'home_pitcher_era': 4.12,
        'away_runs_per_game': 4.8,
        'home_runs_per_game': 4.5,
    }

    # Generate English article
    print("\n" + "=" * 80)
    print("ENGLISH ARTICLE (Substack)")
    print("=" * 80)
    english_article = generator.generate_english_article(game_data)
    print(english_article)

    # Generate Japanese article
    print("\n" + "=" * 80)
    print("JAPANESE ARTICLE (note.com)")
    print("=" * 80)
    japanese_article = generator.generate_japanese_article(game_data)
    print(japanese_article)

    # Show API call structure
    print("\n" + "=" * 80)
    print("API CALL EXAMPLE")
    print("=" * 80)
    api_call = generator.create_api_call_example(game_data)
    import json
    print(json.dumps(api_call, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
