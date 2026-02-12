"""
Moneyball Dojo Google Sheets Schema (v2) - Event Log Architecture
====================================================================

This module defines the append-only event log schema for the Moneyball Dojo
prediction and betting tracking system. All sheets follow the principle of
immutable records: new data is appended, never updated, preserving the
historical record for model evaluation and audit trails.

Key Design Principles:
- Append-only: Each row is immutable once written
- model_version tracking: Predictions tied to specific model versions
- Edge calculation: Model probability - market implied probability (critical metric)
- Timestamped: All transactions include precise timestamps
- Normalized: game_id links related data across sheets
"""

from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from dataclasses import dataclass
import json


# =============================================================================
# Helper Functions
# =============================================================================

def calculate_implied_probability(american_odds: int) -> float:
    """
    Convert American odds to implied probability.
    
    Args:
        american_odds: Odds in American format (e.g., -110, +150)
    
    Returns:
        Implied probability as a decimal (0.0 to 1.0)
    
    Examples:
        -110 → 0.524 (52.4%)
        +150 → 0.400 (40.0%)
        -200 → 0.667 (66.7%)
    """
    if american_odds < 0:
        # Negative odds: probability = |odds| / (|odds| + 100)
        return abs(american_odds) / (abs(american_odds) + 100)
    else:
        # Positive odds: probability = 100 / (odds + 100)
        return 100 / (american_odds + 100)


def calculate_edge(model_probability: float, american_odds: int) -> float:
    """
    Calculate the edge between model prediction and market odds.
    
    Edge = Model Probability - Implied Probability
    
    Positive edge means the model thinks the outcome is more likely than
    the market does (good betting opportunity).
    
    Args:
        model_probability: Model's predicted probability (0.0 to 1.0)
        american_odds: Market odds in American format
    
    Returns:
        Edge as a decimal (e.g., 0.052 = 5.2% edge)
    """
    implied_prob = calculate_implied_probability(american_odds)
    return model_probability - implied_prob


def american_to_decimal(american_odds: int) -> float:
    """
    Convert American odds to decimal odds.
    
    Args:
        american_odds: Odds in American format
    
    Returns:
        Decimal odds (e.g., 1.91 for -110)
    """
    if american_odds < 0:
        return 100 / abs(american_odds) + 1
    else:
        return american_odds / 100 + 1


# =============================================================================
# Schema Definitions (Dataclasses)
# =============================================================================

@dataclass
class GameRow:
    """Game metadata sheet row"""
    game_id: str
    date: str  # YYYY-MM-DD
    league: str  # MLB, NFL, etc.
    home_team: str
    away_team: str
    start_time: str  # HH:MM EST
    venue: str
    weather_temp: int  # Fahrenheit
    weather_wind: int  # mph
    created_at: str = None  # ISO timestamp
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow().isoformat() + "Z"


@dataclass
class OddsRow:
    """Odds tracking sheet row (append-only)"""
    game_id: str
    market: str  # ML (Moneyline), RL (Run Line), Total, etc.
    odds_open: int  # American odds
    odds_close: int  # American odds
    source: str  # DraftKings, FanDuel, etc.
    timestamp: str  # ISO format, when odds were recorded


@dataclass
class PredictionRow:
    """Model predictions sheet row (append-only)"""
    game_id: str
    model_version: str  # e.g., "v2.1_2024_01_15"
    win_prob: float  # Model's predicted win probability (0.0-1.0)
    total_prob: float  # Predicted total score probability (for total bets)
    edge: float  # Calculated edge (model_prob - implied_prob)
    confidence_tier: str  # STRONG, MODERATE, WEAK
    pick: str  # "HOME", "AWAY", "OVER", "UNDER", "PASS"
    reasoning: str  # Brief explanation of the pick
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow().isoformat() + "Z"


@dataclass
class BetRow:
    """Actual bets placed sheet row (append-only)"""
    game_id: str
    bet_type: str  # Moneyline, Total, Spread, etc.
    stake_units: float  # Number of units risked
    odds_taken: int  # American odds when bet was placed
    result: str  # W (Win), L (Loss), P (Push/Void)
    pnl_units: float  # Profit/loss in units
    pick: str  # What was the pick
    resolved_at: str  # ISO timestamp when result was known
    created_at: str = None  # When bet was created
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow().isoformat() + "Z"


@dataclass
class ArticleRow:
    """Published articles sheet row (append-only)"""
    game_id: str
    platform: str  # Substack, note.com, etc.
    url: str
    publish_time: str  # ISO timestamp
    model_version: str  # Which model version was featured
    article_type: str  # digest, deep_dive, recap
    language: str  # en, ja
    audience_tier: str  # free, paid, all
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow().isoformat() + "Z"


@dataclass
class KPIDailyRow:
    """Daily KPI metrics sheet row (time-series)"""
    date: str  # YYYY-MM-DD
    roi_cumulative: float  # Cumulative return on investment (%)
    clv_avg: float  # Average customer lifetime value ($)
    winrate: float  # Win rate as decimal (0.0-1.0)
    total_picks: int  # Total picks made this season
    total_bets_placed: int  # How many of those were actually bet
    subscribers_free: int
    subscribers_paid: int
    churn_rate: float  # Daily churn as decimal
    yesterday_record: str  # e.g., "3-1" for wins-losses
    yesterday_roi: float  # Daily ROI for yesterday
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow().isoformat() + "Z"


# =============================================================================
# Schema Printer
# =============================================================================

def print_schema_documentation():
    """
    Print a nicely formatted description of the entire schema.
    """
    schema_doc = """
╔══════════════════════════════════════════════════════════════════════════════╗
║         MONEYBALL DOJO SHEETS SCHEMA v2 - EVENT LOG ARCHITECTURE            ║
╚══════════════════════════════════════════════════════════════════════════════╝

DESIGN PRINCIPLE: Append-Only, Immutable Records
────────────────────────────────────────────────
All sheets follow an event log pattern: data is NEVER updated, only appended.
This preserves the historical record for model evaluation and audit trails.


SHEET 1: games (Master game reference)
───────────────────────────────────────
PRIMARY KEY: game_id
FREQUENCY: Created once per game

Columns:
  • game_id (TEXT, UNIQUE): e.g., "MLB_2024_01_15_NYY_BOS"
  • date (TEXT): Game date in YYYY-MM-DD format
  • league (TEXT): MLB, NFL, NBA, NHL, etc.
  • home_team (TEXT): Home team code
  • away_team (TEXT): Away team code
  • start_time (TEXT): Start time in HH:MM EST
  • venue (TEXT): Stadium/arena name
  • weather_temp (NUMBER): Temperature in Fahrenheit
  • weather_wind (NUMBER): Wind speed in mph
  • created_at (TIMESTAMP): ISO format UTC timestamp

NOTES:
  - This is the only mutable sheet (can be corrected if weather data changes)
  - Foreign key for all other tables
  - Contains game metadata needed for analysis


SHEET 2: odds (Market odds tracking - append-only)
────────────────────────────────────────────────────
PRIMARY KEY: (game_id, market, timestamp, source)
FREQUENCY: Append multiple times per game as odds move

Columns:
  • game_id (TEXT): Links to games sheet
  • market (TEXT): ML, RL, Total, Spread, etc.
  • odds_open (NUMBER): Opening odds in American format
  • odds_close (NUMBER): Closing odds in American format
  • source (TEXT): DraftKings, FanDuel, BetMGM, etc.
  • timestamp (TIMESTAMP): ISO format when recorded

NOTES:
  - Record odds movement throughout the day
  - Used to calculate edge (model_prob vs implied_prob)
  - Multiple rows per game as market evolves


SHEET 3: predictions (Model predictions - append-only)
─────────────────────────────────────────────────────────
PRIMARY KEY: (game_id, model_version)
FREQUENCY: One row per game per model version released

Columns:
  • game_id (TEXT): Links to games sheet
  • model_version (TEXT): e.g., "v2.1_2024_01_15"
  • win_prob (NUMBER): Model's probability 0.0-1.0
  • total_prob (NUMBER): Total score prediction probability
  • edge (NUMBER): Model prob - implied prob (critical!)
  • confidence_tier (TEXT): STRONG, MODERATE, WEAK
  • pick (TEXT): HOME, AWAY, OVER, UNDER, PASS
  • reasoning (TEXT): Why the model made this pick
  • created_at (TIMESTAMP): ISO format UTC timestamp

NOTES:
  - Never update: if model improves, new version gets new row
  - This preserves the prediction record for backtesting
  - edge = win_prob - calculate_implied_probability(odds_close)
  - Confidence tier guides bet sizing


SHEET 4: bets (Actual bets placed - append-only)
──────────────────────────────────────────────────
PRIMARY KEY: (game_id, bet_type, created_at)
FREQUENCY: One row per bet placed

Columns:
  • game_id (TEXT): Links to games sheet
  • bet_type (TEXT): Moneyline, Total, Spread, etc.
  • stake_units (NUMBER): Units risked (bankroll management)
  • odds_taken (NUMBER): American odds when placed
  • result (TEXT): W, L, P (Win, Loss, Push)
  • pnl_units (NUMBER): Profit/loss in units
  • pick (TEXT): What the actual pick was
  • resolved_at (TIMESTAMP): When result became known
  • created_at (TIMESTAMP): When bet was placed

NOTES:
  - Only contains bets we actually placed (not all picks)
  - Used for ROI, win rate, and performance tracking
  - pnl_units = stake_units * decimal_odds if WIN, else negative


SHEET 5: articles (Published content - append-only)
───────────────────────────────────────────────────────
PRIMARY KEY: (game_id, platform, publish_time)
FREQUENCY: Variable per game

Columns:
  • game_id (TEXT): Primary game featured (links to games)
  • platform (TEXT): Substack, note.com, Twitter, etc.
  • url (TEXT): Direct link to published article
  • publish_time (TIMESTAMP): When published
  • model_version (TEXT): Which model version is featured
  • article_type (TEXT): digest, deep_dive, recap
  • language (TEXT): en, ja
  • audience_tier (TEXT): free, paid, all
  • created_at (TIMESTAMP): ISO format UTC timestamp

NOTES:
  - Daily digests will have game_id = "DAILY_DIGEST_YYYY_MM_DD"
  - Tracks content output and model attribution
  - Used for subscriber engagement analysis


SHEET 6: kpi_daily (Daily performance metrics - time-series)
──────────────────────────────────────────────────────────────
PRIMARY KEY: date
FREQUENCY: One row per day

Columns:
  • date (TEXT): YYYY-MM-DD
  • roi_cumulative (NUMBER): Cumulative season ROI as % (e.g., 12.5)
  • clv_avg (NUMBER): Average customer lifetime value ($)
  • winrate (NUMBER): Win rate as decimal (e.g., 0.565)
  • total_picks (NUMBER): Total picks made this season
  • total_bets_placed (NUMBER): How many picks were actually bet
  • subscribers_free (NUMBER): Free tier count
  • subscribers_paid (NUMBER): Paid tier count
  • churn_rate (NUMBER): Daily churn as decimal (e.g., 0.01 = 1%)
  • yesterday_record (TEXT): e.g., "3-1" for bets won-lost
  • yesterday_roi (NUMBER): Daily ROI for previous day
  • created_at (TIMESTAMP): ISO format UTC timestamp

NOTES:
  - Snapshot metrics for dashboard and reporting
  - Used for KPI tracking and business health
  - Calculated from bets sheet each evening


EDGE CALCULATION (Critical Concept)
───────────────────────────────────
Edge is the gap between our model's probability and the market's implied prob:

    edge = model_probability - implied_probability

Implied probability comes from American odds using:
  - Negative odds: probability = |odds| / (|odds| + 100)
  - Positive odds: probability = 100 / (odds + 100)

EXAMPLES:
  • Model says 55% to win, market offers -110 (52.4% implied)
    → Edge = 0.55 - 0.524 = 0.026 = 2.6% edge (positive, bet it!)
  
  • Model says 40% to win, market offers +150 (40% implied)
    → Edge = 0.40 - 0.40 = 0.0 = 0% edge (no value, PASS)


DATA INTEGRITY NOTES
────────────────────
• All timestamps are ISO 8601 UTC (ending with 'Z')
• game_id is the foreign key across all tables
• Never UPDATE existing rows; only INSERT new ones
• For corrections, append new row with correction flag
• Monthly archives recommended (BigQuery for analysis)

"""
    print(schema_doc)


# =============================================================================
# Sample Data Generator
# =============================================================================

def generate_sample_data() -> Dict[str, List[Dict]]:
    """
    Generate realistic sample data for each sheet.
    
    Returns:
        Dictionary with sheet names as keys and lists of row dicts as values
    """
    base_date = datetime(2024, 1, 15)
    
    # Sample Games
    games = [
        {
            "game_id": "MLB_2024_01_15_NYY_BOS",
            "date": "2024-01-15",
            "league": "MLB",
            "home_team": "BOS",
            "away_team": "NYY",
            "start_time": "19:10 EST",
            "venue": "Fenway Park",
            "weather_temp": 45,
            "weather_wind": 12,
            "created_at": base_date.isoformat() + "Z"
        },
        {
            "game_id": "MLB_2024_01_15_LAD_SD",
            "date": "2024-01-15",
            "league": "MLB",
            "home_team": "SD",
            "away_team": "LAD",
            "start_time": "19:40 EST",
            "venue": "Petco Park",
            "weather_temp": 68,
            "weather_wind": 8,
            "created_at": base_date.isoformat() + "Z"
        },
    ]
    
    # Sample Odds
    odds = [
        {
            "game_id": "MLB_2024_01_15_NYY_BOS",
            "market": "ML",
            "odds_open": -105,
            "odds_close": -110,
            "source": "DraftKings",
            "timestamp": (base_date + timedelta(hours=8)).isoformat() + "Z"
        },
        {
            "game_id": "MLB_2024_01_15_NYY_BOS",
            "market": "Total",
            "odds_open": -110,
            "odds_close": -110,
            "source": "DraftKings",
            "timestamp": (base_date + timedelta(hours=8)).isoformat() + "Z"
        },
        {
            "game_id": "MLB_2024_01_15_LAD_SD",
            "market": "ML",
            "odds_open": -125,
            "odds_close": -120,
            "source": "FanDuel",
            "timestamp": (base_date + timedelta(hours=7)).isoformat() + "Z"
        },
    ]
    
    # Sample Predictions
    predictions = [
        {
            "game_id": "MLB_2024_01_15_NYY_BOS",
            "model_version": "v2.1_2024_01_15",
            "win_prob": 0.568,
            "total_prob": 0.452,
            "edge": 0.044,  # 56.8% - 52.4% (implied from -110)
            "confidence_tier": "STRONG",
            "pick": "AWAY",
            "reasoning": "NYY has better pitching matchup and positive edge vs implied prob",
            "created_at": (base_date + timedelta(hours=5)).isoformat() + "Z"
        },
        {
            "game_id": "MLB_2024_01_15_LAD_SD",
            "model_version": "v2.1_2024_01_15",
            "win_prob": 0.625,
            "total_prob": 0.512,
            "edge": 0.089,  # 62.5% - 53.6% (implied from -120)
            "confidence_tier": "MODERATE",
            "pick": "AWAY",
            "reasoning": "LAD's recent form positive but not overwhelming edge",
            "created_at": (base_date + timedelta(hours=5)).isoformat() + "Z"
        },
    ]
    
    # Sample Bets
    bets = [
        {
            "game_id": "MLB_2024_01_15_NYY_BOS",
            "bet_type": "Moneyline",
            "stake_units": 2.0,
            "odds_taken": -110,
            "result": "W",
            "pnl_units": 1.82,  # 2.0 * (100/110) ~= 1.82 units profit
            "pick": "AWAY",
            "resolved_at": (base_date + timedelta(hours=13)).isoformat() + "Z",
            "created_at": (base_date + timedelta(hours=6)).isoformat() + "Z"
        },
        {
            "game_id": "MLB_2024_01_15_LAD_SD",
            "bet_type": "Moneyline",
            "stake_units": 1.5,
            "odds_taken": -120,
            "result": "L",
            "pnl_units": -1.5,
            "pick": "AWAY",
            "resolved_at": (base_date + timedelta(hours=14)).isoformat() + "Z",
            "created_at": (base_date + timedelta(hours=6)).isoformat() + "Z"
        },
    ]
    
    # Sample Articles
    articles = [
        {
            "game_id": "DAILY_DIGEST_2024_01_15",
            "platform": "Substack",
            "url": "https://moneyballdojo.substack.com/p/daily-digest-jan-15",
            "publish_time": (base_date + timedelta(hours=9)).isoformat() + "Z",
            "model_version": "v2.1_2024_01_15",
            "article_type": "digest",
            "language": "en",
            "audience_tier": "all",
            "created_at": (base_date + timedelta(hours=9)).isoformat() + "Z"
        },
        {
            "game_id": "DAILY_DIGEST_2024_01_15",
            "platform": "note.com",
            "url": "https://note.com/moneyballdojo/n/jan15-digest",
            "publish_time": (base_date + timedelta(hours=9, minutes=15)).isoformat() + "Z",
            "model_version": "v2.1_2024_01_15",
            "article_type": "digest",
            "language": "ja",
            "audience_tier": "all",
            "created_at": (base_date + timedelta(hours=9, minutes=15)).isoformat() + "Z"
        },
    ]
    
    # Sample Daily KPI
    kpi = [
        {
            "date": "2024-01-15",
            "roi_cumulative": 8.5,
            "clv_avg": 245.50,
            "winrate": 0.562,
            "total_picks": 47,
            "total_bets_placed": 32,
            "subscribers_free": 1250,
            "subscribers_paid": 380,
            "churn_rate": 0.008,
            "yesterday_record": "3-1",
            "yesterday_roi": 1.2,
            "created_at": (base_date + timedelta(hours=22)).isoformat() + "Z"
        },
    ]
    
    return {
        "games": games,
        "odds": odds,
        "predictions": predictions,
        "bets": bets,
        "articles": articles,
        "kpi_daily": kpi,
    }


def print_sample_data():
    """Print sample data in a formatted way"""
    data = generate_sample_data()
    
    print("\n" + "="*80)
    print("SAMPLE DATA FOR MONEYBALL DOJO SHEETS")
    print("="*80 + "\n")
    
    for sheet_name, rows in data.items():
        print(f"\nSHEET: {sheet_name}")
        print("─" * 80)
        
        for i, row in enumerate(rows, 1):
            print(f"\nRow {i}:")
            for key, value in row.items():
                print(f"  {key:25} : {value}")
    
    print("\n" + "="*80 + "\n")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # Print schema documentation
    print_schema_documentation()
    
    # Print sample data
    print_sample_data()
    
    # Demonstrate edge calculation
    print("\n" + "="*80)
    print("EDGE CALCULATION EXAMPLES")
    print("="*80 + "\n")
    
    test_cases = [
        (0.568, -110),
        (0.625, -120),
        (0.500, -110),
        (0.450, +150),
        (0.600, +100),
    ]
    
    for model_prob, odds in test_cases:
        implied = calculate_implied_probability(odds)
        edge = calculate_edge(model_prob, odds)
        status = "✓ BET" if edge > 0.02 else "✗ PASS"
        print(f"Model: {model_prob*100:5.1f}% | Odds: {odds:5} | "
              f"Implied: {implied*100:5.1f}% | Edge: {edge*100:+5.1f}% | {status}")
    
    print("\n")
