"""
Google Sheets Database Schema for Moneyball Dojo
================================================
Defines the complete data schema for all sheets in the prediction tracking system.
These sheets serve as the central database for:
- Daily game predictions
- Game results and betting outcomes
- Model performance metrics
- Article generation pipeline

Usage:
    from sheets_schema import SHEETS_SCHEMA, describe_schema, create_headers

    # Get schema for a specific sheet
    schema = SHEETS_SCHEMA['Daily Predictions']

    # Print human-readable schema description
    describe_schema()

    # Get column headers for sheet initialization
    headers = create_headers('Daily Predictions')
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class ColumnDef:
    """Definition of a single column in a sheet."""
    name: str
    data_type: str
    description: str
    example: str
    validation: str = None  # Optional: data validation rule


# ============================================================================
# SHEET 1: DAILY PREDICTIONS
# ============================================================================
DAILY_PREDICTIONS_SCHEMA = [
    ColumnDef(
        name='date',
        data_type='DATE',
        description='Game date (YYYY-MM-DD format)',
        example='2024-06-15'
    ),
    ColumnDef(
        name='game_id',
        data_type='STRING',
        description='Unique game identifier (MLB schedule ID)',
        example='mlb_20240615_001'
    ),
    ColumnDef(
        name='away_team',
        data_type='STRING',
        description='Away team abbreviation',
        example='NYY',
        validation='Must be valid MLB team code'
    ),
    ColumnDef(
        name='home_team',
        data_type='STRING',
        description='Home team abbreviation',
        example='BOS',
        validation='Must be valid MLB team code'
    ),
    ColumnDef(
        name='away_pitcher',
        data_type='STRING',
        description='Away team starting pitcher name',
        example='Gerrit Cole'
    ),
    ColumnDef(
        name='home_pitcher',
        data_type='STRING',
        description='Home team starting pitcher name',
        example='Garrett Whitlock'
    ),
    ColumnDef(
        name='model_probability',
        data_type='DECIMAL',
        description='Model-predicted probability of home team win (0.0-1.0)',
        example='0.62',
        validation='0 <= value <= 1'
    ),
    ColumnDef(
        name='pick',
        data_type='STRING',
        description='Recommended pick: HOME or AWAY',
        example='HOME',
        validation='Must be HOME or AWAY'
    ),
    ColumnDef(
        name='line',
        data_type='DECIMAL',
        description='Opening betting line (home team favored)',
        example='-110'
    ),
    ColumnDef(
        name='confidence_tier',
        data_type='STRING',
        description='Confidence level of prediction: HIGH, MEDIUM, or LOW',
        example='HIGH',
        validation='Must be HIGH, MEDIUM, or LOW'
    ),
    ColumnDef(
        name='article_assigned',
        data_type='BOOLEAN',
        description='Whether article has been assigned for this game',
        example='TRUE'
    ),
    ColumnDef(
        name='notes',
        data_type='STRING',
        description='Additional notes or observations about the matchup',
        example='Key injury: Home team missing lead cleanup hitter'
    ),
]


# ============================================================================
# SHEET 2: RESULTS
# ============================================================================
RESULTS_SCHEMA = [
    ColumnDef(
        name='date',
        data_type='DATE',
        description='Game date',
        example='2024-06-15'
    ),
    ColumnDef(
        name='game_id',
        data_type='STRING',
        description='Unique game identifier (link to Daily Predictions)',
        example='mlb_20240615_001'
    ),
    ColumnDef(
        name='away_team',
        data_type='STRING',
        description='Away team',
        example='NYY'
    ),
    ColumnDef(
        name='home_team',
        data_type='STRING',
        description='Home team',
        example='BOS'
    ),
    ColumnDef(
        name='away_score',
        data_type='INTEGER',
        description='Final away team score',
        example='5'
    ),
    ColumnDef(
        name='home_score',
        data_type='INTEGER',
        description='Final home team score',
        example='4'
    ),
    ColumnDef(
        name='prediction',
        data_type='STRING',
        description='Original prediction from Daily Predictions sheet',
        example='HOME',
        validation='HOME or AWAY'
    ),
    ColumnDef(
        name='actual_result',
        data_type='STRING',
        description='Actual outcome of game',
        example='AWAY',
        validation='HOME or AWAY'
    ),
    ColumnDef(
        name='hit',
        data_type='BOOLEAN',
        description='Whether prediction was correct',
        example='FALSE'
    ),
    ColumnDef(
        name='units_wagered',
        data_type='DECIMAL',
        description='Units wagered on this game',
        example='2.5'
    ),
    ColumnDef(
        name='units_won',
        data_type='DECIMAL',
        description='Units won or lost (negative = loss)',
        example='-2.5'
    ),
    ColumnDef(
        name='roi',
        data_type='DECIMAL',
        description='Return on investment for this game',
        example='-100.0'
    ),
    ColumnDef(
        name='cumulative_roi',
        data_type='DECIMAL',
        description='Running cumulative ROI across all games',
        example='12.5',
        validation='Updated daily based on cumulative wins/losses'
    ),
]


# ============================================================================
# SHEET 3: MODEL PERFORMANCE
# ============================================================================
MODEL_PERFORMANCE_SCHEMA = [
    ColumnDef(
        name='date',
        data_type='DATE',
        description='Reporting date (week end date)',
        example='2024-06-21'
    ),
    ColumnDef(
        name='week_number',
        data_type='INTEGER',
        description='Week number of season',
        example='12'
    ),
    ColumnDef(
        name='total_picks',
        data_type='INTEGER',
        description='Total number of predictions issued this week',
        example='28'
    ),
    ColumnDef(
        name='wins',
        data_type='INTEGER',
        description='Number of correct predictions',
        example='18'
    ),
    ColumnDef(
        name='losses',
        data_type='INTEGER',
        description='Number of incorrect predictions',
        example='10'
    ),
    ColumnDef(
        name='win_rate',
        data_type='DECIMAL',
        description='Percentage of predictions that were correct (0-1)',
        example='0.643'
    ),
    ColumnDef(
        name='ats_record',
        data_type='STRING',
        description='Against the spread record (Wins-Losses format)',
        example='18-10'
    ),
    ColumnDef(
        name='units_wagered',
        data_type='DECIMAL',
        description='Total units wagered this week',
        example='28.0'
    ),
    ColumnDef(
        name='units_won',
        data_type='DECIMAL',
        description='Net units won or lost',
        example='3.5'
    ),
    ColumnDef(
        name='roi',
        data_type='DECIMAL',
        description='Weekly return on investment percentage',
        example='12.5'
    ),
    ColumnDef(
        name='cumulative_roi',
        data_type='DECIMAL',
        description='Cumulative ROI from start of season',
        example='28.3'
    ),
    ColumnDef(
        name='high_confidence_record',
        data_type='STRING',
        description='Record on HIGH confidence picks only',
        example='12-3'
    ),
    ColumnDef(
        name='medium_confidence_record',
        data_type='STRING',
        description='Record on MEDIUM confidence picks',
        example='5-4'
    ),
    ColumnDef(
        name='low_confidence_record',
        data_type='STRING',
        description='Record on LOW confidence picks',
        example='1-3'
    ),
]


# ============================================================================
# SHEET 4: ARTICLE QUEUE
# ============================================================================
ARTICLE_QUEUE_SCHEMA = [
    ColumnDef(
        name='date',
        data_type='DATE',
        description='Publish date for article',
        example='2024-06-15'
    ),
    ColumnDef(
        name='game_id',
        data_type='STRING',
        description='Associated game ID (link to Daily Predictions)',
        example='mlb_20240615_001'
    ),
    ColumnDef(
        name='away_team',
        data_type='STRING',
        description='Away team',
        example='NYY'
    ),
    ColumnDef(
        name='home_team',
        data_type='STRING',
        description='Home team',
        example='BOS'
    ),
    ColumnDef(
        name='matchup_title',
        data_type='STRING',
        description='Game matchup title for article',
        example='Yankees vs. Red Sox: ALDS Game 5 Decider'
    ),
    ColumnDef(
        name='article_status',
        data_type='STRING',
        description='Status: DRAFT, READY_FOR_REVIEW, PUBLISHED, SKIPPED',
        example='READY_FOR_REVIEW',
        validation='Must be valid status'
    ),
    ColumnDef(
        name='english_article_status',
        data_type='STRING',
        description='Status of English (Substack) article',
        example='PUBLISHED'
    ),
    ColumnDef(
        name='japanese_article_status',
        data_type='STRING',
        description='Status of Japanese (note.com) article',
        example='PUBLISHED'
    ),
    ColumnDef(
        name='substack_url',
        data_type='STRING',
        description='Published Substack article URL',
        example='https://moneyballdojo.substack.com/p/yankees-vs-redsox-042'
    ),
    ColumnDef(
        name='note_url',
        data_type='STRING',
        description='Published note.com article URL',
        example='https://note.com/moneyballdojo/n/n1234567890abcdef'
    ),
    ColumnDef(
        name='generated_by_claude',
        data_type='BOOLEAN',
        description='Whether article was generated by Claude API',
        example='TRUE'
    ),
    ColumnDef(
        name='prompt_version',
        data_type='STRING',
        description='Version of article generation prompt used',
        example='v2.1'
    ),
    ColumnDef(
        name='model_prediction',
        data_type='DECIMAL',
        description='Original model prediction probability',
        example='0.62'
    ),
    ColumnDef(
        name='confidence_tier',
        data_type='STRING',
        description='Confidence tier of prediction',
        example='HIGH'
    ),
    ColumnDef(
        name='notes',
        data_type='STRING',
        description='Internal notes about article generation or edits',
        example='Added context on bullpen matchups'
    ),
]


# ============================================================================
# COMPLETE SCHEMA DICTIONARY
# ============================================================================
SHEETS_SCHEMA: Dict[str, List[ColumnDef]] = {
    'Daily Predictions': DAILY_PREDICTIONS_SCHEMA,
    'Results': RESULTS_SCHEMA,
    'Model Performance': MODEL_PERFORMANCE_SCHEMA,
    'Article Queue': ARTICLE_QUEUE_SCHEMA,
}


def create_headers(sheet_name: str) -> List[str]:
    """
    Get column headers for a specific sheet.

    Args:
        sheet_name: Name of the sheet

    Returns:
        List of column names in order
    """
    if sheet_name not in SHEETS_SCHEMA:
        raise ValueError(f"Unknown sheet: {sheet_name}")

    return [col.name for col in SHEETS_SCHEMA[sheet_name]]


def get_schema(sheet_name: str) -> List[ColumnDef]:
    """Get the complete schema for a sheet."""
    if sheet_name not in SHEETS_SCHEMA:
        raise ValueError(f"Unknown sheet: {sheet_name}")

    return SHEETS_SCHEMA[sheet_name]


def describe_schema(sheet_name: str = None) -> str:
    """
    Generate a human-readable description of the schema.

    Args:
        sheet_name: Specific sheet to describe, or None for all sheets

    Returns:
        Formatted string description
    """
    description = ""

    if sheet_name:
        sheets_to_describe = {sheet_name: SHEETS_SCHEMA[sheet_name]}
    else:
        sheets_to_describe = SHEETS_SCHEMA

    for sname, schema in sheets_to_describe.items():
        description += f"\n{'=' * 80}\n"
        description += f"SHEET: {sname}\n"
        description += f"{'=' * 80}\n\n"

        for col in schema:
            description += f"Column: {col.name}\n"
            description += f"  Type: {col.data_type}\n"
            description += f"  Description: {col.description}\n"
            description += f"  Example: {col.example}\n"
            if col.validation:
                description += f"  Validation: {col.validation}\n"
            description += "\n"

    return description


def get_validation_rules(sheet_name: str) -> Dict[str, str]:
    """
    Get data validation rules for a sheet.

    Args:
        sheet_name: Name of the sheet

    Returns:
        Dictionary mapping column names to validation rules
    """
    schema = get_schema(sheet_name)
    rules = {}

    for col in schema:
        if col.validation:
            rules[col.name] = col.validation

    return rules


def export_schema_to_json() -> Dict:
    """
    Export complete schema as JSON-serializable dictionary.

    Returns:
        Dictionary representation of schema
    """
    output = {}

    for sheet_name, columns in SHEETS_SCHEMA.items():
        output[sheet_name] = []

        for col in columns:
            output[sheet_name].append({
                'name': col.name,
                'type': col.data_type,
                'description': col.description,
                'example': col.example,
                'validation': col.validation,
            })

    return output


def main():
    """Display complete schema documentation."""
    print(describe_schema())

    # Show header examples
    print("\n\n" + "=" * 80)
    print("SHEET HEADERS")
    print("=" * 80)

    for sheet_name in SHEETS_SCHEMA.keys():
        headers = create_headers(sheet_name)
        print(f"\n{sheet_name}:")
        print(", ".join(headers))

    # Show validation rules
    print("\n\n" + "=" * 80)
    print("VALIDATION RULES")
    print("=" * 80)

    for sheet_name in SHEETS_SCHEMA.keys():
        rules = get_validation_rules(sheet_name)
        if rules:
            print(f"\n{sheet_name}:")
            for col_name, rule in rules.items():
                print(f"  {col_name}: {rule}")


if __name__ == '__main__':
    main()
