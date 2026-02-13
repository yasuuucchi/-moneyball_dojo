"""
MLB Prediction Data Pipeline Demo
==================================
A working prototype that demonstrates the complete data pipeline for MLB game predictions.
This script:
1. Fetches recent MLB batting and pitching stats using pybaseball
2. Constructs team-level features for prediction
3. Engineers game-specific features
4. Trains an XGBoost model on historical data
5. Generates predictions with confidence scores
6. Outputs results to CSV (simulating Google Sheets export)
"""

import argparse
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Try to import pybaseball for real data; fall back to synthetic data if unavailable
try:
    from pybaseball import playerid_lookup, playerid_reverse_lookup, statcast
    PYBASEBALL_AVAILABLE = True
except ImportError:
    PYBASEBALL_AVAILABLE = False
    print("Warning: pybaseball not installed. Using synthetic data for demo.")


class MLBDataPipeline:
    """
    Complete MLB prediction data pipeline.
    Handles data fetching, feature engineering, model training, and prediction generation.
    """

    def __init__(self, output_dir='./output'):
        """Initialize the pipeline."""
        self.output_dir = output_dir
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.team_stats = {}
        self.predictions_df = None

    def fetch_mlb_data(self, season=2024):
        """
        Fetch recent MLB data using pybaseball.
        In production, this would fetch live data from baseball-reference.com.

        Args:
            season: MLB season year (default: 2024)

        Returns:
            tuple: (batting_stats, pitching_stats) DataFrames
        """
        print(f"[Pipeline] Fetching MLB {season} season data...")

        if not PYBASEBALL_AVAILABLE:
            return self._generate_synthetic_data(season)

        try:
            # Use pybaseball's batting_stats and pitching_stats functions
            from pybaseball import batting_stats as pb_batting, pitching_stats as pb_pitching

            batting_data = pb_batting(season, qual=100)  # min 100 AB
            pitching_data = pb_pitching(season, qual=30)  # min 30 IP

            # Normalize column names
            batting_data = batting_data.rename(columns={'Tm': 'Team'}) if 'Tm' in batting_data.columns else batting_data
            pitching_data = pitching_data.rename(columns={'Tm': 'Team'}) if 'Tm' in pitching_data.columns else pitching_data

            print(f"[Pipeline] Successfully fetched {len(batting_data)} batters and {len(pitching_data)} pitchers")
            return batting_data, pitching_data

        except Exception as e:
            print(f"[Pipeline] Error fetching data: {e}. Using synthetic data.")
            return self._generate_synthetic_data(season)

    def _generate_synthetic_data(self, season=2024):
        """Generate synthetic MLB data for demo purposes."""
        teams = ['NYY', 'BOS', 'TB', 'BAL', 'TOR', 'NYM', 'ATL', 'WSH', 'PHI', 'MIA',
                 'LAD', 'SD', 'SF', 'ARI', 'COL', 'MIL', 'CHC', 'STL', 'PIT', 'CIN',
                 'HOU', 'LAA', 'OAK', 'SEA', 'TEX', 'KC', 'MIN', 'CWS', 'DET', 'TB']

        # Generate synthetic batting stats
        np.random.seed(42)
        batting_stats = pd.DataFrame({
            'Name': [f'Player_{i}' for i in range(200)],
            'Team': np.random.choice(teams, 200),
            'AB': np.random.randint(300, 600, 200),
            'H': np.random.randint(70, 200, 200),
            'HR': np.random.randint(0, 50, 200),
            'RBI': np.random.randint(20, 120, 200),
            'AVG': np.random.uniform(0.220, 0.330, 200),
            'OBP': np.random.uniform(0.280, 0.400, 200),
            'SLG': np.random.uniform(0.330, 0.550, 200),
        })

        # Generate synthetic pitching stats
        pitching_stats = pd.DataFrame({
            'Name': [f'Pitcher_{i}' for i in range(100)],
            'Team': np.random.choice(teams, 100),
            'W': np.random.randint(0, 20, 100),
            'L': np.random.randint(0, 15, 100),
            'ERA': np.random.uniform(2.5, 5.5, 100),
            'IP': np.random.uniform(50, 220, 100),
            'SO': np.random.randint(30, 250, 100),
            'WHIP': np.random.uniform(0.9, 1.6, 100),
        })

        return batting_stats, pitching_stats

    def compute_team_stats(self, batting_stats, pitching_stats):
        """
        Compute aggregate team-level statistics from player stats.

        Args:
            batting_stats: DataFrame with batting statistics
            pitching_stats: DataFrame with pitching statistics

        Returns:
            dict: Team statistics keyed by team abbreviation
        """
        print("[Pipeline] Computing team-level statistics...")

        team_stats = {}

        # Group batting stats by team
        for team in batting_stats['Team'].unique():
            team_batters = batting_stats[batting_stats['Team'] == team]
            team_pitchers = pitching_stats[pitching_stats['Team'] == team]

            # Aggregate batting statistics
            team_avg = team_batters['AVG'].mean() if 'AVG' in team_batters.columns else 0.260
            team_obp = team_batters['OBP'].mean() if 'OBP' in team_batters.columns else 0.320
            team_slg = team_batters['SLG'].mean() if 'SLG' in team_batters.columns else 0.410
            team_hr = team_batters['HR'].sum() if 'HR' in team_batters.columns else 100

            # Aggregate pitching statistics
            team_era = team_pitchers['ERA'].mean() if 'ERA' in team_pitchers.columns else 4.0
            team_whip = team_pitchers['WHIP'].mean() if 'WHIP' in team_pitchers.columns else 1.2
            team_so_rate = (team_pitchers['SO'].sum() / team_pitchers['IP'].sum()) * 9 if 'IP' in team_pitchers.columns else 8.5

            team_stats[team] = {
                'batting_avg': team_avg,
                'obp': team_obp,
                'slg': team_slg,
                'home_runs': team_hr,
                'era': team_era,
                'whip': team_whip,
                'k9': team_so_rate,
                'wins': len(team_pitchers),  # proxy for team strength
            }

        self.team_stats = team_stats
        print(f"[Pipeline] Computed stats for {len(team_stats)} teams")
        return team_stats

    def generate_game_features(self, away_team, home_team, away_pitcher_era=4.0, home_pitcher_era=4.0):
        """
        Engineer features for a specific game matchup.

        Args:
            away_team: Away team abbreviation
            home_team: Home team abbreviation
            away_pitcher_era: Away pitcher's ERA
            home_pitcher_era: Home pitcher's ERA

        Returns:
            dict: Feature dictionary for the game
        """
        away_stats = self.team_stats.get(away_team, self._default_team_stats())
        home_stats = self.team_stats.get(home_team, self._default_team_stats())

        features = {
            'away_batting_avg': away_stats['batting_avg'],
            'away_obp': away_stats['obp'],
            'away_slg': away_stats['slg'],
            'away_era_allowed': home_pitcher_era,
            'home_batting_avg': home_stats['batting_avg'],
            'home_obp': home_stats['obp'],
            'home_slg': home_stats['slg'],
            'home_era_allowed': away_pitcher_era,
            'away_pitcher_era': away_pitcher_era,
            'home_pitcher_era': home_pitcher_era,
            'home_field_advantage': 1.0,  # Home team always has advantage
            'runs_differential': (home_stats['batting_avg'] - away_stats['batting_avg']) * 1000,
            'pitching_quality_gap': away_pitcher_era - home_pitcher_era,
        }

        return features

    def _default_team_stats(self):
        """Return default team statistics."""
        return {
            'batting_avg': 0.260,
            'obp': 0.320,
            'slg': 0.410,
            'home_runs': 100,
            'era': 4.0,
            'whip': 1.2,
            'k9': 8.5,
            'wins': 20,
        }

    def create_training_dataset(self, num_samples=500):
        """
        Create a synthetic training dataset for model training.
        In production, this would use historical game outcomes.

        Args:
            num_samples: Number of training samples to generate

        Returns:
            tuple: (X_train, y_train) DataFrames for training
        """
        print(f"[Pipeline] Generating {num_samples} training samples...")

        teams = list(self.team_stats.keys())
        if not teams:
            teams = ['NYY', 'BOS', 'TB', 'LAD', 'SF']

        training_data = []

        for _ in range(num_samples):
            away_team = np.random.choice(teams)
            home_team = np.random.choice(teams)

            away_pitcher_era = np.random.uniform(2.5, 5.5)
            home_pitcher_era = np.random.uniform(2.5, 5.5)

            features = self.generate_game_features(away_team, home_team, away_pitcher_era, home_pitcher_era)

            # Synthetic outcome: home team wins more often (home field advantage)
            home_win_prob = 0.55 + (home_pitcher_era - away_pitcher_era) * 0.05
            home_win_prob = np.clip(home_win_prob, 0.3, 0.7)
            home_win = 1 if np.random.random() < home_win_prob else 0

            features['home_team'] = home_team
            features['away_team'] = away_team
            features['home_win'] = home_win

            training_data.append(features)

        df = pd.DataFrame(training_data)
        return df

    def train_model(self, training_df, test_size=0.2):
        """
        Train XGBoost model on historical game data.

        Args:
            training_df: DataFrame with training data
            test_size: Proportion of data to use for testing

        Returns:
            dict: Training metrics and model info
        """
        print("[Pipeline] Training XGBoost model...")

        # Prepare features
        feature_cols = [col for col in training_df.columns
                       if col not in ['home_team', 'away_team', 'home_win']]

        X = training_df[feature_cols].copy()
        y = training_df['home_win'].copy()

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model
        self.model = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            verbosity=0
        )

        self.model.fit(X_train_scaled, y_train, verbose=False)
        self.feature_columns = feature_cols

        # Evaluate
        train_accuracy = self.model.score(X_train_scaled, y_train)
        test_accuracy = self.model.score(X_test_scaled, y_test)

        metrics = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'n_samples': len(training_df),
            'n_features': len(feature_cols),
        }

        print(f"[Pipeline] Model trained: Train Acc={train_accuracy:.3f}, Test Acc={test_accuracy:.3f}")
        return metrics

    def predict_game(self, away_team, home_team, away_pitcher_name='TBA', home_pitcher_name='TBA',
                    away_pitcher_era=4.0, home_pitcher_era=4.0):
        """
        Generate prediction for a specific game.

        Args:
            away_team: Away team abbreviation
            home_team: Home team abbreviation
            away_pitcher_name: Away pitcher name
            home_pitcher_name: Home pitcher name
            away_pitcher_era: Away pitcher ERA
            home_pitcher_era: Home pitcher ERA

        Returns:
            dict: Prediction with probability and confidence
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")

        # Generate features
        features = self.generate_game_features(away_team, home_team, away_pitcher_era, home_pitcher_era)

        # Prepare feature vector
        feature_vector = np.array([features[col] for col in self.feature_columns]).reshape(1, -1)
        feature_vector_scaled = self.scaler.transform(feature_vector)

        # Get prediction and probability
        prediction = self.model.predict(feature_vector_scaled)[0]
        probability = self.model.predict_proba(feature_vector_scaled)[0]

        home_win_prob = probability[1]
        away_win_prob = probability[0]

        # Determine confidence tier
        max_prob = max(home_win_prob, away_win_prob)
        if max_prob >= 0.65:
            confidence_tier = 'HIGH'
        elif max_prob >= 0.55:
            confidence_tier = 'MEDIUM'
        else:
            confidence_tier = 'LOW'

        # Make pick (home team favored if prob > 0.5)
        pick = 'HOME' if home_win_prob > 0.5 else 'AWAY'

        result = {
            'away_team': away_team,
            'home_team': home_team,
            'away_pitcher': away_pitcher_name,
            'home_pitcher': home_pitcher_name,
            'home_win_probability': home_win_prob,
            'away_win_probability': away_win_prob,
            'pick': pick,
            'confidence_tier': confidence_tier,
            'confidence_score': max_prob,
        }

        return result

    def generate_predictions(self, schedule_df):
        """
        Generate predictions for a list of games.

        Args:
            schedule_df: DataFrame with game schedule
            Columns: game_id, date, away_team, home_team, away_pitcher, home_pitcher,
                    away_pitcher_era, home_pitcher_era

        Returns:
            DataFrame: Predictions for all games
        """
        print(f"[Pipeline] Generating predictions for {len(schedule_df)} games...")

        predictions = []

        for idx, row in schedule_df.iterrows():
            try:
                pred = self.predict_game(
                    away_team=row['away_team'],
                    home_team=row['home_team'],
                    away_pitcher_name=row.get('away_pitcher', 'TBA'),
                    home_pitcher_name=row.get('home_pitcher', 'TBA'),
                    away_pitcher_era=row.get('away_pitcher_era', 4.0),
                    home_pitcher_era=row.get('home_pitcher_era', 4.0)
                )

                pred['game_id'] = row.get('game_id', f'game_{idx}')
                pred['date'] = row.get('date', datetime.now().strftime('%Y-%m-%d'))
                predictions.append(pred)

            except Exception as e:
                print(f"[Pipeline] Error predicting game {idx}: {e}")
                continue

        self.predictions_df = pd.DataFrame(predictions)
        return self.predictions_df

    def export_predictions(self, filename=None):
        """
        Export predictions to CSV (simulating Google Sheets output).

        Args:
            filename: Output CSV filename

        Returns:
            str: Path to saved file
        """
        if self.predictions_df is None:
            raise ValueError("No predictions generated. Call generate_predictions() first.")

        if filename is None:
            filename = f'daily_predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'

        filepath = f'{self.output_dir}/{filename}'

        # Format for Google Sheets
        export_df = self.predictions_df[[
            'date', 'away_team', 'home_team', 'away_pitcher', 'home_pitcher',
            'home_win_probability', 'pick', 'confidence_tier', 'confidence_score'
        ]].copy()

        export_df.rename(columns={
            'home_win_probability': 'model_probability',
            'confidence_score': 'confidence'
        }, inplace=True)

        export_df.to_csv(filepath, index=False)
        print(f"[Pipeline] Predictions exported to {filepath}")

        return filepath


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='MLB Prediction Data Pipeline')
    parser.add_argument('--output-dir', default='./output', help='Output directory for predictions')
    parser.add_argument('--season', type=int, default=2024, help='MLB season year')
    parser.add_argument('--test-mode', default='false', help='Run in test mode (true/false)')
    args = parser.parse_args()

    output_dir = args.output_dir
    season = args.season

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("MLB PREDICTION DATA PIPELINE DEMO")
    print("=" * 70)

    # Initialize pipeline
    pipeline = MLBDataPipeline(output_dir=output_dir)

    # Step 1: Fetch data
    batting_stats, pitching_stats = pipeline.fetch_mlb_data(season=season)
    print(f"\nBatting Stats Shape: {batting_stats.shape}")
    print(f"Pitching Stats Shape: {pitching_stats.shape}")

    # Step 2: Compute team stats
    team_stats = pipeline.compute_team_stats(batting_stats, pitching_stats)
    print(f"\nSample Team Stats (NYY): {team_stats.get('NYY', 'N/A')}")

    # Step 3: Create training data
    training_df = pipeline.create_training_dataset(num_samples=500)
    print(f"\nTraining Data Shape: {training_df.shape}")
    print(f"Training Data Columns: {list(training_df.columns)}")

    # Step 4: Train model
    metrics = pipeline.train_model(training_df, test_size=0.2)
    print(f"\nModel Metrics: {metrics}")

    # Step 5: Generate sample schedule
    teams = ['NYY', 'BOS', 'TB', 'LAD', 'SF', 'CHC', 'ATL']
    schedule = []
    for i in range(10):
        game_date = (datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d')
        schedule.append({
            'game_id': f'mlb_{game_date}_{i:03d}',
            'date': game_date,
            'away_team': np.random.choice(teams),
            'home_team': np.random.choice(teams),
            'away_pitcher': f'Pitcher_{np.random.randint(1, 50)}',
            'home_pitcher': f'Pitcher_{np.random.randint(1, 50)}',
            'away_pitcher_era': np.random.uniform(2.5, 5.5),
            'home_pitcher_era': np.random.uniform(2.5, 5.5),
        })

    schedule_df = pd.DataFrame(schedule)

    # Step 6: Generate predictions
    predictions_df = pipeline.generate_predictions(schedule_df)
    print(f"\nGenerated Predictions:\n{predictions_df.head()}")

    # Step 7: Export results
    output_file = pipeline.export_predictions('daily_predictions.csv')
    print(f"\nExported to: {output_file}")

    # Summary statistics
    print("\n" + "=" * 70)
    print("PREDICTION SUMMARY")
    print("=" * 70)
    print(f"Total Predictions: {len(predictions_df)}")
    print(f"High Confidence Picks: {len(predictions_df[predictions_df['confidence_tier'] == 'HIGH'])}")
    print(f"Home Picks: {len(predictions_df[predictions_df['pick'] == 'HOME'])}")
    print(f"Away Picks: {len(predictions_df[predictions_df['pick'] == 'AWAY'])}")
    print(f"Average Confidence: {predictions_df['confidence_score'].mean():.3f}")


if __name__ == '__main__':
    main()
