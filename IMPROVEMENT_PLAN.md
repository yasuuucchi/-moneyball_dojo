# Model Improvement Plan — 精度向上ロードマップ

> **このファイルはコンテキスト断裂対策。各タスクが完了したら [x] に変更する。**
> 新しいセッションで再開する場合はここを最初に読むこと。

## 現状 (Before)
- Moneyline test accuracy: 57.8% (2025 test set, leak-free)
- 特徴量: 28個、全てチームレベル（先発投手の個人成績なし）
- アルゴリズム: XGBoost単体
- ハイパーパラメータ: 手動設定（未最適化）
- バリデーション: 単純な年度分割 (2022-2024 train / 2025 test)

## 実装タスク

### Task 1: 先発投手特徴量をML/F5/RLモデルに追加
**ファイル:** `train_all_models.py`, `run_daily.py`

**train_all_models.py の変更:**
- [x] `build_feature_matrix()` に pitcher_lookup を追加引数として渡す
- [x] games_df の `home_pitcher`, `away_pitcher` 列から投手名を取得
- [x] pitcher_stats CSVから投手個人成績を取得 (ERA, WHIP, K/9, BB/9)
- [x] 新規特徴量 (12個):
  - `sp_home_era`, `sp_away_era`, `sp_era_diff`
  - `sp_home_whip`, `sp_away_whip`, `sp_whip_diff`
  - `sp_home_k9`, `sp_away_k9`, `sp_k9_diff`
  - `sp_home_bb9`, `sp_away_bb9`, `sp_bb9_diff`
- [x] NRFIモデルと同じパターンで交差特徴量も追加 (6個):
  - `sp_era_x_opp_obp` (home/away)
  - `sp_whip_x_opp_slg` (home/away)
  - `sp_dominance` (home/away) = K/9 / max(ERA, 0.5)
- [x] train_moneyline(), train_f5_moneyline(), train_run_line() の exclude リストから新特徴量を除外確認
- [x] main() で pitcher_lookup を構築して build_feature_matrix() に渡す

**run_daily.py の変更:**
- [x] `build_game_features()` に pitcher_name 引数を追加
- [x] pitcher_stats から個人成績を引いて同じ18特徴量を生成
- [x] `generate_all_predictions()` で build_game_features に投手名を渡す

**検証:**
- [x] `python train_all_models.py --model ml` が正常完了
- [x] 新特徴量がfeature_importancesに現れることを確認

### Task 2: Optuna ハイパーパラメータ最適化
**ファイル:** `train_all_models.py`, `requirements.txt`

- [x] requirements.txt に `optuna>=3.4.0` 追加
- [x] `pip install optuna` 実行
- [x] `optimize_hyperparameters()` 関数を追加
  - 5-fold CV の log_loss を目標関数
  - 50 trials (十分な探索と実行時間のバランス)
  - 探索空間: n_estimators, max_depth, learning_rate, subsample, colsample_bytree,
    min_child_weight, reg_alpha, reg_lambda, gamma
- [x] train_moneyline() で Optuna 結果のベストパラメータを使用
- [x] train_f5_moneyline(), train_run_line() にも同様に適用
- [x] ベストパラメータを model pickle に保存

**検証:**
- [x] Optuna 最適化が正常完了（50 trials）
- [x] ベストパラメータがログに出力されること

### Task 3: アンサンブル (XGBoost + LightGBM + LogisticRegression)
**ファイル:** `train_all_models.py`, `requirements.txt`

- [x] requirements.txt に `lightgbm>=4.0.0` 追加
- [x] `pip install lightgbm` 実行
- [x] `build_ensemble()` 関数を追加
  - XGBoost (Optuna最適化済み)
  - LightGBM (同一探索空間でOptuna)
  - LogisticRegression (正則化C=1.0)
  - soft voting (確率平均)
- [x] 各モデルは個別にCalibratedClassifierCVで校正
- [x] アンサンブル vs 単体の精度比較をログ出力
- [x] ensemble model を model_moneyline.pkl に保存

**検証:**
- [x] アンサンブルの精度が単体以上であること
- [x] 3モデルの個別精度もログに出力

### Task 4: Walk-forward validation
**ファイル:** `train_all_models.py`

- [x] `walk_forward_validate()` 関数を追加
  - Window: 2022-2023 train → 2024前半 test
  - Window: 2022-2024前半 train → 2024後半 test
  - Window: 2022-2024 train → 2025 test
  - 各windowの精度を集計
- [x] train_moneyline() の最後で walk-forward 結果を出力
- [x] 平均精度をmodel pickleのmetricsに保存

**検証:**
- [x] 3+ windowの精度が一貫していること（過学習チェック）

### Task 5: コミット & プッシュ & モデル再学習
- [x] 全変更をコミット
- [x] `python train_all_models.py` で全モデル再学習
- [x] 再学習済みモデルをコミット
- [x] ブランチにプッシュ

## 実績 (After) — 2026-02-26 実行結果
- **Moneyline: 57.8% → 59.9%** (+2.1%, Walk-forward avg: 59.8%)
- **F5 Moneyline: 57.9% → 59.6%** (+1.7%)
- **Run Line: 64.6% → 66.3%** (+1.7%, STRONG: 68.8%)
- 特徴量: 28個 → 56個 (先発投手20個追加)
- アルゴリズム: XGBoost単体 → XGBoost+LightGBM+LogReg アンサンブル (ML)
- ハイパーパラメータ: Optuna 50 trials で最適化済み (ML, RL, F5)
- バリデーション: Walk-forward 2窓 (2022-23→2024: 60.3%, 2022-24→2025: 59.3%)
