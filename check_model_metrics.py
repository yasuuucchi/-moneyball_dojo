
import pickle
from pathlib import Path
import pandas as pd

models_dir = Path('/Users/taiki/moneyball_dojo/models')
project_dir = Path('/Users/taiki/moneyball_dojo')

print("="*80)
print(f"{'Model Name':<25} | {'Metric 1':<15} | {'Metric 2':<15} | {'Features':<8}")
print("-" * 80)

# 既存のモデルファイルをすべてチェック
model_files = sorted(models_dir.glob('*.pkl'))
# legacy model.pkl もチェック
if (project_dir / 'model.pkl').exists():
    model_files.append(project_dir / 'model.pkl')

for pkl_file in model_files:
    try:
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        
        name = data.get('model_type', pkl_file.stem)
        metrics = data.get('metrics', {})
        n_feat = data.get('n_features', 0)
        
        if 'pitcher_k' in pkl_file.stem:
            m1 = f"MAE: {metrics.get('k_mae', 0):.3f}"
            m2 = f"R2: {metrics.get('k_r2', 0):.3f}"
        elif 'pitcher_outs' in pkl_file.stem:
            m1 = f"MAE: {metrics.get('outs_mae', 0):.3f}"
            m2 = f"R2: {metrics.get('outs_r2', 0):.3f}"
        elif 'batter' in pkl_file.stem:
            m1 = f"H_MAE: {metrics.get('hit_mae', 0):.3f}"
            m2 = f"HR_MAE: {metrics.get('hr_mae', 0):.4f}"
        elif 'over_under' in pkl_file.stem:
            m1 = f"Acc: {metrics.get('accuracy', 0):.4f}"
            m2 = f"MAE: {metrics.get('reg_mae', 0):.3f}"
        else:
            m1 = f"Acc: {metrics.get('accuracy', 0):.4f}"
            m2 = f"AUC: {metrics.get('auc', 0):.4f}"
            
        print(f"{name:<25} | {m1:<15} | {m2:<15} | {n_feat:<8}")
    except Exception as e:
        print(f"{pkl_file.name:<25} | Error loading: {str(e)[:30]}")

print("="*80)
