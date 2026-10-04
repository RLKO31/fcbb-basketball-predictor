# ==============================================================================
# FC BAYERN BASKETBALL MODEL V3: ALL-IN-ONE STANDALONE KAGGLE PIPELINE
# Predicts Match Winners (Win Proba) & Exact Point Scores across BBL, EuroLeague, Pokal
# ==============================================================================
# Instructions for Kaggle:
# 1. Open a new Kaggle notebook (or existing one)
# 2. Copy and paste this ENTIRE script into the first code cell
# 3. Press Shift + Enter (Run)
# No external file uploads required! The dataset is automatically loaded or extracted.
# ==============================================================================

import os
import sys
import io
import gzip
import base64
import json
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    RandomForestRegressor,
    HistGradientBoostingRegressor
)
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error
)
import matplotlib.pyplot as plt

print("=" * 80)
print(" FC BAYERN BASKETBALL MODEL V3 (TACTICAL & 1D TIME-DECAY + SCORE REGRESSORS)")
print("=" * 80)

# --- STEP 1: LOAD OR EXTRACT MULTI-COMPETITION DATASET ---
csv_candidates = [
    'fcbb_multicomp_processed.csv',
    '/kaggle/input/fcbb-multicomp/fcbb_multicomp_processed.csv',
    'basketball_prediction/data/processed/fcbb_multicomp_processed.csv'
]

df = None
for p in csv_candidates:
    if os.path.exists(p):
        try:
            df = pd.read_csv(p)
            print(f"[Data] Loaded dataset from: {p} ({len(df)} rows)")
            break
        except Exception:
            pass

if df is None:
    print("[Data] No external CSV found. Decompressing embedded 430-match dataset...")
    EMBEDDED_B64 = "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"
    raw_bytes = gzip.decompress(base64.b64decode(EMBEDDED_B64))
    df = pd.read_csv(io.BytesIO(raw_bytes))
    print(f"[Data] Successfully restored embedded multi-competition dataset ({len(df)} matches across 2019-2025)!")

df['date_dt'] = pd.to_datetime(df['date'])
df = df.sort_values('date_dt').reset_index(drop=True)

print(f"\nTotal Matches: {len(df)}")
print("Competition Breakdown:", df['competition'].value_counts().to_dict())
print("Season Breakdown:", df['season'].value_counts().to_dict())
print(f"FC Bayern Historical Win Rate: {df['bayern_win'].mean()*100:.1f}%\n")

# --- STEP 2: 1D TIME-DECAY & TACTICAL FEATURE ENGINEERING ---
print("--- Step 2: Feature Engineering (1D Time-Decay & Tactical Possession Control) ---")
comp_weights = {'EuroLeague': 1.25, 'BBL': 1.00, 'BBL-Pokal': 1.05}
df['comp_tier'] = df['competition'].map(comp_weights).fillna(1.0)

team_histories = {}
gamma = 0.85        # Game recency decay
lambda_days = 0.015 # Calendar day decay

new_cols = {}
for W in range(1, 11):
    for pfx in ['home', 'away']:
        new_cols[f'{pfx}_pts_decay_{W}'] = []
        new_cols[f'{pfx}_pts_conceded_decay_{W}'] = []
        new_cols[f'{pfx}_pace_decay_{W}'] = []
        new_cols[f'{pfx}_ortg_decay_{W}'] = []
        new_cols[f'{pfx}_drtg_decay_{W}'] = []
        new_cols[f'{pfx}_net_rtg_decay_{W}'] = []
        new_cols[f'{pfx}_win_rate_decay_{W}'] = []
    new_cols[f'net_rtg_diff_{W}'] = []
    new_cols[f'pace_projected_{W}'] = []
    new_cols[f'off_def_mismatch_home_{W}'] = []
    new_cols[f'off_def_mismatch_away_{W}'] = []

for idx, row in df.iterrows():
    ht = row['home_team']
    at = row['away_team']
    dt = row['date_dt']
    
    def get_stats(team, W, cur_date):
        hist = team_histories.get(team, [])
        if not hist:
            return {'pts': 78.0, 'ptsc': 78.0, 'pace': 72.0, 'ortg': 108.0, 'drtg': 108.0, 'net_rtg': 0.0, 'win': 0.50}
        rec = hist[-W:]
        n = len(rec)
        w = np.array([np.exp(-lambda_days * max((cur_date - e['date']).days, 1)) * (gamma ** (n - 1 - k)) * e.get('tier', 1.0) for k, e in enumerate(rec)])
        w = w / np.sum(w) if np.sum(w) > 0 else np.ones(n) / n
        pts = np.sum([e['pts'] for e in rec] * w)
        ptsc = np.sum([e['ptsc'] for e in rec] * w)
        pace = np.sum([e['pace'] for e in rec] * w)
        ortg = np.sum([e['ortg'] for e in rec] * w)
        drtg = np.sum([e['drtg'] for e in rec] * w)
        win = np.sum([1.0 if e['pts'] > e['ptsc'] else 0.0 for e in rec] * w)
        return {'pts': pts, 'ptsc': ptsc, 'pace': pace, 'ortg': ortg, 'drtg': drtg, 'net_rtg': ortg - drtg, 'win': win}
        
    for W in range(1, 11):
        hs = get_stats(ht, W, dt)
        as_ = get_stats(at, W, dt)
        new_cols[f'home_pts_decay_{W}'].append(hs['pts'])
        new_cols[f'home_pts_conceded_decay_{W}'].append(hs['ptsc'])
        new_cols[f'home_pace_decay_{W}'].append(hs['pace'])
        new_cols[f'home_ortg_decay_{W}'].append(hs['ortg'])
        new_cols[f'home_drtg_decay_{W}'].append(hs['drtg'])
        new_cols[f'home_net_rtg_decay_{W}'].append(hs['net_rtg'])
        new_cols[f'home_win_rate_decay_{W}'].append(hs['win'])
        
        new_cols[f'away_pts_decay_{W}'].append(as_['pts'])
        new_cols[f'away_pts_conceded_decay_{W}'].append(as_['ptsc'])
        new_cols[f'away_pace_decay_{W}'].append(as_['pace'])
        new_cols[f'away_ortg_decay_{W}'].append(as_['ortg'])
        new_cols[f'away_drtg_decay_{W}'].append(as_['drtg'])
        new_cols[f'away_net_rtg_decay_{W}'].append(as_['net_rtg'])
        new_cols[f'away_win_rate_decay_{W}'].append(as_['win'])
        
        new_cols[f'net_rtg_diff_{W}'].append(hs['net_rtg'] - as_['net_rtg'])
        new_cols[f'pace_projected_{W}'].append((hs['pace'] + as_['pace']) / 2.0)
        new_cols[f'off_def_mismatch_home_{W}'].append(hs['ortg'] - as_['drtg'])
        new_cols[f'off_def_mismatch_away_{W}'].append(as_['ortg'] - hs['drtg'])
        
    team_histories.setdefault(ht, []).append({'date': dt, 'pts': row['home_score'], 'ptsc': row['away_score'], 'pace': row['pace'], 'ortg': row['home_ortg'], 'drtg': row['away_ortg'], 'tier': row['comp_tier']})
    team_histories.setdefault(at, []).append({'date': dt, 'pts': row['away_score'], 'ptsc': row['home_score'], 'pace': row['pace'], 'ortg': row['away_ortg'], 'drtg': row['home_ortg'], 'tier': row['comp_tier']})

cols_df = pd.DataFrame(new_cols)
df = pd.concat([df, cols_df], axis=1)

df['home_is_b2b'] = (df['home_rest'] <= 2).astype(int)
df['away_is_b2b'] = (df['away_rest'] <= 2).astype(int)
df['b2b_diff'] = df['away_is_b2b'] - df['home_is_b2b']
df['is_euroleague'] = (df['competition'] == 'EuroLeague').astype(int)
df['is_bbl'] = (df['competition'] == 'BBL').astype(int)
df['target_home_win'] = (df['home_score'] > df['away_score']).astype(int)

# --- STEP 3: CHRONOLOGICAL SPLIT & WINDOW SEARCH W in [1..10] ---
print("--- Step 3: Tactical Horizon Grid Search W in [1..10] ---")
train_seasons = ["2019-2020", "2020-2021", "2021-2022", "2022-2023"]
val_seasons = ["2023-2024"]
test_seasons = ["2024-2025"]

train_mask = df['season'].isin(train_seasons)
val_mask = df['season'].isin(val_seasons)
test_mask = df['season'].isin(test_seasons)
train_pool_mask = df['season'].isin(train_seasons + val_seasons)

def get_feature_cols(W):
    return [
        f'home_pts_decay_{W}', f'away_pts_decay_{W}',
        f'home_pts_conceded_decay_{W}', f'away_pts_conceded_decay_{W}',
        f'home_pace_decay_{W}', f'away_pace_decay_{W}',
        f'home_ortg_decay_{W}', f'away_ortg_decay_{W}',
        f'home_drtg_decay_{W}', f'away_drtg_decay_{W}',
        f'home_net_rtg_decay_{W}', f'away_net_rtg_decay_{W}',
        f'home_win_rate_decay_{W}', f'away_win_rate_decay_{W}',
        f'net_rtg_diff_{W}', f'pace_projected_{W}',
        f'off_def_mismatch_home_{W}', f'off_def_mismatch_away_{W}',
        'rest_diff', 'b2b_diff', 'comp_tier', 'is_euroleague', 'is_bbl'
    ]

best_w = 8
best_acc = 0.0
for W in range(1, 11):
    f_cols = get_feature_cols(W)
    X_tr = StandardScaler().fit_transform(df.loc[train_mask, f_cols].values)
    X_v = StandardScaler().fit(df.loc[train_mask, f_cols].values).transform(df.loc[val_mask, f_cols].values)
    y_tr = df.loc[train_mask, 'target_home_win'].values
    y_v = df.loc[val_mask, 'target_home_win'].values
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42).fit(X_tr, y_tr)
    hgb = HistGradientBoostingClassifier(max_iter=80, max_depth=3, random_state=42).fit(X_tr, y_tr)
    p_blend = 0.5 * rf.predict_proba(X_v)[:, 1] + 0.5 * hgb.predict_proba(X_v)[:, 1]
    acc = accuracy_score(y_v, (p_blend >= 0.5).astype(int)) * 100.0
    print(f"  Horizon W = {W:2d} matches | Validation Accuracy: {acc:.2f}%")
    if acc > best_acc:
        best_acc = acc
        best_w = W

print(f"\nOptimal Tactical Horizon Selected: W* = {best_w} matches (Val Acc: {best_acc:.2f}%)\n")

# --- STEP 4: FIT MODEL V3 & STACKED EXACT SCORE REGRESSORS ---
# --- STEP 4: FIT MODEL V3 & MULTIPLICATIVE DISPERSED REGRESSORS (STRATEGY 1 & 2) ---
print("--- Step 4: Training Winner Blend & Multiplicative Score Regressors (Strategy 1 & 2) ---")
opt_cols = get_feature_cols(best_w)
scaler = StandardScaler()
X_pool = scaler.fit_transform(df.loc[train_pool_mask, opt_cols].values)
X_test = scaler.transform(df.loc[test_mask, opt_cols].values)

y_pool_win = df.loc[train_pool_mask, 'target_home_win'].values
y_test_win = df.loc[test_mask, 'target_home_win'].values
y_pool_hs = df.loc[train_pool_mask, 'home_score'].values
y_pool_as = df.loc[train_pool_mask, 'away_score'].values
y_test_hs = df.loc[test_mask, 'home_score'].values
y_test_as = df.loc[test_mask, 'away_score'].values

y_pool_pace = df.loc[train_pool_mask, 'pace'].values
y_pool_oh = df.loc[train_pool_mask, 'home_ortg'].values
y_pool_oa = df.loc[train_pool_mask, 'away_ortg'].values

# Winner Classifier Blend
rf_final = RandomForestClassifier(n_estimators=150, max_depth=5, min_samples_leaf=3, random_state=42).fit(X_pool, y_pool_win)
hgb_final = HistGradientBoostingClassifier(max_iter=100, max_depth=4, learning_rate=0.04, min_samples_leaf=5, random_state=42).fit(X_pool, y_pool_win)

p_test = 0.5 * rf_final.predict_proba(X_test)[:, 1] + 0.5 * hgb_final.predict_proba(X_test)[:, 1]
test_acc = accuracy_score(y_test_win, (p_test >= 0.5).astype(int)) * 100.0
test_loss = log_loss(y_test_win, np.column_stack([1 - p_test, p_test]))
print(f"Final Model v3 Classification Test Accuracy: {test_acc:.2f}% (Log Loss: {test_loss:.4f})")

# Strategy 1: Multiplicative Component Regressors (Pace x Efficiency)
p_pool = 0.5 * rf_final.predict_proba(X_pool)[:, 1] + 0.5 * hgb_final.predict_proba(X_pool)[:, 1]
X_pool_fused = np.hstack([X_pool, p_pool.reshape(-1, 1), (1 - p_pool).reshape(-1, 1)])
X_test_fused = np.hstack([X_test, p_test.reshape(-1, 1), (1 - p_test).reshape(-1, 1)])

reg_pace = HistGradientBoostingRegressor(max_iter=80, max_depth=3, random_state=42).fit(X_pool_fused, y_pool_pace)
reg_oh = HistGradientBoostingRegressor(max_iter=100, max_depth=4, random_state=42).fit(X_pool_fused, y_pool_oh)
reg_oa = HistGradientBoostingRegressor(max_iter=100, max_depth=4, random_state=42).fit(X_pool_fused, y_pool_oa)

pred_pace_test = reg_pace.predict(X_test_fused)
pred_oh_test = reg_oh.predict(X_test_fused)
pred_oa_test = reg_oa.predict(X_test_fused)

raw_pred_hs = pred_pace_test * (pred_oh_test / 100.0)
raw_pred_as = pred_pace_test * (pred_oa_test / 100.0)

# Strategy 2: Calibrated Dispersion Expansion (Affine Scaling)
raw_pool_hs = reg_pace.predict(X_pool_fused) * (reg_oh.predict(X_pool_fused) / 100.0)
raw_pool_as = reg_pace.predict(X_pool_fused) * (reg_oa.predict(X_pool_fused) / 100.0)

mean_h = float(np.mean(y_pool_hs))
mean_a = float(np.mean(y_pool_as))
scale_h = float(np.clip((np.std(y_pool_hs) / np.std(raw_pool_hs)) * 0.85, 1.25, 1.85))
scale_a = float(np.clip((np.std(y_pool_as) / np.std(raw_pool_as)) * 0.85, 1.25, 1.85))

disp_pred_hs = mean_h + scale_h * (raw_pred_hs - mean_h)
disp_pred_as = mean_a + scale_a * (raw_pred_as - mean_a)

# Winner consistency alignment with classification probabilities
for i in range(len(disp_pred_hs)):
    pw = p_test[i]
    if pw >= 0.50 and disp_pred_hs[i] <= disp_pred_as[i]:
        disp_pred_hs[i] = disp_pred_as[i] + max(1.0, (pw - 0.50) * 16.0)
    elif pw < 0.50 and disp_pred_as[i] <= disp_pred_hs[i]:
        disp_pred_as[i] = disp_pred_hs[i] + max(1.0, (0.50 - pw) * 16.0)

mae_h = mean_absolute_error(y_test_hs, disp_pred_hs)
mae_a = mean_absolute_error(y_test_as, disp_pred_as)
mae_margin = mean_absolute_error(y_test_hs - y_test_as, disp_pred_hs - disp_pred_as)
spread_winner_acc = accuracy_score(y_test_win, (disp_pred_hs > disp_pred_as).astype(int)) * 100.0

print(f"\nExact Score Regression Metrics with Calibrated Dispersion (Test Season 2024-2025):")
print(f"  - Home Points MAE : {mae_h:.2f} pts (RMSE: {np.sqrt(mean_squared_error(y_test_hs, disp_pred_hs)):.2f}) | Std: {np.std(disp_pred_hs):.2f} pts (Actual Std: {np.std(y_test_hs):.2f})")
print(f"  - Away Points MAE : {mae_a:.2f} pts (RMSE: {np.sqrt(mean_squared_error(y_test_as, disp_pred_as)):.2f}) | Std: {np.std(disp_pred_as):.2f} pts (Actual Std: {np.std(y_test_as):.2f})")
print(f"  - Score Ranges    : Home [{np.min(disp_pred_hs):.1f} - {np.max(disp_pred_hs):.1f}] | Away [{np.min(disp_pred_as):.1f} - {np.max(disp_pred_as):.1f}]")
print(f"  - Margin MAE      : {mae_margin:.2f} pts")
print(f"  - Spread Implied Winner Accuracy: {spread_winner_acc:.2f}%\n")

# --- STEP 5: PREVIEW TEST MATCH EXACT PREDICTIONS ---
eval_df = df.loc[test_mask, ['date', 'competition', 'home_team', 'away_team', 'home_score', 'away_score']].copy()
eval_df['pred_home'] = np.round(disp_pred_hs).astype(int)
eval_df['pred_away'] = np.round(disp_pred_as).astype(int)
eval_df['home_win_prob'] = np.round(p_test * 100, 1)

print("Sample Predictions vs Ground Truth (Unseen 2024-2025 matches):")
for idx, r in eval_df.tail(6).iterrows():
    print(f"  [{r['date']}] {r['home_team']} vs {r['away_team']} ({r['competition']})")
    print(f"     Actual: {r['home_score']} - {r['away_score']} | Predicted: {r['pred_home']} - {r['pred_away']} (Home Win Prob: {r['home_win_prob']}%)")

# --- STEP 6: PLOT CALIBRATION ---
try:
    plt.figure(figsize=(9.5, 4.8))
    plt.scatter(eval_df['home_score'], eval_df['pred_home'], color='#dc2626', alpha=0.75, s=45, label='Home Points')
    plt.scatter(eval_df['away_score'], eval_df['pred_away'], color='#2563eb', alpha=0.75, s=45, label='Away Points')
    plt.plot([55, 115], [55, 115], 'k--', alpha=0.55, linewidth=1.5, label='Perfect Calibration (x=y)')
    plt.xlabel('Actual Points', fontsize=11, fontweight='bold')
    plt.ylabel('Model v3 Predicted Points (Strategy 1+2)', fontsize=11, fontweight='bold')
    plt.title('FC Bayern Basketball: Calibrated Score Dispersion (2024-2025 Test Season)', fontsize=12, fontweight='bold')
    plt.xlim(55, 120)
    plt.ylim(55, 120)
    plt.legend(frameon=True, facecolor='white', framealpha=0.9)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig('fcbb_calibration_plot.png', dpi=160)
    plt.close()
    print("\nCalibration chart rendered and saved as fcbb_calibration_plot.png.")
except Exception as e:
    print("Plotting skipped:", e)

print("\n" + "=" * 80)
print(" KAGGLE EXECUTION COMPLETE! MODEL V3 READY FOR INFERENCE.")
print("=" * 80)
