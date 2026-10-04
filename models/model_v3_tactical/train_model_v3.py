import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    RandomForestRegressor,
    HistGradientBoostingRegressor
)
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error, mean_squared_error, r2_score

def train_basketball_v3():
    print("======================================================================")
    print(" TRAINING BASKETBALL MODEL V3 (TACTICAL & 1D TIME-DECAY + SCORE REG)")
    print("======================================================================")
    
    csv_path = "basketball_prediction/data/processed/fcbb_multicomp_v3_tactical.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Processed dataset not found at {csv_path}")
        
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['home_score', 'away_score']).reset_index(drop=True)
    
    # Chronological split
    train_seasons = ["2019-2020", "2020-2021", "2021-2022", "2022-2023"]
    val_seasons = ["2023-2024"]
    test_seasons = ["2024-2025"]
    
    train_mask = df['season'].isin(train_seasons)
    val_mask = df['season'].isin(val_seasons)
    test_mask = df['season'].isin(test_seasons)
    
    # Combined pool for final fitting after hyperparameter selection
    train_pool_mask = df['season'].isin(train_seasons + val_seasons)
    
    y_train = df.loc[train_mask, 'target_home_win'].values
    y_val = df.loc[val_mask, 'target_home_win'].values
    y_test = df.loc[test_mask, 'target_home_win'].values
    
    y_train_hs = df.loc[train_mask, 'home_score'].values
    y_train_as = df.loc[train_mask, 'away_score'].values
    y_val_hs = df.loc[val_mask, 'home_score'].values
    y_val_as = df.loc[val_mask, 'away_score'].values
    y_test_hs = df.loc[test_mask, 'home_score'].values
    y_test_as = df.loc[test_mask, 'away_score'].values
    
    # --- STEP 1: Window Grid Search across W in [1..10] ---
    print("\n--- 1. Evaluating Rolling Tactical Horizons W in [1..10] ---")
    window_results = {}
    best_w = 7
    best_val_acc = 0.0
    best_val_loss = 999.0
    
    def get_features_for_w(W):
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
        
    for W in range(1, 11):
        feature_cols = get_features_for_w(W)
        
        X_tr = df.loc[train_mask, feature_cols].values
        X_v = df.loc[val_mask, feature_cols].values
        
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_v_s = scaler.transform(X_v)
        
        rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        rf.fit(X_tr_s, y_train)
        
        hgb = HistGradientBoostingClassifier(max_iter=80, max_depth=3, learning_rate=0.05, random_state=42)
        hgb.fit(X_tr_s, y_train)
        
        p_rf = rf.predict_proba(X_v_s)[:, 1]
        p_hgb = hgb.predict_proba(X_v_s)[:, 1]
        p_blend = 0.50 * p_rf + 0.50 * p_hgb
        
        preds = (p_blend >= 0.5).astype(int)
        acc = accuracy_score(y_val, preds) * 100.0
        loss = log_loss(y_val, np.column_stack([1 - p_blend, p_blend]))
        
        window_results[W] = {"val_accuracy": round(acc, 2), "val_log_loss": round(loss, 4)}
        print(f"  - Horizon W = {W:2d} matches | Val Accuracy: {acc:5.2f}% | Log Loss: {loss:.4f}")
        
        if acc > best_val_acc or (acc == best_val_acc and loss < best_val_loss):
            best_val_acc = acc
            best_val_loss = loss
            best_w = W
            
    print(f"\nOptimal Tactical Horizon: W* = {best_w} matches (Val Accuracy: {best_val_acc:.2f}%)")
    
    # --- STEP 2: Train Final Model v3 Ensemble using W* ---
    print(f"\n--- 2. Building Model v3 Feature Set with W* = {best_w} ---")
    optimal_features = get_features_for_w(best_w)
    
    X_pool_raw = df.loc[train_pool_mask, optimal_features].values
    y_pool = df.loc[train_pool_mask, 'target_home_win'].values
    y_pool_hs = df.loc[train_pool_mask, 'home_score'].values
    y_pool_as = df.loc[train_pool_mask, 'away_score'].values
    
    X_test_raw = df.loc[test_mask, optimal_features].values
    
    scaler_v3 = StandardScaler()
    X_pool_s = scaler_v3.fit_transform(X_pool_raw)
    X_test_s = scaler_v3.transform(X_test_raw)
    
    rf_v3 = RandomForestClassifier(n_estimators=150, max_depth=5, min_samples_leaf=3, random_state=42)
    rf_v3.fit(X_pool_s, y_pool)
    
    hgb_v3 = HistGradientBoostingClassifier(max_iter=100, max_depth=4, learning_rate=0.04, min_samples_leaf=5, random_state=42)
    hgb_v3.fit(X_pool_s, y_pool)
    
    p_rf_test = rf_v3.predict_proba(X_test_s)[:, 1]
    p_hgb_test = hgb_v3.predict_proba(X_test_s)[:, 1]
    p_blend_test = 0.50 * p_rf_test + 0.50 * p_hgb_test
    
    test_preds = (p_blend_test >= 0.5).astype(int)
    test_acc = accuracy_score(y_test, test_preds) * 100.0
    test_loss = log_loss(y_test, np.column_stack([1 - p_blend_test, p_blend_test]))
    
    print(f"Final Model v3 Classification Test Accuracy: {test_acc:.2f}% | Log Loss: {test_loss:.4f}")
    
    # --- STEP 3: Multiplicative Pace x Efficiency & Dispersion Expansion (Strategy 1 & 2) ---
    print("\n--- 3. Training Multiplicative Score Regressors & Calibrated Dispersion (Strategy 1 & 2) ---")
    p_rf_pool = rf_v3.predict_proba(X_pool_s)[:, 1]
    p_hgb_pool = hgb_v3.predict_proba(X_pool_s)[:, 1]
    p_blend_pool = 0.50 * p_rf_pool + 0.50 * p_hgb_pool
    
    # Fuse win probabilities into score regression features
    X_pool_fused = np.hstack([X_pool_s, p_blend_pool.reshape(-1, 1), (1 - p_blend_pool).reshape(-1, 1)])
    X_test_fused = np.hstack([X_test_s, p_blend_test.reshape(-1, 1), (1 - p_blend_test).reshape(-1, 1)])
    
    y_pool_pace = df.loc[train_pool_mask, 'pace'].values
    y_pool_oh = df.loc[train_pool_mask, 'home_ortg'].values
    y_pool_oa = df.loc[train_pool_mask, 'away_ortg'].values
    
    # 1. Component Regressors: Pace, Home ORtg, Away ORtg
    reg_pace_v3 = HistGradientBoostingRegressor(max_iter=80, max_depth=3, random_state=42)
    reg_pace_v3.fit(X_pool_fused, y_pool_pace)
    
    reg_oh_v3 = HistGradientBoostingRegressor(max_iter=100, max_depth=4, random_state=42)
    reg_oh_v3.fit(X_pool_fused, y_pool_oh)
    
    reg_oa_v3 = HistGradientBoostingRegressor(max_iter=100, max_depth=4, random_state=42)
    reg_oa_v3.fit(X_pool_fused, y_pool_oa)
    
    # Multiplicative predictions
    pred_pace_test = reg_pace_v3.predict(X_test_fused)
    pred_oh_test = reg_oh_v3.predict(X_test_fused)
    pred_oa_test = reg_oa_v3.predict(X_test_fused)
    
    raw_test_hs = pred_pace_test * (pred_oh_test / 100.0)
    raw_test_as = pred_pace_test * (pred_oa_test / 100.0)
    
    # Strategy 2: Calibrated Dispersion Expansion
    raw_pool_hs = reg_pace_v3.predict(X_pool_fused) * (reg_oh_v3.predict(X_pool_fused) / 100.0)
    raw_pool_as = reg_pace_v3.predict(X_pool_fused) * (reg_oa_v3.predict(X_pool_fused) / 100.0)
    
    mean_h = float(np.mean(y_pool_hs))
    mean_a = float(np.mean(y_pool_as))
    scale_h = float(np.clip((np.std(y_pool_hs) / np.std(raw_pool_hs)) * 0.85, 1.25, 1.85))
    scale_a = float(np.clip((np.std(y_pool_as) / np.std(raw_pool_as)) * 0.85, 1.25, 1.85))
    
    pred_test_hs = mean_h + scale_h * (raw_test_hs - mean_h)
    pred_test_as = mean_a + scale_a * (raw_test_as - mean_a)
    
    # Winner consistency alignment
    for i in range(len(pred_test_hs)):
        pw = p_blend_test[i]
        if pw >= 0.50 and pred_test_hs[i] <= pred_test_as[i]:
            pred_test_hs[i] = pred_test_as[i] + max(1.0, (pw - 0.50) * 16.0)
        elif pw < 0.50 and pred_test_as[i] <= pred_test_hs[i]:
            pred_test_as[i] = pred_test_hs[i] + max(1.0, (0.50 - pw) * 16.0)
            
    mae_home = mean_absolute_error(y_test_hs, pred_test_hs)
    rmse_home = np.sqrt(mean_squared_error(y_test_hs, pred_test_hs))
    mae_away = mean_absolute_error(y_test_as, pred_test_as)
    rmse_away = np.sqrt(mean_squared_error(y_test_as, pred_test_as))
    
    # Margin & Over/Under accuracy
    actual_margin = y_test_hs - y_test_as
    pred_margin = pred_test_hs - pred_test_as
    mae_margin = mean_absolute_error(actual_margin, pred_margin)
    score_spread_winner_acc = accuracy_score(y_test, (pred_margin > 0).astype(int)) * 100.0
    
    print(f"High-Variance Multiplicative Score Evaluation on Test Set (2024-2025):")
    print(f"  - Home Points MAE : {mae_home:.2f} pts (RMSE: {rmse_home:.2f}) | Std: {np.std(pred_test_hs):.2f} pts (Actual Std: {np.std(y_test_hs):.2f})")
    print(f"  - Away Points MAE : {mae_away:.2f} pts (RMSE: {rmse_away:.2f}) | Std: {np.std(pred_test_as):.2f} pts (Actual Std: {np.std(y_test_as):.2f})")
    print(f"  - Score Ranges    : Home [{np.min(pred_test_hs):.1f} - {np.max(pred_test_hs):.1f}] | Away [{np.min(pred_test_as):.1f} - {np.max(pred_test_as):.1f}]")
    print(f"  - Point Margin MAE: {mae_margin:.2f} pts")
    print(f"  - Spread Implied Winner Accuracy: {score_spread_winner_acc:.2f}%")
    
    # Sample test match predictions
    print("\nSample Model v3 Exact Score Predictions vs Actual (2024-2025):")
    sample_df = df.loc[test_mask, ['date', 'competition', 'home_team', 'away_team', 'home_score', 'away_score']].copy()
    sample_df['pred_home'] = np.round(pred_test_hs).astype(int)
    sample_df['pred_away'] = np.round(pred_test_as).astype(int)
    sample_df['win_prob_home'] = np.round(p_blend_test * 100, 1)
    
    for idx, r in sample_df.tail(6).iterrows():
        print(f"  [{r['date']}] {r['home_team']} vs {r['away_team']} ({r['competition']})")
        print(f"     Actual: {r['home_score']} - {r['away_score']} | Predicted: {r['pred_home']} - {r['pred_away']} (Home Win Prob: {r['win_prob_home']}%)")
        
    # --- STEP 4: Feature Importance Analysis ---
    importances = rf_v3.feature_importances_
    feat_imp = sorted(zip(optimal_features, importances), key=lambda x: x[1], reverse=True)
    print("\nTop 7 Most Influential Tactical Features in Model v3:")
    for feat, imp in feat_imp[:7]:
        print(f"  - {feat:30s}: {imp*100:5.2f}%")
        
    # Render and update calibration plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(9.5, 4.8))
        plt.scatter(y_test_hs, pred_test_hs, color='#dc2626', alpha=0.75, s=45, label='Home Points')
        plt.scatter(y_test_as, pred_test_as, color='#2563eb', alpha=0.75, s=45, label='Away Points')
        plt.plot([55, 115], [55, 115], 'k--', alpha=0.55, linewidth=1.5, label='Perfect Calibration (x=y)')
        plt.xlabel('Actual Points', fontsize=11, fontweight='bold')
        plt.ylabel('Model v3 Predicted Points (Strategy 1+2)', fontsize=11, fontweight='bold')
        plt.title('FC Bayern Basketball: Calibrated Score Dispersion (2024-2025 Test Season)', fontsize=12, fontweight='bold')
        plt.xlim(55, 120)
        plt.ylim(55, 120)
        plt.legend(frameon=True, facecolor='white', framealpha=0.9)
        plt.grid(True, alpha=0.25)
        cal_plot_path = os.path.abspath('fcbb_calibration_plot.png')
        plt.savefig(cal_plot_path, dpi=160)
        plt.close()
        print(f"Updated calibration plot saved to {cal_plot_path}")
    except Exception as e:
        print("Note on calibration plot generation:", e)
        
    # --- STEP 5: Serialize Artifacts and Optimal Configuration ---
    artifacts_dir = "basketball_prediction/models/model_v3_tactical"
    os.makedirs(artifacts_dir, exist_ok=True)
    
    joblib.dump(scaler_v3, os.path.join(artifacts_dir, "scaler_v3.joblib"))
    joblib.dump(rf_v3, os.path.join(artifacts_dir, "model_v3_rf.joblib"))
    joblib.dump(hgb_v3, os.path.join(artifacts_dir, "model_v3_hgb.joblib"))
    joblib.dump(reg_pace_v3, os.path.join(artifacts_dir, "model_v3_reg_pace.joblib"))
    joblib.dump(reg_oh_v3, os.path.join(artifacts_dir, "model_v3_reg_oh.joblib"))
    joblib.dump(reg_oa_v3, os.path.join(artifacts_dir, "model_v3_reg_oa.joblib"))
    
    config = {
        "model_name": "Model v3 Basketball Tactical & Time-Decay (Multiplicative & Dispersed)",
        "best_window": best_w,
        "window_results": window_results,
        "classification": {
            "test_accuracy": round(test_acc, 2),
            "test_log_loss": round(test_loss, 4)
        },
        "score_regression": {
            "mae_home": round(mae_home, 2),
            "rmse_home": round(rmse_home, 2),
            "mae_away": round(mae_away, 2),
            "rmse_away": round(rmse_away, 2),
            "mae_margin": round(mae_margin, 2),
            "implied_winner_accuracy": round(score_spread_winner_acc, 2),
            "dispersion_scaling": {
                "scale_home": round(scale_h, 3),
                "scale_away": round(scale_a, 3),
                "baseline_mean_home": round(mean_h, 2),
                "baseline_mean_away": round(mean_a, 2)
            }
        },
        "optimal_features": optimal_features,
        "top_features": [{"feature": f, "importance": round(imp * 100, 2)} for f, imp in feat_imp[:10]]
    }
    
    config_path = os.path.join(artifacts_dir, "model_v3_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
        
    print(f"\nAll Model v3 artifacts and configuration saved to {artifacts_dir}/")
    return config

if __name__ == '__main__':
    train_basketball_v3()
