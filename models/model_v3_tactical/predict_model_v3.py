import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime

def run_predictions():
    print("=== Generating Model v3 Predictions for Basketball Fixtures ===")
    
    artifacts_dir = "basketball_prediction/models/model_v3_tactical"
    config_path = os.path.join(artifacts_dir, "model_v3_config.json")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found at {config_path}. Run train_model_v3.py first.")
        
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
        
    optimal_features = config["optimal_features"]
    best_w = config["best_window"]
    
    scaler = joblib.load(os.path.join(artifacts_dir, "scaler_v3.joblib"))
    rf = joblib.load(os.path.join(artifacts_dir, "model_v3_rf.joblib"))
    hgb = joblib.load(os.path.join(artifacts_dir, "model_v3_hgb.joblib"))
    reg_home = joblib.load(os.path.join(artifacts_dir, "model_v3_reg_home.joblib"))
    reg_away = joblib.load(os.path.join(artifacts_dir, "model_v3_reg_away.joblib"))
    
    # Load processed historical dataset
    df_proc = pd.read_csv("basketball_prediction/data/processed/fcbb_multicomp_v3_tactical.csv")
    
    # Define upcoming / featured marquee matchups across BBL, EuroLeague, and Pokal
    fixtures = [
        # BBL Clashes
        {
            "id": "bbl_2025_alba",
            "competition": "BBL",
            "round": "Round 29 - Regular Season",
            "date": "2025-05-11",
            "home_team": "FC Bayern München",
            "away_team": "ALBA Berlin",
            "venue": "BMW Park, Munich",
            "home_rest": 3,
            "away_rest": 6
        },
        {
            "id": "bbl_2025_ulm",
            "competition": "BBL",
            "round": "Round 30 - Regular Season",
            "date": "2025-05-15",
            "home_team": "ratiopharm ulm",
            "away_team": "FC Bayern München",
            "venue": "ratiopharm arena, Neu-Ulm",
            "home_rest": 5,
            "away_rest": 4
        },
        {
            "id": "bbl_2025_bonn",
            "competition": "BBL",
            "round": "Round 31 - Regular Season",
            "date": "2025-05-18",
            "home_team": "FC Bayern München",
            "away_team": "Telekom Baskets Bonn",
            "venue": "BMW Park, Munich",
            "home_rest": 3,
            "away_rest": 7
        },
        {
            "id": "bbl_2025_chemnitz",
            "competition": "BBL",
            "round": "Round 32 - Regular Season",
            "date": "2025-05-22",
            "home_team": "NINERS Chemnitz",
            "away_team": "FC Bayern München",
            "venue": "Messe Chemnitz",
            "home_rest": 6,
            "away_rest": 4
        },
        {
            "id": "bbl_2025_wuerzburg",
            "competition": "BBL",
            "round": "Round 33 - Regular Season",
            "date": "2025-05-25",
            "home_team": "FC Bayern München",
            "away_team": "Würzburg Baskets",
            "venue": "BMW Park, Munich",
            "home_rest": 3,
            "away_rest": 7
        },
        # EuroLeague Playoff / Final 4 Marquee Matches
        {
            "id": "el_2025_real",
            "competition": "EuroLeague",
            "round": "Playoffs Round 1",
            "date": "2025-05-08",
            "home_team": "Real Madrid",
            "away_team": "FC Bayern München",
            "venue": "WiZink Center, Madrid",
            "home_rest": 4,
            "away_rest": 3
        },
        {
            "id": "el_2025_pana",
            "competition": "EuroLeague",
            "round": "Playoffs Round 2",
            "date": "2025-05-13",
            "home_team": "FC Bayern München",
            "away_team": "Panathinaikos AKTOR Athens",
            "venue": "SAP Garden, Munich",
            "home_rest": 2,
            "away_rest": 4
        },
        {
            "id": "el_2025_fener",
            "competition": "EuroLeague",
            "round": "Playoffs Round 3",
            "date": "2025-05-16",
            "home_team": "Fenerbahce Beko Istanbul",
            "away_team": "FC Bayern München",
            "venue": "Ulker Sports Arena, Istanbul",
            "home_rest": 4,
            "away_rest": 3
        },
        {
            "id": "el_2025_oly",
            "competition": "EuroLeague",
            "round": "EuroLeague Final Four",
            "date": "2025-05-23",
            "home_team": "FC Bayern München",
            "away_team": "Olympiacos Piraeus",
            "venue": "Etihad Arena / Final Four Host",
            "home_rest": 5,
            "away_rest": 5
        },
        {
            "id": "el_2025_barca",
            "competition": "EuroLeague",
            "round": "EuroLeague Marquee",
            "date": "2025-05-28",
            "home_team": "FC Barcelona",
            "away_team": "FC Bayern München",
            "venue": "Palau Blaugrana, Barcelona",
            "home_rest": 4,
            "away_rest": 3
        },
        # BBL-Pokal Knockout Clash
        {
            "id": "pokal_2025_final",
            "competition": "BBL-Pokal",
            "round": "TOP FOUR Final",
            "date": "2025-06-01",
            "home_team": "FC Bayern München",
            "away_team": "ALBA Berlin",
            "venue": "BMW Park, Munich (Host)",
            "home_rest": 1,
            "away_rest": 1
        }
    ]
    
    # Precompute baseline tactical profiles from latest state in df_proc
    team_profiles = {}
    for team in set(df_proc['home_team']).union(set(df_proc['away_team'])):
        h_sub = df_proc[df_proc['home_team'] == team]
        a_sub = df_proc[df_proc['away_team'] == team]
        
        pts_scored = []
        pts_conceded = []
        pace_list = []
        wins = []
        
        for idx, r in h_sub.iterrows():
            pts_scored.append(r['home_score'])
            pts_conceded.append(r['away_score'])
            pace_list.append(r['pace'])
            wins.append(1 if r['home_score'] > r['away_score'] else 0)
            
        for idx, r in a_sub.iterrows():
            pts_scored.append(r['away_score'])
            pts_conceded.append(r['home_score'])
            pace_list.append(r['pace'])
            wins.append(1 if r['away_score'] > r['home_score'] else 0)
            
        if pts_scored:
            avg_pts = np.mean(pts_scored[-8:])
            avg_ptsc = np.mean(pts_conceded[-8:])
            avg_pace = np.mean(pace_list[-8:])
            avg_win = np.mean(wins[-8:])
        else:
            avg_pts, avg_ptsc, avg_pace, avg_win = 78.0, 78.0, 72.0, 0.50
            
        ortg = round((avg_pts / avg_pace) * 100, 1)
        drtg = round((avg_ptsc / avg_pace) * 100, 1)
        team_profiles[team] = {
            'pts': avg_pts,
            'pts_conceded': avg_ptsc,
            'pace': avg_pace,
            'ortg': ortg,
            'drtg': drtg,
            'net_rtg': ortg - drtg,
            'win_rate': avg_win
        }
        
    predictions_output = []
    
    for f_idx, fix in enumerate(fixtures):
        ht = fix['home_team']
        at = fix['away_team']
        comp = fix['competition']
        
        hp = team_profiles.get(ht, team_profiles['FC Bayern München'])
        ap = team_profiles.get(at, {'pts': 77.0, 'pts_conceded': 79.0, 'pace': 71.5, 'ortg': 107.0, 'drtg': 110.0, 'net_rtg': -3.0, 'win_rate': 0.45})
        
        h_rest = fix['home_rest']
        a_rest = fix['away_rest']
        rest_diff = h_rest - a_rest
        h_b2b = 1 if h_rest <= 2 else 0
        a_b2b = 1 if a_rest <= 2 else 0
        b2b_diff = a_b2b - h_b2b
        
        comp_tier = 1.25 if comp == 'EuroLeague' else (1.05 if comp == 'BBL-Pokal' else 1.0)
        is_el = 1 if comp == 'EuroLeague' else 0
        is_bbl = 1 if comp == 'BBL' else 0
        
        # Build feature vector matching optimal_features
        feat_vals = []
        for col in optimal_features:
            if col == f'home_pts_decay_{best_w}': feat_vals.append(hp['pts'])
            elif col == f'away_pts_decay_{best_w}': feat_vals.append(ap['pts'])
            elif col == f'home_pts_conceded_decay_{best_w}': feat_vals.append(hp['pts_conceded'])
            elif col == f'away_pts_conceded_decay_{best_w}': feat_vals.append(ap['pts_conceded'])
            elif col == f'home_pace_decay_{best_w}': feat_vals.append(hp['pace'])
            elif col == f'away_pace_decay_{best_w}': feat_vals.append(ap['pace'])
            elif col == f'home_ortg_decay_{best_w}': feat_vals.append(hp['ortg'])
            elif col == f'away_ortg_decay_{best_w}': feat_vals.append(ap['ortg'])
            elif col == f'home_drtg_decay_{best_w}': feat_vals.append(hp['drtg'])
            elif col == f'away_drtg_decay_{best_w}': feat_vals.append(ap['drtg'])
            elif col == f'home_net_rtg_decay_{best_w}': feat_vals.append(hp['net_rtg'])
            elif col == f'away_net_rtg_decay_{best_w}': feat_vals.append(ap['net_rtg'])
            elif col == f'home_win_rate_decay_{best_w}': feat_vals.append(hp['win_rate'])
            elif col == f'away_win_rate_decay_{best_w}': feat_vals.append(ap['win_rate'])
            elif col == f'net_rtg_diff_{best_w}': feat_vals.append(hp['net_rtg'] - ap['net_rtg'])
            elif col == f'pace_projected_{best_w}': feat_vals.append((hp['pace'] + ap['pace']) / 2.0)
            elif col == f'off_def_mismatch_home_{best_w}': feat_vals.append(hp['ortg'] - ap['drtg'])
            elif col == f'off_def_mismatch_away_{best_w}': feat_vals.append(ap['ortg'] - hp['drtg'])
            elif col == 'rest_diff': feat_vals.append(rest_diff)
            elif col == 'b2b_diff': feat_vals.append(b2b_diff)
            elif col == 'comp_tier': feat_vals.append(comp_tier)
            elif col == 'is_euroleague': feat_vals.append(is_el)
            elif col == 'is_bbl': feat_vals.append(is_bbl)
            else: feat_vals.append(0.0)
            
        x_raw = np.array(feat_vals).reshape(1, -1)
        x_s = scaler.transform(x_raw)
        
        p_rf = rf.predict_proba(x_s)[0, 1]
        p_hgb = hgb.predict_proba(x_s)[0, 1]
        p_home_win = round(0.50 * p_rf + 0.50 * p_hgb, 3)
        p_away_win = round(1.0 - p_home_win, 3)
        
        # Exact score regression
        x_fused = np.hstack([x_s, np.array([[p_home_win, p_away_win]])])
        pred_hs = float(reg_home.predict(x_fused)[0])
        pred_as = float(reg_away.predict(x_fused)[0])
        
        # Enforce consistency between win probability and score margin
        pred_score_h = int(np.round(pred_hs))
        pred_score_a = int(np.round(pred_as))
        if p_home_win > 0.52 and pred_score_h <= pred_score_a:
            pred_score_h = pred_score_a + 2
        elif p_home_win < 0.48 and pred_score_a <= pred_score_h:
            pred_score_a = pred_score_h + 2
            
        proj_total = pred_score_h + pred_score_a
        proj_margin = pred_score_h - pred_score_a
        proj_pace = round((hp['pace'] + ap['pace']) / 2.0, 1)
        
        is_bayern_home = (ht == "FC Bayern München")
        bayern_prob = p_home_win if is_bayern_home else p_away_win
        bayern_pts = pred_score_h if is_bayern_home else pred_score_a
        opp_pts = pred_score_a if is_bayern_home else pred_score_h
        opp_name = at if is_bayern_home else ht
        
        record = {
            "id": fix["id"],
            "competition": comp,
            "round": fix["round"],
            "date": fix["date"],
            "venue": fix["venue"],
            "home_team": ht,
            "away_team": at,
            "opponent": opp_name,
            "is_bayern_home": is_bayern_home,
            "probabilities": {
                "home_win": round(p_home_win * 100, 1),
                "away_win": round(p_away_win * 100, 1),
                "bayern_win": round(bayern_prob * 100, 1),
                "confidence": "High" if bayern_prob > 0.70 or bayern_prob < 0.30 else "Moderate"
            },
            "predicted_score": {
                "home": pred_score_h,
                "away": pred_score_a,
                "bayern": bayern_pts,
                "opponent": opp_pts,
                "formatted": f"{pred_score_h} : {pred_score_a}",
                "bayern_formatted": f"FC Bayern {bayern_pts} : {opp_pts} {opp_name}",
                "margin": proj_margin,
                "total_points": proj_total,
                "over_under_line": round(proj_total - 0.5, 1)
            },
            "tactical_factors": {
                "projected_pace": proj_pace,
                "rest_days": {
                    "home": h_rest,
                    "away": a_rest,
                    "bayern": h_rest if is_bayern_home else a_rest,
                    "opponent": a_rest if is_bayern_home else h_rest,
                    "advantage": "FC Bayern" if (is_bayern_home and rest_diff > 0) or (not is_bayern_home and rest_diff < 0) else opp_name
                },
                "net_rating_diff": round(hp['net_rtg'] - ap['net_rtg'], 1),
                "home_off_rating": round(hp['ortg'], 1),
                "away_off_rating": round(ap['ortg'], 1),
                "home_def_rating": round(hp['drtg'], 1),
                "away_def_rating": round(ap['drtg'], 1)
            }
        }
        predictions_output.append(record)
        print(f"[{comp}] {ht} vs {at} -> Predicted: {pred_score_h} - {pred_score_a} (Bayern Win: {bayern_prob*100:.1f}%)")
        
    out_json = "basketball_prediction/dashboard/predictions.json"
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(predictions_output, f, indent=4)
        
    print(f"\nSaved {len(predictions_output)} predictions to {out_json}")
    return predictions_output

if __name__ == '__main__':
    run_predictions()
