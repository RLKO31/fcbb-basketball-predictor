import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime

def get_resolved_path(rel_path):
    base = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(base, "..", ".."))
    candidates = [
        os.path.join(repo_root, rel_path),
        os.path.join(repo_root, "basketball_prediction", rel_path),
        os.path.join(os.getcwd(), rel_path),
        os.path.join(os.getcwd(), "basketball_prediction", rel_path)
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return os.path.abspath(candidates[0])

def predict_official_season_26_27():
    print("=== Predicting Official 2026/2027 Multi-Competition Season (Model v3) ===")
    
    artifacts_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(artifacts_dir, "model_v3_config.json")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
        
    optimal_features = config["optimal_features"]
    best_w = config["best_window"]
    
    scaler = joblib.load(os.path.join(artifacts_dir, "scaler_v3.joblib"))
    rf = joblib.load(os.path.join(artifacts_dir, "model_v3_rf.joblib"))
    hgb = joblib.load(os.path.join(artifacts_dir, "model_v3_hgb.joblib"))
    reg_pace = joblib.load(os.path.join(artifacts_dir, "model_v3_reg_pace.joblib"))
    reg_oh = joblib.load(os.path.join(artifacts_dir, "model_v3_reg_oh.joblib"))
    reg_oa = joblib.load(os.path.join(artifacts_dir, "model_v3_reg_oa.joblib"))
    
    disp_config = config["score_regression"].get("dispersion_scaling", {
        "scale_home": 1.45, "scale_away": 1.45, "baseline_mean_home": 82.0, "baseline_mean_away": 78.0
    })
    scale_h = float(disp_config["scale_home"])
    scale_a = float(disp_config["scale_away"])
    base_mean_h = float(disp_config["baseline_mean_home"])
    base_mean_a = float(disp_config["baseline_mean_away"])
    
    # Load official 2026/2027 schedule
    sched_path = get_resolved_path("data/processed/schedule_2026_2027_official.csv")
    df_sched = pd.read_csv(sched_path)
    df_sched['date_dt'] = pd.to_datetime(df_sched['date'])
    df_sched = df_sched.sort_values('date_dt').reset_index(drop=True)
    
    # Load historical processed data to initialize team tactical baseline profiles
    hist_path = get_resolved_path("data/processed/fcbb_multicomp_v3_tactical.csv")
    df_hist = pd.read_csv(hist_path)
    df_hist['date_dt'] = pd.to_datetime(df_hist['date'])
    
    team_histories = {}
    for idx, r in df_hist.sort_values('date_dt').iterrows():
        ht = r['home_team']
        at = r['away_team']
        dt = r['date_dt']
        tier = r.get('comp_tier', 1.0)
        
        team_histories.setdefault(ht, []).append({
            'date': dt,
            'pts': r['home_score'],
            'ptsc': r['away_score'],
            'pace': r['pace'],
            'ortg': r['home_ortg'],
            'drtg': r['away_ortg'],
            'tier': tier
        })
        team_histories.setdefault(at, []).append({
            'date': dt,
            'pts': r['away_score'],
            'ptsc': r['home_score'],
            'pace': r['pace'],
            'ortg': r['away_ortg'],
            'drtg': r['home_ortg'],
            'tier': tier
        })
        
    gamma = 0.85
    lambda_days = 0.015
    
    predictions_26_27 = []
    
    bbl_record = {"wins": 0, "losses": 0, "pts_for": 0, "pts_against": 0}
    el_record = {"wins": 0, "losses": 0, "pts_for": 0, "pts_against": 0}
    pokal_record = {"wins": 0, "losses": 0, "pts_for": 0, "pts_against": 0}
    
    for idx, row in df_sched.iterrows():
        ht = row['home_team']
        at = row['away_team']
        dt = row['date_dt']
        comp = row['competition']
        r_name = row['round']
        venue = row['venue']
        m_id = row['match_id']
        is_already_played = bool(row.get('played', False)) and pd.notna(row.get('actual_home_score'))
        
        h_rest = row['home_rest']
        a_rest = row['away_rest']
        rest_diff = h_rest - a_rest
        h_b2b = 1 if h_rest <= 2 else 0
        a_b2b = 1 if a_rest <= 2 else 0
        b2b_diff = a_b2b - h_b2b
        
        comp_tier = 1.25 if comp == 'EuroLeague' else (1.05 if comp == 'BBL-Pokal' else 1.0)
        is_el = 1 if comp == 'EuroLeague' else 0
        is_bbl = 1 if comp == 'BBL' else 0
        
        def get_current_stats(team, W, cur_date):
            hist = team_histories.get(team, [])
            if not hist:
                return {'pts': 78.0, 'ptsc': 78.0, 'pace': 72.0, 'ortg': 108.0, 'drtg': 108.0, 'net_rtg': 0.0, 'win': 0.50}
            rec = hist[-W:]
            n = len(rec)
            weights = []
            for k, e in enumerate(rec):
                d_ago = max((cur_date - e['date']).days, 1)
                w = np.exp(-lambda_days * d_ago) * (gamma ** (n - 1 - k)) * e.get('tier', 1.0)
                weights.append(w)
            w = np.array(weights)
            w = w / np.sum(w) if np.sum(w) > 0 else np.ones(n) / n
            
            pts = np.sum([e['pts'] for e in rec] * w)
            ptsc = np.sum([e['ptsc'] for e in rec] * w)
            pace = np.sum([e['pace'] for e in rec] * w)
            ortg = np.sum([e['ortg'] for e in rec] * w)
            drtg = np.sum([e['drtg'] for e in rec] * w)
            win = np.sum([1.0 if e['pts'] > e['ptsc'] else 0.0 for e in rec] * w)
            return {'pts': pts, 'ptsc': ptsc, 'pace': pace, 'ortg': ortg, 'drtg': drtg, 'net_rtg': ortg - drtg, 'win': win}
            
        hs = get_current_stats(ht, best_w, dt)
        as_ = get_current_stats(at, best_w, dt)
        
        # Build features
        feat_vals = []
        for col in optimal_features:
            if col == f'home_pts_decay_{best_w}': feat_vals.append(hs['pts'])
            elif col == f'away_pts_decay_{best_w}': feat_vals.append(as_['pts'])
            elif col == f'home_pts_conceded_decay_{best_w}': feat_vals.append(hs['ptsc'])
            elif col == f'away_pts_conceded_decay_{best_w}': feat_vals.append(as_['ptsc'])
            elif col == f'home_pace_decay_{best_w}': feat_vals.append(hs['pace'])
            elif col == f'away_pace_decay_{best_w}': feat_vals.append(as_['pace'])
            elif col == f'home_ortg_decay_{best_w}': feat_vals.append(hs['ortg'])
            elif col == f'away_ortg_decay_{best_w}': feat_vals.append(as_['ortg'])
            elif col == f'home_drtg_decay_{best_w}': feat_vals.append(hs['drtg'])
            elif col == f'away_drtg_decay_{best_w}': feat_vals.append(as_['drtg'])
            elif col == f'home_net_rtg_decay_{best_w}': feat_vals.append(hs['net_rtg'])
            elif col == f'away_net_rtg_decay_{best_w}': feat_vals.append(as_['net_rtg'])
            elif col == f'home_win_rate_decay_{best_w}': feat_vals.append(hs['win'])
            elif col == f'away_win_rate_decay_{best_w}': feat_vals.append(as_['win'])
            elif col == f'net_rtg_diff_{best_w}': feat_vals.append(hs['net_rtg'] - as_['net_rtg'])
            elif col == f'pace_projected_{best_w}': feat_vals.append((hs['pace'] + as_['pace']) / 2.0)
            elif col == f'off_def_mismatch_home_{best_w}': feat_vals.append(hs['ortg'] - as_['drtg'])
            elif col == f'off_def_mismatch_away_{best_w}': feat_vals.append(as_['ortg'] - hs['drtg'])
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
        p_home_win = float(0.50 * p_rf + 0.50 * p_hgb)
        p_away_win = 1.0 - p_home_win
        
        # Multiplicative Pace x Efficiency (Strategy 1)
        x_fused = np.hstack([x_s, np.array([[p_home_win, p_away_win]])])
        pred_pace_m = float(reg_pace.predict(x_fused)[0])
        pred_oh_m = float(reg_oh.predict(x_fused)[0])
        pred_oa_m = float(reg_oa.predict(x_fused)[0])
        
        raw_pred_hs = pred_pace_m * (pred_oh_m / 100.0)
        raw_pred_as = pred_pace_m * (pred_oa_m / 100.0)
        
        # Calibrated Dispersion Expansion (Strategy 2)
        disp_hs = base_mean_h + scale_h * (raw_pred_hs - base_mean_h)
        disp_as = base_mean_a + scale_a * (raw_pred_as - base_mean_a)
        
        pred_score_h = int(np.round(disp_hs))
        pred_score_a = int(np.round(disp_as))
        
        # Ensure consistency with classification winner
        if p_home_win >= 0.50 and pred_score_h <= pred_score_a:
            pred_score_h = pred_score_a + max(1, int(np.round((p_home_win - 0.50) * 16)))
        elif p_home_win < 0.50 and pred_score_a <= pred_score_h:
            pred_score_a = pred_score_h + max(1, int(np.round((0.50 - p_home_win) * 16)))
            
        pred_margin = pred_score_h - pred_score_a
        pred_total = pred_score_h + pred_score_a
        proj_pace = round((hs['pace'] + as_['pace']) / 2.0, 1)
        
        is_bayern_home = (ht == "FC Bayern München")
        bayern_prob = p_home_win if is_bayern_home else p_away_win
        bayern_pred_pts = pred_score_h if is_bayern_home else pred_score_a
        opp_pred_pts = pred_score_a if is_bayern_home else pred_score_h
        opp_name = at if is_bayern_home else ht
        pred_winner = ht if pred_score_h > pred_score_a else at
        bayern_pred_won = (bayern_pred_pts > opp_pred_pts)
        
        # Check actual score if played
        actual_score_obj = None
        comparison_obj = None
        
        if is_already_played:
            status_label = "FINAL (PLAYED)"
            act_h = int(row['actual_home_score'])
            act_a = int(row['actual_away_score'])
            act_margin = act_h - act_a
            act_total = act_h + act_a
            act_winner = ht if act_h > act_a else at
            act_bayern_won = (act_h > act_a) if is_bayern_home else (act_a > act_h)
            winner_hit = (act_winner == pred_winner)
            
            actual_score_obj = {
                "home": act_h,
                "away": act_a,
                "bayern": act_h if is_bayern_home else act_a,
                "opponent": act_a if is_bayern_home else act_h,
                "formatted": f"{act_h} : {act_a}",
                "bayern_formatted": f"FC Bayern {act_h if is_bayern_home else act_a} : {act_a if is_bayern_home else act_h} {opp_name}",
                "margin": act_margin,
                "total_points": act_total
            }
            
            comparison_obj = {
                "winner_hit": winner_hit,
                "actual_winner": act_winner,
                "predicted_winner": pred_winner,
                "diff_home": act_h - pred_score_h,
                "diff_away": act_a - pred_score_a,
                "abs_diff_home": abs(act_h - pred_score_h),
                "abs_diff_away": abs(act_a - pred_score_a),
                "diff_margin": act_margin - pred_margin,
                "abs_diff_margin": abs(act_margin - pred_margin),
                "diff_total": act_total - pred_total
            }
            
            feed_score_h = act_h
            feed_score_a = act_a
            feed_bayern_won = act_bayern_won
            feed_bayern_pts = act_h if is_bayern_home else act_a
            feed_opp_pts = act_a if is_bayern_home else act_h
        else:
            status_label = "UPCOMING (FORECAST)"
            feed_score_h = pred_score_h
            feed_score_a = pred_score_a
            feed_bayern_won = bayern_pred_won
            feed_bayern_pts = bayern_pred_pts
            feed_opp_pts = opp_pred_pts
        
        # Tallies
        rec_target = el_record if comp == 'EuroLeague' else (bbl_record if comp == 'BBL' else pokal_record)
        if feed_bayern_won:
            rec_target['wins'] += 1
        else:
            rec_target['losses'] += 1
        rec_target['pts_for'] += feed_bayern_pts
        rec_target['pts_against'] += feed_opp_pts
        
        # Rolling update for subsequent matches
        h_ortg = round((feed_score_h / proj_pace) * 100, 1)
        a_ortg = round((feed_score_a / proj_pace) * 100, 1)
        team_histories.setdefault(ht, []).append({'date': dt, 'pts': feed_score_h, 'ptsc': feed_score_a, 'pace': proj_pace, 'ortg': h_ortg, 'drtg': a_ortg, 'tier': comp_tier})
        team_histories.setdefault(at, []).append({'date': dt, 'pts': feed_score_a, 'ptsc': feed_score_h, 'pace': proj_pace, 'ortg': a_ortg, 'drtg': h_ortg, 'tier': comp_tier})
        
        predictions_26_27.append({
            "match_id": m_id,
            "competition": comp,
            "round": r_name,
            "date": row['date'],
            "month": row['date'][:7],
            "venue": venue,
            "status": status_label,
            "is_played": is_already_played,
            "home_team": ht,
            "away_team": at,
            "opponent": opp_name,
            "is_bayern_home": is_bayern_home,
            "probabilities": {
                "home_win": round(p_home_win * 100, 1),
                "away_win": round(p_away_win * 100, 1),
                "bayern_win": round(bayern_prob * 100, 1),
                "tip": "BAYERN WIN" if bayern_pred_won else "OPPONENT WIN",
                "confidence": "High" if bayern_prob >= 70 or bayern_prob <= 30 else "Competitive"
            },
            "predicted_score": {
                "home": pred_score_h,
                "away": pred_score_a,
                "bayern": bayern_pred_pts,
                "opponent": opp_pred_pts,
                "formatted": f"{pred_score_h} : {pred_score_a}",
                "bayern_formatted": f"FC Bayern {bayern_pred_pts} : {opp_pred_pts} {opp_name}",
                "margin": pred_margin,
                "total_points": pred_total,
                "over_under_line": round(pred_total - 0.5, 1)
            },
            "actual_score": actual_score_obj,
            "comparison": comparison_obj,
            "tactical_factors": {
                "projected_pace": proj_pace,
                "rest_days": {
                    "home": h_rest,
                    "away": a_rest,
                    "bayern": h_rest if is_bayern_home else a_rest,
                    "opponent": a_rest if is_bayern_home else h_rest,
                    "advantage": "FC Bayern" if (is_bayern_home and rest_diff > 0) or (not is_bayern_home and rest_diff < 0) else opp_name
                },
                "net_rating_diff": round(hs['net_rtg'] - as_['net_rtg'], 1),
                "home_off_rating": round(hs['ortg'], 1),
                "away_off_rating": round(as_['ortg'], 1),
                "home_def_rating": round(hs['drtg'], 1),
                "away_def_rating": round(as_['drtg'], 1)
            }
        })
        
    bbl_games = bbl_record['wins'] + bbl_record['losses']
    el_games = el_record['wins'] + el_record['losses']
    pokal_games = pokal_record['wins'] + pokal_record['losses']
    tot_games = bbl_games + el_games + pokal_games
    tot_wins = bbl_record['wins'] + el_record['wins'] + pokal_record['wins']
    
    # Calculate performance tracking on played games
    played_fixtures = [m for m in predictions_26_27 if m['is_played']]
    num_played = len(played_fixtures)
    if num_played > 0:
        winner_hits = sum(1 for m in played_fixtures if m['comparison']['winner_hit'])
        winner_acc = round((winner_hits / num_played) * 100, 1)
        mean_err_h = round(np.mean([m['comparison']['abs_diff_home'] for m in played_fixtures]), 1)
        mean_err_a = round(np.mean([m['comparison']['abs_diff_away'] for m in played_fixtures]), 1)
        mean_err_margin = round(np.mean([m['comparison']['abs_diff_margin'] for m in played_fixtures]), 1)
        mean_err_total = round(np.mean([abs(m['comparison']['diff_total']) for m in played_fixtures]), 1)
    else:
        winner_hits = 0
        winner_acc = 0.0
        mean_err_h = 0.0
        mean_err_a = 0.0
        mean_err_margin = 0.0
        mean_err_total = 0.0
        
    try:
        from zoneinfo import ZoneInfo
        now_de = datetime.now(ZoneInfo("Europe/Berlin"))
        sync_ts_str = now_de.strftime("%Y-%m-%d %H:%M CEST")
    except Exception:
        sync_ts_str = datetime.now().strftime("%Y-%m-%d %H:%M CEST")

    summary = {
        "season": "2026-2027",
        "generated_date": datetime.now().strftime("%Y-%m-%d"),
        "last_sync_timestamp": sync_ts_str,
        "total_matches": tot_games,
        "performance_tracking": {
            "played_matches": num_played,
            "winner_hits": winner_hits,
            "winner_accuracy": winner_acc,
            "mean_home_error": mean_err_h,
            "mean_away_error": mean_err_a,
            "mean_spread_error": mean_err_margin,
            "mean_total_error": mean_err_total
        },
        "overall": {
            "wins": tot_wins,
            "losses": tot_games - tot_wins,
            "win_rate": round((tot_wins / tot_games) * 100, 1),
            "avg_points_scored": round((bbl_record['pts_for'] + el_record['pts_for'] + pokal_record['pts_for']) / tot_games, 1),
            "avg_points_conceded": round((bbl_record['pts_against'] + el_record['pts_against'] + pokal_record['pts_against']) / tot_games, 1)
        },
        "competitions": {
            "BBL": {
                "games": bbl_games,
                "wins": bbl_record['wins'],
                "losses": bbl_record['losses'],
                "win_rate": round((bbl_record['wins'] / bbl_games) * 100, 1),
                "avg_points": round(bbl_record['pts_for'] / bbl_games, 1),
                "avg_conceded": round(bbl_record['pts_against'] / bbl_games, 1),
                "projected_finish": "1st Seed (Championship Favorite)"
            },
            "EuroLeague": {
                "games": el_games,
                "wins": el_record['wins'],
                "losses": el_record['losses'],
                "win_rate": round((el_record['wins'] / el_games) * 100, 1),
                "avg_points": round(el_record['pts_for'] / el_games, 1),
                "avg_conceded": round(el_record['pts_against'] / el_games, 1),
                "projected_finish": "Play-In / Top 8 Contender"
            },
            "BBL-Pokal": {
                "games": pokal_games,
                "wins": pokal_record['wins'],
                "losses": pokal_record['losses'],
                "win_rate": round((pokal_record['wins'] / pokal_games) * 100, 1),
                "avg_points": round(pokal_record['pts_for'] / pokal_games, 1),
                "avg_conceded": round(pokal_record['pts_against'] / pokal_games, 1),
                "projected_finish": "TOP FOUR Champion"
            }
        },
        "fixtures": predictions_26_27
    }
    
    repo_root = os.path.abspath(os.path.join(artifacts_dir, "..", ".."))
    out_json = os.path.join(repo_root, "dashboard", "predictions_26_27.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
        
    alt_out_json = os.path.join(repo_root, "basketball_prediction", "dashboard", "predictions_26_27.json")
    if os.path.exists(os.path.dirname(alt_out_json)):
        with open(alt_out_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)
        
    print(f"\nSUCCESS: Saved official 2026/2027 season forecast ({tot_games} matches) to {out_json}")
    print(f"Performance Tracking ({num_played} games played):")
    print(f"  - Winner Prediction Accuracy : {winner_acc}% ({winner_hits}/{num_played})")
    print(f"  - Mean Absolute Error (Home) : {mean_err_h} pts")
    print(f"  - Mean Absolute Error (Away) : {mean_err_a} pts")
    print(f"  - Mean Absolute Spread Error : {mean_err_margin} pts")
    print(f"Total Record      : {tot_wins}W - {tot_games - tot_wins}L")
    return summary

if __name__ == '__main__':
    predict_official_season_26_27()
