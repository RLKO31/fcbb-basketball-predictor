import pandas as pd
import numpy as np
import os
from datetime import datetime

def engineer_basketball_v3_features(
    input_csv="basketball_prediction/data/processed/fcbb_multicomp_processed.csv",
    output_csv="basketball_prediction/data/processed/fcbb_multicomp_v3_tactical.csv"
):
    print("=== Running Feature Engineering v3 (Basketball Tactical & 1D Time-Decay) ===")
    df = pd.read_csv(input_csv)
    df['date_dt'] = pd.to_datetime(df['date'])
    df = df.sort_values('date_dt').reset_index(drop=True)
    
    # Competition tier weight: EuroLeague is highest intensity, BBL is national league, Pokal knockout
    comp_weights = {
        'EuroLeague': 1.25,
        'BBL': 1.00,
        'BBL-Pokal': 1.05
    }
    df['comp_tier'] = df['competition'].map(comp_weights).fillna(1.0)
    
    # Store team histories chronologically
    team_histories = {}
    
    # Decay parameters
    gamma = 0.85  # Game recency decay
    lambda_days = 0.015  # Calendar day decay
    
    # Columns dictionary for window features
    new_columns = {}
    for W in range(1, 11):
        for prefix in ['home', 'away']:
            new_columns[f'{prefix}_pts_decay_{W}'] = []
            new_columns[f'{prefix}_pts_conceded_decay_{W}'] = []
            new_columns[f'{prefix}_pace_decay_{W}'] = []
            new_columns[f'{prefix}_ortg_decay_{W}'] = []
            new_columns[f'{prefix}_drtg_decay_{W}'] = []
            new_columns[f'{prefix}_net_rtg_decay_{W}'] = []
            new_columns[f'{prefix}_win_rate_decay_{W}'] = []
            
        new_columns[f'net_rtg_diff_{W}'] = []
        new_columns[f'pace_projected_{W}'] = []
        new_columns[f'off_def_mismatch_home_{W}'] = []
        new_columns[f'off_def_mismatch_away_{W}'] = []
        
    for idx, row in df.iterrows():
        h_team = row['home_team']
        a_team = row['away_team']
        match_date = row['date_dt']
        
        def get_weighted_stats(team, W, current_date):
            history = team_histories.get(team, [])
            if not history:
                # Default baseline values for new/unseen teams
                return {
                    'pts': 78.0,
                    'pts_conceded': 78.0,
                    'pace': 72.0,
                    'ortg': 108.0,
                    'drtg': 108.0,
                    'net_rtg': 0.0,
                    'win_rate': 0.50
                }
                
            recent = history[-W:]
            n = len(recent)
            
            weights = []
            for k_idx, entry in enumerate(recent):
                days_ago = max((current_date - entry['date']).days, 1)
                # Exponential calendar time decay combined with positional game decay
                w = np.exp(-lambda_days * days_ago) * (gamma ** (n - 1 - k_idx)) * entry.get('tier', 1.0)
                weights.append(w)
                
            weights = np.array(weights)
            w_sum = np.sum(weights)
            if w_sum == 0:
                weights = np.ones(n) / n
            else:
                weights = weights / w_sum
                
            pts_vals = np.array([e['pts'] for e in recent])
            ptsc_vals = np.array([e['pts_conceded'] for e in recent])
            pace_vals = np.array([e['pace'] for e in recent])
            ortg_vals = np.array([e['ortg'] for e in recent])
            drtg_vals = np.array([e['drtg'] for e in recent])
            win_vals = np.array([1.0 if e['pts'] > e['pts_conceded'] else 0.0 for e in recent])
            
            pts_w = np.sum(pts_vals * weights)
            ptsc_w = np.sum(ptsc_vals * weights)
            pace_w = np.sum(pace_vals * weights)
            ortg_w = np.sum(ortg_vals * weights)
            drtg_w = np.sum(drtg_vals * weights)
            net_rtg_w = ortg_w - drtg_w
            win_rate_w = np.sum(win_vals * weights)
            
            return {
                'pts': pts_w,
                'pts_conceded': ptsc_w,
                'pace': pace_w,
                'ortg': ortg_w,
                'drtg': drtg_w,
                'net_rtg': net_rtg_w,
                'win_rate': win_rate_w
            }
            
        for W in range(1, 11):
            h_stat = get_weighted_stats(h_team, W, match_date)
            a_stat = get_weighted_stats(a_team, W, match_date)
            
            new_columns[f'home_pts_decay_{W}'].append(round(h_stat['pts'], 2))
            new_columns[f'home_pts_conceded_decay_{W}'].append(round(h_stat['pts_conceded'], 2))
            new_columns[f'home_pace_decay_{W}'].append(round(h_stat['pace'], 2))
            new_columns[f'home_ortg_decay_{W}'].append(round(h_stat['ortg'], 2))
            new_columns[f'home_drtg_decay_{W}'].append(round(h_stat['drtg'], 2))
            new_columns[f'home_net_rtg_decay_{W}'].append(round(h_stat['net_rtg'], 2))
            new_columns[f'home_win_rate_decay_{W}'].append(round(h_stat['win_rate'], 3))
            
            new_columns[f'away_pts_decay_{W}'].append(round(a_stat['pts'], 2))
            new_columns[f'away_pts_conceded_decay_{W}'].append(round(a_stat['pts_conceded'], 2))
            new_columns[f'away_pace_decay_{W}'].append(round(a_stat['pace'], 2))
            new_columns[f'away_ortg_decay_{W}'].append(round(a_stat['ortg'], 2))
            new_columns[f'away_drtg_decay_{W}'].append(round(a_stat['drtg'], 2))
            new_columns[f'away_net_rtg_decay_{W}'].append(round(a_stat['net_rtg'], 2))
            new_columns[f'away_win_rate_decay_{W}'].append(round(a_stat['win_rate'], 3))
            
            # Matchup features
            new_columns[f'net_rtg_diff_{W}'].append(round(h_stat['net_rtg'] - a_stat['net_rtg'], 2))
            new_columns[f'pace_projected_{W}'].append(round((h_stat['pace'] + a_stat['pace']) / 2.0, 2))
            new_columns[f'off_def_mismatch_home_{W}'].append(round(h_stat['ortg'] - a_stat['drtg'], 2))
            new_columns[f'off_def_mismatch_away_{W}'].append(round(a_stat['ortg'] - h_stat['drtg'], 2))
            
        # Update team histories with this game
        h_entry = {
            'date': match_date,
            'pts': row['home_score'],
            'pts_conceded': row['away_score'],
            'pace': row['pace'],
            'ortg': row['home_ortg'],
            'drtg': row['away_ortg'],
            'tier': row['comp_tier']
        }
        a_entry = {
            'date': match_date,
            'pts': row['away_score'],
            'pts_conceded': row['home_score'],
            'pace': row['pace'],
            'ortg': row['away_ortg'],
            'drtg': row['home_ortg'],
            'tier': row['comp_tier']
        }
        
        team_histories.setdefault(h_team, []).append(h_entry)
        team_histories.setdefault(a_team, []).append(a_entry)
        
    for k, v in new_columns.items():
        df[k] = v
        
    # Additional Context Features
    # Rest disparity & Back-to-Back flags
    df['home_is_b2b'] = (df['home_rest'] <= 2).astype(int)
    df['away_is_b2b'] = (df['away_rest'] <= 2).astype(int)
    df['b2b_diff'] = df['away_is_b2b'] - df['home_is_b2b'] # positive = home has rest advantage
    
    # Competition One-Hot Encoding
    df['is_euroleague'] = (df['competition'] == 'EuroLeague').astype(int)
    df['is_bbl'] = (df['competition'] == 'BBL').astype(int)
    df['is_pokal'] = (df['competition'] == 'BBL-Pokal').astype(int)
    
    # Target binary outcome: 1 = Home Win, 0 = Away Win
    df['target_home_win'] = (df['home_score'] > df['away_score']).astype(int)
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.drop(columns=['date_dt']).to_csv(output_csv, index=False)
    print(f"Engineered {len(new_columns)} time-decay tactical features.")
    print(f"Saved feature dataset: {output_csv} ({len(df)} rows, {df.shape[1]} columns)")
    return df

if __name__ == '__main__':
    engineer_basketball_v3_features()
