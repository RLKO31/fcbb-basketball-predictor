import os
import sys
import json
from datetime import datetime
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from euroleague_api.schedule import Schedule
from euroleague_api.game_stats import GameStats
from basketball_prediction.models.model_v3_tactical.predict_season_26_27 import predict_official_season_26_27
from basketball_prediction.dashboard.compile_dashboard import compile_dashboard

REGISTRY_PATH = "basketball_prediction/data/actual_results_registry.json"
SCHEDULE_PATH = "basketball_prediction/data/processed/schedule_2026_2027_official.csv"

def load_registry():
    if os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"_description": "Verified registry of actual results", "last_sync": "", "results": {}}

def save_registry(registry):
    registry["last_sync"] = datetime.now().isoformat()
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=4)

def sync_euroleague_results(registry):
    print(">>> [1/4] Checking official EuroLeague API for completed 2026/27 games...")
    results = registry.get("results", {})
    try:
        s = Schedule('E')
        df_sched = s.get_schedule(2026)
        bayern_games = df_sched[df_sched.astype(str).apply(lambda row: row.str.contains('Bayern', case=False).any(), axis=1)].copy()
        
        gs = GameStats('E')
        updated_count = 0
        
        for idx, r in bayern_games.iterrows():
            gday = int(r['gameday'])
            match_id = f"el_2627_rd{gday}"
            is_played = str(r.get('played', '')).lower() == 'true'
            gcode = r.get('gamecode', '')
            # Game code numeric
            g_num = int(gcode.split('_')[-1]) if '_' in str(gcode) else None
            
            if is_played and g_num is not None:
                # If not recorded or need refresh
                if match_id not in results or not results[match_id].get('home_score'):
                    print(f"  Fetching EuroLeague Round {gday} box score (Game #{g_num})...")
                    rep = gs.get_game_report(2026, g_num)
                    h_score = int(rep['local.score'].iloc[0])
                    a_score = int(rep['road.score'].iloc[0])
                    
                    results[match_id] = {
                        "match_id": match_id,
                        "competition": "EuroLeague",
                        "round": f"EuroLeague Round {gday}",
                        "date": datetime.strptime(r['date'].strip(), "%b %d, %Y").strftime("%Y-%m-%d"),
                        "home_team": str(r['hometeam']).title(),
                        "away_team": str(r['awayteam']).title(),
                        "home_score": h_score,
                        "away_score": a_score,
                        "played": True,
                        "source": "euroleague_api"
                    }
                    updated_count += 1
                    print(f"  -> Recorded Round {gday}: {results[match_id]['home_team']} {h_score} : {a_score} {results[match_id]['away_team']}")
                    
        print(f"  EuroLeague Sync Complete: {updated_count} new result(s) updated.")
    except Exception as e:
        print(f"  Note during EuroLeague API sync: {e}")
        
    registry["results"] = results
    return registry

def sync_schedule_csv(registry):
    print(">>> [2/4] Synchronizing schedule dataset with verified match outcomes...")
    if not os.path.exists(SCHEDULE_PATH):
        raise FileNotFoundError(f"Schedule CSV missing: {SCHEDULE_PATH}")
        
    df_sched = pd.read_csv(SCHEDULE_PATH)
    results = registry.get("results", {})
    
    updated_rows = 0
    for idx, row in df_sched.iterrows():
        m_id = row['match_id']
        if m_id in results and results[m_id].get('played'):
            res = results[m_id]
            df_sched.at[idx, 'played'] = True
            df_sched.at[idx, 'actual_home_score'] = float(res['home_score'])
            df_sched.at[idx, 'actual_away_score'] = float(res['away_score'])
            updated_rows += 1
            
    df_sched.to_csv(SCHEDULE_PATH, index=False)
    print(f"  Schedule CSV updated: {updated_rows} match(es) marked as FINAL with ground truth.")

def update_daily():
    print("================================================================================")
    print(f" FC BAYERN BASKETBALL - DAILY RESULTS SYNC & DASHBOARD UPDATE")
    print(f" Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("================================================================================")
    
    # 1. Load registry and sync
    registry = load_registry()
    registry = sync_euroleague_results(registry)
    save_registry(registry)
    
    # 2. Update CSV
    sync_schedule_csv(registry)
    
    # 3. Re-run Model v3 prediction & evaluation
    print(">>> [3/4] Re-running Model v3 Season Simulation & Scorecard Generation...")
    summary = predict_official_season_26_27()
    
    # 4. Recompile dashboard
    print(">>> [4/4] Recompiling Interactive Web Dashboard...")
    compile_dashboard()
    
    # Summary report
    perf = summary.get("performance_tracking", {})
    played_fixtures = [m for m in summary.get("fixtures", []) if m.get("is_played")]
    
    print("\n" + "=" * 80)
    print(" DAILY SYNC SUMMARY: MODEL V3 PREDICTIONS VS. ACTUAL SCORES")
    print("=" * 80)
    print(f"Total Games Played : {perf.get('played_matches', 0)}")
    print(f"Winner Accuracy    : {perf.get('winner_accuracy', 0)}% ({perf.get('winner_hits', 0)}/{perf.get('played_matches', 0)} Correct)")
    print(f"Point Error Home   : ±{perf.get('mean_home_error', 0)} pts")
    print(f"Point Error Away   : ±{perf.get('mean_away_error', 0)} pts")
    print(f"Point Spread MAE   : ±{perf.get('mean_spread_error', 0)} pts")
    print("-" * 80)
    print(f"{'Date':<11} | {'Matchup':<42} | {'Actual':<9} | {'Predicted':<10} | {'Winner':<8} | {'Delta'}")
    print("-" * 80)
    for m in played_fixtures:
        comp = m['comparison']
        act_str = m['actual_score']['formatted']
        pred_str = m['predicted_score']['formatted']
        hit_str = "HIT [OK]" if comp['winner_hit'] else "MISS [X]"
        match_str = f"{m['home_team']} vs {m['away_team']}"
        if len(match_str) > 42:
            match_str = match_str[:39] + "..."
        delta_str = f"H:{comp['diff_home']:+d} A:{comp['diff_away']:+d} (Spr:{comp['diff_margin']:+d})"
        print(f"{m['date']:<11} | {match_str:<42} | {act_str:<9} | {pred_str:<10} | {hit_str:<8} | {delta_str}")
    print("=" * 80)
    print(f" Interactive Dashboard Updated: basketball_prediction/dashboard/index.html")
    print("================================================================================\n")

if __name__ == '__main__':
    update_daily()
