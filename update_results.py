import os
import sys
import json
import re
import requests
from datetime import datetime
import pandas as pd

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
PARENT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from euroleague_api.schedule import Schedule
from euroleague_api.game_stats import GameStats

try:
    from models.model_v3_tactical.predict_season_26_27 import predict_official_season_26_27
except ImportError:
    from basketball_prediction.models.model_v3_tactical.predict_season_26_27 import predict_official_season_26_27

try:
    from dashboard.compile_dashboard import compile_dashboard
except ImportError:
    from basketball_prediction.dashboard.compile_dashboard import compile_dashboard

def get_resolved_path(rel_path):
    candidates = [
        os.path.join(BASE_DIR, rel_path),
        os.path.join(BASE_DIR, "basketball_prediction", rel_path),
        os.path.join(PARENT_DIR, rel_path),
        os.path.join(PARENT_DIR, "basketball_prediction", rel_path),
        os.path.join(os.getcwd(), rel_path)
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return os.path.abspath(candidates[0])

REGISTRY_PATH = get_resolved_path("data/actual_results_registry.json")
SCHEDULE_PATH = get_resolved_path("data/processed/schedule_2026_2027_official.csv")

def load_registry():
    if os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"_description": "Verified registry of actual results", "last_sync": "", "results": {}}

def save_registry(registry):
    registry["last_sync"] = datetime.now().isoformat()
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=4)
    alt = os.path.join(BASE_DIR, "basketball_prediction", "data", "actual_results_registry.json")
    if os.path.exists(os.path.dirname(alt)):
        with open(alt, "w", encoding="utf-8") as f:
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


def normalize_bbl_team(name):
    if not name:
        return name
    name = str(name).strip()
    mapping = {
        "FC Bayern München Basketball": "FC Bayern München",
        "Fitness First Würzburg Baskets": "Würzburg Baskets",
        "ALBA BERLIN": "ALBA Berlin",
        "SKYLINERS": "SKYLINERS Frankfurt",
        "RASTA Vechta": "RASTA Vechta",
        "SYNTAINICS MBC": "SYNTAINICS MBC",
        "Telekom Baskets Bonn": "Telekom Baskets Bonn",
        "Basketball Löwen Braunschweig": "Basketball Löwen Braunschweig",
        "Science City Jena": "Science City Jena",
    }
    return mapping.get(name, name)

def _parse_bbl_date(iso_str):
    try:
        from zoneinfo import ZoneInfo
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return dt.astimezone(ZoneInfo("Europe/Berlin")).strftime("%Y-%m-%d")
    except Exception:
        return str(iso_str)[:10]

def sync_bbl_results(registry):
    print(">>> [1b/4] Checking official easyCredit BBL page for completed & upcoming 2026/27 games...")
    results = registry.get("results", {})
    upcoming_updates = {}
    try:
        r = requests.get(
            "https://www.easycredit-bbl.de/teams/486/2026",
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=15
        )
        r.raise_for_status()
        m = re.search(r'__NEXT_DATA__[^>]*>(.*?)</script>', r.text, re.S)
        if not m:
            print("  Note: __NEXT_DATA__ not found on easycredit-bbl.de")
            return registry, upcoming_updates

        pp = json.loads(m.group(1))["props"]["pageProps"]
        df_sched = pd.read_csv(SCHEDULE_PATH) if os.path.exists(SCHEDULE_PATH) else pd.DataFrame()
        updated_count = 0

        # 1. Process finished games
        for item in pp.get("games", {}).get("items", []):
            if item.get("status") != "OFFICIAL":
                continue
            res = item.get("result") or {}
            h_score = res.get("homeTeamFinalScore")
            a_score = res.get("guestTeamFinalScore")
            if h_score is None or a_score is None:
                continue

            g_date = _parse_bbl_date(item.get("scheduledTime", ""))
            h_team = normalize_bbl_team((item.get("homeTeam") or {}).get("name", ""))
            a_team = normalize_bbl_team((item.get("guestTeam") or {}).get("name", ""))
            venue = (item.get("venue") or {}).get("name", "")
            comp_raw = str(item.get("competition", "BBL")).upper()
            comp = "BBL-Pokal" if "POKAL" in comp_raw else "BBL"
            mday = item.get("matchDay")

            # Match against schedule CSV by date + competition
            match_id = None
            if not df_sched.empty:
                cand = df_sched[(df_sched["date"] == g_date) & (df_sched["competition"].isin(["BBL", "BBL-Pokal"]))]
                if not cand.empty:
                    match_id = str(cand.iloc[0]["match_id"])
            if not match_id and mday is not None:
                match_id = f"bbl_2627_md{mday}"

            if match_id and (match_id not in results or not results[match_id].get("home_score")):
                results[match_id] = {
                    "match_id": match_id,
                    "competition": comp,
                    "round": f"BBL Matchday {mday}" if mday else comp,
                    "date": g_date,
                    "home_team": h_team,
                    "away_team": a_team,
                    "venue": venue,
                    "home_score": int(h_score),
                    "away_score": int(a_score),
                    "played": True,
                    "source": "easycredit_bbl_web"
                }
                updated_count += 1
                print(f"  -> Recorded BBL ({g_date}): {h_team} {int(h_score)} : {int(a_score)} {a_team}")

        # 2. Collect upcoming scheduled games to keep CSV aligned
        for item in pp.get("scheduledGames", {}).get("items", []):
            g_date = _parse_bbl_date(item.get("scheduledTime", ""))
            h_team = normalize_bbl_team((item.get("homeTeam") or {}).get("name", ""))
            a_team = normalize_bbl_team((item.get("guestTeam") or {}).get("name", ""))
            venue = (item.get("venue") or {}).get("name", "")
            if g_date and h_team and a_team:
                upcoming_updates[g_date] = {
                    "home_team": h_team,
                    "away_team": a_team,
                    "venue": venue,
                }

        print(f"  easyCredit BBL Sync Complete: {updated_count} new result(s) updated.")
    except Exception as e:
        print(f"  Note during easyCredit BBL sync: {e}")

    registry["results"] = results
    return registry, upcoming_updates

def sync_schedule_csv(registry, bbl_upcoming=None):
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
            if res.get('source') == 'easycredit_bbl_web':
                if res.get('home_team'):
                    df_sched.at[idx, 'home_team'] = res['home_team']
                if res.get('away_team'):
                    df_sched.at[idx, 'away_team'] = res['away_team']
                if res.get('venue'):
                    df_sched.at[idx, 'venue'] = res['venue']
            updated_rows += 1
        elif bbl_upcoming and row.get('competition') in ('BBL', 'BBL-Pokal') and row.get('date') in bbl_upcoming:
            up = bbl_upcoming[row['date']]
            df_sched.at[idx, 'home_team'] = up['home_team']
            df_sched.at[idx, 'away_team'] = up['away_team']
            if up.get('venue'):
                df_sched.at[idx, 'venue'] = up['venue']
            
    df_sched.to_csv(SCHEDULE_PATH, index=False)
    alt_sched = os.path.join(BASE_DIR, "basketball_prediction", "data", "processed", "schedule_2026_2027_official.csv")
    if os.path.exists(os.path.dirname(alt_sched)):
        df_sched.to_csv(alt_sched, index=False)
    print(f"  Schedule CSV updated: {updated_rows} match(es) marked as FINAL with ground truth.")

def update_daily():
    print("================================================================================")
    print(f" FC BAYERN BASKETBALL - DAILY RESULTS SYNC & DASHBOARD UPDATE")
    print(f" Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("================================================================================")
    
    # 1. Load registry and sync
    registry = load_registry()
    registry = sync_euroleague_results(registry)
    registry, bbl_upcoming = sync_bbl_results(registry)
    save_registry(registry)
    
    # 2. Update CSV
    sync_schedule_csv(registry, bbl_upcoming)
    
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
    print(f" Interactive Dashboard Updated: index.html & dashboard/index.html")
    print("================================================================================\n")

if __name__ == '__main__':
    update_daily()
