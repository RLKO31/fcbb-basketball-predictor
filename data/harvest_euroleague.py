import concurrent.futures
import time
import os
import pandas as pd
from euroleague_api.schedule import Schedule
from euroleague_api.game_stats import GameStats

def fetch_single_game(args):
    season, g_code, round_val = args
    gs = GameStats('E')
    try:
        rep = gs.get_game_report(season, g_code)
        date_val = str(rep['date'].values[0])
        h_team = rep['local.club.name'].values[0]
        a_team = rep['road.club.name'].values[0]
        h_score = int(rep['local.score'].values[0])
        a_score = int(rep['road.score'].values[0])
        round_name = rep['Round'].values[0] if 'Round' in rep.columns else round_val
        
        # Default pace / ortg
        poss = 71.5
        h_ortg = round((h_score / poss) * 100, 1)
        a_ortg = round((a_score / poss) * 100, 1)
        
        return {
            'competition': 'EuroLeague',
            'season': f'{season}-{season+1}',
            'round': f'Round {round_name}',
            'date': date_val[:10],
            'home_team': h_team,
            'away_team': a_team,
            'home_score': h_score,
            'away_score': a_score,
            'pace': round(poss, 1),
            'home_ortg': h_ortg,
            'away_ortg': a_ortg,
            'home_drtg': a_ortg,
            'away_drtg': h_ortg
        }
    except Exception as e:
        return None

def harvest_euroleague_bayern_fast():
    print('=== Fast Parallel Harvesting EuroLeague Data for FC Bayern (2019-2024) ===')
    s = Schedule('E')
    all_euroleague_matches = []
    seasons = [2019, 2020, 2021, 2022, 2023, 2024]
    
    for season in seasons:
        sched = s.get_schedule(season)
        mask = sched.astype(str).apply(lambda row: row.str.contains('Bayern', case=False).any(), axis=1)
        bayern_sched = sched[mask].copy()
        
        tasks = []
        for idx, row in bayern_sched.iterrows():
            if str(row.get('played', '')).lower() == 'true':
                tasks.append((season, row['game'], row.get('gameday', '')))
                
        print(f'Season {season}-{season+1}: Fetching {len(tasks)} matches in parallel...')
        t0 = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            res = list(executor.map(fetch_single_game, tasks))
            
        valid = [r for r in res if r is not None]
        all_euroleague_matches.extend(valid)
        print(f'  -> Retrieved {len(valid)} games in {time.time()-t0:.2f}s')
        
    df_el = pd.DataFrame(all_euroleague_matches)
    df_el = df_el.sort_values('date').reset_index(drop=True)
    os.makedirs('basketball_prediction/data/raw', exist_ok=True)
    out_path = 'basketball_prediction/data/raw/euroleague_bayern_raw.csv'
    df_el.to_csv(out_path, index=False)
    print(f'\nTOTAL EuroLeague matches harvested: {len(df_el)}')
    print(f'Saved to: {out_path}')

if __name__ == '__main__':
    harvest_euroleague_bayern_fast()
