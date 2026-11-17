import json
import os
import re
from datetime import datetime
import requests
from euroleague_api.schedule import Schedule
from euroleague_api.standings import Standings
from euroleague_api.game_stats import GameStats


def _parse_bbl_date(iso_str):
    try:
        from zoneinfo import ZoneInfo
        dt = datetime.fromisoformat(str(iso_str).replace("Z", "+00:00"))
        return dt.astimezone(ZoneInfo("Europe/Berlin")).strftime("%Y-%m-%d")
    except Exception:
        return str(iso_str)[:10]


def _extract_next_data(url):
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
    r.raise_for_status()
    m = re.search(r'__NEXT_DATA__[^>]*>(.*?)</script>', r.text, re.S)
    if not m:
        return {}
    return json.loads(m.group(1)).get("props", {}).get("pageProps", {})


def _match_team_name(target, candidate):
    t = target.lower().strip()
    c = candidate.lower().strip()
    if t == c or t in c or c in t:
        return True
    # Strip common prefixes/suffixes
    for tok in ["fc ", "bc ", "basketball", "segafredo", "meridianbet", "beko", "aktor", "mozzart bet", "playtika", "fitness first", "syntax", "emporio armani", "ea7"]:
        t = t.replace(tok, " ")
        c = c.replace(tok, " ")
    t_words = {w for w in t.split() if len(w) > 2}
    c_words = {w for w in c.split() if len(w) > 2}
    return len(t_words & c_words) > 0


def fetch_euroleague_standings_and_schedule():
    s = Schedule("E")
    df_sched = s.get_schedule(2026)
    played_df = df_sched[df_sched["played"].astype(str).str.lower() == "true"].copy()
    latest_round = int(played_df["gameday"].astype(int).max()) if not played_df.empty else 1

    st = Standings("E")
    df_st = st.get_standings(2026, latest_round)
    standings = []
    for _, row in df_st.iterrows():
        team_name = str(row.get("club.name", "")).strip()
        form_raw = row.get("last5Form", [])
        if isinstance(form_raw, str):
            form_list = [x.strip().upper() for x in form_raw.replace(",", " ").split() if x.strip()]
        elif isinstance(form_raw, list):
            form_list = [str(x).strip().upper() for x in form_raw]
        else:
            form_list = []

        standings.append({
            "rank": int(row.get("position", 0)),
            "team": team_name,
            "code": str(row.get("club.code", "")),
            "crest": str(row.get("club.images.crest", "")),
            "played": int(row.get("gamesPlayed", 0)),
            "won": int(row.get("gamesWon", 0)),
            "lost": int(row.get("gamesLost", 0)),
            "win_pct": str(row.get("winPercentage", "0%")),
            "pts_for": int(row.get("pointsFor", 0)),
            "pts_against": int(row.get("pointsAgainst", 0)),
            "diff": int(row.get("pointsDifference", 0)),
            "home_record": str(row.get("homeRecord", "")),
            "away_record": str(row.get("awayRecord", "")),
            "last5_form": form_list[-5:],
            "is_bayern": "bayern" in team_name.lower(),
        })
    standings.sort(key=lambda x: x["rank"])
    return standings, latest_round, played_df


def fetch_bbl_standings():
    pp = _extract_next_data("https://www.easycredit-bbl.de/saison/tabelle")
    widgets = pp.get("preloadedWidgetData", {})
    if not widgets:
        return []
    val = list(widgets.values())[0]
    standings = []
    for item in val.get("standings", []):
        st_team = item.get("seasonTeam") or {}
        team_name = str(st_team.get("name", "")).strip()
        team_id = str(st_team.get("teamId", st_team.get("id", "")))
        outcomes = item.get("last10Outcomes") or []
        standings.append({
            "rank": int(item.get("rank", 0)),
            "team": team_name,
            "team_id": team_id,
            "tlc": str(st_team.get("tlc", "")),
            "logo": str(st_team.get("logoUrl", "")),
            "played": int(item.get("totalGames", 0)),
            "won": int(item.get("totalVictories", 0)),
            "lost": int(item.get("totalLosses", 0)),
            "table_pts": f"{item.get('totalPointsMade', 0)}:{item.get('totalPointsAgainst', 0)}",
            "win_pct": f"{round(float(item.get('totalVictoryPercentage', 0)), 1)}%",
            "pts_for": int(item.get("totalGoalsMade", 0)),
            "pts_against": int(item.get("totalGoalsAgainst", 0)),
            "diff": int(item.get("totalGoalDifference", 0)),
            "streak": f"{item.get('streak', '')}{item.get('streakCount', '')}",
            "last5_form": [str(x).upper() for x in outcomes[-5:]],
            "is_bayern": team_id == "486" or "bayern" in team_name.lower(),
        })
    standings.sort(key=lambda x: x["rank"])
    return standings


def fetch_euroleague_opponent_last5(opponent_name, played_df, standings):
    opp_games = played_df[
        played_df.apply(
            lambda r: _match_team_name(opponent_name, str(r.get("hometeam", "")))
            or _match_team_name(opponent_name, str(r.get("awayteam", ""))),
            axis=1,
        )
    ].copy()
    opp_games["gday_int"] = opp_games["gameday"].astype(int)
    opp_games = opp_games.sort_values("gday_int", ascending=False).head(5)

    gs = GameStats("E")
    recent_matches = []
    for _, r in opp_games.iterrows():
        gcode = str(r.get("gamecode", ""))
        g_num = int(gcode.split("_")[-1]) if "_" in gcode else None
        if g_num is None:
            continue
        try:
            rep = gs.get_game_report(2026, g_num)
            h_score = int(rep["local.score"].iloc[0])
            a_score = int(rep["road.score"].iloc[0])
            h_team = str(r["hometeam"]).title()
            a_team = str(r["awayteam"]).title()
            opp_is_home = _match_team_name(opponent_name, h_team)
            won = (h_score > a_score) if opp_is_home else (a_score > h_score)
            g_date = datetime.strptime(str(r["date"]).strip(), "%b %d, %Y").strftime("%Y-%m-%d")
            recent_matches.append({
                "date": g_date,
                "round": f"Round {int(r['gameday'])}",
                "home_team": h_team,
                "away_team": a_team,
                "home_score": h_score,
                "away_score": a_score,
                "opp_score": h_score if opp_is_home else a_score,
                "other_score": a_score if opp_is_home else h_score,
                "other_team": a_team if opp_is_home else h_team,
                "is_home": opp_is_home,
                "result": "W" if won else "L",
            })
        except Exception:
            continue

    st_row = next((s for s in standings if _match_team_name(opponent_name, s["team"])), None)
    return recent_matches, st_row


def fetch_bbl_opponent_last5(opponent_name, standings):
    st_row = next((s for s in standings if _match_team_name(opponent_name, s["team"])), None)
    if not st_row or not st_row.get("team_id"):
        return [], st_row

    team_id = st_row["team_id"]
    recent_matches = []
    seen_ids = set()

    def _add_item(item):
        gid = str(item.get("id", ""))
        if not gid or gid in seen_ids or item.get("status") != "OFFICIAL":
            return
        res = item.get("result") or {}
        h_score = res.get("homeTeamFinalScore")
        a_score = res.get("guestTeamFinalScore")
        if h_score is None or a_score is None:
            return
        h_team = str((item.get("homeTeam") or {}).get("name", "")).strip()
        a_team = str((item.get("guestTeam") or {}).get("name", "")).strip()
        h_id = str((item.get("homeTeam") or {}).get("teamId", ""))
        a_id = str((item.get("guestTeam") or {}).get("teamId", ""))
        if h_id != team_id and a_id != team_id:
            return
        seen_ids.add(gid)
        opp_is_home = (h_id == team_id)
        h_score_i, a_score_i = int(h_score), int(a_score)
        won = (h_score_i > a_score_i) if opp_is_home else (a_score_i > h_score_i)
        mday = item.get("matchDay")
        recent_matches.append({
            "date": _parse_bbl_date(item.get("scheduledTime", "")),
            "round": f"MD {mday}" if mday else str(item.get("competition", "BBL")),
            "home_team": h_team,
            "away_team": a_team,
            "home_score": h_score_i,
            "away_score": a_score_i,
            "opp_score": h_score_i if opp_is_home else a_score_i,
            "other_score": a_score_i if opp_is_home else h_score_i,
            "other_team": a_team if opp_is_home else h_team,
            "is_home": opp_is_home,
            "result": "W" if won else "L",
        })

    try:
        pp = _extract_next_data(f"https://www.easycredit-bbl.de/teams/{team_id}/2026")
        for item in pp.get("games", {}).get("items", []):
            _add_item(item)
    except Exception:
        pass

    if len(recent_matches) < 5:
        try:
            pp2 = _extract_next_data("https://www.easycredit-bbl.de/saison/aktuelle-spiele")
            w = list(pp2.get("preloadedWidgetData", {}).values())[0]
            for item in w.get("finishedGames", {}).get("items", []):
                _add_item(item)
        except Exception:
            pass

    recent_matches.sort(key=lambda x: x["date"], reverse=True)
    return recent_matches[:5], st_row


def build_league_intel(next_match, out_path):
    print(">>> [2b/4] Fetching Live BBL & EuroLeague Standings + Opponent Last-5 Form...")
    intel = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "euroleague_round": 0,
        "standings": {"EuroLeague": [], "BBL": []},
        "upcoming_opponent": None,
    }

    el_standings, played_df = [], None
    try:
        el_standings, el_rd, played_df = fetch_euroleague_standings_and_schedule()
        intel["euroleague_round"] = el_rd
        intel["standings"]["EuroLeague"] = el_standings
        print(f"  EuroLeague Standings synced ({len(el_standings)} teams, Round {el_rd}).")
    except Exception as e:
        print(f"  Note during EuroLeague standings sync: {e}")

    bbl_standings = []
    try:
        bbl_standings = fetch_bbl_standings()
        intel["standings"]["BBL"] = bbl_standings
        print(f"  easyCredit BBL Standings synced ({len(bbl_standings)} teams).")
    except Exception as e:
        print(f"  Note during BBL standings sync: {e}")

    if next_match:
        opp = next_match.get("opponent", "")
        comp = next_match.get("competition", "")
        recent_matches = []
        opp_standing = None

        if comp == "EuroLeague" and played_df is not None:
            recent_matches, opp_standing = fetch_euroleague_opponent_last5(opp, played_df, el_standings)
        elif comp in ("BBL", "BBL-Pokal") and bbl_standings:
            recent_matches, opp_standing = fetch_bbl_opponent_last5(opp, bbl_standings)

        wins_5 = sum(1 for m in recent_matches if m["result"] == "W")
        losses_5 = len(recent_matches) - wins_5
        intel["upcoming_opponent"] = {
            "opponent": opp,
            "competition": comp,
            "match_date": next_match.get("date", ""),
            "round": next_match.get("round", ""),
            "standing": opp_standing,
            "last5_summary": {
                "games": len(recent_matches),
                "wins": wins_5,
                "losses": losses_5,
                "avg_pts_for": round(sum(m["opp_score"] for m in recent_matches) / len(recent_matches), 1) if recent_matches else 0.0,
                "avg_pts_against": round(sum(m["other_score"] for m in recent_matches) / len(recent_matches), 1) if recent_matches else 0.0,
            },
            "recent_matches": recent_matches,
        }
        print(f"  Upcoming Opponent ({opp}, {comp}): fetched {len(recent_matches)} recent league match(es).")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(intel, f, indent=4)
    return intel
