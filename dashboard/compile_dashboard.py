import json
import os
import pandas as pd

def get_resolved_path(rel_path):
    base = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(base, ".."))
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

def compile_dashboard():
    print("=== Compiling FC Bayern Basketball 2026/2027 Season Dashboard ===")
    
    preds_path = get_resolved_path("dashboard/predictions_26_27.json")
    config_path = get_resolved_path("models/model_v3_tactical/model_v3_config.json")
    
    if not os.path.exists(preds_path):
        raise FileNotFoundError(f"Predictions not found at {preds_path}. Run predict_season_26_27.py first.")
        
    with open(preds_path, "r", encoding="utf-8") as f:
        season_data = json.load(f)
        
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
        
    # JSON payloads for embedding
    season_json = json.dumps(season_data)
    config_json = json.dumps(config)
    
    perf = season_data.get("performance_tracking", {
        "played_matches": 0,
        "winner_hits": 0,
        "winner_accuracy": 0.0,
        "mean_home_error": 0.0,
        "mean_away_error": 0.0,
        "mean_spread_error": 0.0,
        "mean_total_error": 0.0
    })
    
    sync_ts = season_data.get("last_sync_timestamp", "2026-10-02 17:15 CEST")
    
    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FC Bayern Basketball // Season 2026/2027 Model Forecast & Live Tracker</title>
    
    <!-- Google Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet">
    
    <!-- Tailwind CSS CDN -->
    <script src="https://cdn.tailwindcss.com"></script>
    
    <!-- Lucide Icons -->
    <script src="https://unpkg.com/lucide@latest"></script>
    
    <script>
        tailwind.config = {{
            darkMode: 'class',
            theme: {{
                extend: {{
                    colors: {{
                        bayern: {{
                            red: '#dc2626',
                            crimson: '#991b1b',
                            gold: '#f59e0b',
                            night: '#080c14',
                            card: 'rgba(15, 23, 42, 0.78)',
                            border: 'rgba(255, 255, 255, 0.08)'
                        }}
                    }},
                    fontFamily: {{
                        outfit: ['Outfit', 'sans-serif'],
                        space: ['Space Grotesk', 'monospace']
                    }}
                }}
            }}
        }}
    </script>
    
    <style>
        /* Smooth transitions for theme switching */
        body, .glass-card, header, footer, section, div, span, button, input {{
            transition: background-color 0.25s ease, border-color 0.25s ease, color 0.25s ease;
        }}

        /* Default Light Mode (Light Blue Atmosphere) */
        body {{
            background-color: #eaf3fb;
            background-image: 
                radial-gradient(circle at 12% 15%, rgba(220, 38, 38, 0.08) 0%, transparent 40%),
                radial-gradient(circle at 88% 85%, rgba(245, 158, 11, 0.08) 0%, transparent 40%),
                radial-gradient(circle at 50% 50%, rgba(186, 230, 253, 0.60) 0%, transparent 75%);
            color: #1e293b;
            font-family: 'Outfit', sans-serif;
            min-height: 100vh;
        }}

        .glass-card {{
            background: rgba(255, 255, 255, 0.92);
            backdrop-filter: blur(16px);
            border: 1px solid rgba(186, 230, 253, 0.95);
            box-shadow: 0 4px 20px -2px rgba(14, 116, 144, 0.08), 0 2px 6px -1px rgba(15, 23, 42, 0.04);
        }}
        .glass-card:hover {{
            border-color: rgba(220, 38, 38, 0.45);
            box-shadow: 0 8px 25px -4px rgba(220, 38, 38, 0.12), 0 4px 10px -2px rgba(14, 116, 144, 0.06);
        }}

        .glow-red {{
            box-shadow: 0 0 25px -5px rgba(220, 38, 38, 0.25);
        }}
        .glow-emerald {{
            box-shadow: 0 0 25px -5px rgba(16, 185, 129, 0.22);
        }}

        /* Light Mode Specific Overrides */
        body:not(.dark) header {{
            border-bottom-color: rgba(186, 230, 253, 0.9);
        }}
        body:not(.dark) footer {{
            border-top-color: rgba(186, 230, 253, 0.9);
        }}
        body:not(.dark) .text-white {{
            color: #0f172a !important;
        }}
        body:not(.dark) .text-slate-200 {{
            color: #1e293b !important;
        }}
        body:not(.dark) .text-slate-300 {{
            color: #334155 !important;
        }}
        body:not(.dark) .text-slate-400 {{
            color: #475569 !important;
        }}
        body:not(.dark) .text-slate-500 {{
            color: #64748b !important;
        }}
        body:not(.dark) .border-slate-800,
        body:not(.dark) .border-slate-800\\/80,
        body:not(.dark) .border-slate-800\\/90,
        body:not(.dark) .border-slate-800\\/60,
        body:not(.dark) .border-slate-700 {{
            border-color: #cbd5e1 !important;
        }}
        body:not(.dark) .divide-slate-800 > :not([hidden]) ~ :not([hidden]) {{
            border-color: #e2e8f0 !important;
        }}
        body:not(.dark) .bg-slate-900,
        body:not(.dark) .bg-slate-900\\/95,
        body:not(.dark) .bg-slate-900\\/90,
        body:not(.dark) .bg-slate-900\\/80,
        body:not(.dark) .bg-slate-900\\/70 {{
            background-color: #f8fbff !important;
            border-color: #dbeafe !important;
        }}
        body:not(.dark) .bg-slate-800,
        body:not(.dark) .bg-slate-800\\/80 {{
            background-color: #e2e8f0 !important;
            color: #1e293b !important;
        }}

        /* Badges in Light Mode */
        body:not(.dark) .badge-el {{
            background: #fef3c7;
            color: #b45309;
            border: 1px solid #fde68a;
        }}
        body:not(.dark) .badge-bbl {{
            background: #fee2e2;
            color: #b91c1c;
            border: 1px solid #fca5a5;
        }}
        body:not(.dark) .badge-pokal {{
            background: #d1fae5;
            color: #047857;
            border: 1px solid #6ee7b7;
        }}
        body:not(.dark) .bg-red-950,
        body:not(.dark) .bg-red-950\\/80,
        body:not(.dark) .bg-red-950\\/50 {{
            background-color: #fee2e2 !important;
            color: #b91c1c !important;
            border-color: #fca5a5 !important;
        }}
        body:not(.dark) .bg-emerald-950,
        body:not(.dark) .bg-emerald-950\\/90 {{
            background-color: #d1fae5 !important;
            color: #065f46 !important;
            border-color: #a7f3d0 !important;
        }}
        body:not(.dark) .text-emerald-400 {{
            color: #059669 !important;
        }}
        body:not(.dark) .text-amber-400 {{
            color: #d97706 !important;
        }}
        body:not(.dark) .text-red-400 {{
            color: #dc2626 !important;
        }}

        /* Elements that must ALWAYS have white text */
        .bg-red-600,
        .bg-red-600 *,
        .tab-btn.bg-red-600,
        .bg-gradient-to-br,
        .bg-gradient-to-br * {{
            color: #ffffff !important;
        }}

        /* Search input in light mode */
        body:not(.dark) #search-input {{
            background-color: #ffffff !important;
            border-color: #bae6fd !important;
            color: #0f172a !important;
        }}
        body:not(.dark) #search-input::placeholder {{
            color: #94a3b8 !important;
        }}

        /* Filter Tab Bar in light mode */
        body:not(.dark) .tab-btn:not(.bg-red-600) {{
            color: #475569 !important;
        }}
        body:not(.dark) .tab-btn:not(.bg-red-600):hover {{
            color: #0f172a !important;
            background-color: rgba(241, 245, 249, 0.9);
        }}

        /* Modal in light mode */
        body:not(.dark) #legal-modal {{
            background-color: rgba(15, 23, 42, 0.65) !important;
        }}
        body:not(.dark) #legal-modal .glass-card {{
            background: #ffffff !important;
            border-color: #cbd5e1 !important;
        }}
        body:not(.dark) #legal-modal button.rounded-full {{
            background-color: #f1f5f9 !important;
            color: #64748b !important;
        }}
        body:not(.dark) #legal-modal button.rounded-full:hover {{
            background-color: #e2e8f0 !important;
            color: #0f172a !important;
        }}

        /* ---------------- DARK MODE (Active when body has .dark) ---------------- */
        body.dark {{
            background-color: #060911;
            background-image: 
                radial-gradient(circle at 12% 15%, rgba(220, 38, 38, 0.15) 0%, transparent 40%),
                radial-gradient(circle at 88% 85%, rgba(245, 158, 11, 0.10) 0%, transparent 40%),
                radial-gradient(circle at 50% 50%, rgba(30, 41, 59, 0.25) 0%, transparent 70%);
            color: #f1f5f9;
        }}
        body.dark .glass-card {{
            background: rgba(15, 23, 42, 0.78);
            backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            box-shadow: none;
        }}
        body.dark .glass-card:hover {{
            border-color: rgba(220, 38, 38, 0.4);
        }}
        body.dark .glow-red {{
            box-shadow: 0 0 25px -5px rgba(220, 38, 38, 0.35);
        }}
        body.dark .glow-emerald {{
            box-shadow: 0 0 25px -5px rgba(16, 185, 129, 0.30);
        }}
        body.dark .badge-el {{
            background: linear-gradient(135deg, rgba(245, 158, 11, 0.2), rgba(217, 119, 6, 0.2));
            color: #fbbf24;
            border: 1px solid rgba(245, 158, 11, 0.4);
        }}
        body.dark .badge-bbl {{
            background: linear-gradient(135deg, rgba(220, 38, 38, 0.2), rgba(185, 28, 28, 0.2));
            color: #f87171;
            border: 1px solid rgba(220, 38, 38, 0.4);
        }}
        body.dark .badge-pokal {{
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.2), rgba(5, 150, 105, 0.2));
            color: #34d399;
            border: 1px solid rgba(16, 185, 129, 0.4);
        }}
    </style>
</head>
<body class="p-4 md:p-8">
    <script>
        // Early theme initialization to avoid flash
        if (localStorage.getItem('fcbb_theme') === 'dark') {{
            document.body.classList.add('dark');
            document.documentElement.classList.add('dark');
        }}
    </script>

    <!-- Top Navigation Header -->
    <header class="max-w-7xl mx-auto mb-6 flex flex-col md:flex-row md:items-center justify-between gap-4 pb-6 border-b border-slate-800">
        <div class="flex items-center gap-4">
            <div class="w-14 h-14 rounded-2xl bg-gradient-to-br from-red-600 via-red-700 to-amber-600 flex items-center justify-center font-black text-2xl shadow-lg glow-red">
                🏀
            </div>
            <div>
                <div class="flex items-center gap-3">
                    <h1 class="text-2xl md:text-3xl font-black tracking-tight text-white font-space">FC BAYERN BASKETBALL</h1>
                    <span class="px-3 py-0.5 rounded-full text-xs font-bold uppercase tracking-wider bg-red-950/80 text-red-400 border border-red-800/60 shadow">Season 2026/2027</span>
                </div>
                <p class="text-sm text-slate-400">Model v3 Full-Season Forecast & Daily Actual Score Comparison: BBL · EuroLeague · Pokal</p>
            </div>
        </div>
        
        <div class="flex flex-wrap items-center gap-3">
            <button id="theme-toggle-btn" onclick="toggleTheme()" class="glass-card px-3.5 py-1.5 rounded-xl flex items-center gap-2 hover:border-amber-400/60 transition cursor-pointer text-xs font-mono font-bold shadow-sm" title="Modus wechseln (Hell / Dunkel)">
                <span id="theme-icon" class="text-sm">🌙</span>
                <span id="theme-text">Dark Mode</span>
            </button>
            <div class="glass-card px-3.5 py-1.5 rounded-xl flex items-center gap-2 border-emerald-500/30">
                <span class="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse"></span>
                <span class="text-xs font-mono text-emerald-400 font-bold">Daily Sync Active</span>
                <span class="text-[11px] text-slate-400 font-mono">({sync_ts})</span>
            </div>
            <div class="glass-card px-3.5 py-1.5 rounded-xl flex items-center gap-2">
                <i data-lucide="shield-check" class="w-4 h-4 text-emerald-400"></i>
                <span class="text-xs font-mono text-slate-300">Model Acc: {config["classification"]["test_accuracy"]}%</span>
            </div>
            <button onclick="openLegalModal()" class="glass-card px-3.5 py-1.5 rounded-xl flex items-center gap-2 hover:border-red-500/50 transition cursor-pointer text-xs font-mono text-slate-300">
                <i data-lucide="scale" class="w-4 h-4 text-amber-400"></i>
                <span>Impressum &amp; Legal</span>
            </button>
        </div>
    </header>

    <!-- Main Content Container -->
    <main class="max-w-7xl mx-auto space-y-6">

        <!-- LIVE MODEL PERFORMANCE & REALITY SCORECARD (NEW) -->
        <section class="glass-card rounded-2xl p-5 border border-emerald-500/30 glow-emerald">
            <div class="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-4 pb-3 border-b border-slate-800/80">
                <div class="flex items-center gap-3">
                    <div class="w-9 h-9 rounded-xl bg-emerald-500/20 text-emerald-400 flex items-center justify-center font-bold text-lg border border-emerald-500/40">
                        🎯
                    </div>
                    <div>
                        <div class="flex items-center gap-2">
                            <h2 class="text-base font-bold font-space text-white">Model v3 vs. Actual Reality (Live 2026/27 Evaluation)</h2>
                            <span class="px-2 py-0.5 rounded text-[10px] font-mono font-bold uppercase bg-emerald-950 text-emerald-300 border border-emerald-800">
                                {perf["played_matches"]} Matches Played
                            </span>
                        </div>
                        <p class="text-xs text-slate-400">Continuous daily tracking of model predictions against official ground-truth results.</p>
                    </div>
                </div>
                <div class="text-right font-mono text-xs text-slate-400">
                    Next Sync: Daily Scheduled &bull; Source: <span class="text-slate-200">EuroLeague API + BBL</span>
                </div>
            </div>

            <!-- Scorecard KPI Grid -->
            <div class="grid grid-cols-2 sm:grid-cols-4 gap-3">
                <div class="bg-slate-900/90 rounded-xl p-3 border border-slate-800">
                    <div class="text-[11px] uppercase tracking-wider text-slate-400 font-mono">Winner Accuracy</div>
                    <div class="text-2xl font-black font-space text-emerald-400 mt-0.5">{perf["winner_accuracy"]}%</div>
                    <div class="text-[10px] text-slate-400 font-mono">{perf["winner_hits"]} / {perf["played_matches"]} Correct Tips</div>
                </div>

                <div class="bg-slate-900/90 rounded-xl p-3 border border-slate-800">
                    <div class="text-[11px] uppercase tracking-wider text-slate-400 font-mono">Home Point Error</div>
                    <div class="text-2xl font-black font-space text-white mt-0.5">&plusmn;{perf["mean_home_error"]} <span class="text-xs font-normal text-slate-400">pts</span></div>
                    <div class="text-[10px] text-slate-400 font-mono">Mean Absolute Error</div>
                </div>

                <div class="bg-slate-900/90 rounded-xl p-3 border border-slate-800">
                    <div class="text-[11px] uppercase tracking-wider text-slate-400 font-mono">Away Point Error</div>
                    <div class="text-2xl font-black font-space text-white mt-0.5">&plusmn;{perf["mean_away_error"]} <span class="text-xs font-normal text-slate-400">pts</span></div>
                    <div class="text-[10px] text-slate-400 font-mono">Mean Absolute Error</div>
                </div>

                <div class="bg-slate-900/90 rounded-xl p-3 border border-slate-800">
                    <div class="text-[11px] uppercase tracking-wider text-slate-400 font-mono">Spread Margin MAE</div>
                    <div class="text-2xl font-black font-space text-amber-400 mt-0.5">&plusmn;{perf["mean_spread_error"]} <span class="text-xs font-normal text-slate-400">pts</span></div>
                    <div class="text-[10px] text-slate-400 font-mono">Total pts MAE: &plusmn;{perf["mean_total_error"]} pts</div>
                </div>
            </div>
        </section>

        <!-- 2026/2027 Season Projection Summary Cards -->
        <section class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <!-- Overall Record -->
            <div class="glass-card p-5 rounded-2xl relative overflow-hidden border-t-2 border-t-red-500">
                <div class="text-xs font-medium text-slate-400 uppercase tracking-wider mb-1">Overall 2026/27 Record</div>
                <div class="text-3xl font-black font-space text-white">{season_data["overall"]["wins"]}W - {season_data["overall"]["losses"]}L</div>
                <div class="text-xs text-emerald-400 font-bold mt-1">{season_data["overall"]["win_rate"]}% Win Rate ({season_data["total_matches"]} Games)</div>
                <div class="text-[11px] text-slate-500 mt-2 font-mono">Avg: {season_data["overall"]["avg_points_scored"]} pts for / {season_data["overall"]["avg_points_conceded"]} against</div>
            </div>

            <!-- BBL Projection -->
            <div class="glass-card p-5 rounded-2xl relative overflow-hidden border-t-2 border-t-red-600">
                <div class="flex items-center justify-between mb-1">
                    <span class="text-xs font-medium text-slate-400 uppercase tracking-wider">easyCredit BBL</span>
                    <span class="badge-bbl text-[10px] px-2 py-0.2 rounded font-mono">{season_data["competitions"]["BBL"]["games"]} Games</span>
                </div>
                <div class="text-3xl font-black font-space text-red-400">{season_data["competitions"]["BBL"]["wins"]}W - {season_data["competitions"]["BBL"]["losses"]}L</div>
                <div class="text-xs text-slate-300 font-bold mt-1">{season_data["competitions"]["BBL"]["projected_finish"]}</div>
                <div class="text-[11px] text-slate-500 mt-2 font-mono">Win Rate: {season_data["competitions"]["BBL"]["win_rate"]}% &bull; Avg: {season_data["competitions"]["BBL"]["avg_points"]} pts</div>
            </div>

            <!-- EuroLeague Projection -->
            <div class="glass-card p-5 rounded-2xl relative overflow-hidden border-t-2 border-t-amber-500">
                <div class="flex items-center justify-between mb-1">
                    <span class="text-xs font-medium text-slate-400 uppercase tracking-wider">EuroLeague</span>
                    <span class="badge-el text-[10px] px-2 py-0.2 rounded font-mono">{season_data["competitions"]["EuroLeague"]["games"]} Games</span>
                </div>
                <div class="text-3xl font-black font-space text-amber-400">{season_data["competitions"]["EuroLeague"]["wins"]}W - {season_data["competitions"]["EuroLeague"]["losses"]}L</div>
                <div class="text-xs text-slate-300 font-bold mt-1">{season_data["competitions"]["EuroLeague"]["projected_finish"]}</div>
                <div class="text-[11px] text-slate-500 mt-2 font-mono">Win Rate: {season_data["competitions"]["EuroLeague"]["win_rate"]}% &bull; Avg: {season_data["competitions"]["EuroLeague"]["avg_points"]} pts</div>
            </div>

            <!-- BBL-Pokal Projection -->
            <div class="glass-card p-5 rounded-2xl relative overflow-hidden border-t-2 border-t-emerald-500">
                <div class="flex items-center justify-between mb-1">
                    <span class="text-xs font-medium text-slate-400 uppercase tracking-wider">BBL-Pokal</span>
                    <span class="badge-pokal text-[10px] px-2 py-0.2 rounded font-mono">{season_data["competitions"]["BBL-Pokal"]["games"]} Games</span>
                </div>
                <div class="text-3xl font-black font-space text-emerald-400">{season_data["competitions"]["BBL-Pokal"]["wins"]}W - {season_data["competitions"]["BBL-Pokal"]["losses"]}L</div>
                <div class="text-xs text-slate-300 font-bold mt-1">{season_data["competitions"]["BBL-Pokal"]["projected_finish"]}</div>
                <div class="text-[11px] text-slate-500 mt-2 font-mono">Win Rate: {season_data["competitions"]["BBL-Pokal"]["win_rate"]}% &bull; Avg: {season_data["competitions"]["BBL-Pokal"]["avg_points"]} pts</div>
            </div>
        </section>

        <!-- Marquee Spotlight Match -->
        <section id="marquee-section" class="glass-card rounded-3xl p-6 md:p-8 border border-red-500/20 glow-red">
            <!-- Dynamically populated via JS -->
        </section>

        <!-- Filter Controls & Search -->
        <section class="flex flex-col md:flex-row items-center justify-between gap-4">
            <div class="flex flex-wrap items-center gap-2 bg-slate-900/90 p-1.5 rounded-2xl border border-slate-800">
                <button onclick="setCompFilter('ALL')" id="tab-ALL" class="tab-btn px-3.5 py-1.5 rounded-xl text-xs font-bold uppercase transition bg-red-600 text-white shadow-md">All (76)</button>
                <button onclick="setCompFilter('PLAYED')" id="tab-PLAYED" class="tab-btn px-3.5 py-1.5 rounded-xl text-xs font-bold uppercase transition text-emerald-400 hover:text-white border border-emerald-500/30">Played &amp; Compared ({perf["played_matches"]})</button>
                <button onclick="setCompFilter('UPCOMING')" id="tab-UPCOMING" class="tab-btn px-3.5 py-1.5 rounded-xl text-xs font-bold uppercase transition text-slate-400 hover:text-white">Upcoming ({season_data["total_matches"] - perf["played_matches"]})</button>
                <button onclick="setCompFilter('BBL')" id="tab-BBL" class="tab-btn px-3.5 py-1.5 rounded-xl text-xs font-bold uppercase transition text-slate-400 hover:text-white">easyCredit BBL (34)</button>
                <button onclick="setCompFilter('EuroLeague')" id="tab-EuroLeague" class="tab-btn px-3.5 py-1.5 rounded-xl text-xs font-bold uppercase transition text-slate-400 hover:text-white">EuroLeague (38)</button>
                <button onclick="setCompFilter('BBL-Pokal')" id="tab-BBL-Pokal" class="tab-btn px-3.5 py-1.5 rounded-xl text-xs font-bold uppercase transition text-slate-400 hover:text-white">BBL-Pokal (4)</button>
            </div>
            
            <div class="flex items-center gap-3 w-full md:w-auto">
                <div class="relative flex-1 md:w-64">
                    <i data-lucide="search" class="w-4 h-4 text-slate-500 absolute left-3 top-1/2 -translate-y-1/2"></i>
                    <input type="text" id="search-input" onkeyup="applySearch(this.value)" placeholder="Search opponent..." class="w-full bg-slate-900 border border-slate-800 rounded-xl pl-9 pr-3 py-2 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-red-500">
                </div>
                <div class="text-xs text-slate-400 font-mono whitespace-nowrap">
                    Showing <span id="match-count" class="text-white font-bold">{season_data["total_matches"]}</span> matches
                </div>
            </div>
        </section>

        <!-- Fixture Cards Grid (All 76 matches of 2026/27) -->
        <section id="fixtures-grid" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
            <!-- Dynamically rendered via JS -->
        </section>

        <!-- Tactical Insights & Diagnostics -->
        <section class="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <!-- Feature Importance -->
            <div class="glass-card rounded-3xl p-6 lg:col-span-2">
                <div class="flex items-center justify-between mb-4">
                    <h3 class="text-lg font-bold font-space text-white flex items-center gap-2">
                        <i data-lucide="bar-chart-3" class="w-5 h-5 text-red-400"></i>
                        Model v3 Tactical Feature Weights
                    </h3>
                    <span class="text-xs text-slate-400 font-mono">Window W* = {config["best_window"]}</span>
                </div>
                <p class="text-xs text-slate-400 mb-6">
                    Feature importances driving win probability and stacked score regressions for 2026/2027 fixtures.
                </p>
                <div class="space-y-3" id="feature-importance-list">
                    <!-- Populated via JS -->
                </div>
            </div>

            <!-- Model Specifications -->
            <div class="glass-card rounded-3xl p-6">
                <h3 class="text-lg font-bold font-space text-white mb-4 flex items-center gap-2">
                    <i data-lucide="cpu" class="w-5 h-5 text-amber-400"></i>
                    Model v3 Specs
                </h3>
                <div class="space-y-3 text-xs font-mono">
                    <div class="p-3 rounded-xl bg-slate-900/80 border border-slate-800">
                        <div class="text-slate-400 mb-1">Architecture</div>
                        <div class="text-white font-bold">Random Forest + HistGB Blend</div>
                    </div>
                    <div class="p-3 rounded-xl bg-slate-900/80 border border-slate-800">
                        <div class="text-slate-400 mb-1">Score Prediction</div>
                        <div class="text-white font-bold">Stacked Dual Point Regressors</div>
                    </div>
                    <div class="p-3 rounded-xl bg-slate-900/80 border border-slate-800">
                        <div class="text-slate-400 mb-1">Holdout Test Accuracy</div>
                        <div class="text-emerald-400 font-bold">{config["classification"]["test_accuracy"]}% (Log Loss: {config["classification"]["test_log_loss"]})</div>
                    </div>
                    <div class="p-3 rounded-xl bg-slate-900/80 border border-slate-800">
                        <div class="text-slate-400 mb-1">Spread Implied Accuracy</div>
                        <div class="text-amber-400 font-bold">{config["score_regression"]["implied_winner_accuracy"]}%</div>
                    </div>
                </div>
            </div>
        </section>


        <!-- Impressum, Legal & Data Sources Section -->
        <section id="legal-section" class="grid grid-cols-1 lg:grid-cols-2 gap-6 pt-4">
            
            <!-- Card 1: Impressum & Rechtliche Hinweise -->
            <div class="glass-card rounded-3xl p-6 border border-slate-800/80">
                <div class="flex items-center gap-3 mb-4 pb-3 border-b border-slate-800">
                    <div class="w-8 h-8 rounded-xl bg-amber-500/20 text-amber-400 flex items-center justify-center font-bold border border-amber-500/30">
                        ⚖️
                    </div>
                    <div>
                        <h3 class="text-base font-bold font-space text-white">Impressum &amp; Rechtliche Hinweise</h3>
                        <p class="text-xs text-slate-400">Angaben gemäß § 5 DDG (Digitale-Dienste-Gesetz)</p>
                    </div>
                </div>

                <div class="space-y-4 text-xs leading-relaxed text-slate-300">
                    <div>
                        <span class="font-bold text-white block mb-0.5">Betreiber / Entwickler</span>
                        <p class="text-slate-400">
                            <strong>Ralf Oppel</strong><br>
                            Akademisches Forschungsprojekt: Technische Hochschule Würzburg-Schweinfurt (THWS)<br>
                            Kontakt / E-Mail: <a href="mailto:ralf.oppel@study.thws.de" class="text-red-400 hover:underline">ralf.oppel@study.thws.de</a><br>
                            Standort: Deutschland
                        </p>
                    </div>

                    <div>
                        <span class="font-bold text-white block mb-0.5">Verantwortlich für Inhalte (§ 18 Abs. 2 MStV)</span>
                        <p class="text-slate-400">Ralf Oppel (Anschrift auf Anfrage via E-Mail)</p>
                    </div>

                    <div class="p-3.5 rounded-xl bg-red-950/40 border border-red-800/60">
                        <span class="font-bold text-red-400 flex items-center gap-1.5 mb-1 text-xs">
                            <i data-lucide="shield-alert" class="w-4 h-4"></i> Wichtiger Haftungsausschluss: Keine Wettberatung
                        </span>
                        <p class="text-[11px] text-red-300/90 leading-normal">
                            Dieses Projekt ist ein rein wissenschaftliches KI-Forschungs- und Demonstrationsprojekt im Rahmen eines universitären Studien-Portfolios. 
                            Sämtliche Vorhersagen, Gewinnwahrscheinlichkeiten und Punkt-Prognosen basieren auf statistischen Machine-Learning-Algorithmen (Stacked Dual Point Regressors, Random Forest, Time-Decay) 
                            und stellen <strong>keine Anlageberatung, Spielprognose oder Wettempfehlung</strong> dar. Es wird keinerlei Gewähr für Richtigkeit oder finanzielle Ergebnisse übernommen.
                        </p>
                    </div>

                    <div>
                        <span class="font-bold text-white block mb-0.5">Marken- &amp; Namenshinweis (Trademarks)</span>
                        <p class="text-[11px] text-slate-400">
                            „FC Bayern München Basketball“, „easyCredit BBL“, „Turkish Airlines EuroLeague“ und sämtliche Vereins- und Wettbewerbsbezeichnungen sind geschützte Marken der jeweiligen Rechteinhaber (FC Bayern München Basketball GmbH, Basketball Bundesliga GmbH, Euroleague Basketball). 
                            Die Nennung erfolgt rein redaktionell und nominativ zur visuellen und sachlichen Zuordnung der Begegnungen im Rahmen eines nicht-kommerziellen Portfolioprojekts gem. § 23 MarkenG. Es besteht keinerlei offizielle Partnerschaft oder Lizenzierung.
                        </p>
                    </div>
                </div>
            </div>

            <!-- Card 2: Datenquellen & Methodik -->
            <div class="glass-card rounded-3xl p-6 border border-slate-800/80">
                <div class="flex items-center gap-3 mb-4 pb-3 border-b border-slate-800">
                    <div class="w-8 h-8 rounded-xl bg-red-500/20 text-red-400 flex items-center justify-center font-bold border border-red-500/30">
                        📊
                    </div>
                    <div>
                        <h3 class="text-base font-bold font-space text-white">Datenquellen, APIs &amp; Methodik</h3>
                        <p class="text-xs text-slate-400">Verwendete Schnittstellen &amp; TDM-Rechtsgrundlage</p>
                    </div>
                </div>

                <div class="space-y-3.5 text-xs text-slate-300">
                    <div class="p-3 rounded-xl bg-slate-900/80 border border-slate-800">
                        <div class="flex items-center justify-between mb-1">
                            <span class="font-bold text-white">EuroLeague Live &amp; Stats API</span>
                            <span class="text-[10px] font-mono font-bold px-2 py-0.5 rounded bg-amber-500/20 text-amber-300 border border-amber-500/30">Offizielle Spielberichte</span>
                        </div>
                        <p class="text-[11px] text-slate-400 mb-1">
                            Abruf von offiziellen EuroLeague-Spielterminen, Box-Scores und Ergebnissen über die EuroLeague REST-Schnittstelle (<code class="text-slate-300 font-mono">euroleague-api</code>).
                        </p>
                        <span class="text-[10px] text-amber-400/80 font-mono">Attribution: Public game reports provided by Euroleague Basketball.</span>
                    </div>

                    <div class="p-3 rounded-xl bg-slate-900/80 border border-slate-800">
                        <div class="flex items-center justify-between mb-1">
                            <span class="font-bold text-white">easyCredit BBL &amp; BBL-Pokal</span>
                            <span class="text-[10px] font-mono font-bold px-2 py-0.5 rounded bg-red-500/20 text-red-300 border border-red-500/30">Nationale Wettbewerbe</span>
                        </div>
                        <p class="text-[11px] text-slate-400 mb-1">
                            Historische Saisondaten (2019–2025) und 2026/27 Spielpläne für Basketball-Bundesliga und Pokal. Faktualergebnis-Daten unterliegen keinem urheberrechtlichen Schutz.
                        </p>
                    </div>

                    <div class="p-3 rounded-xl bg-slate-900/80 border border-slate-800">
                        <div class="flex items-center justify-between mb-1">
                            <span class="font-bold text-white">Text- &amp; Data-Mining (UrhG § 44b / § 60d)</span>
                            <span class="text-[10px] font-mono font-bold px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">Gesetzliche Erlaubnis</span>
                        </div>
                        <p class="text-[11px] text-slate-400">
                            Die Aggregation historischer Boxscores zur Berechnung von rollierenden Time-Decay-Metriken (Pace, ORtg, DRtg, Net-Rating) und Modellschulung erfolgt im Einklang mit den Schrankenbestimmungen für Text- und Data-Mining zur nicht-kommerziellen Forschung und Portfolio-Demonstration.
                        </p>
                    </div>

                    <div class="text-[11px] text-slate-500 font-mono pt-1">
                        Open-Source Software: Python 3.12 &bull; Scikit-Learn &bull; Tailwind CSS &bull; Lucide Icons &bull; Kaggle Notebooks
                    </div>
                </div>
            </div>

        </section>

    </main>

    <!-- Embedded JSON Data Payload -->
    <script>
        const DATA = {season_json};
        const CONFIG = {config_json};
        const ALL_FIXTURES = DATA.fixtures;
        
        let currentFilter = 'ALL';
        let searchQuery = '';

        function setCompFilter(filterKey) {{
            currentFilter = filterKey;
            document.querySelectorAll('.tab-btn').forEach(btn => {{
                btn.className = 'tab-btn px-3.5 py-1.5 rounded-xl text-xs font-bold uppercase transition text-slate-400 hover:text-white';
            }});
            const activeTab = document.getElementById('tab-' + filterKey);
            if (activeTab) {{
                activeTab.className = 'tab-btn px-3.5 py-1.5 rounded-xl text-xs font-bold uppercase transition bg-red-600 text-white shadow-md';
            }}
            renderFixtures();
        }}

        function applySearch(query) {{
            searchQuery = query.toLowerCase().trim();
            renderFixtures();
        }}

        function getCompBadge(comp) {{
            if (comp === 'EuroLeague') return '<span class="badge-el px-2.5 py-0.5 rounded-lg text-xs font-bold uppercase">EuroLeague</span>';
            if (comp === 'BBL') return '<span class="badge-bbl px-2.5 py-0.5 rounded-lg text-xs font-bold uppercase">easyCredit BBL</span>';
            return '<span class="badge-pokal px-2.5 py-0.5 rounded-lg text-xs font-bold uppercase">BBL-Pokal</span>';
        }}

        function renderMarquee() {{
            // Select the next upcoming match
            const marquee = ALL_FIXTURES.find(m => !m.is_played) || ALL_FIXTURES[0];
            if (!marquee) return;

            const isBayernFav = marquee.probabilities.bayern_win >= 50;
            const marqueeEl = document.getElementById('marquee-section');

            marqueeEl.innerHTML = `
                <div class="flex flex-col lg:flex-row items-center justify-between gap-8">
                    <div class="w-full lg:w-1/3 text-center lg:text-left">
                        <div class="flex items-center justify-center lg:justify-start gap-2 mb-2">
                            ${{getCompBadge(marquee.competition)}}
                            <span class="text-xs text-slate-400 font-mono">${{marquee.round}}</span>
                            <span class="px-2.5 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider bg-red-950 text-red-400 border border-red-800/60 animate-pulse">TONIGHT'S MATCH</span>
                        </div>
                        <h2 class="text-3xl md:text-4xl font-black font-space text-white tracking-tight">${{marquee.home_team}} <span class="text-red-500">vs</span> ${{marquee.away_team}}</h2>
                        <p class="text-xs text-slate-400 mt-2 flex items-center justify-center lg:justify-start gap-1">
                            <i data-lucide="map-pin" class="w-3.5 h-3.5 text-slate-500"></i> ${{marquee.venue}} &bull; ${{marquee.date}} (20:00 CEST)
                        </p>
                        
                        <div class="mt-6 inline-flex items-center gap-2 px-3 py-1.5 rounded-xl bg-slate-900 border border-slate-800 text-xs font-mono">
                            <i data-lucide="clock" class="w-3.5 h-3.5 text-amber-400"></i>
                            <span>Rest: Bayern (${{marquee.tactical_factors.rest_days.bayern}}d) vs ${{marquee.opponent}} (${{marquee.tactical_factors.rest_days.opponent}}d)</span>
                        </div>
                    </div>

                    <!-- Center Expected Score Dial -->
                    <div class="w-full lg:w-1/3 flex flex-col items-center justify-center bg-slate-900/70 p-6 rounded-2xl border border-slate-800/90 shadow-inner">
                        <div class="text-xs uppercase tracking-widest text-slate-400 font-medium mb-1">Model v3 Projected Score</div>
                        <div class="text-5xl md:text-6xl font-black font-space tracking-tight text-white flex items-center gap-4">
                            <span class="${{marquee.home_team.includes('Bayern') ? 'text-red-500' : 'text-slate-200'}}">${{marquee.predicted_score.home}}</span>
                            <span class="text-slate-600">:</span>
                            <span class="${{marquee.away_team.includes('Bayern') ? 'text-red-500' : 'text-slate-200'}}">${{marquee.predicted_score.away}}</span>
                        </div>
                        <div class="flex items-center gap-4 mt-3 text-xs font-mono text-slate-400">
                            <span>Spread: <b class="text-white">${{marquee.predicted_score.margin > 0 ? '+' : ''}}${{marquee.predicted_score.margin}} pts</b></span>
                            <span>&bull;</span>
                            <span>Total O/U: <b class="text-white">${{marquee.predicted_score.total_points}}</b></span>
                            <span>&bull;</span>
                            <span>Pace: <b class="text-white">${{marquee.tactical_factors.projected_pace}}</b></span>
                        </div>
                    </div>

                    <!-- Win Probability Dial -->
                    <div class="w-full lg:w-1/3 bg-slate-900/70 p-6 rounded-2xl border border-slate-800/90">
                        <div class="flex items-center justify-between text-xs font-mono text-slate-400 mb-2">
                            <span>Bayern Win Proba</span>
                            <span class="text-xl font-bold font-space ${{isBayernFav ? 'text-emerald-400' : 'text-red-400'}}">${{marquee.probabilities.bayern_win}}%</span>
                        </div>
                        <div class="w-full h-3.5 bg-slate-800 rounded-full overflow-hidden p-0.5 border border-slate-700/50">
                            <div class="h-full rounded-full transition-all duration-1000 bg-gradient-to-r from-red-600 to-emerald-500" style="width: ${{marquee.probabilities.bayern_win}}%"></div>
                        </div>
                        <div class="flex items-center justify-between text-xs text-slate-500 mt-2 font-mono">
                            <span>${{marquee.home_team}} (${{marquee.probabilities.home_win}}%)</span>
                            <span>${{marquee.away_team}} (${{marquee.probabilities.away_win}}%)</span>
                        </div>
                        <div class="mt-4 pt-3 border-t border-slate-800/60 flex items-center justify-between text-xs">
                            <span class="text-slate-400">Model Tip:</span>
                            <span class="font-mono font-bold text-white uppercase px-2 py-0.5 rounded bg-slate-800">${{marquee.probabilities.tip}}</span>
                        </div>
                    </div>
                </div>
            `;
            lucide.createIcons();
        }}

        function renderFixtures() {{
            const container = document.getElementById('fixtures-grid');
            
            const filtered = ALL_FIXTURES.filter(m => {{
                let matchesCategory = true;
                if (currentFilter === 'ALL') matchesCategory = true;
                else if (currentFilter === 'PLAYED') matchesCategory = m.is_played;
                else if (currentFilter === 'UPCOMING') matchesCategory = !m.is_played;
                else matchesCategory = (m.competition === currentFilter);
                
                const matchesSearch = !searchQuery || 
                    m.home_team.toLowerCase().includes(searchQuery) || 
                    m.away_team.toLowerCase().includes(searchQuery) ||
                    m.round.toLowerCase().includes(searchQuery);
                    
                return matchesCategory && matchesSearch;
            }});
            
            document.getElementById('match-count').innerText = filtered.length;
            container.innerHTML = '';

            filtered.forEach(m => {{
                const isBayernFav = m.probabilities.bayern_win >= 50;
                const card = document.createElement('div');
                card.className = 'glass-card rounded-2xl p-5 flex flex-col justify-between transition-all hover:scale-[1.01] hover:border-slate-700';

                if (m.is_played && m.actual_score && m.comparison) {{
                    // PLAYED MATCH WITH COMPARISON
                    const comp = m.comparison;
                    const winnerHit = comp.winner_hit;
                    
                    const hitBadge = winnerHit
                        ? '<span class="px-2 py-0.5 rounded text-[10px] font-bold uppercase bg-emerald-950 text-emerald-300 border border-emerald-700 flex items-center gap-1"><i data-lucide="check" class="w-3 h-3"></i> Winner Hit</span>'
                        : '<span class="px-2 py-0.5 rounded text-[10px] font-bold uppercase bg-red-950 text-red-300 border border-red-700 flex items-center gap-1"><i data-lucide="x" class="w-3 h-3"></i> Miss</span>';

                    card.innerHTML = `
                        <div>
                            <div class="flex items-center justify-between gap-2 mb-3">
                                <div class="flex items-center gap-2">
                                    ${{getCompBadge(m.competition)}}
                                    <span class="px-2 py-0.5 rounded text-[10px] font-bold uppercase bg-emerald-950/90 text-emerald-400 border border-emerald-800/80">Final</span>
                                    ${{hitBadge}}
                                </div>
                                <span class="text-xs text-slate-400 font-mono">${{m.date}}</span>
                            </div>

                            <div class="text-xs font-semibold text-slate-400 mb-1">${{m.round}}</div>
                            <div class="text-base font-bold font-space text-white tracking-tight flex items-center justify-between">
                                <span class="${{m.home_team.includes('Bayern') ? 'text-red-400 font-extrabold' : ''}}">${{m.home_team}}</span>
                            </div>
                            <div class="text-base font-bold font-space text-white tracking-tight flex items-center justify-between mb-4">
                                <span class="${{m.away_team.includes('Bayern') ? 'text-red-400 font-extrabold' : ''}}">${{m.away_team}}</span>
                            </div>

                            <!-- Side-by-Side Actual vs Predicted Comparison Box -->
                            <div class="bg-slate-900/95 rounded-xl p-3 mb-3 border border-slate-800 divide-y divide-slate-800">
                                <div class="grid grid-cols-2 gap-2 pb-2.5">
                                    <!-- Left: Actual Score -->
                                    <div class="border-r border-slate-800 pr-2">
                                        <div class="text-[10px] uppercase tracking-wider text-emerald-400 font-mono font-bold flex items-center gap-1">
                                            <span class="w-1.5 h-1.5 rounded-full bg-emerald-400"></span> Actual Result
                                        </div>
                                        <div class="text-2xl font-black font-space text-white mt-0.5">${{m.actual_score.formatted}}</div>
                                        <div class="text-[10px] text-slate-400 font-mono truncate">Winner: ${{comp.actual_winner}}</div>
                                    </div>

                                    <!-- Right: Predicted Score -->
                                    <div class="pl-2">
                                        <div class="text-[10px] uppercase tracking-wider text-amber-400 font-mono font-bold flex items-center gap-1">
                                            <span class="w-1.5 h-1.5 rounded-full bg-amber-400"></span> Model v3
                                        </div>
                                        <div class="text-2xl font-black font-space text-slate-300 mt-0.5">${{m.predicted_score.formatted}}</div>
                                        <div class="text-[10px] text-slate-400 font-mono">Tip: ${{m.probabilities.tip}}</div>
                                    </div>
                                </div>

                                <!-- Accuracy Delta Metrics -->
                                <div class="pt-2 flex items-center justify-between text-[11px] font-mono text-slate-400">
                                    <span>Point Diff: <b class="text-slate-200">H:${{comp.diff_home > 0 ? '+' : ''}}${{comp.diff_home}} / A:${{comp.diff_away > 0 ? '+' : ''}}${{comp.diff_away}}</b></span>
                                    <span>Spread Diff: <b class="${{Math.abs(comp.diff_margin) <= 10 ? 'text-emerald-400' : 'text-amber-400'}}">${{comp.diff_margin > 0 ? '+' : ''}}${{comp.diff_margin}} pts</b></span>
                                </div>
                            </div>
                        </div>

                        <!-- Card Footer -->
                        <div class="pt-2.5 border-t border-slate-800/80 flex items-center justify-between text-[11px] font-mono text-slate-400">
                            <span>Total Pts: <b class="text-slate-200">${{m.actual_score.total_points}}</b> (Pred: ${{m.predicted_score.total_points}})</span>
                            <span class="px-2 py-0.5 rounded bg-slate-800 text-emerald-300 font-bold uppercase text-[10px]">Verified Result</span>
                        </div>
                    `;
                }} else {{
                    // UPCOMING MATCH (MODEL FORECAST)
                    card.innerHTML = `
                        <div>
                            <div class="flex items-center justify-between gap-2 mb-3">
                                <div class="flex items-center gap-2">
                                    ${{getCompBadge(m.competition)}}
                                    <span class="px-2 py-0.5 rounded text-[10px] font-bold uppercase bg-slate-800/80 text-amber-400 border border-amber-900/30">Upcoming</span>
                                </div>
                                <span class="text-xs text-slate-400 font-mono">${{m.date}}</span>
                            </div>

                            <div class="text-xs font-semibold text-slate-400 mb-1">${{m.round}}</div>
                            <div class="text-base font-bold font-space text-white tracking-tight flex items-center justify-between">
                                <span class="${{m.home_team.includes('Bayern') ? 'text-red-400 font-extrabold' : ''}}">${{m.home_team}}</span>
                            </div>
                            <div class="text-base font-bold font-space text-white tracking-tight flex items-center justify-between mb-4">
                                <span class="${{m.away_team.includes('Bayern') ? 'text-red-400 font-extrabold' : ''}}">${{m.away_team}}</span>
                            </div>

                            <!-- Expected Score Box -->
                            <div class="bg-slate-900/90 rounded-xl p-3 mb-4 border border-slate-800 flex items-center justify-between">
                                <div>
                                    <div class="text-[10px] uppercase tracking-wider text-slate-500 font-mono">Model v3 Projected Score</div>
                                    <div class="text-2xl font-black font-space text-white">${{m.predicted_score.formatted}}</div>
                                </div>
                                <div class="text-right">
                                    <div class="text-[10px] uppercase tracking-wider text-slate-500 font-mono">Total O/U</div>
                                    <div class="text-sm font-bold font-space text-amber-400">${{m.predicted_score.total_points}} <span class="text-slate-500 text-xs font-normal">pts</span></div>
                                </div>
                            </div>

                            <!-- Win Probability Bar -->
                            <div class="mb-3">
                                <div class="flex items-center justify-between text-xs font-mono mb-1">
                                    <span class="text-slate-400">Bayern Win Proba</span>
                                    <span class="font-bold ${{isBayernFav ? 'text-emerald-400' : 'text-red-400'}}">${{m.probabilities.bayern_win}}%</span>
                                </div>
                                <div class="w-full h-2 bg-slate-800 rounded-full overflow-hidden">
                                    <div class="h-full rounded-full bg-gradient-to-r ${{isBayernFav ? 'from-amber-500 to-emerald-500' : 'from-red-600 to-amber-500'}}" style="width: ${{m.probabilities.bayern_win}}%"></div>
                                </div>
                            </div>
                        </div>

                        <!-- Card Footer: Tactical Metrics -->
                        <div class="pt-3 border-t border-slate-800/80 flex items-center justify-between text-[11px] font-mono text-slate-400">
                            <span>Pace: <b class="text-slate-200">${{m.tactical_factors.projected_pace}}</b></span>
                            <span>Rest: <b class="text-slate-200">${{m.tactical_factors.rest_days.advantage.includes('Bayern') ? 'Bayern (+)' : 'Opp (+)'}}</b></span>
                            <span class="px-2 py-0.5 rounded bg-slate-800 text-slate-200 font-bold uppercase text-[10px]">${{m.probabilities.tip}}</span>
                        </div>
                    `;
                }}

                container.appendChild(card);
            }});
            lucide.createIcons();
        }}

        function renderFeatureImportance() {{
            const container = document.getElementById('feature-importance-list');
            const topFeats = CONFIG.top_features || [];
            
            container.innerHTML = '';
            topFeats.forEach(f => {{
                const row = document.createElement('div');
                row.className = 'flex items-center gap-3 text-xs';
                row.innerHTML = `
                    <div class="w-44 truncate font-mono text-slate-300" title="${{f.feature}}">${{f.feature}}</div>
                    <div class="flex-1 h-2 bg-slate-800 rounded-full overflow-hidden">
                        <div class="h-full rounded-full bg-gradient-to-r from-red-500 to-amber-400" style="width: ${{Math.min(f.importance * 8, 100)}}%"></div>
                    </div>
                    <div class="w-12 text-right font-mono font-bold text-white">${{f.importance}}%</div>
                `;
                container.appendChild(row);
            }});
        }}


        function openLegalModal() {{
            const m = document.getElementById('legal-modal');
            if (m) m.classList.remove('hidden');
        }}
        function closeLegalModal() {{
            const m = document.getElementById('legal-modal');
            if (m) m.classList.add('hidden');
        }}
        // Close modal on click outside
        window.addEventListener('click', (e) => {{
            const m = document.getElementById('legal-modal');
            if (e.target === m) closeLegalModal();
        }});

        // Theme Manager (Light Mode Standard / Dark Mode toggle)
        function getInitialTheme() {{
            const saved = localStorage.getItem('fcbb_theme');
            return (saved === 'dark') ? 'dark' : 'light';
        }}

        function applyTheme(theme) {{
            const isDark = (theme === 'dark');
            if (isDark) {{
                document.body.classList.add('dark');
                document.documentElement.classList.add('dark');
            }} else {{
                document.body.classList.remove('dark');
                document.documentElement.classList.remove('dark');
            }}
            updateThemeControls(isDark);
        }}

        function updateThemeControls(isDark) {{
            const icon = document.getElementById('theme-icon');
            const text = document.getElementById('theme-text');
            const fIcon = document.getElementById('theme-footer-icon');
            const fText = document.getElementById('theme-footer-text');
            
            if (icon) icon.innerText = isDark ? '☀️' : '🌙';
            if (text) text.innerText = isDark ? 'Light Mode' : 'Dark Mode';
            if (fIcon) fIcon.innerText = isDark ? '☀️' : '🌙';
            if (fText) fText.innerText = isDark ? 'Light Mode' : 'Dark Mode';
        }}

        function toggleTheme() {{
            const current = document.body.classList.contains('dark') ? 'dark' : 'light';
            const next = current === 'dark' ? 'light' : 'dark';
            localStorage.setItem('fcbb_theme', next);
            applyTheme(next);
        }}

        // Initialize Dashboard
        window.addEventListener('DOMContentLoaded', () => {{
            applyTheme(getInitialTheme());
            renderMarquee();
            renderFixtures();
            renderFeatureImportance();
            lucide.createIcons();
        }});
    </script>

    <!-- Legal Modal (Backdrop overlay) -->
    <div id="legal-modal" class="fixed inset-0 z-50 hidden bg-black/80 backdrop-blur-md flex items-center justify-center p-4">
        <div class="glass-card max-w-2xl w-full rounded-3xl p-6 border border-slate-700 shadow-2xl max-h-[90vh] overflow-y-auto space-y-4">
            <div class="flex items-center justify-between pb-3 border-b border-slate-800">
                <div class="flex items-center gap-2">
                    <span class="text-xl">⚖️</span>
                    <h3 class="text-lg font-bold font-space text-white">Impressum &amp; Rechtliche Hinweise</h3>
                </div>
                <button onclick="closeLegalModal()" class="w-8 h-8 rounded-full bg-slate-800 flex items-center justify-center text-slate-400 hover:text-white hover:bg-slate-700 transition">
                    &times;
                </button>
            </div>

            <div class="text-xs space-y-3.5 text-slate-300">
                <div>
                    <strong class="text-white block mb-0.5 font-space text-sm">Angaben gemäß § 5 DDG (ehemals TMG)</strong>
                    <p class="text-slate-400">
                        <strong>Ralf Oppel</strong><br>
                        Akademisches Portfolio-Projekt: Technische Hochschule Würzburg-Schweinfurt (THWS)<br>
                        E-Mail: <a href="mailto:ralf.oppel@study.thws.de" class="text-amber-400 hover:underline">ralf.oppel@study.thws.de</a><br>
                        Standort: Deutschland
                    </p>
                </div>

                <div class="p-3 rounded-xl bg-red-950/50 border border-red-800/80">
                    <strong class="text-red-400 block mb-1">Keine Wettberatung / Haftungsausschluss (GlüStV)</strong>
                    <p class="text-[11px] text-red-200/90 leading-normal">
                        Alle hier dargestellten Punkt- und Siegwahrscheinlichkeiten sind Ergebnisse eines maschinellen Lernmodells (Model v3 Tactical). Sie dienen rein wissenschaftlichen Demonstrations- und Bildungszwecken. Es handelt sich ausdrücklich um keine Sportwetten- oder Anlageberatung.
                    </p>
                </div>

                <div>
                    <strong class="text-white block mb-0.5">Urheberrecht &amp; Markenzeichen (§ 23 MarkenG)</strong>
                    <p class="text-[11px] text-slate-400">
                        Namen von Vereinen, Ligen und Wettbewerben sind geschützte Marken ihrer jeweiligen Eigentümer und werden rein nominativ und redaktionell verwendet. Es besteht keine geschäftliche Verbindung zu FC Bayern Basketball, der BBL oder der EuroLeague.
                    </p>
                </div>

                <div>
                    <strong class="text-white block mb-0.5">Datenquellen</strong>
                    <p class="text-[11px] text-slate-400">
                        Offizielle Spielberichte via <span class="text-slate-200">EuroLeague REST API</span> &amp; BBL Liga-Archive. Verarbeitet unter den gesetzlichen TDM-Schranken (§ 44b / § 60d UrhG).
                    </p>
                </div>
            </div>

            <div class="pt-3 border-t border-slate-800 text-right">
                <button onclick="closeLegalModal()" class="px-5 py-2 rounded-xl bg-red-600 hover:bg-red-500 font-bold text-xs text-white transition font-mono">
                    Schließen
                </button>
            </div>
        </div>
    </div>

    <!-- Responsive Footer -->
    <footer class="max-w-7xl mx-auto mt-12 pt-8 pb-12 border-t border-slate-800/80 text-center text-xs text-slate-500 space-y-3">
        <div class="flex flex-wrap items-center justify-center gap-4 text-xs font-mono text-slate-400">
            <a href="#marquee-section" class="hover:text-white transition">Marquee Game</a>
            <span>&bull;</span>
            <a href="#fixtures-grid" class="hover:text-white transition">Alle 76 Spiele</a>
            <span>&bull;</span>
            <a href="#feature-importance-list" class="hover:text-white transition">Modell-Faktoren</a>
            <span>&bull;</span>
            <button onclick="openLegalModal()" class="hover:text-amber-400 transition text-amber-400/90 font-bold cursor-pointer">Impressum &amp; Datenschutz</button>
            <span>&bull;</span>
            <button onclick="toggleTheme()" class="hover:text-amber-400 transition cursor-pointer font-bold inline-flex items-center gap-1.5">
                <span id="theme-footer-icon">🌙</span> <span id="theme-footer-text">Dark Mode</span>
            </button>
            <span>&bull;</span>
            <a href="https://github.com/RLKO31/fcbb-basketball-predictor" target="_blank" class="hover:text-white transition flex items-center gap-1 inline-flex">
                <i data-lucide="github" class="w-3.5 h-3.5"></i> GitHub Repo
            </a>
        </div>
        <p class="max-w-2xl mx-auto text-[11px] text-slate-500">
            &copy; 2026 FC Bayern Basketball AI Predictor &bull; Ralf Oppel &bull; THWS Portfolio Project<br>
            Rein wissenschaftliches KI-Demonstrationsprojekt &bull; Keine Wettberatung
        </p>
    </footer>

</body>
</html>
"""
    
    base = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(base, ".."))
    
    # 1. Output to root index.html (Primary entrypoint for GitHub Pages)
    root_html = os.path.join(repo_root, "index.html")
    with open(root_html, "w", encoding="utf-8") as f:
        f.write(html_template)
    print(f"Generated root GitHub Pages entrypoint: {root_html} ({len(html_template)} bytes)")
        
    # 2. Output to dashboard/index.html
    dash_html = os.path.join(repo_root, "dashboard", "index.html")
    os.makedirs(os.path.dirname(dash_html), exist_ok=True)
    with open(dash_html, "w", encoding="utf-8") as f:
        f.write(html_template)
        
    # 3. Output to basketball_prediction/dashboard/index.html if parent folder exists
    alt_html = os.path.join(repo_root, "basketball_prediction", "dashboard", "index.html")
    if os.path.exists(os.path.dirname(alt_html)):
        with open(alt_html, "w", encoding="utf-8") as f:
            f.write(html_template)
        
    print(f"Compiled standalone 2026/2027 dashboard successfully.")

if __name__ == '__main__':
    compile_dashboard()
