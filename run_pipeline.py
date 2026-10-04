import os
import sys
import subprocess
import time

def run_step(description, command):
    print(f'\n>>> STEP: {description}')
    t0 = time.time()
    subprocess.run([sys.executable] + command, check=True)
    print(f'--- Completed in {time.time() - t0:.2f}s ---')

def main():
    print('================================================================================')
    print(' FC BAYERN BASKETBALL MODEL V3 PREDICTION PIPELINE (SEASON 2026/2027)')
    print('================================================================================')
    
    # 1. Data Ingestion & Multi-Competition Harmonization
    run_step('1. Ingest & Harmonize Data (EuroLeague, BBL, BBL-Pokal)', ['basketball_prediction/data/data_loader.py'])
    
    # 2. 1D Time-Decay & Tactical Feature Engineering
    run_step('2. Engineer Model v3 Tactical & Time-Decay Features', ['basketball_prediction/models/model_v3_tactical/feature_engineer_v3.py'])
    
    # 3. Model Training & Stacked Score Regressors
    run_step('3. Train Model v3 & Stacked Score Regressors', ['basketball_prediction/models/model_v3_tactical/train_model_v3.py'])
    
    # 4. Generate 2026/2027 Schedule & Predict Full Season
    run_step('4. Fetch & Compile Official 2026/2027 Schedule', ['scratch/make_official_26_27_schedule.py'])
    run_step('5. Predict Season 2026/2027 (76 matches)', ['basketball_prediction/models/model_v3_tactical/predict_season_26_27.py'])
    
    # 5. Compile Interactive Web Dashboard
    run_step('6. Recompile Standalone HTML Dashboard', ['basketball_prediction/dashboard/compile_dashboard.py'])
    
    print('\n================================================================================')
    print(' PIPELINE FINISHED SUCCESSFULLY!')
    print(' View the interactive dashboard at: basketball_prediction/dashboard/index.html')
    print(' Kaggle script ready at: basketball_prediction/models/model_v3_tactical/fcbb_kaggle_standalone.py')
    print('================================================================================')

if __name__ == '__main__':
    main()
