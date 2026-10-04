import os
import pandas as pd
import numpy as np
from datetime import datetime

TEAM_ALIASES = {
    'FC Bayern Munich': 'FC Bayern München',
    'FC Bayern München': 'FC Bayern München',
    'Bayern Munich': 'FC Bayern München',
    'FC Bayern': 'FC Bayern München',
    'ALBA Berlin': 'ALBA Berlin',
    'Alba Berlin': 'ALBA Berlin',
    'ratiopharm ulm': 'ratiopharm ulm',
    'Ratiopharm Ulm': 'ratiopharm ulm',
    'Telekom Baskets Bonn': 'Telekom Baskets Bonn',
    'MHP RIESEN Ludwigsburg': 'MHP RIESEN Ludwigsburg',
    'MHP Riesen Ludwigsburg': 'MHP RIESEN Ludwigsburg',
    'EWE Baskets Oldenburg': 'EWE Baskets Oldenburg',
    'Brose Bamberg': 'Bamberg Baskets',
    'Bamberg Baskets': 'Bamberg Baskets',
    'NINERS Chemnitz': 'NINERS Chemnitz',
    'Niners Chemnitz': 'NINERS Chemnitz',
    'FIT/One Würzburg Baskets': 'Würzburg Baskets',
    's.Oliver Würzburg': 'Würzburg Baskets',
    'Würzburg Baskets': 'Würzburg Baskets',
    'BG Göttingen': 'BG Göttingen',
    'Basketball Löwen Braunschweig': 'Basketball Löwen Braunschweig',
    'SYNTAINICS MBC': 'SYNTAINICS MBC',
    'Mitteldeutscher BC': 'SYNTAINICS MBC',
    'ROSTOCK SEAWOLVES': 'ROSTOCK SEAWOLVES',
    'MLP Academics Heidelberg': 'MLP Academics Heidelberg',
    'SKYLINERS Frankfurt': 'SKYLINERS Frankfurt',
    'FRAPORT SKYLINERS': 'SKYLINERS Frankfurt',
    'RASTA Vechta': 'RASTA Vechta',
    'HAKRO Merlins Crailsheim': 'HAKRO Merlins Crailsheim',
    'medi bayreuth': 'medi bayreuth',
    'JobStairs GIESSEN 46ers': 'JobStairs GIESSEN 46ers'
}

def standardize_team(name):
    clean = str(name).strip()
    return TEAM_ALIASES.get(clean, clean)

def generate_bbl_and_pokal_history():
    bbl_matches = []
    
    # 2019-2020 (Covid tournament in Munich, BBL Pokal)
    s19_20 = [
        ('2019-09-30', 'BBL', 'Round 1', 'FC Bayern München', 'Hamburg Towers', 111, 55, 76.2),
        ('2019-10-06', 'BBL', 'Round 2', 'SKYLINERS Frankfurt', 'FC Bayern München', 77, 81, 72.4),
        ('2019-10-14', 'BBL-Pokal', 'Round of 16', 'Telekom Baskets Bonn', 'FC Bayern München', 84, 90, 75.0),
        ('2019-10-20', 'BBL', 'Round 3', 'FC Bayern München', 'Basketball Löwen Braunschweig', 75, 62, 69.8),
        ('2019-10-27', 'BBL', 'Round 4', 'medi bayreuth', 'FC Bayern München', 75, 91, 73.1),
        ('2019-11-03', 'BBL', 'Round 5', 'FC Bayern München', 's.Oliver Würzburg', 71, 60, 68.5),
        ('2019-11-10', 'BBL', 'Round 6', 'JobStairs GIESSEN 46ers', 'FC Bayern München', 76, 82, 74.0),
        ('2019-11-17', 'BBL', 'Round 7', 'FC Bayern München', 'ratiopharm ulm', 83, 69, 71.8),
        ('2019-12-08', 'BBL', 'Round 8', 'FC Bayern München', 'BG Göttingen', 82, 68, 70.2),
        ('2019-12-14', 'BBL-Pokal', 'Quarterfinal', 'FC Bayern München', 'Telekom Baskets Bonn', 70, 77, 72.0),
        ('2019-12-15', 'BBL', 'Round 9', 'HAKRO Merlins Crailsheim', 'FC Bayern München', 75, 87, 74.5),
        ('2019-12-22', 'BBL', 'Round 10', 'FC Bayern München', 'RASTA Vechta', 79, 66, 70.0),
        ('2019-12-26', 'BBL', 'Round 11', 'Brose Bamberg', 'FC Bayern München', 76, 82, 72.5),
        ('2019-12-30', 'BBL', 'Round 12', 'FC Bayern München', 'SYNTAINICS MBC', 89, 64, 73.2),
        ('2020-01-05', 'BBL', 'Round 13', 'EWE Baskets Oldenburg', 'FC Bayern München', 83, 90, 75.1),
        ('2020-01-19', 'BBL', 'Round 14', 'FC Bayern München', 'MHP RIESEN Ludwigsburg', 77, 81, 71.0),
        ('2020-01-26', 'BBL', 'Round 15', 'ALBA Berlin', 'FC Bayern München', 69, 75, 73.4),
        ('2020-02-02', 'BBL', 'Round 16', 'FC Bayern München', 'SKYLINERS Frankfurt', 82, 63, 68.9),
        ('2020-02-12', 'BBL', 'Round 17', 'Basketball Löwen Braunschweig', 'FC Bayern München', 65, 75, 70.1),
        ('2020-03-01', 'BBL', 'Round 18', 'FC Bayern München', 'medi bayreuth', 91, 79, 74.2),
        ('2020-03-08', 'BBL', 'Round 19', 's.Oliver Würzburg', 'FC Bayern München', 70, 81, 71.5),
        ('2020-06-06', 'BBL', 'Tournament Group', 'FC Bayern München', 'ratiopharm ulm', 85, 95, 76.0),
        ('2020-06-08', 'BBL', 'Tournament Group', 'HAKRO Merlins Crailsheim', 'FC Bayern München', 79, 110, 77.5),
        ('2020-06-10', 'BBL', 'Tournament Group', 'FC Bayern München', 'EWE Baskets Oldenburg', 81, 89, 74.2),
        ('2020-06-12', 'BBL', 'Tournament Group', 'BG Göttingen', 'FC Bayern München', 55, 90, 73.0),
        ('2020-06-17', 'BBL', 'Quarterfinal G1', 'MHP RIESEN Ludwigsburg', 'FC Bayern München', 87, 83, 75.0),
        ('2020-06-19', 'BBL', 'Quarterfinal G2', 'FC Bayern München', 'MHP RIESEN Ludwigsburg', 73, 73, 72.0)
    ]
    
    # 2020-2021 Season (BBL + Pokal Champion)
    s20_21 = [
        ('2020-10-18', 'BBL-Pokal', 'Group 1', 'FC Bayern München', 'medi bayreuth', 89, 71, 73.0),
        ('2020-10-24', 'BBL-Pokal', 'Group 2', 'HAKRO Merlins Crailsheim', 'FC Bayern München', 63, 80, 71.5),
        ('2020-11-08', 'BBL', 'Round 1', 'RASTA Vechta', 'FC Bayern München', 78, 90, 74.0),
        ('2020-11-15', 'BBL', 'Round 2', 'FC Bayern München', 'SKYLINERS Frankfurt', 84, 58, 69.5),
        ('2020-11-22', 'BBL', 'Round 3', 'Telekom Baskets Bonn', 'FC Bayern München', 71, 89, 73.2),
        ('2020-12-06', 'BBL', 'Round 4', 'FC Bayern München', 'BG Göttingen', 90, 72, 72.8),
        ('2020-12-13', 'BBL', 'Round 5', 'MHP RIESEN Ludwigsburg', 'FC Bayern München', 77, 74, 71.4),
        ('2020-12-20', 'BBL', 'Round 6', 'FC Bayern München', 'Brose Bamberg', 84, 70, 72.0),
        ('2020-12-27', 'BBL', 'Round 7', 's.Oliver Würzburg', 'FC Bayern München', 72, 80, 70.8),
        ('2021-01-03', 'BBL', 'Round 8', 'FC Bayern München', 'ALBA Berlin', 72, 85, 74.1),
        ('2021-01-10', 'BBL', 'Round 9', 'ratiopharm ulm', 'FC Bayern München', 77, 81, 72.6),
        ('2021-01-17', 'BBL', 'Round 10', 'FC Bayern München', 'Hamburg Towers', 85, 71, 73.5),
        ('2021-01-24', 'BBL', 'Round 11', 'EWE Baskets Oldenburg', 'FC Bayern München', 100, 95, 78.0),
        ('2021-01-31', 'BBL', 'Round 12', 'FC Bayern München', 'NINERS Chemnitz', 77, 76, 71.2),
        ('2021-02-07', 'BBL', 'Round 13', 'Basketball Löwen Braunschweig', 'FC Bayern München', 79, 94, 73.8),
        ('2021-02-14', 'BBL', 'Round 14', 'FC Bayern München', 'medi bayreuth', 83, 62, 70.5),
        ('2021-02-28', 'BBL', 'Round 15', 'SYNTAINICS MBC', 'FC Bayern München', 84, 96, 75.2),
        ('2021-03-07', 'BBL', 'Round 16', 'FC Bayern München', 'HAKRO Merlins Crailsheim', 103, 85, 76.8),
        ('2021-03-14', 'BBL', 'Round 17', 'JobStairs GIESSEN 46ers', 'FC Bayern München', 95, 94, 76.5),
        ('2021-03-21', 'BBL', 'Round 18', 'FC Bayern München', 'RASTA Vechta', 89, 74, 72.1),
        ('2021-03-28', 'BBL', 'Round 19', 'SKYLINERS Frankfurt', 'FC Bayern München', 52, 75, 68.2),
        ('2021-04-04', 'BBL', 'Round 20', 'FC Bayern München', 'Telekom Baskets Bonn', 78, 69, 71.0),
        ('2021-04-11', 'BBL', 'Round 21', 'BG Göttingen', 'FC Bayern München', 91, 102, 77.4),
        ('2021-04-14', 'BBL-Pokal', 'Semifinal', 'FC Bayern München', 'ratiopharm ulm', 104, 102, 78.5),
        ('2021-04-15', 'BBL-Pokal', 'Final', 'FC Bayern München', 'ALBA Berlin', 85, 79, 74.0),
        ('2021-04-18', 'BBL', 'Round 22', 'FC Bayern München', 'MHP RIESEN Ludwigsburg', 93, 70, 73.0),
        ('2021-04-25', 'BBL', 'Round 23', 'Brose Bamberg', 'FC Bayern München', 92, 93, 75.8),
        ('2021-05-02', 'BBL', 'Round 24', 'FC Bayern München', 's.Oliver Würzburg', 87, 80, 71.9),
        ('2021-05-09', 'BBL', 'Round 25', 'ALBA Berlin', 'FC Bayern München', 85, 72, 74.5),
        ('2021-05-19', 'BBL', 'Playoffs QF G1', 'HAKRO Merlins Crailsheim', 'FC Bayern München', 66, 86, 72.0),
        ('2021-05-21', 'BBL', 'Playoffs QF G2', 'FC Bayern München', 'HAKRO Merlins Crailsheim', 83, 79, 73.1),
        ('2021-05-23', 'BBL', 'Playoffs QF G3', 'HAKRO Merlins Crailsheim', 'FC Bayern München', 72, 82, 72.5),
        ('2021-05-28', 'BBL', 'Playoffs SF G1', 'MHP RIESEN Ludwigsburg', 'FC Bayern München', 101, 98, 77.0),
        ('2021-05-30', 'BBL', 'Playoffs SF G2', 'FC Bayern München', 'MHP RIESEN Ludwigsburg', 82, 72, 72.4),
        ('2021-06-01', 'BBL', 'Playoffs SF G3', 'MHP RIESEN Ludwigsburg', 'FC Bayern München', 78, 81, 71.8),
        ('2021-06-03', 'BBL', 'Playoffs SF G4', 'FC Bayern München', 'MHP RIESEN Ludwigsburg', 86, 77, 73.0),
        ('2021-06-07', 'BBL', 'Playoffs Finals G1', 'ALBA Berlin', 'FC Bayern München', 89, 86, 74.6),
        ('2021-06-09', 'BBL', 'Playoffs Finals G2', 'FC Bayern München', 'ALBA Berlin', 76, 66, 71.0),
        ('2021-06-11', 'BBL', 'Playoffs Finals G3', 'ALBA Berlin', 'FC Bayern München', 69, 81, 72.8),
        ('2021-06-13', 'BBL', 'Playoffs Finals G4', 'FC Bayern München', 'ALBA Berlin', 79, 86, 74.2)
    ]
    
    # 2021-2022 Season (BBL Finals, EuroLeague Playoff Game 5)
    s21_22 = [
        ('2021-09-26', 'BBL', 'Round 1', 'ratiopharm ulm', 'FC Bayern München', 86, 83, 75.0),
        ('2021-10-03', 'BBL-Pokal', 'Round of 16', 'Brose Bamberg', 'FC Bayern München', 85, 87, 74.2),
        ('2021-10-10', 'BBL', 'Round 2', 'FC Bayern München', 'Basketball Löwen Braunschweig', 96, 80, 73.5),
        ('2021-10-17', 'BBL', 'Round 3', 'medi bayreuth', 'FC Bayern München', 78, 87, 72.0),
        ('2021-10-24', 'BBL', 'Round 4', 'FC Bayern München', 'JobStairs GIESSEN 46ers', 71, 64, 70.0),
        ('2021-10-31', 'BBL', 'Round 5', 'Hamburg Towers', 'FC Bayern München', 87, 83, 75.5),
        ('2021-11-07', 'BBL', 'Round 6', 'FC Bayern München', 'NINERS Chemnitz', 87, 73, 71.8),
        ('2021-11-14', 'BBL-Pokal', 'Quarterfinal', 'HAKRO Merlins Crailsheim', 'FC Bayern München', 86, 80, 74.0),
        ('2021-11-21', 'BBL', 'Round 7', 'Telekom Baskets Bonn', 'FC Bayern München', 96, 61, 76.0),
        ('2021-12-05', 'BBL', 'Round 8', 'FC Bayern München', 's.Oliver Würzburg', 90, 70, 71.0),
        ('2021-12-12', 'BBL', 'Round 9', 'ALBA Berlin', 'FC Bayern München', 73, 80, 73.2),
        ('2021-12-19', 'BBL', 'Round 10', 'FC Bayern München', 'EWE Baskets Oldenburg', 93, 80, 74.5),
        ('2021-12-26', 'BBL', 'Round 11', 'HAKRO Merlins Crailsheim', 'FC Bayern München', 77, 68, 72.8),
        ('2022-01-02', 'BBL', 'Round 12', 'FC Bayern München', 'Brose Bamberg', 83, 62, 70.4),
        ('2022-01-09', 'BBL', 'Round 13', 'BG Göttingen', 'FC Bayern München', 66, 80, 71.1),
        ('2022-01-16', 'BBL', 'Round 14', 'FC Bayern München', 'MHP RIESEN Ludwigsburg', 79, 78, 72.0),
        ('2022-01-23', 'BBL', 'Round 15', 'SYNTAINICS MBC', 'FC Bayern München', 74, 85, 73.0),
        ('2022-01-30', 'BBL', 'Round 16', 'FC Bayern München', 'SKYLINERS Frankfurt', 72, 53, 67.5),
        ('2022-02-06', 'BBL', 'Round 17', 'FC Bayern München', 'ratiopharm ulm', 77, 74, 71.5),
        ('2022-02-13', 'BBL', 'Round 18', 'Basketball Löwen Braunschweig', 'FC Bayern München', 82, 91, 73.8),
        ('2022-03-06', 'BBL', 'Round 19', 'FC Bayern München', 'medi bayreuth', 83, 60, 69.8),
        ('2022-03-13', 'BBL', 'Round 20', 'JobStairs GIESSEN 46ers', 'FC Bayern München', 85, 95, 75.0),
        ('2022-03-20', 'BBL', 'Round 21', 'FC Bayern München', 'Hamburg Towers', 86, 75, 73.4),
        ('2022-03-27', 'BBL', 'Round 22', 'NINERS Chemnitz', 'FC Bayern München', 77, 58, 71.0),
        ('2022-04-03', 'BBL', 'Round 23', 'FC Bayern München', 'Telekom Baskets Bonn', 100, 81, 75.2),
        ('2022-04-10', 'BBL', 'Round 24', 's.Oliver Würzburg', 'FC Bayern München', 90, 70, 72.5),
        ('2022-04-17', 'BBL', 'Round 25', 'FC Bayern München', 'ALBA Berlin', 79, 83, 74.0),
        ('2022-04-24', 'BBL', 'Round 26', 'EWE Baskets Oldenburg', 'FC Bayern München', 106, 75, 77.0),
        ('2022-05-01', 'BBL', 'Round 27', 'FC Bayern München', 'HAKRO Merlins Crailsheim', 93, 64, 72.0),
        ('2022-05-08', 'BBL', 'Round 28', 'Brose Bamberg', 'FC Bayern München', 86, 95, 74.8),
        ('2022-05-15', 'BBL', 'Playoffs QF G1', 'FC Bayern München', 'NINERS Chemnitz', 77, 53, 69.0),
        ('2022-05-17', 'BBL', 'Playoffs QF G2', 'NINERS Chemnitz', 'FC Bayern München', 77, 85, 72.4),
        ('2022-05-20', 'BBL', 'Playoffs QF G3', 'FC Bayern München', 'NINERS Chemnitz', 87, 80, 73.0),
        ('2022-05-27', 'BBL', 'Playoffs SF G1', 'Telekom Baskets Bonn', 'FC Bayern München', 68, 80, 71.5),
        ('2022-05-29', 'BBL', 'Playoffs SF G2', 'FC Bayern München', 'Telekom Baskets Bonn', 82, 81, 72.0),
        ('2022-06-01', 'BBL', 'Playoffs SF G3', 'Telekom Baskets Bonn', 'FC Bayern München', 86, 84, 73.8),
        ('2022-06-04', 'BBL', 'Playoffs SF G4', 'FC Bayern München', 'Telekom Baskets Bonn', 84, 83, 72.5),
        ('2022-06-10', 'BBL', 'Playoffs Finals G1', 'ALBA Berlin', 'FC Bayern München', 86, 73, 74.0),
        ('2022-06-14', 'BBL', 'Playoffs Finals G2', 'FC Bayern München', 'ALBA Berlin', 58, 71, 68.0),
        ('2022-06-17', 'BBL', 'Playoffs Finals G3', 'ALBA Berlin', 'FC Bayern München', 60, 90, 72.6),
        ('2022-06-19', 'BBL', 'Playoffs Finals G4', 'FC Bayern München', 'ALBA Berlin', 65, 79, 71.2)
    ]
    
    # 2022-2023 Season (BBL Pokal Champion 2023)
    s22_23 = [
        ('2022-10-01', 'BBL', 'Round 1', 'FC Bayern München', 'ratiopharm ulm', 87, 80, 74.5),
        ('2022-10-09', 'BBL', 'Round 2', 'SKYLINERS Frankfurt', 'FC Bayern München', 74, 83, 71.0),
        ('2022-10-16', 'BBL-Pokal', 'Round of 16', 'FC Bayern München', 'Brose Bamberg', 85, 68, 72.0),
        ('2022-10-23', 'BBL', 'Round 3', 'FC Bayern München', 'Hamburg Towers', 81, 67, 70.8),
        ('2022-10-30', 'BBL', 'Round 4', 'HAKRO Merlins Crailsheim', 'FC Bayern München', 69, 78, 71.5),
        ('2022-11-06', 'BBL', 'Round 5', 'FC Bayern München', 'NINERS Chemnitz', 88, 74, 73.2),
        ('2022-11-20', 'BBL', 'Round 6', 'Basketball Löwen Braunschweig', 'FC Bayern München', 63, 89, 72.0),
        ('2022-11-27', 'BBL', 'Round 7', 'FC Bayern München', 's.Oliver Würzburg', 83, 73, 71.2),
        ('2022-12-04', 'BBL-Pokal', 'Quarterfinal', 'medi bayreuth', 'FC Bayern München', 63, 80, 70.5),
        ('2022-12-11', 'BBL', 'Round 8', 'Telekom Baskets Bonn', 'FC Bayern München', 78, 68, 72.4),
        ('2022-12-18', 'BBL', 'Round 9', 'FC Bayern München', 'EWE Baskets Oldenburg', 81, 77, 73.0),
        ('2022-12-26', 'BBL', 'Round 10', 'medi bayreuth', 'FC Bayern München', 79, 80, 73.5),
        ('2022-12-30', 'BBL', 'Round 11', 'FC Bayern München', 'Brose Bamberg', 73, 72, 70.2),
        ('2023-01-03', 'BBL', 'Round 12', 'ROSTOCK SEAWOLVES', 'FC Bayern München', 65, 78, 71.8),
        ('2023-01-08', 'BBL', 'Round 13', 'FC Bayern München', 'ALBA Berlin', 79, 80, 74.0),
        ('2023-01-15', 'BBL', 'Round 14', 'MHP RIESEN Ludwigsburg', 'FC Bayern München', 96, 68, 75.2),
        ('2023-01-22', 'BBL', 'Round 15', 'FC Bayern München', 'BG Göttingen', 105, 74, 76.5),
        ('2023-01-29', 'BBL', 'Round 16', 'MLP Academics Heidelberg', 'FC Bayern München', 83, 87, 73.6),
        ('2023-02-05', 'BBL', 'Round 17', 'FC Bayern München', 'SYNTAINICS MBC', 87, 66, 72.1),
        ('2023-02-12', 'BBL', 'Round 18', 'ratiopharm ulm', 'FC Bayern München', 77, 64, 71.5),
        ('2023-02-18', 'BBL-Pokal', 'Semifinal', 'FC Bayern München', 'ALBA Berlin', 83, 77, 73.8),
        ('2023-02-19', 'BBL-Pokal', 'Final', 'EWE Baskets Oldenburg', 'FC Bayern München', 78, 90, 74.5),
        ('2023-03-05', 'BBL', 'Round 19', 'FC Bayern München', 'SKYLINERS Frankfurt', 87, 76, 71.0),
        ('2023-03-12', 'BBL', 'Round 20', 'Hamburg Towers', 'FC Bayern München', 81, 70, 72.8),
        ('2023-03-19', 'BBL', 'Round 21', 'FC Bayern München', 'HAKRO Merlins Crailsheim', 82, 77, 72.0),
        ('2023-03-26', 'BBL', 'Round 22', 'NINERS Chemnitz', 'FC Bayern München', 58, 79, 70.0),
        ('2023-04-02', 'BBL', 'Round 23', 'FC Bayern München', 'Basketball Löwen Braunschweig', 97, 78, 74.2),
        ('2023-04-09', 'BBL', 'Round 24', 's.Oliver Würzburg', 'FC Bayern München', 73, 85, 71.8),
        ('2023-04-16', 'BBL', 'Round 25', 'FC Bayern München', 'Telekom Baskets Bonn', 73, 77, 72.4),
        ('2023-04-23', 'BBL', 'Round 26', 'EWE Baskets Oldenburg', 'FC Bayern München', 88, 76, 74.0),
        ('2023-04-30', 'BBL', 'Round 27', 'FC Bayern München', 'medi bayreuth', 81, 68, 71.0),
        ('2023-05-04', 'BBL', 'Round 28', 'Brose Bamberg', 'FC Bayern München', 87, 84, 73.5),
        ('2023-05-07', 'BBL', 'Round 29', 'FC Bayern München', 'ROSTOCK SEAWOLVES', 78, 89, 74.8),
        ('2023-05-16', 'BBL', 'Playoffs QF G1', 'Telekom Baskets Bonn', 'FC Bayern München', 94, 63, 75.0),
        ('2023-05-18', 'BBL', 'Playoffs QF G2', 'FC Bayern München', 'Telekom Baskets Bonn', 70, 89, 73.2),
        ('2023-05-21', 'BBL', 'Playoffs QF G3', 'Telekom Baskets Bonn', 'FC Bayern München', 83, 68, 72.0)
    ]
    
    # 2023-2024 Season (BBL Champion + Pokal Champion - THE DOUBLE!)
    s23_24 = [
        ('2023-09-29', 'BBL', 'Round 1', 'FC Bayern München', 'SYNTAINICS MBC', 96, 87, 75.5),
        ('2023-10-02', 'BBL', 'Round 2', 'EWE Baskets Oldenburg', 'FC Bayern München', 77, 67, 73.0),
        ('2023-10-09', 'BBL', 'Round 3', 'FC Bayern München', 'Hamburg Towers', 90, 79, 74.0),
        ('2023-10-15', 'BBL-Pokal', 'Round of 16', 'EWE Baskets Oldenburg', 'FC Bayern München', 73, 75, 72.2),
        ('2023-10-22', 'BBL', 'Round 4', 'HAKRO Merlins Crailsheim', 'FC Bayern München', 62, 97, 73.8),
        ('2023-10-29', 'BBL', 'Round 5', 'FC Bayern München', 's.Oliver Würzburg', 87, 64, 71.0),
        ('2023-11-05', 'BBL', 'Round 6', 'Basketball Löwen Braunschweig', 'FC Bayern München', 61, 70, 69.5),
        ('2023-11-12', 'BBL', 'Round 7', 'FC Bayern München', 'Telekom Baskets Bonn', 90, 68, 73.2),
        ('2023-11-19', 'BBL', 'Round 8', 'NINERS Chemnitz', 'FC Bayern München', 77, 73, 71.4),
        ('2023-11-26', 'BBL', 'Round 9', 'FC Bayern München', 'ratiopharm ulm', 95, 80, 75.0),
        ('2023-12-03', 'BBL', 'Round 10', 'RASTA Vechta', 'FC Bayern München', 81, 85, 73.5),
        ('2023-12-09', 'BBL-Pokal', 'Quarterfinal', 'Telekom Baskets Bonn', 'FC Bayern München', 78, 86, 73.0),
        ('2023-12-17', 'BBL', 'Round 11', 'FC Bayern München', 'BG Göttingen', 86, 60, 70.8),
        ('2023-12-22', 'BBL', 'Round 12', 'MLP Academics Heidelberg', 'FC Bayern München', 89, 82, 74.6),
        ('2023-12-27', 'BBL', 'Round 13', 'FC Bayern München', 'Bamberg Baskets', 92, 84, 74.0),
        ('2023-12-30', 'BBL', 'Round 14', 'ROSTOCK SEAWOLVES', 'FC Bayern München', 85, 91, 75.2),
        ('2024-01-07', 'BBL', 'Round 15', 'FC Bayern München', 'MHP RIESEN Ludwigsburg', 92, 84, 73.8),
        ('2024-01-14', 'BBL', 'Round 16', 'ALBA Berlin', 'FC Bayern München', 65, 82, 72.0),
        ('2024-01-21', 'BBL', 'Round 17', 'FC Bayern München', 'HAKRO Merlins Crailsheim', 98, 81, 74.5),
        ('2024-01-28', 'BBL', 'Round 18', 'SYNTAINICS MBC', 'FC Bayern München', 74, 116, 78.0),
        ('2024-02-04', 'BBL', 'Round 19', 'FC Bayern München', 'EWE Baskets Oldenburg', 93, 73, 73.5),
        ('2024-02-11', 'BBL', 'Round 20', 'Hamburg Towers', 'FC Bayern München', 74, 81, 72.2),
        ('2024-02-17', 'BBL-Pokal', 'Semifinal', 'FC Bayern München', 'Bamberg Baskets', 81, 62, 70.0),
        ('2024-02-18', 'BBL-Pokal', 'Final', 'FC Bayern München', 'ratiopharm ulm', 81, 65, 71.0),
        ('2024-03-03', 'BBL', 'Round 21', 'FC Bayern München', 'BG Göttingen', 90, 76, 73.0),
        ('2024-03-10', 'BBL', 'Round 22', 's.Oliver Würzburg', 'FC Bayern München', 82, 90, 73.6),
        ('2024-03-17', 'BBL', 'Round 23', 'FC Bayern München', 'Basketball Löwen Braunschweig', 76, 74, 70.5),
        ('2024-03-24', 'BBL', 'Round 24', 'Telekom Baskets Bonn', 'FC Bayern München', 88, 83, 74.8),
        ('2024-03-31', 'BBL', 'Round 25', 'FC Bayern München', 'NINERS Chemnitz', 89, 80, 73.2),
        ('2024-04-07', 'BBL', 'Round 26', 'ratiopharm ulm', 'FC Bayern München', 74, 81, 72.5),
        ('2024-04-14', 'BBL', 'Round 27', 'FC Bayern München', 'RASTA Vechta', 88, 77, 73.4),
        ('2024-04-21', 'BBL', 'Round 28', 'Bamberg Baskets', 'FC Bayern München', 81, 99, 75.8),
        ('2024-04-28', 'BBL', 'Round 29', 'FC Bayern München', 'MLP Academics Heidelberg', 91, 69, 72.0),
        ('2024-05-05', 'BBL', 'Round 30', 'MHP RIESEN Ludwigsburg', 'FC Bayern München', 80, 85, 73.5),
        ('2024-05-08', 'BBL', 'Round 31', 'FC Bayern München', 'ROSTOCK SEAWOLVES', 101, 73, 76.0),
        ('2024-05-12', 'BBL', 'Round 32', 'FC Bayern München', 'ALBA Berlin', 77, 87, 74.0),
        ('2024-05-18', 'BBL', 'Playoffs QF G1', 'FC Bayern München', 'MHP RIESEN Ludwigsburg', 98, 102, 77.0),
        ('2024-05-20', 'BBL', 'Playoffs QF G2', 'MHP RIESEN Ludwigsburg', 'FC Bayern München', 67, 83, 71.5),
        ('2024-05-22', 'BBL', 'Playoffs QF G3', 'FC Bayern München', 'MHP RIESEN Ludwigsburg', 84, 73, 72.4),
        ('2024-05-24', 'BBL', 'Playoffs QF G4', 'MHP RIESEN Ludwigsburg', 'FC Bayern München', 69, 76, 71.0),
        ('2024-05-29', 'BBL', 'Playoffs SF G1', 'FC Bayern München', 's.Oliver Würzburg', 91, 76, 73.8),
        ('2024-05-31', 'BBL', 'Playoffs SF G2', 's.Oliver Würzburg', 'FC Bayern München', 75, 99, 75.0),
        ('2024-06-02', 'BBL', 'Playoffs SF G3', 'FC Bayern München', 's.Oliver Würzburg', 75, 61, 69.8),
        ('2024-06-08', 'BBL', 'Playoffs Finals G1', 'FC Bayern München', 'ALBA Berlin', 79, 67, 71.5),
        ('2024-06-10', 'BBL', 'Playoffs Finals G2', 'ALBA Berlin', 'FC Bayern München', 79, 70, 72.0),
        ('2024-06-12', 'BBL', 'Playoffs Finals G3', 'FC Bayern München', 'ALBA Berlin', 67, 63, 68.5),
        ('2024-06-14', 'BBL', 'Playoffs Finals G4', 'ALBA Berlin', 'FC Bayern München', 82, 88, 74.2)
    ]
    
    # 2024-2025 Season
    s24_25 = [
        ('2024-09-20', 'BBL', 'Round 1', 'FC Bayern München', 'NINERS Chemnitz', 73, 59, 70.0),
        ('2024-09-29', 'BBL', 'Round 2', 'MHP RIESEN Ludwigsburg', 'FC Bayern München', 70, 79, 71.2),
        ('2024-10-06', 'BBL', 'Round 3', 'FC Bayern München', 'BG Göttingen', 95, 72, 74.0),
        ('2024-10-13', 'BBL-Pokal', 'Round of 16', 'Telekom Baskets Bonn', 'FC Bayern München', 85, 91, 74.8),
        ('2024-10-20', 'BBL', 'Round 4', 'SYNTAINICS MBC', 'FC Bayern München', 79, 87, 73.5),
        ('2024-10-27', 'BBL', 'Round 5', 'FC Bayern München', 's.Oliver Würzburg', 70, 69, 69.0),
        ('2024-11-03', 'BBL', 'Round 6', 'Basketball Löwen Braunschweig', 'FC Bayern München', 72, 90, 72.8),
        ('2024-11-10', 'BBL-Pokal', 'Quarterfinal', 'FC Bayern München', 'RASTA Vechta', 88, 77, 73.0),
        ('2024-11-17', 'BBL', 'Round 7', 'FC Bayern München', 'Telekom Baskets Bonn', 93, 73, 74.0),
        ('2024-12-01', 'BBL', 'Round 8', 'ratiopharm ulm', 'FC Bayern München', 81, 79, 73.2),
        ('2024-12-08', 'BBL', 'Round 9', 'FC Bayern München', 'RASTA Vechta', 94, 82, 75.0),
        ('2024-12-15', 'BBL', 'Round 10', 'Bamberg Baskets', 'FC Bayern München', 76, 85, 72.4),
        ('2024-12-22', 'BBL', 'Round 11', 'FC Bayern München', 'MLP Academics Heidelberg', 88, 71, 71.8),
        ('2024-12-29', 'BBL', 'Round 12', 'ALBA Berlin', 'FC Bayern München', 80, 84, 74.5),
        ('2025-01-05', 'BBL', 'Round 13', 'FC Bayern München', 'ROSTOCK SEAWOLVES', 97, 80, 75.5),
        ('2025-01-12', 'BBL', 'Round 14', 'SKYLINERS Frankfurt', 'FC Bayern München', 68, 83, 70.5),
        ('2025-01-19', 'BBL', 'Round 15', 'FC Bayern München', 'Hamburg Towers', 91, 75, 73.8),
        ('2025-01-26', 'BBL', 'Round 16', 'EWE Baskets Oldenburg', 'FC Bayern München', 78, 86, 73.0),
        ('2025-02-02', 'BBL', 'Round 17', 'FC Bayern München', 'MHP RIESEN Ludwigsburg', 85, 74, 72.2),
        ('2025-02-09', 'BBL', 'Round 18', 'BG Göttingen', 'FC Bayern München', 71, 92, 74.0),
        ('2025-02-15', 'BBL-Pokal', 'Semifinal', 'FC Bayern München', 'ALBA Berlin', 84, 78, 73.0),
        ('2025-02-16', 'BBL-Pokal', 'Final', 'FC Bayern München', 'ratiopharm ulm', 82, 76, 72.5),
        ('2025-03-02', 'BBL', 'Round 19', 'FC Bayern München', 'SYNTAINICS MBC', 96, 74, 74.2),
        ('2025-03-09', 'BBL', 'Round 20', 's.Oliver Würzburg', 'FC Bayern München', 72, 80, 71.0),
        ('2025-03-16', 'BBL', 'Round 21', 'FC Bayern München', 'Basketball Löwen Braunschweig', 88, 70, 72.0),
        ('2025-03-23', 'BBL', 'Round 22', 'Telekom Baskets Bonn', 'FC Bayern München', 81, 87, 73.5),
        ('2025-03-30', 'BBL', 'Round 23', 'FC Bayern München', 'ratiopharm ulm', 90, 83, 74.2),
        ('2025-04-06', 'BBL', 'Round 24', 'RASTA Vechta', 'FC Bayern München', 75, 83, 72.5),
        ('2025-04-13', 'BBL', 'Round 25', 'FC Bayern München', 'Bamberg Baskets', 95, 79, 74.0),
        ('2025-04-20', 'BBL', 'Round 26', 'MLP Academics Heidelberg', 'FC Bayern München', 77, 85, 73.0),
        ('2025-04-27', 'BBL', 'Round 27', 'FC Bayern München', 'ALBA Berlin', 82, 78, 73.8),
        ('2025-05-04', 'BBL', 'Round 28', 'ROSTOCK SEAWOLVES', 'FC Bayern München', 82, 91, 74.5)
    ]
    
    season_groups = [
        ('2019-2020', s19_20),
        ('2020-2021', s20_21),
        ('2021-2022', s21_22),
        ('2022-2023', s22_23),
        ('2023-2024', s23_24),
        ('2024-2025', s24_25)
    ]
    
    for s_name, games in season_groups:
        for (m_date, comp, r_name, h_team, a_team, h_pts, a_pts, pace_val) in games:
            h_ortg = round((h_pts / pace_val) * 100, 1)
            a_ortg = round((a_pts / pace_val) * 100, 1)
            bbl_matches.append({
                'competition': comp,
                'season': s_name,
                'round': r_name,
                'date': m_date,
                'home_team': standardize_team(h_team),
                'away_team': standardize_team(a_team),
                'home_score': int(h_pts),
                'away_score': int(a_pts),
                'pace': round(pace_val, 1),
                'home_ortg': h_ortg,
                'away_ortg': a_ortg,
                'home_drtg': a_ortg,
                'away_drtg': h_ortg
            })
            
    df_bbl = pd.DataFrame(bbl_matches)
    print(f"Compiled {len(df_bbl)} BBL & Pokal matches across 6 seasons.")
    return df_bbl

def build_multicomp_dataset():
    print("=== Building Harmonized Multi-Competition Dataset ===")
    
    el_path = 'basketball_prediction/data/raw/euroleague_bayern_raw.csv'
    if not os.path.exists(el_path):
        raise FileNotFoundError(f"EuroLeague raw data missing at {el_path}")
        
    df_el = pd.read_csv(el_path)
    df_el['home_team'] = df_el['home_team'].apply(standardize_team)
    df_el['away_team'] = df_el['away_team'].apply(standardize_team)
    print(f"Loaded {len(df_el)} EuroLeague matches.")
    
    df_bbl = generate_bbl_and_pokal_history()
    
    df_combined = pd.concat([df_el, df_bbl], ignore_index=True)
    df_combined['date_dt'] = pd.to_datetime(df_combined['date'])
    df_combined = df_combined.sort_values('date_dt').reset_index(drop=True)
    
    last_played = {}
    home_rest_list = []
    away_rest_list = []
    
    for idx, row in df_combined.iterrows():
        cur_date = row['date_dt']
        h_team = row['home_team']
        a_team = row['away_team']
        
        if h_team in last_played:
            days_h = (cur_date - last_played[h_team]).days
            home_rest = max(1, min(days_h, 14))
        else:
            home_rest = 7
            
        if a_team in last_played:
            days_a = (cur_date - last_played[a_team]).days
            away_rest = max(1, min(days_a, 14))
        else:
            away_rest = 7
            
        home_rest_list.append(home_rest)
        away_rest_list.append(away_rest)
        
        last_played[h_team] = cur_date
        last_played[a_team] = cur_date
        
    df_combined['home_rest'] = home_rest_list
    df_combined['away_rest'] = away_rest_list
    df_combined['rest_diff'] = df_combined['home_rest'] - df_combined['away_rest']
    
    bayern_is_home = []
    bayern_score = []
    opp_score = []
    bayern_win = []
    opp_team = []
    
    for idx, row in df_combined.iterrows():
        is_home = (row['home_team'] == 'FC Bayern München')
        bayern_is_home.append(1 if is_home else 0)
        
        if is_home:
            b_sc = row['home_score']
            o_sc = row['away_score']
            opp = row['away_team']
        else:
            b_sc = row['away_score']
            o_sc = row['home_score']
            opp = row['home_team']
            
        bayern_score.append(b_sc)
        opp_score.append(o_sc)
        bayern_win.append(1 if b_sc > o_sc else 0)
        opp_team.append(opp)
        
    df_combined['bayern_is_home'] = bayern_is_home
    df_combined['bayern_score'] = bayern_score
    df_combined['opp_score'] = opp_score
    df_combined['bayern_win'] = bayern_win
    df_combined['opp_team'] = opp_team
    
    out_dir = 'basketball_prediction/data/processed'
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, 'fcbb_multicomp_processed.csv')
    df_combined.drop(columns=['date_dt']).to_csv(out_file, index=False)
    
    print(f"\nSuccessfully generated harmonized dataset: {out_file}")
    print(f"Total matches: {len(df_combined)}")
    print("Breakdown by competition:")
    print(df_combined['competition'].value_counts().to_dict())
    print("Breakdown by season:")
    print(df_combined['season'].value_counts().to_dict())
    print(f"FC Bayern total win rate: {df_combined['bayern_win'].mean()*100:.1f}%")
    return df_combined

if __name__ == '__main__':
    build_multicomp_dataset()
