import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
from collections import defaultdict
import datetime

# Import the nfl_data_py library
import nfl_data_py as nfl

# (Fantasy Point calculation functions are unchanged)
SCORING_RULES = {'pass_yd': 0.04, 'pass_td': 4, 'int': -2, 'rush_yd': 0.1, 'rush_td': 6, 'rec': 1, 'rec_yd': 0.1, 'rec_td': 6, 'fumble_lost': -2}
def calculate_passer_fp(play_row, is_td=False):
    points = play_row.get('passing_yards', 0) * SCORING_RULES['pass_yd'] + play_row.get('interception', 0) * SCORING_RULES['int']
    if play_row.get('passer_player_id') == play_row.get('rusher_player_id'):
        points += play_row.get('rushing_yards', 0) * SCORING_RULES['rush_yd'] + play_row.get('fumble_lost', 0) * SCORING_RULES['fumble_lost']
    if is_td: points += SCORING_RULES['pass_td']
    return points
def calculate_rusher_fp(play_row, is_td=False):
    points = play_row.get('rushing_yards', 0) * SCORING_RULES['rush_yd'] + play_row.get('fumble_lost', 0) * SCORING_RULES['fumble_lost']
    if is_td: points += SCORING_RULES['rush_td']
    return points
def calculate_receiver_fp(play_row, is_td=False):
    points = play_row.get('reception', 0) * SCORING_RULES['rec'] + play_row.get('receiving_yards', 0) * SCORING_RULES['rec_yd'] + play_row.get('fumble_lost', 0) * SCORING_RULES['fumble_lost']
    if is_td: points += SCORING_RULES['rec_td']
    return points

# --- Data Loading and Preprocessing ---
def fetch_and_prep_data(years: list):
    print(f"Downloading data for seasons: {years}...")
    try:
        df = nfl.import_pbp_data(years=years, downcast=True, cache=False)
        rosters_df = nfl.import_seasonal_rosters(years=years)
        games_df = nfl.import_schedules(years=years)
        print("Download complete.")
    except Exception as e:
        print(f"An error occurred during data download: {e}")
        return None, None, None
    
    print("Processing data...")
    player_info = rosters_df[['player_id', 'player_name', 'position']].dropna().drop_duplicates(subset=['player_id'])
    id_to_name_map = pd.Series(player_info.player_name.values, index=player_info.player_id).to_dict()
    id_to_pos_map = pd.Series(player_info.position.values, index=player_info.player_id).to_dict()

    DATA_COLS = ['passing_yards', 'pass_touchdown', 'interception', 'rushing_yards', 'rush_touchdown', 'reception', 'receiving_yards', 'receiving_touchdown', 'fumble_lost', 'sack', 'safety', 'blocked_kick', 'touchdown', 'td_team']
    missing_cols = [col for col in DATA_COLS if col not in df.columns]
    if missing_cols:
        df = pd.concat([df, pd.DataFrame(0, index=df.index, columns=missing_cols)], axis=1)
    df[DATA_COLS] = df[DATA_COLS].fillna(0)

    game_info = games_df[['game_id', 'roof', 'wind', 'temp']]
    df = pd.merge(df, game_info, on='game_id', how='left')
    
    weather_defaults = {'roof': 'outdoors', 'wind': 0, 'temp': 70}
    cols_to_add = {col: default for col, default in weather_defaults.items() if col not in df.columns}
    if cols_to_add:
        df = df.assign(**cols_to_add)

    df['is_dome'] = (df['roof'].isin(['dome', 'closed', 'retractable'])).astype(int)
    df['wind'] = pd.to_numeric(df['wind'], errors='coerce').fillna(0)
    df['precip'] = (pd.to_numeric(df['temp'], errors='coerce').fillna(70) < 32).astype(int)

    tendencies = {}
    skill_plays = df[df['play_type'].isin(['pass', 'run']) & (df['season_type'] == 'REG')].copy()
    games_played = df.groupby('posteam')['game_id'].nunique()

    for team, team_df in skill_plays.groupby('posteam'):
        total_plays, pass_plays, rush_plays = len(team_df), team_df[team_df['play_type'] == 'pass'], team_df[team_df['play_type'] == 'run']
        target_counts, carry_counts = pass_plays['receiver_player_id'].value_counts(), rush_plays['rusher_player_id'].value_counts()
        tendencies[team] = {
            'plays_per_game': total_plays / games_played.get(team, 1),
            'pass_rate': len(pass_plays) / total_plays,
            'target_shares': {id_to_name_map.get(k, k): v / target_counts.sum() for k, v in target_counts.items() if v > 0 and pd.notna(k)},
            'carry_shares': {id_to_name_map.get(k, k): v / carry_counts.sum() for k, v in carry_counts.items() if v > 0 and pd.notna(k)}
        }
    
    fpa_df = df[df['down'].isin([1, 2, 3, 4])].copy()
    fpa_df['passer_fp'] = fpa_df.apply(lambda r: calculate_passer_fp(r, is_td=True), axis=1)
    fpa_df['rusher_fp'] = fpa_df.apply(lambda r: calculate_rusher_fp(r, is_td=True), axis=1)
    fpa_df['receiver_fp'] = fpa_df.apply(lambda r: calculate_receiver_fp(r, is_td=True), axis=1)
    game_defs = []
    for game_id, game_df in fpa_df.groupby('game_id'):
        for team in game_df['posteam'].unique():
            off_df = game_df[game_df['posteam'] == team].copy(); def_team = off_df['defteam'].iloc[0] if not off_df.empty else None
            if def_team is None: continue
            off_df['rusher_pos'] = off_df['rusher_player_id'].map(id_to_pos_map); off_df['receiver_pos'] = off_df['receiver_player_id'].map(id_to_pos_map)
            game_defs.append({'defteam': def_team, 'fpa_qb': off_df['passer_fp'].sum(), 'fpa_rb': off_df[off_df['rusher_pos']=='RB']['rusher_fp'].sum(), 'fpa_wr': off_df[off_df['receiver_pos']=='WR']['receiver_fp'].sum(), 'fpa_te': off_df[off_df['receiver_pos']=='TE']['receiver_fp'].sum()})
    def_stats = pd.DataFrame(game_defs).groupby('defteam').mean(numeric_only=True).reset_index().rename(columns={'defteam':'team'})
    
    # (DST Rate Calculation is removed as per your request)

    df['yardline_100'] = df['yardline_100'].fillna(50).astype(int)
    df = pd.merge(df, def_stats, left_on='defteam', right_on='team', how='left')
    
    processed_plays = []
    action_plays = df[df['down'].isin([1, 2, 3, 4]) & df['play_type'].isin(['pass', 'run'])].copy()
    for _, play in action_plays.iterrows():
        base_info = {'down': play['down'], 'ydstogo': play['ydstogo'], 'yardline_100': play['yardline_100'], 'touchdown': play['touchdown'], 'is_dome': play['is_dome'], 'wind': play['wind'], 'precip': play['precip']}
        def add_record(pos, fp, fpa_col):
            record = base_info.copy(); record.update({'position': pos, 'fantasy_points': fp, 'def_fpa_to_pos': play.get(fpa_col, 0)})
            processed_plays.append(record)
        if pd.notna(play['passer_player_id']): add_record('QB', calculate_passer_fp(play, is_td=False), 'fpa_qb')
        if pd.notna(play['receiver_player_id']):
            player_pos = id_to_pos_map.get(play['receiver_player_id'], 'WR')
            add_record('TE' if player_pos == 'TE' else 'WR', calculate_receiver_fp(play, is_td=False), 'fpa_te' if player_pos == 'TE' else 'fpa_wr')
        if pd.notna(play['rusher_player_id']) and play['passer_player_id'] != play['rusher_player_id']: add_record('RB', calculate_rusher_fp(play, is_td=False), 'fpa_rb')
    
    df_offense = pd.DataFrame(processed_plays).fillna(0)
    return df_offense, def_stats, tendencies

# --- Train Predictive Models ---
def train_models(df_offense, quantiles={'floor': 0.1, 'median': 0.5, 'ceiling': 0.9}):
    models = defaultdict(dict); positions = ['QB', 'RB', 'WR', 'TE'] 
    features = ['down', 'ydstogo', 'yardline_100', 'def_fpa_to_pos', 'is_dome', 'wind', 'precip']
    
    print("--- Training Touchdown Probability Model ---")
    td_X = df_offense[features]; td_y = df_offense['touchdown']
    td_model = LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced', max_iter=1000)
    td_model.fit(td_X, td_y); models['TD_Prob'] = td_model
    
    for pos in positions:
        print(f"--- Training models for position: {pos} ---")
        pos_df = df_offense[df_offense['position'] == pos]
        if len(pos_df) < 200: continue
        X = pos_df[features]; y = pos_df['fantasy_points']
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        for name, q in quantiles.items():
            print(f"  Training {name} model (quantile={q})...")
            model = GradientBoostingRegressor(loss='quantile', alpha=q, n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
            model.fit(X_train, y_train); models[pos][name] = model
    return models

# --- Main Execution ---
if __name__ == "__main__":
    today = datetime.datetime.now()
    current_year = today.year
    
    if today.month < 9:
        end_year = current_year
    else:
        end_year = current_year + 1
        
    YEARS_TO_DOWNLOAD = list(range(end_year - 4, end_year))

    offense_df, def_stats, tendencies = fetch_and_prep_data(YEARS_TO_DOWNLOAD)
    if offense_df is not None and not offense_df.empty:
        trained_models = train_models(offense_df)
        if trained_models:
            model_bundle = {'models': trained_models, 'def_stats': def_stats.set_index('team'), 'tendencies': tendencies}
            joblib.dump(model_bundle, 'nfl_efp_quantile_models.pkl')
            print("\nModel bundle saved to 'nfl_efp_quantile_models.pkl'")