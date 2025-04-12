import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from scipy.stats import poisson
import time
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

current_date = pd.to_datetime("2025-04-11")

def scrape_match_logs(url, team_name, competition):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124'}
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"Failed to fetch {url}: {response.status_code}")
        return None
    
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table',{'id': 'matchlogs_for'})
    if not table:
        print(f"No match logs table found for {team_name} - {competition}")
        return None
    
    headers = [th.text.strip() for th in table.find('thead').find_all('th')]
    rows = []
    for tr in table.find('tbody').find_all('tr'):
        cells = [td.text.strip() for td in tr.find_all('td')]
        date = tr.find('th').text.strip()
        if cells and date:
            rows.append([date] + cells)

    df = pd.DataFrame(rows, columns=['Date'] + headers[1:])
    df['Competition'] = competition
    df['Team'] = team_name
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df[df['Date'] <= current_date]

    numeric_cols = ['GF', 'xG', 'GA', 'xGA', 'Poss']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    df['Is_Home'] = df['Venue'].apply(lambda x: 1 if x == 'Home' else 0)
    return df

urls = [
    ("https://fbref.com/en/squads/18bb7c10/2024-2025/all_comps/Arsenal-Stats-All-Competitions", "Arsenal", "UCL"),
    ("https://fbref.com/en/squads/18bb7c10/2024-2025/Arsenal-Stats", "Arsenal", "Premier League"),
    ("https://fbref.com/en/squads/18bb7c10/2023-2024/all_comps/Arsenal-Stats-All-Competitions", "Arsenal", "UCL"),
    ("https://fbref.com/en/squads/18bb7c10/2023-2024/Arsenal-Stats", "Arsenal", "Premier League"),
    ("https://fbref.com/en/squads/53a2f082/2024-2025/all_comps/Real-Madrid-Stats-All-Competitions", "Real Madrid", "UCL"),
    ("https://fbref.com/en/squads/53a2f082/2024-2025/Real-Madrid-Stats", "Real Madrid", "La Liga"),
    ("https://fbref.com/en/squads/53a2f082/2023-2024/all_comps/Real-Madrid-Stats-All-Competitions", "Real Madrid", "UCL"),
    ("https://fbref.com/en/squads/53a2f082/2023-2024/Real-Madrid-Stats", "Real Madrid", "La Liga"),
]

all_matches = []
for url, team, comp in urls:
    print(f"Scraping {team} - {comp}")
    df = scrape_match_logs(url, team, comp)
    if df is not None:
        all_matches.append(df)
    time.sleep(3)
if not all_matches:
    raise ValueError("No match data scraped")

match_data = pd.concat(all_matches, ignore_index=True)
match_data = match_data[(match_data['GF'] > 0) | (match_data['GA'] > 0) | (match_data['xG'] > 0) | (match_data['xGA'] > 0)]
match_data = match_data.sort_values('Date')

latest_arsenal = match_data[match_data['Team'] == 'Arsenal'].tail(50)
latest_madrid = match_data[match_data['Team'] == 'Arsenal'].tail(50)

if len(latest_arsenal) < 50 or len(latest_madrid) < 50:
    print(f"Warning: Arsenal has {len(latest_arsenal)} games, Madrid has {len(latest_madrid)} games")

print(f"Latest Arsenal raw data (last 5 of 50):\n{latest_arsenal[['Date', 'GF', 'xG', 'GA', 'xGA', 'Is_Home']].tail()}")
print(f"Latest Madrid raw data (last 5 of 50):\n{latest_madrid[['Date', 'GF', 'xG', 'GA', 'xGA', 'Is_Home']].tail()}")

def get_outcome(row):
    if row['Result'] == 'W':
        return 2
    elif row['Result'] == 'D':
        return 1
    else:
        return 0
    
match_data['Outcome'] = match_data.apply(get_outcome, axis=1)

def compute_form_features(group, cols, n=5):
    group = group.sort_index()
    weights = np.linspace(0.5, 1.5, n)
    for col in cols:
        group[f'rolling_avg_{col}'] = group[col].shift(1).rolling(window=n).apply(lambda x: np.average(x, weights=weights[-len(x):]), raw=True).fillna(0)
    return group

form_cols = ['GF', 'GA', 'xG', 'xGA', 'Poss']
match_data = match_data.groupby('Team').apply(compute_form_features, cols=form_cols, include_groups=False).reset_index()

X_cols = [f'rolling_avg_{col}' for col in form_cols] + ['Is_Home']
X = match_data[X_cols].fillna(0)
y = match_data['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
xgb = XGBClassifier(random_state=42, objective='multi:softpob', num_class=3, min_child_weight=3)
param_grid = {
    'n_estimators': [200, 300],
    'max_depth': [ 10, 15],
    'learning_rate': [0.05, 0.1]
}
grid_search = GridSearchCV(xgb, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)
y_pred = grid_search.best_estimator_.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Loss', 'Draw', 'Win']))
print(f"Model Accuracy: {accuracy * 100:.1f}%")

def compute_form(df, cols, n=50):
    weights = np.linspace(0.5, 1.5, n)
    form = {}
    for col in cols:
        form[f'avg_{col}'] = np.average(df[col], weights=weights[-len(df):])
        form[f'avg_{col}_home'] = np.average(df[col].where(df['Is_Home'] == 1, 0), weights=weights[-len(df):]) if df['Is_Home'].sum() > 0 else 0
        form[f'avg_{col}_away'] = np.average(df[col].where(df['Is_Home'] == 0, 0), weights=weights[-len(df):]) if(len(df) - df['Is_Home'].sum()) > 0 else 0
    return form

form_cols = ['GF', 'GA', 'xG', 'xGA']
arsenal_form = compute_form(latest_arsenal, form_cols)
madrid_form = compute_form(latest_madrid, form_cols)

madrid_lambda = max(madrid_form.get('avg_GF_home', madrid_form.get('avg_xG_home', 2)), 1.5) * 1.2
arsenal_lambda = max(arsenal_form.get('avg_GF_away', arsenal_form.get('avg_xG_away', 1)), 0.5)

print(f"Madrid expected goals: {madrid_lambda:.2f}")
print(f"Arsenal expected goals: {arsenal_lambda:.2f}")

prob_madrid_4plus = 1 - poisson.cdf(3, madrid_lambda)
prob_arsenal_0 = poisson.pmf(0, arsenal_lambda)
prob_4_0_or_better = prob_madrid_4plus * prob_arsenal_0

print(f"Probability Real Madrid win 4-0+ to qualify to next round: {prob_4_0_or_better * 100:.2f}%")



