# utils.py
import os
import io
import base64
import warnings

import pandas as pd
import numpy as np
from dateutil import parser
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename

warnings.filterwarnings("ignore")

# ——————————————————————————————————————
# Data-Processing & Modeling Utilities
# ——————————————————————————————————————

def parse_mixed_date(d):
    try:
        return parser.parse(d, fuzzy=True, dayfirst=True)
    except:
        return pd.NaT

def hour_to_str(h):
    if pd.isna(h):
        return None
    hour = int(h)
    minute = int(round((h - hour) * 60))
    return f"{hour:02d}:{minute:02d}"

def preprocess_data(df):
    """Clean, parse dates, add features, label-encode categoricals."""
    df = df.copy()
    df.columns = df.columns.str.strip()
    df['Parsed_Shoot_Date'] = df['Dates de shoot'].astype(str).apply(parse_mixed_date)
    df.dropna(subset=['Parsed_Shoot_Date'], inplace=True)

    df['number_of_characters'] = df['Message'].astype(str).apply(len)
    df['is_optimal_length'] = df['number_of_characters'].between(70, 149).astype(int)

    base_features = [
        "Type d'opération", "Type de SMS", "Secteur", "Nom partenaire",
        "Type de lien court", "orientation du sms",
        "incitation (rating 0-1)", "number_of_characters"
    ]
    numeric_cols = ['incitation (rating 0-1)', 'number_of_characters']
    target = 'Tx de clic unique'
    categorical_cols = list(set(base_features) - set(numeric_cols))

    # Clean numeric and target columns
    for col in numeric_cols + [target]:
        df[col] = (
            df[col].astype(str)
                   .str.replace(',', '.')
                   .replace({'-': np.nan, '': np.nan})
                   .astype(float)
        )
    df.dropna(subset=[target], inplace=True)

    # Time features
    df['Heure'] = df['Parsed_Shoot_Date'].dt.hour + df['Parsed_Shoot_Date'].dt.minute/60
    df['Jour']  = df['Parsed_Shoot_Date'].dt.strftime('%A')
    df['Mois']  = df['Parsed_Shoot_Date'].dt.month
    df['sin_hour'] = np.sin(2*np.pi*df['Heure']/24)
    df['cos_hour'] = np.cos(2*np.pi*df['Heure']/24)

    # Label-encode categoricals
    encoders = {}
    for col in categorical_cols + ['Jour']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    feature_list = base_features + ['Heure','Jour','Mois','sin_hour','cos_hour']
    return df, feature_list, target, encoders

def train_model(X, y):
    model = XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def find_top_similar(df, idx, similar_features, top_n=10):
    row = df.iloc[idx]
    mask = np.ones(len(df), dtype=bool)
    mask[idx] = False
    for c in similar_features:
        mask &= df[c] == row[c]
    candidates = df[mask]
    if candidates.empty:
        return []
    top = candidates.sort_values('Predicted_Success', ascending=False).head(top_n)
    seen = set(); results = []
    for _, r in top.iterrows():
        t = round(r['Heure'], 2)
        if t not in seen:
            seen.add(t)
            results.append((r['Predicted_Success'], r['Heure']))
        if len(results) == 3:
            break
    return results

def process_results(df):
    similar = ["Type d'opération","Type de SMS","Type de lien court",
               "orientation du sms","Secteur"]
    all_scores, all_hours = [], []
    for i in range(len(df)):
        pred_score = df.at[i, 'Predicted_Success']
        sim = find_top_similar(df, i, similar, top_n=10)
        # possibly inject the row’s own prediction
        scores = [s for s,_ in sim]
        entries = list(sim)
        if all(pred_score > s for s in scores):
            hr = df.at[i,'Heure']
            if round(hr,2) not in {round(h,2) for _,h in sim}:
                entries.append((pred_score, hr))
        # sort and pick top 3 unique times
        uniq = []
        seen = set()
        for s,h in sorted(entries, key=lambda x:-x[0]):
            t=round(h,2)
            if t not in seen:
                seen.add(t)
                uniq.append((s,h))
            if len(uniq)==3: break
        while len(uniq)<3:
            uniq.append((None,None))
        all_scores.append([f"{s*100:.1f}%" if s is not None else None for s,_ in uniq])
        all_hours .append([hour_to_str(h) for _,h in uniq])
    df['Top_1_Success'], df['Top_1_Heure'] = zip(*all_scores), zip(*all_hours)
    # format predicted
    df['Predicted_Success_Display'] = df['Predicted_Success'].apply(lambda x:f"{x*100:.1f}%")
    return df

def create_visualizations(df):
    graphs = {}
    # By-hour plot
    fig,ax = plt.subplots(figsize=(8,4))
    hour_grp = df.groupby(df['Heure'].round()).mean()
    ax.plot(hour_grp.index, hour_grp['Predicted_Success']*100, marker='o')
    ax.set_xlabel("Hour"); ax.set_ylabel("Success %"); ax.set_title("Success by Hour")
    buf = io.BytesIO(); fig.savefig(buf,format='png'); buf.seek(0)
    graphs['hour_success'] = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    # Distribution
    fig,ax = plt.subplots(figsize=(8,4))
    ax.hist(df['Predicted_Success']*100, bins=20)
    ax.set_xlabel("Success %"); ax.set_ylabel("Frequency"); ax.set_title("Success Distribution")
    buf = io.BytesIO(); fig.savefig(buf,format='png'); buf.seek(0)
    graphs['success_dist'] = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    return graphs
