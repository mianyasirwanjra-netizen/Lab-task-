# ============================================================
# TITANIC SURVIVAL AI SYSTEM (ALL IN ONE)
# Run: python "yasir fyp titanic.py"
# ============================================================

import subprocess
import sys

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

for pkg in ["flask", "pandas", "scikit-learn", "numpy"]:
    try:
        __import__(pkg)
    except:
        install(pkg)

import pandas as pd
import numpy as np
import pickle
import os
from io import StringIO
from flask import Flask, request, render_template_string
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "titanic_model.pkl")
CSV_PATH   = os.path.join(BASE_DIR, "yasir test.csv")

BACKUP_DATA = """survived,pclass,sex,age,sibsp,parch,fare,embarked
0,3,male,22,1,0,7.25,S
1,1,female,38,1,0,71.2833,C
1,3,female,26,0,0,7.925,S
1,1,female,35,1,0,53.1,S
0,3,male,35,0,0,8.05,S
0,3,male,27,0,0,8.4583,Q
0,1,male,54,0,0,51.8625,S
0,3,male,2,3,1,21.075,S
1,3,female,27,0,2,11.1333,S
1,2,female,14,1,0,30.0708,C
1,3,female,4,1,1,16.7,S
1,1,female,58,0,0,26.55,S
0,3,male,20,0,0,8.05,S
0,3,male,39,1,5,31.275,S
0,3,female,14,0,0,7.8542,S
1,2,female,55,0,0,16,S
0,3,male,2,4,1,29.125,Q
1,2,male,23,0,0,13,S
0,3,female,31,1,0,18,S
1,3,female,22,0,0,7.225,C
0,2,male,35,0,0,26,S
1,2,male,34,0,0,13,S
1,3,female,15,0,0,8.0292,Q
1,1,male,28,0,0,35.5,S
0,3,female,8,3,1,21.075,S
1,3,female,38,1,5,31.275,S
0,3,male,19,0,0,7.8958,S
0,1,male,40,0,0,27.7208,C
1,3,female,28,0,0,7.8792,Q
0,3,male,28,0,0,7.8958,S
0,1,male,40,0,0,146.5208,C
1,1,female,28,1,0,82.1708,C
1,3,female,24,0,0,7.8958,S
0,2,male,28,0,0,10.5,S
0,1,male,32,0,0,30,C
1,3,female,20,1,0,15.7417,C
0,3,male,28,0,0,7.775,S
1,2,female,17,0,0,12,S
0,3,male,26,0,0,7.8958,S
1,3,female,28,0,0,7.8958,S
0,3,male,29,0,0,7.8958,S
1,1,female,45,1,0,83.475,S
0,3,male,28,0,0,8.4583,Q
1,2,female,28,0,0,26,S
0,3,male,45,0,0,8.05,S
0,3,male,65,0,1,61.9792,C
1,1,male,28,1,0,35.5,S
0,3,male,19,0,0,7.8958,S
1,3,female,19,0,0,7.8792,Q
0,3,male,50,2,0,133.65,S
1,1,female,28,0,0,28.7125,C
0,3,male,25,1,2,7.925,S
0,3,male,26,0,0,7.8958,S
1,2,female,36,1,0,17.4,S
0,1,male,28,0,0,27.7208,C
0,3,male,33,0,0,7.8958,S
0,2,male,28,0,0,10.5,S
1,2,female,58,0,0,146.5208,C
0,3,male,28,0,0,7.75,Q
1,1,female,50,0,0,28.7125,C
0,3,male,28,0,0,7.75,Q
0,2,male,28,0,0,13,S
1,3,female,28,0,0,9.5,S
0,1,male,38,0,0,0,S
1,1,female,18,1,0,90,Q
0,2,male,28,0,0,13,S
1,2,female,28,0,0,13,S
1,1,female,36,0,0,135.6333,C
0,3,male,42,1,0,52,S
0,3,male,26,0,0,7.8958,S
1,2,female,24,0,0,13,S
0,1,male,36,0,1,512.3292,C
1,3,female,24,2,0,65,S
1,2,female,33,0,0,13,S
1,1,female,40,0,0,153.4625,S
1,1,female,36,1,2,120,S
1,3,female,32,0,0,13,S
1,1,female,38,0,0,227.525,C
1,3,female,38,1,0,16.1,S
1,1,female,28,1,0,82.1708,C
1,1,female,28,0,0,59.4,C
0,3,male,45,0,0,7.8,S
1,2,female,29,1,0,26,S
1,1,female,58,1,0,113.275,C
1,3,female,28,1,0,16.1,S
1,1,female,60,1,0,75.25,C
1,1,female,44,0,1,57.9792,C
1,2,female,28,0,0,13,S
1,1,female,49,0,0,25.9292,C
1,1,female,52,1,1,79.65,S
1,3,female,28,0,0,7.75,Q
1,1,male,36,1,2,120,S
1,2,female,28,0,0,13,S
1,1,female,48,1,0,76.7292,C
0,1,male,64,1,4,263,S
0,1,male,32,0,0,30,C
0,1,male,25,0,0,26,S
0,3,male,40,0,0,7.8958,S
0,3,male,55,0,0,8.05,S
0,3,male,30,0,0,8.6625,S
1,1,female,43,1,0,211.3375,S
0,3,male,18,0,0,8.05,S
1,2,female,26,0,0,26,S
1,1,female,24,0,0,69.3,C
0,2,male,28,0,0,13,S
1,3,female,16,0,0,7.8542,S
1,3,female,44,1,0,7.8542,S
0,3,male,60,0,0,7.8958,S
1,1,female,42,0,0,227.525,C
0,2,male,29,0,0,13,S
0,3,male,45,0,0,8.05,S
1,1,female,35,1,0,83.475,S
0,3,male,32,0,0,8.05,S
0,3,male,23,0,0,7.925,S
1,1,female,54,1,0,78.2667,C
0,3,male,38,0,0,7.8958,S
1,2,female,22,0,0,10.5167,S
0,3,male,31,0,0,8.6625,S
1,1,female,27,0,2,211.3375,S
0,3,male,34,0,0,8.05,S
1,1,female,24,0,0,83.1583,C
1,2,female,45,0,1,26.25,S
0,3,male,42,0,0,7.55,S
1,3,female,21,0,0,7.8542,S
0,3,male,22,0,0,7.8958,S
0,3,male,37,0,0,7.8542,S
0,3,male,29,0,0,7.8958,S
0,3,male,43,0,0,7.8958,S
0,3,male,33,0,0,7.8958,S
0,3,male,18,0,0,8.05,S
0,3,male,30,0,0,7.8958,S
1,2,female,30,0,0,12.475,S
0,3,male,33,0,0,7.8958,S
1,1,female,24,0,0,56.9292,C
0,3,male,36,0,0,7.8958,S
0,2,male,41,0,0,15.0458,C
0,3,male,32,0,0,7.8542,S
1,3,female,25,0,0,7.8542,S
0,3,male,18,0,0,7.8958,S
0,3,male,35,0,0,7.8958,S
0,3,male,25,0,0,7.8958,S
1,2,female,28,0,0,26,S
0,3,male,31,0,0,7.8958,S
0,3,male,40,0,0,7.8958,S
1,1,female,30,0,0,93.5,S
0,3,male,30,0,0,56.4958,S
0,1,male,44,2,0,90,Q
1,1,female,28,0,0,151.55,S
1,1,female,47,1,1,52.5542,S
1,2,female,21,0,0,73.5,S
1,1,female,34,1,0,57.0,S
1,1,female,31,1,0,113.275,C
1,1,female,18,0,0,79.65,S
1,2,female,36,0,0,13,S
0,3,male,55,0,0,8.05,S
0,1,male,40,0,0,31,C
1,1,female,50,1,0,106.425,C
0,3,male,28,0,0,7.8958,S
0,2,male,30,0,0,13,S
1,1,female,36,0,0,135.6333,C
1,2,female,24,0,0,13,S
1,1,female,45,1,0,164.8667,S
1,1,female,55,0,0,25.7417,C
1,3,female,28,0,0,7.8792,Q
1,1,female,40,0,0,153.4625,S
1,1,female,46,0,0,75.2417,C
"""

if not os.path.exists(MODEL_PATH):
    print("Model train ho raha hai...")
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        df.columns = [c.strip().lower() for c in df.columns]
        if 'survived' not in df.columns:
            df = pd.read_csv(StringIO(BACKUP_DATA))
    else:
        df = pd.read_csv(StringIO(BACKUP_DATA))

    needed = ['survived','pclass','sex','age','sibsp','parch','fare','embarked']
    df = df[[c for c in needed if c in df.columns]]
    df['age']      = pd.to_numeric(df['age'],      errors='coerce').fillna(28)
    df['fare']     = pd.to_numeric(df['fare'],     errors='coerce').fillna(32)
    df['pclass']   = pd.to_numeric(df['pclass'],   errors='coerce').fillna(3)
    df['sibsp']    = pd.to_numeric(df['sibsp'],    errors='coerce').fillna(0)
    df['parch']    = pd.to_numeric(df['parch'],    errors='coerce').fillna(0)
    df['survived'] = pd.to_numeric(df['survived'], errors='coerce')
    df['embarked'] = df['embarked'].fillna('S').astype(str)
    df['sex']      = df['sex'].fillna('male').astype(str)
    df.dropna(inplace=True)

    le_sex = LabelEncoder()
    le_emb = LabelEncoder()
    df['sex']      = le_sex.fit_transform(df['sex'])
    df['embarked'] = le_emb.fit_transform(df['embarked'])

    X = df.drop("survived", axis=1)
    y = df["survived"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    print(f"✅ Model ready! Accuracy: {acc*100:.1f}%")
    pickle.dump(clf, open(MODEL_PATH, "wb"))
else:
    print(f"✅ Model load ho gaya!")

app   = Flask(__name__)
model = pickle.load(open(MODEL_PATH, "rb"))

html_page = """
<!DOCTYPE html>
<html>
<head>
<title>Titanic Survival Predictor</title>
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=DM+Sans:wght@400;500&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'DM Sans', sans-serif;
    background: #0a0f1e;
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 20px;
    background-image:
      radial-gradient(ellipse at 20% 50%, rgba(14,165,233,0.08) 0%, transparent 60%),
      radial-gradient(ellipse at 80% 20%, rgba(99,102,241,0.08) 0%, transparent 60%);
  }
  .card {
    width: 100%;
    max-width: 460px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 20px;
    padding: 40px 36px;
    backdrop-filter: blur(12px);
    box-shadow: 0 32px 80px rgba(0,0,0,0.5);
  }
  .ship-icon { font-size: 2.4rem; margin-bottom: 8px; }
  h1 { font-family: 'Playfair Display', serif; font-size: 1.9rem; color: #f0f9ff; line-height: 1.2; margin-bottom: 4px; }
  .subtitle { font-size: 0.82rem; color: #64748b; margin-bottom: 28px; letter-spacing: 0.05em; text-transform: uppercase; }
  .field { margin-bottom: 14px; }
  label { display: block; font-size: 0.75rem; font-weight: 500; color: #94a3b8; margin-bottom: 5px; letter-spacing: 0.04em; text-transform: uppercase; }
  input, select { width: 100%; padding: 11px 14px; background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.1); border-radius: 10px; color: #e2e8f0; font-size: 0.92rem; font-family: 'DM Sans', sans-serif; transition: border-color 0.2s, box-shadow 0.2s; outline: none; appearance: none; }
  input::placeholder { color: #475569; }
  input:focus, select:focus { border-color: #38bdf8; box-shadow: 0 0 0 3px rgba(56,189,248,0.12); }
  select option { background: #1e293b; color: #e2e8f0; }
  .row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  button { margin-top: 8px; width: 100%; padding: 14px; background: linear-gradient(135deg, #0ea5e9, #6366f1); color: white; border: none; border-radius: 12px; font-size: 1rem; font-weight: 600; font-family: 'DM Sans', sans-serif; cursor: pointer; transition: opacity 0.2s, transform 0.15s; }
  button:hover { opacity: 0.9; transform: translateY(-1px); }
  button:active { transform: translateY(0); }
  .result { margin-top: 22px; padding: 16px 20px; border-radius: 12px; font-size: 1.05rem; font-weight: 600; text-align: center; animation: fadeIn 0.4s ease; }
  .survived { background: rgba(34,197,94,0.12); border: 1px solid rgba(34,197,94,0.3); color: #4ade80; }
  .died     { background: rgba(239,68,68,0.10); border: 1px solid rgba(239,68,68,0.25); color: #f87171; }
  @keyframes fadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }
</style>
</head>
<body>
<div class="card">
  <div class="ship-icon">🚢</div>
  <h1>Titanic Survival<br>Predictor</h1>
  <p class="subtitle">AI Powered · Random Forest</p>
  <form action="/predict" method="post">
    <div class="field">
      <label>Passenger Class</label>
      <select name="pclass">
        <option value="1">1st Class — Upper</option>
        <option value="2">2nd Class — Middle</option>
        <option value="3" selected>3rd Class — Lower</option>
      </select>
    </div>
    <div class="field">
      <label>Gender</label>
      <select name="sex">
        <option value="male">Male</option>
        <option value="female">Female</option>
      </select>
    </div>
    <div class="row">
      <div class="field">
        <label>Age</label>
        <input type="number" name="age" placeholder="e.g. 28" min="0" max="120" required>
      </div>
      <div class="field">
        <label>Fare (£)</label>
        <input type="number" step="0.01" name="fare" placeholder="e.g. 32.50" min="0" value="32">
      </div>
    </div>
    <div class="row">
      <div class="field">
        <label>Siblings / Spouse</label>
        <input type="number" name="sibsp" placeholder="0" min="0" value="0">
      </div>
      <div class="field">
        <label>Parents / Children</label>
        <input type="number" name="parch" placeholder="0" min="0" value="0">
      </div>
    </div>
    <div class="field">
      <label>Port of Embarkation</label>
      <select name="embarked">
        <option value="0">Cherbourg (C)</option>
        <option value="1">Queenstown (Q)</option>
        <option value="2" selected>Southampton (S)</option>
      </select>
    </div>
    <button type="submit">⚡ Predict Survival</button>
  </form>
  {% if prediction %}
    <div class="result {{ css_class }}">{{ prediction }}</div>
  {% endif %}
</div>
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(html_page, prediction=None, css_class="")

@app.route("/predict", methods=["POST"])
def predict():
    features = [
        int(request.form["pclass"]),
        1 if request.form["sex"] == "male" else 0,
        float(request.form["age"]),
        int(request.form.get("sibsp", 0)),
        int(request.form.get("parch", 0)),
        float(request.form.get("fare", 32.0)),
        int(request.form["embarked"])
    ]
    pred = model.predict([features])[0]
    if pred == 1:
        result, css_class = "✅ Passenger Survived!", "survived"
    else:
        result, css_class = "❌ Passenger Did Not Survive", "died"
    return render_template_string(html_page, prediction=result, css_class=css_class)

if __name__ == "__main__":
    import webbrowser
    import threading
    print("🌐 Chrome khul raha hai → http://127.0.0.1:5000")
    threading.Timer(1.5, lambda: webbrowser.open("http://127.0.0.1:5000")).start()
    app.run(debug=False)
    