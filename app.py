import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score

st.title("📊 Covered Call Strike Decision Engine")
st.subheader("Regime-Specific + Walk-Forward Model (Stable Version)")

uploaded_files = st.file_uploader(
    "Upload Excel Files (Same Stock – Multiple Years)",
    type=["csv"],
    accept_multiple_files=True
)

if uploaded_files:

    # =========================================================
    # LOAD & MERGE FILES
    # =========================================================
    dfs = []

    for file in uploaded_files:
        temp_df = pd.read_csv(file)

        price_cols = [
            'OPEN', 'HIGH', 'LOW', 'CLOSE',
            'PREV. CLOSE', 'LTP', 'VWAP',
            '52W H', '52W L', 'VALUE'
        ]

        for col in price_cols:
            if col in temp_df.columns:
                temp_df[col] = (
                    temp_df[col]
                    .astype(str)
                    .str.replace(',', '', regex=False)
                )
                temp_df[col] = pd.to_numeric(temp_df[col], errors='coerce')

        temp_df['DATE'] = pd.to_datetime(temp_df['DATE'], dayfirst=True)
        dfs.append(temp_df)

    df = pd.concat(dfs)
    df = df.sort_values('DATE')
    df = df.drop_duplicates(subset='DATE')

    st.success(f"{len(uploaded_files)} files merged successfully")

    # =========================================================
    # FEATURE ENGINEERING
    # =========================================================
    df['SMA50'] = df['CLOSE'].rolling(50).mean()
    df['SMA200'] = df['CLOSE'].rolling(200).mean()
    df['Trend'] = (df['SMA50'] - df['SMA200']) / df['CLOSE']
    df['Dist_200'] = (df['CLOSE'] - df['SMA200']) / df['SMA200']

    df['Return'] = df['CLOSE'].pct_change()
    df['Vol20'] = df['Return'].rolling(20).std() * np.sqrt(252)
    df['VWAP_Dev'] = (df['CLOSE'] - df['VWAP']) / df['VWAP']
    df['Mom20'] = df['CLOSE'].pct_change(20)

    df['Forward_Return_20d'] = df['CLOSE'].shift(-20) / df['CLOSE'] - 1

    df = df.dropna()

    # =========================================================
    # ROBUST BREAKOUT DEFINITION (Quantile-Based)
    # =========================================================
    breakout_threshold = df['Forward_Return_20d'].quantile(0.70)
    breakdown_threshold = df['Forward_Return_20d'].quantile(0.30)

    df['Breakout'] = (df['Forward_Return_20d'] > breakout_threshold).astype(int)
    df['Breakdown'] = (df['Forward_Return_20d'] < breakdown_threshold).astype(int)

    st.write("Breakout Threshold:", round(breakout_threshold, 4))
    st.write("Breakout Class Balance:")
    st.write(df['Breakout'].value_counts())

    # =========================================================
    # REGIME CLASSIFICATION
    # =========================================================
    df['Regime'] = 1
    df.loc[df['Trend'] > 0.03, 'Regime'] = 2
    df.loc[df['Trend'] < -0.03, 'Regime'] = 0

    features = ['Trend', 'Vol20', 'VWAP_Dev', 'Mom20', 'Dist_200']

    # =========================================================
    # WALK-FORWARD VALIDATION
    # =========================================================
    window = 600
    step = 100

    accuracies, precisions, aucs = [], [], []

    for start in range(0, len(df) - window - step, step):

        train = df.iloc[start:start+window]
        test = df.iloc[start+window:start+window+step]

        X_train = train[features]
        y_train = train['Breakout']
        X_test = test[features]
        y_test = test['Breakout']

        # Skip if single class
        if y_train.nunique() < 2 or y_test.nunique() < 2:
            continue

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = LogisticRegression(class_weight='balanced', max_iter=1000)
        model.fit(X_train_scaled, y_train)

        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        y_pred = (y_prob > 0.5).astype(int)

        accuracies.append(accuracy_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred, zero_division=0))
        aucs.append(roc_auc_score(y_test, y_prob))

    st.subheader("📊 Walk-Forward Performance")

    if len(accuracies) > 0:
        st.write("Accuracy:", f"{np.mean(accuracies)*100:.2f}%")
        st.write("Precision:", f"{np.mean(precisions)*100:.2f}%")
        st.write("ROC AUC:", f"{np.mean(aucs)*100:.2f}%")
    else:
        st.warning("Not enough class variation for walk-forward evaluation.")

    # =========================================================
    # REGIME-SPECIFIC MODELS
    # =========================================================
    regime_models = {}

    for regime in [0, 1, 2]:

        regime_df = df[df['Regime'] == regime]

        if len(regime_df) < 150 or regime_df['Breakout'].nunique() < 2:
            continue

        X_reg = regime_df[features]
        y_reg = regime_df['Breakout']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_reg)

        model = LogisticRegression(class_weight='balanced', max_iter=1000)
        model.fit(X_scaled, y_reg)

        regime_models[regime] = (model, scaler)

    # Train Regime Classifier
    scaler_reg = StandardScaler()
    X_regime_scaled = scaler_reg.fit_transform(df[features])

    regime_model = RandomForestClassifier(n_estimators=200, random_state=42)
    regime_model.fit(X_regime_scaled, df['Regime'])

    # Downside model
    down_model = LogisticRegression(class_weight='balanced', max_iter=1000)
    down_model.fit(X_regime_scaled, df['Breakdown'])

    # =========================================================
    # CURRENT PREDICTION
    # =========================================================
    latest = df[features].iloc[-1:]
    latest_scaled_reg = scaler_reg.transform(latest)

    current_regime = regime_model.predict(latest_scaled_reg)[0]

    if current_regime in regime_models:
        model, scaler = regime_models[current_regime]
        latest_scaled = scaler.transform(latest)
        breakout_prob = model.predict_proba(latest_scaled)[0][1]
    else:
        breakout_prob = df['Breakout'].mean()

    downside_prob = down_model.predict_proba(latest_scaled_reg)[0][1]

    avg_up = df['Breakout'].mean()
    avg_down = df['Breakdown'].mean()

    # =========================================================
    # DECISION ENGINE
    # =========================================================
    up_state = "Elevated" if breakout_prob > avg_up else "Below Normal"
    down_state = "Elevated" if downside_prob > avg_down else "Below Normal"

    regime_map = {0: "Downtrend", 1: "Sideways", 2: "Uptrend"}

    if current_regime == 2:
        decision = "ATM (Risk Control)" if down_state == "Elevated" else \
                   "Far OTM (6–8%)" if up_state == "Elevated" else \
                   "Slight OTM (3–5%)"

    elif current_regime == 1:
        decision = "Slight ITM (2–3%)" if down_state == "Elevated" else "ATM (0–2%)"

    else:
        decision = "ITM (3–5%)" if down_state == "Elevated" else "Slight ITM (2–3%)"

    # =========================================================
    # DISPLAY
    # =========================================================
    st.subheader("📈 Current Market State")
    st.write("Current Regime:", regime_map[current_regime])
    st.write("Upside Breakout Probability:", f"{breakout_prob*100:.2f}%")
    st.write("Downside Breakdown Probability:", f"{downside_prob*100:.2f}%")
    st.write("Upside State:", up_state)
    st.write("Downside State:", down_state)

    st.subheader("🎯 Strike Recommendation")
    st.success(decision)

    st.subheader("📊 Historical Baselines")
    st.write("Average Breakout Probability:", f"{avg_up*100:.2f}%")
    st.write("Average Breakdown Probability:", f"{avg_down*100:.2f}%")

    with st.expander("🔍 Diagnostics"):
        st.write("Last Feature Values")
        st.dataframe(latest)

        st.write("Feature Importance (Regime Model)")
        importance_df = pd.DataFrame({
            "Feature": features,
            "Importance": regime_model.feature_importances_
        })
        st.dataframe(importance_df)
