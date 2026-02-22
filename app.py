import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

st.title("📊 Covered Call Strike Decision Engine")
st.subheader("Single Stock – Multiple Year Files")

uploaded_files = st.file_uploader(
    "Upload Excel Files (Same Stock – Different Years)",
    type=["xlsx"],
    accept_multiple_files=True
)

if uploaded_files:

    dfs = []

    # ===== LOAD & CLEAN EACH FILE =====
    for file in uploaded_files:

        temp_df = pd.read_excel(file)

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

    # ===== STACK VERTICALLY =====
    df = pd.concat(dfs)
    df = df.sort_values('DATE')
    df = df.drop_duplicates(subset='DATE')

    st.success(f"{len(uploaded_files)} files combined successfully")

    # ===== FEATURE ENGINEERING =====
    df['SMA50'] = df['CLOSE'].rolling(50).mean()
    df['SMA200'] = df['CLOSE'].rolling(200).mean()
    df['Trend'] = (df['SMA50'] - df['SMA200']) / df['CLOSE']

    df['Return'] = df['CLOSE'].pct_change()
    df['Vol20'] = df['Return'].rolling(20).std() * np.sqrt(252)

    df['VWAP_Dev'] = (df['CLOSE'] - df['VWAP']) / df['VWAP']

    df['Forward_Return_20d'] = df['CLOSE'].shift(-20) / df['CLOSE'] - 1

    # ===== TARGETS =====
    df['Regime'] = 1
    df.loc[df['Trend'] > 0.03, 'Regime'] = 2
    df.loc[df['Trend'] < -0.03, 'Regime'] = 0

    df['Breakout'] = (df['Forward_Return_20d'] > 0.05).astype(int)
    df['Breakdown'] = (df['Forward_Return_20d'] < -0.05).astype(int)

    df = df.dropna()

    # ===== MODEL TRAINING =====
    features = ['Trend', 'Vol20', 'VWAP_Dev']
    X = df[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    regime_model = RandomForestClassifier(n_estimators=200, random_state=42)
    regime_model.fit(X_scaled, df['Regime'])

    log_model = LogisticRegression()
    log_model.fit(X_scaled, df['Breakout'])

    down_model = LogisticRegression()
    down_model.fit(X_scaled, df['Breakdown'])

    # ===== CURRENT STATE =====
    latest_features = scaler.transform(df[features].iloc[-1:])

    current_regime = regime_model.predict(latest_features)[0]
    breakout_prob = log_model.predict_proba(latest_features)[0][1]
    downside_prob = down_model.predict_proba(latest_features)[0][1]

    avg_up = df['Breakout'].mean()
    avg_down = df['Breakdown'].mean()

    # ===== DECISION ENGINE =====
    up_state = "Elevated" if breakout_prob > avg_up else "Below Normal"
    down_state = "Elevated" if downside_prob > avg_down else "Below Normal"

    if current_regime == 2:
        if down_state == "Elevated":
            decision = "ATM (Risk Control)"
        elif up_state == "Elevated":
            decision = "Far OTM (6–8%)"
        else:
            decision = "Slight OTM (3–5%)"

    elif current_regime == 1:
        if down_state == "Elevated":
            decision = "Slight ITM (2–3%)"
        else:
            decision = "ATM (0–2%)"

    else:
        if down_state == "Elevated":
            decision = "ITM (3–5%)"
        else:
            decision = "Slight ITM (2–3%)"

    regime_map = {0: "Downtrend", 1: "Sideways", 2: "Uptrend"}

    # ===== DISPLAY RESULTS =====
    st.subheader("📈 Current Market State")
    st.write("**Current Regime:**", regime_map[current_regime])
    st.write("**Upside Breakout Probability (>5%):**", f"{breakout_prob*100:.2f}%")
    st.write("**Downside Breakdown Probability (<-5%):**", f"{downside_prob*100:.2f}%")
    st.write("**Upside State:**", up_state)
    st.write("**Downside State:**", down_state)

    st.subheader("🎯 Strike Recommendation")
    st.success(decision)

    st.subheader("📊 Historical Baselines")
    st.write("Average Breakout Probability:", f"{avg_up*100:.2f}%")
    st.write("Average Breakdown Probability:", f"{avg_down*100:.2f}%")

    # ===== DIAGNOSTICS =====
    with st.expander("🔍 Model Diagnostics"):
        st.write("Last Feature Values")
        st.dataframe(df[features].iloc[-1:])

        coef_df = pd.DataFrame({
            "Feature": features,
            "Breakout Coef": log_model.coef_[0],
            "Breakdown Coef": down_model.coef_[0]
        })

        st.dataframe(coef_df)
