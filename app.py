import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score

st.set_page_config(layout="wide")
st.title("📊 Covered Call Strike Decision Engine")
st.caption("Regime-Specific + Walk-Forward Probabilistic Model")

uploaded_files = st.file_uploader(
    "Upload CSV Files (Same Stock – Multiple Years)",
    type=["csv"],
    accept_multiple_files=True
)

if uploaded_files:

    # ========================= LOAD DATA =========================
    dfs = []

    for file in uploaded_files:
        temp_df = pd.read_csv(file)

        price_cols = [
            'OPEN','HIGH','LOW','CLOSE',
            'PREV. CLOSE','LTP','VWAP',
            '52W H','52W L','VALUE'
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

    df = pd.concat(dfs).sort_values('DATE').drop_duplicates('DATE')
    st.success(f"{len(uploaded_files)} files merged successfully")

    # ========================= FEATURE ENGINEERING =========================
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

    breakout_threshold = df['Forward_Return_20d'].quantile(0.70)
    breakdown_threshold = df['Forward_Return_20d'].quantile(0.30)

    df['Breakout'] = (df['Forward_Return_20d'] > breakout_threshold).astype(int)
    df['Breakdown'] = (df['Forward_Return_20d'] < breakdown_threshold).astype(int)

    features = ['Trend','Vol20','VWAP_Dev','Mom20','Dist_200']

    # ========================= WALK-FORWARD =========================
    window = 600
    step = 100
    accuracies, precisions, aucs = [], [], []

    for start in range(0, len(df)-window-step, step):

        train = df.iloc[start:start+window]
        test = df.iloc[start+window:start+window+step]

        if train['Breakout'].nunique()<2 or test['Breakout'].nunique()<2:
            continue

        scaler = StandardScaler()
        X_train = scaler.fit_transform(train[features])
        X_test = scaler.transform(test[features])

        model = LogisticRegression(class_weight='balanced', max_iter=1000)
        model.fit(X_train, train['Breakout'])

        y_prob = model.predict_proba(X_test)[:,1]
        y_pred = (y_prob > 0.5).astype(int)

        accuracies.append(accuracy_score(test['Breakout'], y_pred))
        precisions.append(precision_score(test['Breakout'], y_pred, zero_division=0))
        aucs.append(roc_auc_score(test['Breakout'], y_prob))

    # ========================= MODEL HEALTH DISPLAY =========================
    st.header("🔬 Model Health (Walk-Forward)")

    if len(accuracies)>0:

        col1,col2,col3 = st.columns(3)
        col1.metric("Accuracy", f"{np.mean(accuracies)*100:.2f}%")
        col2.metric("Precision", f"{np.mean(precisions)*100:.2f}%")
        col3.metric("ROC-AUC", f"{np.mean(aucs)*100:.2f}%")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=np.mean(aucs)*100,
            title={'text':"Model Edge (ROC-AUC %)"},
            gauge={
                'axis':{'range':[0,100]},
                'steps':[
                    {'range':[0,50],'color':'lightcoral'},
                    {'range':[50,60],'color':'khaki'},
                    {'range':[60,100],'color':'lightgreen'}
                ]
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Not enough class variation for evaluation.")

    # ========================= REGIME & PROBABILITY =========================
    df['Regime'] = 1
    df.loc[df['Trend']>0.03,'Regime']=2
    df.loc[df['Trend']<-0.03,'Regime']=0

    scaler_reg = StandardScaler()
    X_reg_scaled = scaler_reg.fit_transform(df[features])

    regime_model = RandomForestClassifier(n_estimators=200, random_state=42)
    regime_model.fit(X_reg_scaled, df['Regime'])

    down_model = LogisticRegression(class_weight='balanced', max_iter=1000)
    down_model.fit(X_reg_scaled, df['Breakdown'])

    latest = df[features].iloc[-1:]
    latest_scaled_reg = scaler_reg.transform(latest)

    current_regime = regime_model.predict(latest_scaled_reg)[0]
    breakout_prob = df['Breakout'].mean()
    downside_prob = down_model.predict_proba(latest_scaled_reg)[0][1]

    avg_up = df['Breakout'].mean()
    avg_down = df['Breakdown'].mean()

    regime_map = {0:"Downtrend",1:"Sideways",2:"Uptrend"}

    # ========================= CURRENT STATE DISPLAY =========================
    st.header("📈 Current Market State")

    # --- Correct Probability Assignment ---
    # breakout_prob = Upside probability
    # downside_prob = Downside probability

    col1, col2, col3 = st.columns(3)

    col1.metric("Regime", regime_map[current_regime])
    col2.metric("Downside Probability", f"{downside_prob*100:.2f}%")
    col3.metric("Upside Probability", f"{breakout_prob*100:.2f}%")

    # --- Properly Labeled Probability Chart ---
    prob_df = pd.DataFrame({
        "Direction": ["Upside", "Downside"],
        "Probability": [breakout_prob, downside_prob]  # FIXED ORDER
    })

    st.subheader("Probability Comparison")
    st.bar_chart(
        prob_df.set_index("Direction"),
        use_container_width=True
    )

    # ========================= DECISION =========================
    up_state = "Elevated" if breakout_prob>avg_up else "Normal"
    down_state = "Elevated" if downside_prob>avg_down else "Normal"

    if current_regime==2:
        decision = "ATM (Risk Control)" if down_state=="Elevated" else \
                   "Far OTM (6–8%)" if up_state=="Elevated" else \
                   "Slight OTM (3–5%)"
    elif current_regime==1:
        decision = "Slight ITM (2–3%)" if down_state=="Elevated" else "ATM (0–2%)"
    else:
        decision = "ITM (3–5%)" if down_state=="Elevated" else "Slight ITM (2–3%)"

    st.header("🎯 Strike Recommendation")
    st.success(decision)

    # ========================= HISTORICAL BASELINE =========================
    st.header("📚 Historical Baseline")

    col1,col2 = st.columns(2)
    col1.metric("Avg Breakout", f"{avg_up*100:.2f}%")
    col2.metric("Avg Breakdown", f"{avg_down*100:.2f}%")

    # ========================= DIAGNOSTICS =========================
    with st.expander("🔍 Diagnostics & Explanation"):

        st.subheader("Last Feature Snapshot")
        st.dataframe(latest)

        importance_df = pd.DataFrame({
            "Feature":features,
            "Importance":regime_model.feature_importances_
        })

        st.subheader("Feature Importance")
        st.bar_chart(importance_df.set_index("Feature"))

        st.markdown("""
        ### 📖 How To Interpret This Dashboard

        **Accuracy** → % of correct predictions  
        **Precision** → When breakout predicted, how often correct  
        **ROC-AUC** → Model’s ability to separate breakout vs non-breakout  

        **Regime**
        - Uptrend → Bullish bias
        - Sideways → Range environment
        - Downtrend → Defensive environment

        **Breakout Probability**
        Model-estimated chance of 20-day upside move.

        **Breakdown Probability**
        Risk of 20-day downside move.

        If probability > historical average → Elevated risk/opportunity.

        **Feature Importance**
        Shows which factors influence regime classification most.
        Higher value = stronger influence.

        ⚠️ This is probabilistic — not certainty.
        Always apply position sizing and risk management.
        """)
