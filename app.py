
# Enhanced Streamlit app for Supply Chain Demand Forecasting with Improved ARIMA and UI
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA
from io import BytesIO
from matplotlib.backends.backend_pdf import PdfPages
from streamlit_option_menu import option_menu
import warnings
warnings.filterwarnings("ignore")


# ------------------ LOGIN PAGE ------------------
import streamlit as st

def login():
    st.set_page_config(page_title="Login", layout="centered")
    st.title("🔐 Login to Forecast Dashboard")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "admin123":
            st.session_state.logged_in = True
        else:
            st.error("❌ Invalid credentials. Try again.")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login()
    st.stop()
# ------------------------------------------------

st.set_page_config(page_title="Supply Chain Forecasting", layout="wide")

# Sidebar menu
with st.sidebar:
    selected = option_menu("Navigation", ["Upload", "Forecast", "Inventory", "Download"],
                           icons=["cloud-upload", "graph-up", "box", "download"],
                           menu_icon="cast", default_index=0)

# Main App Logic
st.title("📦 Supply Chain Demand Forecasting Dashboard")

# File upload
uploaded_file = st.sidebar.file_uploader("📁 Upload your CSV data", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")

    try:
        df['week'] = pd.to_datetime(df['week'], format='%y/%m/%d')
    except:
        st.error("❌ Could not parse 'week' column — expected format: YY/MM/DD.")
        st.stop()

    if 'units_sold' not in df.columns:
        st.error("❌ 'units_sold' column missing.")
        st.stop()

    store_options = ['All'] + sorted(df['store_id'].dropna().unique().tolist())
    sku_options = ['All'] + sorted(df['sku_id'].dropna().unique().tolist())
    selected_store = st.sidebar.selectbox("Select Store", store_options)
    selected_sku = st.sidebar.selectbox("Select SKU", sku_options)

    filtered_df = df.copy()
    if selected_store != 'All':
        filtered_df = filtered_df[filtered_df['store_id'] == selected_store]
    if selected_sku != 'All':
        filtered_df = filtered_df[filtered_df['sku_id'] == selected_sku]

    min_date = filtered_df['week'].min()
    max_date = filtered_df['week'].max()
    date_range = st.sidebar.date_input("📆 Select Date Range", [min_date, max_date],
                                       min_value=min_date, max_value=max_date)
    filtered_df = filtered_df[(filtered_df['week'] >= pd.to_datetime(date_range[0])) &
                              (filtered_df['week'] <= pd.to_datetime(date_range[1]))]

    st.subheader("🔍 Filtered Dataset Preview")
    st.write(f"**Rows:** {filtered_df.shape[0]}")
    st.dataframe(filtered_df.head(10))

    model_choice = st.sidebar.selectbox("Choose Forecasting Model", ["Rolling Mean", "ARIMA", "Random Forest"])

    weekly = filtered_df.groupby('week')[['units_sold']].sum().reset_index()
    forecast_col = 'forecast'

    if model_choice == "Rolling Mean":
        weekly[forecast_col] = weekly['units_sold'].rolling(window=4, min_periods=1).mean().shift(1)

    elif model_choice == "ARIMA":
        steps = 8
        model = ARIMA(weekly['units_sold'], order=(1, 1, 1))
        fitted_model = model.fit()
        forecast = fitted_model.forecast(steps=steps)
        future_dates = pd.date_range(start=weekly['week'].iloc[-1] + pd.Timedelta(weeks=1), periods=steps, freq='W')
        forecast_df = pd.DataFrame({'week': future_dates, forecast_col: forecast})
        weekly[forecast_col] = np.nan
        weekly = pd.concat([weekly, forecast_df], ignore_index=True)

    elif model_choice == "Random Forest":
        weekly['week_num'] = weekly['week'].astype('int64') // 10**9
        X = weekly[['week_num']]
        y = weekly['units_sold']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X)
        weekly[forecast_col] = y_pred
        st.subheader("📌 Feature Importance (Random Forest)")
        importances = rf.feature_importances_
        st.bar_chart(pd.Series(importances, index=X.columns))

    st.subheader("📈 Forecast Chart")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(weekly['week'], weekly['units_sold'], label='Actual')
    ax.plot(weekly['week'], weekly[forecast_col], label='Forecast')
    ax.set_title(f"Demand Forecast - {model_choice}")
    ax.legend()
    st.pyplot(fig)

    valid = weekly.dropna()
    y_true = valid['units_sold']
    y_pred = valid[forecast_col]
    if not y_pred.empty:
        col1, col2, col3 = st.columns(3)
        col1.metric("📉 MAE", f"{mean_absolute_error(y_true, y_pred):,.0f}")
        col2.metric("📉 RMSE", f"{np.sqrt(mean_squared_error(y_true, y_pred)):,.0f}")
        col3.metric("📉 MAPE (%)", f"{np.mean(np.abs((y_true - y_pred) / y_true)) * 100:.2f}%")

    st.subheader("📦 Inventory Simulation")
    lead_time = st.slider("Select Lead Time (weeks)", 1, 8, 2)
    safety_stock = st.slider("Select Safety Stock %", 0, 100, 20)
    weekly['inventory'] = weekly[forecast_col].shift(lead_time) * (1 + safety_stock / 100)
    st.line_chart(weekly.set_index('week')[['units_sold', 'inventory']])

    st.subheader("⬇️ Download Forecast Results")
    csv = weekly.to_csv(index=False)
    st.download_button("Download Forecast CSV", csv, file_name="forecast_results.csv", mime="text/csv")

    pdf_buffer = BytesIO()
    with PdfPages(pdf_buffer) as pdf:
        pdf.savefig(fig, bbox_inches='tight')
    pdf_buffer.seek(0)
    st.download_button("Download Forecast Chart PDF", pdf_buffer, file_name="forecast_chart.pdf", mime="application/pdf")

else:
    st.info("👈 Upload a CSV file with 'week', 'units_sold', 'store_id', and 'sku_id' to get started.")
