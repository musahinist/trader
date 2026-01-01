import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import requests

# Set page config
st.set_page_config(page_title="Bitcoin Trading Dashboard", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .positive { color: #00cc00; font-weight: bold; }
    .negative { color: #ff0000; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR - Configuration
# ============================================================================
st.sidebar.title("⚙️ Ayarlar")
st.sidebar.divider()

data_source = st.sidebar.radio(
    "📊 Veri Kaynağı Seç",
    ["🔴 Gerçek Veriler (CoinGecko API)", "🟢 Simüle Edilmiş Veriler"],
    index=0
)

st.sidebar.divider()

if data_source == "🟢 Simüle Edilmiş Veriler":
    st.sidebar.title("Simülasyon Ayarları")
    initial_price = st.sidebar.number_input(
        "Başlangıç BTC Fiyatı ($)",
        min_value=10000,
        max_value=100000,
        value=45000,
        step=1000
    )

    daily_volatility = st.sidebar.slider(
        "Günlük Volatilite (%)",
        min_value=1.0,
        max_value=10.0,
        value=3.0,
        step=0.5
    ) / 100

    daily_drift = st.sidebar.slider(
        "Günlük Beklenen Getiri (%)",
        min_value=-1.0,
        max_value=2.0,
        value=0.05,
        step=0.05
    ) / 100
else:
    initial_price = None
    daily_volatility = None
    daily_drift = None

initial_capital = st.sidebar.number_input(
    "Başlangıç Sermayesi ($)",
    min_value=10000,
    max_value=1000000,
    value=100000,
    step=10000
)

st.sidebar.divider()
st.sidebar.title("📊 Golden Cross Ayarları")
ma7_window = st.sidebar.number_input("7-Day MA Penceresi", min_value=5, max_value=20, value=7)
ma30_window = st.sidebar.number_input("30-Day MA Penceresi", min_value=20, max_value=60, value=30)

st.sidebar.divider()
if st.sidebar.button("🔄 Yeniden Simüle Et", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

# ============================================================================
# FETCH REAL DATA FROM COINGECKO API
# ============================================================================
@st.cache_data(ttl=3600)
def fetch_bitcoin_data_from_api(days=60):
    """Fetch Bitcoin price data from CoinGecko API"""
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {
        "vs_currency": "usd",
        "days": days,
        "interval": "daily"
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        prices = data['prices']
        dates = [datetime.fromtimestamp(p[0]/1000) for p in prices]
        closes = [p[1] for p in prices]
        
        return dates, closes
    except Exception as e:
        st.error(f"❌ API Hatasıтай: {e}")
        return None, None

# ============================================================================
# SIMULATE BITCOIN PRICES (for simulated data)
# ============================================================================
@st.cache_data
def simulate_bitcoin_prices(initial_price, days=60, volatility=0.03, drift=0.0005, seed=None):
    if seed is None:
        seed = int(datetime.now().timestamp())
    np.random.seed(seed)
    
    dates = [datetime(2025, 1, 1) + timedelta(days=i) for i in range(days)]
    prices = [initial_price]
    
    for _ in range(days - 1):
        daily_return = np.random.normal(drift, volatility)
        next_price = prices[-1] * (1 + daily_return)
        prices.append(next_price)
    
    return dates, prices

# ============================================================================
# GOLDEN CROSS ALGORITHM
# ============================================================================
def golden_cross_signals(df, ma7_window=7, ma30_window=30):
    df['MA7'] = df['Close'].rolling(window=ma7_window).mean()
    df['MA30'] = df['Close'].rolling(window=ma30_window).mean()
    df['Signal'] = 0
    
    for i in range(1, len(df)):
        ma7 = df.loc[i, 'MA7']
        ma30 = df.loc[i, 'MA30']
        
        if pd.isna(ma7) or pd.isna(ma30):
            continue
        
        prev_ma7 = df.loc[i-1, 'MA7']
        prev_ma30 = df.loc[i-1, 'MA30']
        
        if pd.isna(prev_ma7) or pd.isna(prev_ma30):
            continue
        
        if prev_ma7 <= prev_ma30 and ma7 > ma30:
            df.loc[i, 'Signal'] = 1  # BUY
        elif prev_ma7 >= prev_ma30 and ma7 < ma30:
            df.loc[i, 'Signal'] = -1  # SELL
    
    return df

# ============================================================================
# PORTFOLIO CLASS
# ============================================================================
class Portfolio:
    def __init__(self, initial_cash=100000):
        self.cash = initial_cash
        self.initial_cash = initial_cash
        self.btc_holdings = 0
        self.trades = []
        self.portfolio_values = []
    
    def execute_trade(self, date, price, signal):
        if signal == 1:
            if self.cash > 0:
                btc_to_buy = self.cash / price
                self.btc_holdings += btc_to_buy
                self.cash = 0
                self.trades.append({
                    'Date': date,
                    'Type': 'BUY',
                    'Price': price,
                    'Amount_BTC': btc_to_buy,
                    'Total_USD': btc_to_buy * price,
                    'BTC_Holdings': self.btc_holdings,
                    'Cash': self.cash
                })
        
        elif signal == -1:
            if self.btc_holdings > 0:
                usd_from_sale = self.btc_holdings * price
                self.cash += usd_from_sale
                self.trades.append({
                    'Date': date,
                    'Type': 'SELL',
                    'Price': price,
                    'Amount_BTC': self.btc_holdings,
                    'Total_USD': usd_from_sale,
                    'BTC_Holdings': 0,
                    'Cash': self.cash
                })
                self.btc_holdings = 0
    
    def calculate_portfolio_value(self, current_price):
        return self.cash + (self.btc_holdings * current_price)
    
    def record_daily_value(self, current_price):
        portfolio_value = self.calculate_portfolio_value(current_price)
        self.portfolio_values.append(portfolio_value)

# Get data based on source
if data_source == "🔴 Gerçek Veriler (CoinGecko API)":
    with st.spinner("📥 CoinGecko API'den veriler alınıyor..."):
        dates, prices = fetch_bitcoin_data_from_api(days=60)
        if dates is None or prices is None:
            st.error("Veriler alınamadı. Lütfen daha sonra tekrar deneyin.")
            st.stop()
    data_info = "**Kaynak:** CoinGecko API (Gerçek Veriler)"
else:
    dates, prices = simulate_bitcoin_prices(initial_price, days=60, volatility=daily_volatility, drift=daily_drift)
    data_info = "**Kaynak:** Simüle Edilmiş Veriler (Geometric Brownian Motion)"

df = pd.DataFrame({'Date': dates, 'Close': prices})
df = golden_cross_signals(df, ma7_window=ma7_window, ma30_window=ma30_window)

# Execute trades
portfolio = Portfolio(initial_cash=initial_capital)
for idx, row in df.iterrows():
    signal = row['Signal']
    if signal != 0:
        portfolio.execute_trade(row['Date'], row['Close'], signal)
    portfolio.record_daily_value(row['Close'])

# ============================================================================
# MAIN TITLE
# ============================================================================
st.title("🚀 Bitcoin Golden Cross Trading Bot")
st.markdown("Gerçekçi volatilite modeli kullanarak Bitcoin trading simülasyonu")
st.markdown(data_info)

# ============================================================================
# KEY METRICS
# ============================================================================
col1, col2, col3, col4 = st.columns(4)

final_price = df.iloc[-1]['Close']
final_portfolio_value = portfolio.calculate_portfolio_value(final_price)
total_return = ((final_portfolio_value - initial_capital) / initial_capital) * 100
btc_return = ((final_price - df.iloc[0]['Close']) / df.iloc[0]['Close']) * 100

with col1:
    st.metric(
        "Başlangıç Fiyatı",
        f"${df.iloc[0]['Close']:,.0f}",
        f"→ ${final_price:,.0f}"
    )

with col2:
    st.metric(
        "Portföy Değeri",
        f"${final_portfolio_value:,.0f}",
        f"{total_return:+.2f}%",
        delta_color="inverse"
    )

with col3:
    st.metric(
        "BTC Holding",
        f"{portfolio.btc_holdings:.6f} BTC",
        f"Nakit: ${portfolio.cash:,.0f}"
    )

with col4:
    st.metric(
        "Trade Sayısı",
        len(portfolio.trades),
        f"Kar/Zarar: ${final_portfolio_value - initial_capital:,.0f}",
        delta_color="inverse"
    )

# ============================================================================
# CHARTS
# ============================================================================
st.divider()
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📈 Fiyat & Hareketli Ortalamalar")
    fig, ax = plt.subplots(figsize=(14, 6), facecolor='white')
    
    ax.plot(df['Date'], df['Close'], label='BTC Fiyatı', linewidth=2, color='#1f77b4', marker='o', markersize=3)
    ax.plot(df['Date'], df['MA7'], label=f'{ma7_window}-Day MA', linewidth=2, color='#ff7f0e', linestyle='--')
    ax.plot(df['Date'], df['MA30'], label=f'{ma30_window}-Day MA', linewidth=2, color='#2ca02c', linestyle='--')
    
    # Mark buy/sell points
    buy_signals = df[df['Signal'] == 1]
    sell_signals = df[df['Signal'] == -1]
    
    if not buy_signals.empty:
        ax.scatter(buy_signals['Date'], buy_signals['Close'], color='green', marker='^', 
                  s=200, label='BUY Sinyali', zorder=5, edgecolors='darkgreen', linewidth=2)
    
    if not sell_signals.empty:
        ax.scatter(sell_signals['Date'], sell_signals['Close'], color='red', marker='v', 
                  s=200, label='SELL Sinyali', zorder=5, edgecolors='darkred', linewidth=2)
    
    ax.set_xlabel('Tarih', fontsize=11)
    ax.set_ylabel('Fiyat ($)', fontsize=11)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

with col2:
    st.subheader("💰 Portföy Değeri")
    fig, ax = plt.subplots(figsize=(6, 6), facecolor='white')
    
    ax.plot(df['Date'], portfolio.portfolio_values, linewidth=2.5, color='#9467bd', marker='o', markersize=4)
    ax.axhline(y=initial_capital, color='red', linestyle='--', linewidth=2, label='Başlangıç', alpha=0.7)
    ax.fill_between(df['Date'], initial_capital, portfolio.portfolio_values, alpha=0.3, color='#9467bd')
    
    ax.set_xlabel('Tarih', fontsize=11)
    ax.set_ylabel('Değer ($)', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

# ============================================================================
# TRADE LEDGER
# ============================================================================
st.divider()
st.subheader("📋 Trade Ledgeri")

if portfolio.trades:
    trades_df = pd.DataFrame(portfolio.trades)
    trades_df['Date'] = trades_df['Date'].dt.strftime('%Y-%m-%d')
    trades_df['Price'] = trades_df['Price'].apply(lambda x: f"${x:,.2f}")
    trades_df['Amount_BTC'] = trades_df['Amount_BTC'].apply(lambda x: f"{x:.6f}")
    trades_df['Total_USD'] = trades_df['Total_USD'].apply(lambda x: f"${x:,.2f}")
    trades_df['BTC_Holdings'] = trades_df['BTC_Holdings'].apply(lambda x: f"{x:.6f}")
    trades_df['Cash'] = trades_df['Cash'].apply(lambda x: f"${x:,.2f}")
    
    st.dataframe(
        trades_df,
        use_container_width=True,
        hide_index=True
    )
else:
    st.warning("⚠️ Bu simülasyonda hiçbir trade yapılmadı. (Golden Cross sinyali oluşmadı)")

# ============================================================================
# PERFORMANCE COMPARISON
# ============================================================================
st.divider()
st.subheader("📊 Strateji Karşılaştırması")

col1, col2, col3 = st.columns(3)

# Golden Cross Strategy
with col1:
    st.markdown("### 🤖 Golden Cross Strateji")
    st.metric("Final Değer", f"${final_portfolio_value:,.0f}")
    st.metric("Getiri", f"{total_return:+.2f}%", delta_color="inverse")

# Buy & Hold Strategy
initial_btc = initial_capital / df.iloc[0]['Close']
buy_hold_value = initial_btc * final_price
buy_hold_return = ((buy_hold_value - initial_capital) / initial_capital) * 100

with col2:
    st.markdown("### 📈 Buy & Hold Strateji")
    st.metric("Final Değer", f"${buy_hold_value:,.0f}")
    st.metric("Getiri", f"{buy_hold_return:+.2f}%", delta_color="inverse")

# Comparison
with col3:
    st.markdown("### ⚖️ Fark")
    difference = final_portfolio_value - buy_hold_value
    difference_pct = total_return - buy_hold_return
    
    st.metric("Mutlak Fark", f"${difference:,.0f}")
    color = "green" if difference > 0 else "red"
    st.markdown(f"<p style='color: {color}; font-size: 24px; font-weight: bold;'>{difference_pct:+.2f}%</p>", 
                unsafe_allow_html=True)

# ============================================================================
# PRICE DATA TABLE
# ============================================================================
st.divider()
st.subheader("📊 Fiyat Verileri")

display_df = df[['Date', 'Close', 'MA7', 'MA30']].copy()
display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
display_df['Close'] = display_df['Close'].apply(lambda x: f"${x:,.2f}")
display_df['MA7'] = display_df['MA7'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "N/A")
display_df['MA30'] = display_df['MA30'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "N/A")

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)

# ============================================================================
# FOOTER
# ============================================================================
st.divider()
st.markdown("""
<div style='text-align: center; color: #888; font-size: 12px; margin-top: 20px;'>
    <p>Bitcoin Golden Cross Trading Bot | Geometric Brownian Motion Simülasyonu</p>
    <p>⚠️ Bu simulator eğitim amaçlıdır. Gerçek yatırım kararları için finansal danışmanla görüşün.</p>
</div>
""", unsafe_allow_html=True)
