import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import requests
import time
import sqlite3
import os

# Set page config
st.set_page_config(page_title="Crypto Trading Dashboard", layout="wide", initial_sidebar_state="expanded")

# Database setup
DB_PATH = "trading_bot.db"

def init_database():
    """Initialize SQLite database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Trades table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pair TEXT NOT NULL,
            trade_date TIMESTAMP NOT NULL,
            trade_type TEXT NOT NULL,
            price REAL NOT NULL,
            amount REAL NOT NULL,
            total_try REAL NOT NULL,
            holdings REAL NOT NULL,
            cash REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Portfolio state table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS portfolio_state (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pair TEXT NOT NULL UNIQUE,
            cash REAL NOT NULL,
            holdings REAL NOT NULL,
            in_position INTEGER NOT NULL,
            last_ma7 REAL,
            last_ma30 REAL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # User settings table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_settings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            setting_key TEXT NOT NULL UNIQUE,
            setting_value TEXT NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

def save_user_settings(selected_pairs, initial_capital, ma7_window, ma30_window, refresh_interval):
    """Save user settings to database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    settings = {
        'selected_pairs': ','.join(selected_pairs),
        'initial_capital': str(initial_capital),
        'ma7_window': str(ma7_window),
        'ma30_window': str(ma30_window),
        'refresh_interval': str(refresh_interval)
    }
    
    for key, value in settings.items():
        cursor.execute('''
            INSERT OR REPLACE INTO user_settings (setting_key, setting_value, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
        ''', (key, value))
    
    conn.commit()
    conn.close()

def load_user_settings():
    """Load user settings from database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('SELECT setting_key, setting_value FROM user_settings')
    settings = dict(cursor.fetchall())
    conn.close()
    
    return settings

# Initialize database
init_database()

# Initialize session state for portfolios
if 'portfolios_state' not in st.session_state:
    st.session_state.portfolios_state = {}
if 'last_signal' not in st.session_state:
    st.session_state.last_signal = {}

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

# Load saved settings
saved_settings = load_user_settings()

# Get default values from saved settings or use defaults
default_pairs = saved_settings.get('selected_pairs', 'BTC/TRY (Bitcoin),PAXG/TRY (Altın)').split(',')
default_capital = int(saved_settings.get('initial_capital', '100000'))
default_ma7 = int(saved_settings.get('ma7_window', '7'))
default_ma30 = int(saved_settings.get('ma30_window', '30'))
default_refresh = int(saved_settings.get('refresh_interval', '300'))

# Select cryptocurrencies
crypto_pairs = st.sidebar.multiselect(
    "📊 Takip Etmek İstediğiniz Kripto Paralar",
    ["BTC/TRY (Bitcoin)", "ETH/TRY (Ethereum)", "IO/TRY (io.net)", "PAXG/TRY (Altın)", "SOL/TRY (Solana)"],
    default=default_pairs
)

if not crypto_pairs:
    st.warning("Lütfen en az bir kripto para seçin!")
    st.stop()

st.sidebar.divider()

initial_capital = st.sidebar.number_input(
    "Başlangıç Sermayesi (TRY)",
    min_value=10000,
    max_value=10000000,
    value=default_capital,
    step=10000
)

st.sidebar.divider()
st.sidebar.title("📊 Golden Cross Ayarları")
ma7_window = st.sidebar.number_input("7-Day MA Penceresi", min_value=5, max_value=20, value=default_ma7)
ma30_window = st.sidebar.number_input("30-Day MA Penceresi", min_value=20, max_value=60, value=default_ma30)

st.sidebar.divider()
st.sidebar.title("🔄 Veri Yenileme")
refresh_interval = st.sidebar.slider("Yenileme Süresi (saniye)", min_value=60, max_value=3600, value=default_refresh, step=60)

# Save settings button
st.sidebar.divider()
if st.sidebar.button("💾 Ayarları Kaydet", use_container_width=True):
    save_user_settings(crypto_pairs, initial_capital, ma7_window, ma30_window, refresh_interval)
    st.sidebar.success("✅ Ayarlar kaydedildi!")

st.sidebar.divider()
if st.sidebar.button("🔄 Verileri Şimdi Yükle", use_container_width=True):
    st.cache_data.clear()
    st.rerun()
st.sidebar.divider()
st.sidebar.title("💾 Veritabanı Bilgisi")

# Get database stats
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute("SELECT COUNT(*) FROM trades")
total_trades = cursor.fetchone()[0]
cursor.execute("SELECT COUNT(DISTINCT pair) FROM trades")
total_pairs = cursor.fetchone()[0]
conn.close()

st.sidebar.metric("Toplam Trade", total_trades)
st.sidebar.metric("Takip Edilen Pair", total_pairs)

if st.sidebar.button("🗑️ Veritabanını Sıfırla", use_container_width=True):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM trades")
    cursor.execute("DELETE FROM portfolio_state")
    conn.commit()
    conn.close()
    st.session_state.portfolios_state = {}
    st.success("✅ Veritabanı sıfırlandı!")
    st.rerun()
# ============================================================================
# FETCH REAL DATA FROM BINANCE API
# ============================================================================
@st.cache_data(ttl=300)
def fetch_crypto_data_binance(symbol, days=60):
    """Fetch cryptocurrency price data from Binance API"""
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": "1d",
        "limit": days
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if not data:
            return None, None
        
        dates = []
        closes = []
        
        for candle in data:
            timestamp = candle[0]
            close_price = float(candle[4])
            
            date = datetime.fromtimestamp(timestamp/1000)
            dates.append(date)
            closes.append(close_price)
        
        return dates, closes
    except Exception as e:
        return None, None

@st.cache_data(ttl=60)
def get_usdt_to_try_rate():
    """Get USDT to TRY exchange rate from Binance"""
    try:
        # Try USDTTRY pair first
        url = "https://api.binance.com/api/v3/ticker/price"
        params = {"symbol": "USDTTRY"}
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        return float(data['price'])
    except:
        try:
            # Fallback: get current rate from CoinGecko and multiply with a safe rate
            url = "https://api.binance.com/api/v3/ticker/price"
            params = {"symbol": "USDT"}
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return 32.5  # Approximate rate fallback
        except:
            return 32.5  # Safe fallback

# Map pair names to Binance symbols
symbol_map = {
    "BTC/TRY (Bitcoin)": "BTCUSDT",
    "ETH/TRY (Ethereum)": "ETHUSDT",
    "IO/TRY (io.net)": "IOUSDT",
    "PAXG/TRY (Altın)": "PAXGUSDT",
    "SOL/TRY (Solana)": "SOLUSDT"
}

# ============================================================================
# GOLDEN CROSS ALGORITHM
# ============================================================================
def golden_cross_signals(df, ma7_window=7, ma30_window=30):
    """Calculate Golden Cross signals"""
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
# PORTFOLIO CLASS WITH AUTO TRADING AND DATABASE
# ============================================================================
class AutoPortfolio:
    def __init__(self, initial_cash=100000, pair_name=""):
        self.cash = initial_cash
        self.initial_cash = initial_cash
        self.holdings = 0
        self.trades = []
        self.portfolio_values = []
        self.pair_name = pair_name
        self.last_ma7 = None
        self.last_ma30 = None
        self.in_position = False
        
        # Load from database if exists
        self.load_from_db()
    
    def load_from_db(self):
        """Load portfolio state from database"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Load portfolio state
        cursor.execute('''
            SELECT cash, holdings, in_position, last_ma7, last_ma30
            FROM portfolio_state
            WHERE pair = ?
        ''', (self.pair_name,))
        
        result = cursor.fetchone()
        if result:
            self.cash, self.holdings, in_pos, self.last_ma7, self.last_ma30 = result
            self.in_position = bool(in_pos)
        
        # Load trade history
        cursor.execute('''
            SELECT trade_date, trade_type, price, amount, total_try, holdings, cash
            FROM trades
            WHERE pair = ?
            ORDER BY trade_date DESC
            LIMIT 100
        ''', (self.pair_name,))
        
        trades_data = cursor.fetchall()
        for row in trades_data:
            self.trades.append({
                'Date': datetime.fromisoformat(row[0]),
                'Type': row[1],
                'Price': row[2],
                'Amount': row[3],
                'Total_TRY': row[4],
                'Holdings': row[5],
                'Cash': row[6]
            })
        
        conn.close()
    
    def save_to_db(self):
        """Save portfolio state to database"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO portfolio_state 
            (pair, cash, holdings, in_position, last_ma7, last_ma30, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ''', (self.pair_name, self.cash, self.holdings, int(self.in_position), 
              self.last_ma7, self.last_ma30))
        
        conn.commit()
        conn.close()
    
    def save_trade_to_db(self, trade):
        """Save trade to database"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO trades 
            (pair, trade_date, trade_type, price, amount, total_try, holdings, cash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (self.pair_name, trade['Date'].isoformat(), trade['Type'], 
              trade['Price'], trade['Amount'], trade['Total_TRY'], 
              trade['Holdings'], trade['Cash']))
        
        conn.commit()
        conn.close()
    
    def check_and_execute_trade(self, date, price, ma7, ma30):
        """Check for signals and execute trades automatically"""
        signal = 0
        
        # Only check if we have previous MAs
        if self.last_ma7 is not None and self.last_ma30 is not None:
            # Golden Cross (BUY signal)
            if self.last_ma7 <= self.last_ma30 and ma7 > ma30 and not self.in_position:
                signal = 1
                self.in_position = True
            
            # Death Cross (SELL signal)
            elif self.last_ma7 >= self.last_ma30 and ma7 < ma30 and self.in_position:
                signal = -1
                self.in_position = False
        
        # Execute trade if signal generated
        if signal == 1:
            if self.cash > 0:
                amount_to_buy = self.cash / price
                self.holdings += amount_to_buy
                self.cash = 0
                trade = {
                    'Date': date,
                    'Type': 'BUY ✅',
                    'Price': price,
                    'Amount': amount_to_buy,
                    'Total_TRY': amount_to_buy * price,
                    'Holdings': self.holdings,
                    'Cash': self.cash
                }
                self.trades.append(trade)
                self.save_trade_to_db(trade)
                st.success(f"🎯 BUY: {self.pair_name} @ ₺{price:,.0f}")
        
        elif signal == -1:
            if self.holdings > 0:
                usd_from_sale = self.holdings * price
                self.cash += usd_from_sale
                trade = {
                    'Date': date,
                    'Type': 'SELL ✅',
                    'Price': price,
                    'Amount': self.holdings,
                    'Total_TRY': usd_from_sale,
                    'Holdings': 0,
                    'Cash': self.cash
                }
                self.trades.append(trade)
                self.save_trade_to_db(trade)
                st.success(f"🎯 SELL: {self.pair_name} @ ₺{price:,.0f}")
                self.holdings = 0
        
        # Update MAs for next check
        self.last_ma7 = ma7
        self.last_ma30 = ma30
        
        # Save state to database
        self.save_to_db()
    
    def calculate_portfolio_value(self, current_price):
        return self.cash + (self.holdings * current_price)
    
    def record_daily_value(self, current_price):
        portfolio_value = self.calculate_portfolio_value(current_price)
        self.portfolio_values.append(portfolio_value)

# Old Portfolio class for simulation


# ============================================================================
# MAIN TITLE
# ============================================================================
st.title("🚀 Crypto Trading Bot - Gerçek Zamanlı")
st.markdown("Binance API ile Türkiye'de doğru TRY fiyatlarını takip edin")
st.markdown(f"⏰ **Son Güncelleme:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================================
# FETCH AND PROCESS DATA FOR EACH PAIR
# ============================================================================
crypto_data = {}
portfolios = {}

# Get USD/TRY rate from Binance
usdt_to_try_rate = get_usdt_to_try_rate()
st.markdown(f"💱 **Güncel USDT/TRY Kuru:** ₺{usdt_to_try_rate:.2f}")
st.sidebar.metric("USDT/TRY Kuru", f"₺{usdt_to_try_rate:.2f}")

with st.spinner("📥 Binance API'den TRY fiyatları alınıyor..."):
    for pair in crypto_pairs:
        symbol = symbol_map[pair]
        dates, prices = fetch_crypto_data_binance(symbol, days=60)
        
        if dates is None or prices is None:
            st.error(f"❌ {pair} için veriler alınamadı!")
            continue
        
        # Convert USDT prices to TRY using actual Binance rate
        prices_try = [price * usdt_to_try_rate for price in prices]
        
        # Create DataFrame
        df = pd.DataFrame({'Date': dates, 'Close': prices_try})
        df['MA7'] = df['Close'].rolling(window=ma7_window).mean()
        df['MA30'] = df['Close'].rolling(window=ma30_window).mean()
        df['Signal'] = 0  # Initialize signal column for charts
        
        # Initialize or restore portfolio from session state
        if pair not in st.session_state.portfolios_state:
            st.session_state.portfolios_state[pair] = AutoPortfolio(initial_cash=initial_capital, pair_name=pair)
        
        portfolio = st.session_state.portfolios_state[pair]
        
        # Check latest signal and execute if needed
        latest_row = df.iloc[-1]
        if pd.notna(latest_row['MA7']) and pd.notna(latest_row['MA30']):
            portfolio.check_and_execute_trade(
                latest_row['Date'],
                latest_row['Close'],
                latest_row['MA7'],
                latest_row['MA30']
            )
        
        # Mark signals in dataframe for visualization
        for i in range(1, len(df)):
            if pd.notna(df.iloc[i]['MA7']) and pd.notna(df.iloc[i]['MA30']):
                prev_ma7 = df.iloc[i-1]['MA7']
                prev_ma30 = df.iloc[i-1]['MA30']
                curr_ma7 = df.iloc[i]['MA7']
                curr_ma30 = df.iloc[i]['MA30']
                
                if pd.notna(prev_ma7) and pd.notna(prev_ma30):
                    if prev_ma7 <= prev_ma30 and curr_ma7 > curr_ma30:
                        df.at[i, 'Signal'] = 1  # BUY
                    elif prev_ma7 >= prev_ma30 and curr_ma7 < curr_ma30:
                        df.at[i, 'Signal'] = -1  # SELL
        
        # Record values for all days
        portfolio.portfolio_values = []
        for _, row in df.iterrows():
            portfolio.record_daily_value(row['Close'])
        
        crypto_data[pair] = df
        portfolios[pair] = portfolio

# ============================================================================
# DISPLAY METRICS FOR EACH PAIR
# ============================================================================
st.divider()
st.subheader("⏰ Otomatik Trading Durumu")

auto_trading_cols = st.columns(len(crypto_pairs))
for idx, pair in enumerate(crypto_pairs):
    if pair not in portfolios:
        continue
    
    portfolio = portfolios[pair]
    with auto_trading_cols[idx]:
        status = "🟢 Pozisyonda" if portfolio.in_position else "⚪ Nakitten"
        st.markdown(f"**{pair}**")
        st.markdown(f"{status}")
        st.markdown(f"Toplam Trade: {len(portfolio.trades)}")

st.divider()

st.subheader("📊 Portföy Özeti")

cols = st.columns(len(crypto_pairs))

for idx, pair in enumerate(crypto_pairs):
    if pair not in crypto_data:
        continue
    
    df = crypto_data[pair]
    portfolio = portfolios[pair]
    
    with cols[idx]:
        st.markdown(f"### {pair}")
        
        final_price = df.iloc[-1]['Close']
        start_price = df.iloc[0]['Close']
        final_portfolio_value = portfolio.calculate_portfolio_value(final_price)
        total_return = ((final_portfolio_value - initial_capital) / initial_capital) * 100
        price_change = ((final_price - start_price) / start_price) * 100
        
        st.metric("Mevcut Fiyat", f"₺{final_price:,.0f}", f"{price_change:+.2f}%")
        st.metric("Portföy Değeri", f"₺{final_portfolio_value:,.0f}", f"{total_return:+.2f}%")
        st.metric("Trade Sayısı", len(portfolio.trades))
        st.metric("Kar/Zarar", f"₺{final_portfolio_value - initial_capital:,.0f}")

# ============================================================================
# CHARTS FOR EACH PAIR
# ============================================================================
st.divider()
st.subheader("📈 Fiyat Grafiği & Hareketli Ortalamalar")

chart_cols = st.columns(len(crypto_pairs))

for idx, pair in enumerate(crypto_pairs):
    if pair not in crypto_data:
        continue
    
    df = crypto_data[pair]
    portfolio = portfolios[pair]
    
    with chart_cols[idx]:
        st.markdown(f"#### {pair}")
        
        fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
        
        ax.plot(df['Date'], df['Close'], label='Fiyat', linewidth=2, color='#1f77b4', marker='o', markersize=3)
        ax.plot(df['Date'], df['MA7'], label=f'{ma7_window}-Day MA', linewidth=2, color='#ff7f0e', linestyle='--')
        ax.plot(df['Date'], df['MA30'], label=f'{ma30_window}-Day MA', linewidth=2, color='#2ca02c', linestyle='--')
        
        # Mark buy/sell points
        buy_signals = df[df['Signal'] == 1]
        sell_signals = df[df['Signal'] == -1]
        
        if not buy_signals.empty:
            ax.scatter(buy_signals['Date'], buy_signals['Close'], color='green', marker='^', 
                      s=200, label='BUY', zorder=5, edgecolors='darkgreen', linewidth=2)
        
        if not sell_signals.empty:
            ax.scatter(sell_signals['Date'], sell_signals['Close'], color='red', marker='v', 
                      s=200, label='SELL', zorder=5, edgecolors='darkred', linewidth=2)
        
        ax.set_xlabel('Tarih', fontsize=10)
        ax.set_ylabel('Fiyat (₺)', fontsize=10)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

# ============================================================================
# PORTFOLIO VALUE CHARTS
# ============================================================================
st.divider()
st.subheader("💰 Portföy Değeri Takibi")

portfolio_cols = st.columns(len(crypto_pairs))

for idx, pair in enumerate(crypto_pairs):
    if pair not in crypto_data:
        continue
    
    df = crypto_data[pair]
    portfolio = portfolios[pair]
    
    with portfolio_cols[idx]:
        st.markdown(f"#### {pair}")
        
        fig, ax = plt.subplots(figsize=(10, 5), facecolor='white')
        
        ax.plot(df['Date'], portfolio.portfolio_values, linewidth=2.5, color='#9467bd', marker='o', markersize=4)
        ax.axhline(y=initial_capital, color='red', linestyle='--', linewidth=2, label='Başlangıç', alpha=0.7)
        ax.fill_between(df['Date'], initial_capital, portfolio.portfolio_values, alpha=0.3, color='#9467bd')
        
        ax.set_xlabel('Tarih', fontsize=10)
        ax.set_ylabel('Değer (₺)', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

# ============================================================================
# TRADE LEDGER FOR EACH PAIR
# ============================================================================
st.divider()
st.subheader("📋 Trade Ledgerleri (Veritabanından)")

for pair in crypto_pairs:
    if pair not in crypto_data:
        continue
    
    portfolio = portfolios[pair]
    
    st.markdown(f"### {pair}")
    
    # Get all trades from database
    conn = sqlite3.connect(DB_PATH)
    trades_from_db = pd.read_sql_query(
        "SELECT * FROM trades WHERE pair = ? ORDER BY trade_date DESC LIMIT 50",
        conn,
        params=(pair,)
    )
    conn.close()
    
    if not trades_from_db.empty:
        # Format for display
        trades_from_db['trade_date'] = pd.to_datetime(trades_from_db['trade_date']).dt.strftime('%Y-%m-%d %H:%M')
        trades_from_db['price'] = trades_from_db['price'].apply(lambda x: f"₺{x:,.2f}")
        trades_from_db['amount'] = trades_from_db['amount'].apply(lambda x: f"{x:.6f}")
        trades_from_db['total_try'] = trades_from_db['total_try'].apply(lambda x: f"₺{x:,.2f}")
        trades_from_db['holdings'] = trades_from_db['holdings'].apply(lambda x: f"{x:.6f}")
        trades_from_db['cash'] = trades_from_db['cash'].apply(lambda x: f"₺{x:,.2f}")
        
        display_cols = ['trade_date', 'trade_type', 'price', 'amount', 'total_try', 'holdings', 'cash']
        st.dataframe(
            trades_from_db[display_cols],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info(f"ℹ️ {pair} için henüz trade yapılmadı.")


# ============================================================================
# PERFORMANCE COMPARISON
# ============================================================================
st.divider()
st.subheader("📊 Strateji Karşılaştırması")

comparison_cols = st.columns(len(crypto_pairs))

for idx, pair in enumerate(crypto_pairs):
    if pair not in crypto_data:
        continue
    
    df = crypto_data[pair]
    portfolio = portfolios[pair]
    
    with comparison_cols[idx]:
        st.markdown(f"#### {pair}")
        
        final_price = df.iloc[-1]['Close']
        start_price = df.iloc[0]['Close']
        final_portfolio_value = portfolio.calculate_portfolio_value(final_price)
        total_return = ((final_portfolio_value - initial_capital) / initial_capital) * 100
        
        # Buy & Hold
        initial_amount = initial_capital / start_price
        buy_hold_value = initial_amount * final_price
        buy_hold_return = ((buy_hold_value - initial_capital) / initial_capital) * 100
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🤖 Golden Cross**")
            st.markdown(f"₺{final_portfolio_value:,.0f}")
            color = 'green' if total_return > 0 else 'red'
            st.markdown(f"<p style='color: {color}; font-size: 16px; font-weight: bold;'>{total_return:+.2f}%</p>", 
                       unsafe_allow_html=True)
        
        with col2:
            st.markdown("**📈 Buy & Hold**")
            st.markdown(f"₺{buy_hold_value:,.0f}")
            color = 'green' if buy_hold_return > 0 else 'red'
            st.markdown(f"<p style='color: {color}; font-size: 16px; font-weight: bold;'>{buy_hold_return:+.2f}%</p>", 
                       unsafe_allow_html=True)
        
        difference = total_return - buy_hold_return
        st.markdown(f"**Fark:** {difference:+.2f}%")

# ============================================================================
# FOOTER
# ============================================================================
st.divider()
st.markdown("""
<div style='text-align: center; color: #888; font-size: 12px; margin-top: 20px;'>
    <p>Crypto Golden Cross Trading Bot | Binance API ile Doğru Fiyatlar</p>
    <p>⚠️ Bu simulator eğitim amaçlıdır. Gerçek yatırım kararları için finansal danışmanla görüşün.</p>
    <p>Veriler her 5 dakikada bir güncellenir. USDT fiyatları Binance'den alınıp TRY'ye çevrilmektedir.</p>
</div>
""", unsafe_allow_html=True)
