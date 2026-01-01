#!/bin/bash

# Crypto Trading Bot - Başlangıç Scripti

echo "🚀 Crypto Trading Bot başlatılıyor..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# İçinde olduğumuz dizini kontrol et
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Virtual environment kontrol et
if [ ! -d "venv" ]; then
    echo "⚠️  Virtual environment bulunamadı!"
    echo "📦 Virtual environment oluşturuluyor..."
    python3 -m venv venv
    
    echo "📥 Gerekli paketler yükleniyor..."
    source venv/bin/activate
    pip install -q pandas numpy matplotlib streamlit requests
    echo "✅ Paketler yüklendi"
else
    echo "✅ Virtual environment bulundu"
fi

# Virtual environment'ı aktifleştir
echo "🔌 Virtual environment aktifleştiriliyor..."
source venv/bin/activate

# Streamlit uygulamasını başlat
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✨ Uygulama başlatılıyor..."
echo ""
echo "📊 Dashboard URL'si: http://localhost:8501"
echo "💡 Tarayıcıda açmak için yukarıdaki linke tıklayın"
echo ""
echo "⏹️  Uygulamayı durdurmak için: Ctrl+C"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Streamlit çalıştır
streamlit run bitcoin_ui_realtime.py
