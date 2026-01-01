# 🚀 Crypto Trading Bot - Golden Cross Strategy

Gerçek zamanlı kripto para ve altın/gümüş fiyatlarını takip eden, Golden Cross stratejisi ile otomatik trading yapan Python tabanlı bot.

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.30+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🎯 Özellikler

- 📊 **Gerçek Zamanlı Fiyat Takibi** - Binance API ile güncel veriler
- 💰 **Çoklu Asset Desteği** - BTC, ETH, IO, PAXG (Altın), SOL
- 🤖 **Otomatik Trading** - Golden Cross stratejisi ile akıllı alım-satım
- 💾 **SQLite Veritabanı** - Tüm trade'ler ve portföy durumu kalıcı olarak saklanır
- 📈 **İnteraktif Grafikler** - Fiyat, hareketli ortalamalar ve sinyaller
- ⚙️ **Özelleştirilebilir Ayarlar** - MA pencere değerleri, sermaye, refresh süresi
- 🇹🇷 **TRY Desteği** - Tüm fiyatlar Türk Lirası cinsinden

## 🛠️ Kurulum

### Gereksinimler

- Python 3.9+
- pip

### Hızlı Başlangıç

1. **Repoyu klonlayın:**
```bash
git clone https://github.com/KULLANICI_ADINIZ/crypto-trading-bot.git
cd crypto-trading-bot
```

2. **Başlatma script'ini çalıştırın:**
```bash
chmod +x run.sh
./run.sh
```

Script otomatik olarak:
- Virtual environment oluşturur
- Gerekli paketleri yükler
- Streamlit uygulamasını başlatır

3. **Tarayıcıda açın:**
```
http://localhost:8501
```

## 📚 Golden Cross Stratejisi Nedir?

**Golden Cross**, teknik analizde kullanılan güçlü bir trading sinyalidir:

- **BUY Sinyali:** Kısa vadeli MA (7-gün) uzun vadeli MA'yı (30-gün) yukarı keser
- **SELL Sinyali:** Kısa vadeli MA uzun vadeli MA'yı aşağı keser (Death Cross)

Bot bu sinyalleri otomatik olarak tespit eder ve trade yapar.

## 🎮 Kullanım

### Sidebar Ayarları

1. **Kripto Para Seçimi** - Takip etmek istediğiniz coin'leri seçin
2. **Başlangıç Sermayesi** - TRY cinsinden başlangıç sermayenizi belirleyin
3. **MA Pencere Değerleri** - Golden Cross için MA değerlerini özelleştirin
4. **Yenileme Süresi** - Verilerin ne sıklıkla güncelleneceğini ayarlayın

### Önemli Butonlar

- 💾 **Ayarları Kaydet** - Tüm ayarlarınız veritabanına kaydedilir
- 🔄 **Yeniden Simüle Et** - Cache'i temizler ve yeni veriler çeker
- 🗑️ **Veritabanını Sıfırla** - Tüm trade'leri ve ayarları sıfırlar

## 📊 Dashboard Bölümleri

1. **Otomatik Trading Durumu** - Hangi coin'de pozisyonda olduğunuzu gösterir
2. **Portföy Özeti** - Mevcut fiyat, portföy değeri, kar/zarar
3. **Fiyat Grafikleri** - Fiyat, MA7, MA30 ve buy/sell sinyalleri
4. **Portföy Değeri Takibi** - Portföy değerinin zaman içindeki değişimi
5. **Trade Ledgeri** - Tüm trade'lerin detaylı geçmişi (veritabanından)
6. **Strateji Karşılaştırması** - Golden Cross vs Buy & Hold

## 💾 Veritabanı Yapısı

Bot SQLite kullanır ve 3 tablo içerir:

- **trades** - Tüm alım-satım işlemleri
- **portfolio_state** - Mevcut portföy durumu
- **user_settings** - Kullanıcı ayarları

Dosya: `trading_bot.db`

## 🔧 Teknik Detaylar

### Kullanılan API'ler

- **Binance API** - USDT pair fiyatları
- **CoinGecko API** - USD/TRY kuru (fallback)

### Paketler

```txt
streamlit>=1.30.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
requests>=2.31.0
```

## ⚠️ Önemli Notlar

- Bu bot **eğitim amaçlıdır** ve gerçek para ile otomatik trading yapmaz
- Gerçek yatırım kararları için finansal danışmanla görüşün
- Geçmiş performans gelecekteki sonuçların garantisi değildir
- API rate limitlerine dikkat edin

## 🤝 Katkıda Bulunma

1. Fork edin
2. Feature branch oluşturun (`git checkout -b feature/AmazingFeature`)
3. Commit edin (`git commit -m 'Add some AmazingFeature'`)
4. Push edin (`git push origin feature/AmazingFeature`)
5. Pull Request açın

## 📝 Lisans

Bu proje MIT lisansı altında lisanslanmıştır.

## 📧 İletişim

Sorularınız için issue açabilirsiniz.

---

**⭐ Beğendiyseniz yıldız vermeyi unutmayın!**
