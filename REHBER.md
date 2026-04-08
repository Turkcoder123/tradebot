# 📘 EURUSD / XAUUSD Scalp-Trend Hibrit Strateji Rehberi

## İçindekiler

1. [Genel Bakış](#genel-bakış)
2. [3 Strateji Versiyonu](#3-strateji-versiyonu)
3. [Giriş Kuralları](#giriş-kuralları)
4. [Risk Yönetimi](#risk-yönetimi)
5. [Slippage ve Spread Loglama](#slippage-ve-spread-loglama)
6. [Backtest (1 Ay)](#backtest-1-ay)
7. [Live Test (Demo Hesap)](#live-test-demo-hesap)
8. [Dosya Yapısı](#dosya-yapısı)
9. [Kurulum ve Çalıştırma](#kurulum-ve-çalıştırma)
10. [Sık Sorulan Sorular](#sık-sorulan-sorular)

---

## Genel Bakış

Bu sistem, EURUSD ve XAUUSD paritelerini **aynı anda** takip eden bir
**scalp-trend hibrit** stratejidir. Temel özellikler:

| Özellik | Değer |
|---------|-------|
| **Semboller** | EURUSD + XAUUSD |
| **Lot** | 0.05 (sabit) |
| **Başlangıç bakiye** | $50 (demo) |
| **Hedef işlem sıklığı** | Günde 10-20 (gerektiğinde, zorunlu değil) |
| **Backtest süresi** | 1 ay |
| **Live test süresi** | 1 gün (24 saat) |
| **Strateji versiyonu** | 3 farklı parametre seti |
| **Loglama** | Slippage + spread dahil detaylı |

### Tasarım İlkeleri

- **Tek birleşik sistem:** Pullback + Momentum + BB Bounce giriş tipleri tek çatıda
- **3 parametre versiyonu:** Agresif / Dengeli / Muhafazakâr
- **Overfitting önleme:** Standart ders kitabı parametreleri
- **Gerçekçi maliyet:** Slippage ve spread simülasyonu
- **Her iki sembol:** EURUSD (pip=0.0001) ve XAUUSD (pip=0.01)

---

## 3 Strateji Versiyonu

### V1 – Agresif

| Parametre | Değer |
|-----------|-------|
| EMA | 20 / 50 |
| MACD | 8 / 17 / 9 |
| SL | 1.0 × ATR |
| TP | 1.5 × ATR |
| Seans | 07:00-19:00 UTC |
| BB Bounce | ✅ Açık |
| EMA tolerans | 0.2% |

**Özellik:** Daha fazla işlem, daha kısa sürede, daha dar SL/TP.

### V2 – Dengeli

| Parametre | Değer |
|-----------|-------|
| EMA | 50 / 200 |
| MACD | 12 / 26 / 9 |
| SL | 1.5 × ATR |
| TP | 3.0 × ATR |
| Seans | 08:00-17:00 UTC |
| BB Bounce | ✅ Açık |
| EMA tolerans | 0.1% |

**Özellik:** Standart parametreler, 1:2 risk-ödül.

### V3 – Muhafazakâr

| Parametre | Değer |
|-----------|-------|
| EMA | 50 / 200 |
| MACD | 12 / 26 / 9 |
| SL | 2.0 × ATR |
| TP | 4.0 × ATR |
| Seans | 09:00-16:00 UTC |
| BB Bounce | ❌ Kapalı |
| EMA tolerans | 0.05% |

**Özellik:** Az ama kaliteli işlem, geniş SL/TP.

---

## Giriş Kuralları

3 giriş tipi tüm versiyonlarda birleşik çalışır:

### Tip 1 – Trend Pullback (Ana Giriş)

```
Trend yönü belirlenir (EMA-hızlı > EMA-yavaş = yükseliş)
  → Fiyat EMA-hızlı'ya geri çekilir
  → Mum EMA-hızlı'nın trend tarafında kapanır
  → MACD histogram trend yönünde
  → İşlem açılır
```

### Tip 2 – Momentum (MACD Zero-Cross)

```
Trend yönü belirlenir
  → MACD histogram sıfır çizgisini trend yönünde geçer
  → Fiyat EMA-hızlı'nın trend tarafında
  → İşlem açılır
```

### Tip 3 – BB Bounce (Scalp Girişi)

```
Trend yönü belirlenir
  → Fiyat Bollinger Band'ın dış bandına değer
  → Band içine geri döner (bounce)
  → RSI kontrolü (aşırı alım/satım filtreleme)
  → İşlem açılır
```

> ⚠️ **Not:** BB Bounce V3'te devre dışıdır. V1 ve V2'de aktiftir.

---

## Risk Yönetimi

| Parametre | Değer |
|-----------|-------|
| **Lot** | 0.05 sabit |
| **Bakiye** | $50 demo |
| **SL/TP** | ATR tabanlı (versiyona göre) |
| **Tek pozisyon** | Sembol + versiyon başına 1 |
| **Margin kontrolü** | Bakiye < $5 ise yeni işlem açılmaz |

### Pip Değerleri (0.05 lot)

| Sembol | Pip boyutu | Pip değeri |
|--------|-----------|------------|
| EURUSD | 0.0001 | $0.50 |
| XAUUSD | 0.01 | $0.05 |

---

## Slippage ve Spread Loglama

Her işlem için kaydedilen detaylar:

| Alan | Açıklama |
|------|----------|
| `signal_price` | Stratejinin hesapladığı sinyal fiyatı |
| `entry_price` | Spread + slippage eklendikten sonraki gerçek giriş |
| `spread_pips` | Girişteki spread (pip) |
| `spread_usd` | Girişteki spread ($) |
| `slippage_entry_pips` | Giriş kayması (pip) |
| `slippage_entry_usd` | Giriş kayması ($) |
| `slippage_exit_pips` | Çıkış kayması (pip, sadece SL'de) |
| `slippage_exit_usd` | Çıkış kayması ($) |
| `total_cost_usd` | Toplam maliyet (spread + slippage) |
| `gross_pnl_usd` | Brüt kar/zarar |
| `net_pnl_usd` | Net kar/zarar (brüt - maliyet) |
| `balance_after` | İşlem sonrası bakiye |
| `duration_bars` | İşlem süresi (bar sayısı) |

### Backtest'te
Slippage, broker-gerçekçi sınırlar içinde rastgele simüle edilir:
- EURUSD: 0-2 pip
- XAUUSD: 0-10 pip

### Live'da
Gerçek slippage kaydedilir: talep edilen fiyat vs doldurma fiyatı.

---

## Backtest (1 Ay)

### Çalıştırma

```bash
# Tüm versiyonlar + tüm semboller, son 30 gün
python unified_backtest.py

# Sadece V1 (Agresif)
python unified_backtest.py --version V1

# Son 60 gün
python unified_backtest.py --days 60
```

### Çıktılar

```
results/
├── backtest_V1_Agresif_EURUSD_<timestamp>.csv
├── backtest_V2_Dengeli_EURUSD_<timestamp>.csv
├── backtest_V3_Muhafazakar_EURUSD_<timestamp>.csv
├── backtest_V1_Agresif_XAUUSD_<timestamp>.csv
├── ... (her versiyon × sembol için)
├── backtest_all_trades_<timestamp>.csv
└── backtest_summary_<timestamp>.json
```

Ekran çıktısında karşılaştırma tablosu gösterilir:

```
  KARŞILAŞTIRMA TABLOSU
  Versiyon           Sembol    İşlem  Kazanma   Net pip     Net $   Maliyet   DD     PF    Bakiye
  V1_Agresif         EURUSD       15    33.3%    -115.1    -57.56     21.98  74.58  0.52     -7.57
  V2_Dengeli         EURUSD        7    14.3%     -96.4    -48.19     10.00  85.41  0.44      1.81
  V3_Muhafazakar     EURUSD        5    20.0%    -116.3    -58.17      7.72  64.53  0.34     -8.17
```

---

## Live Test (Demo Hesap)

### Ön Koşullar

- MetaTrader 5 terminali açık ve demo hesaba giriş yapılmış
- Python bağımlılıkları kurulu
- EURUSD ve XAUUSD sembolleri aktif

### Çalıştırma

```bash
# Varsayılan: V2, 24 saat, hem EURUSD hem XAUUSD
python unified_live.py

# V1 (Agresif), 8 saat
python unified_live.py --version V1 --hours 8

# V3 (Muhafazakâr), 24 saat
python unified_live.py --version V3
```

### Bot Davranışı

1. MT5 terminaline bağlanır
2. Her saat başı (H1 bar kapanışı) sinyal kontrol eder
3. EURUSD ve XAUUSD'yi aynı anda takip eder
4. Sinyal varsa 0.05 lot ile market emri gönderir
5. Gerçek slippage ve spread kaydedilir
6. Belirtilen süre sonunda otomatik durur
7. Tüm sonuçlar `results/` ve `logs/` klasörlerine kaydedilir

### Çıktılar

```
results/
├── live_trades_V2_Dengeli_<timestamp>.csv
└── live_summary_V2_Dengeli_<timestamp>.json

logs/
└── live_V2_<timestamp>.log
```

### Log Formatı

Her işlem için şu detaylar loglanır:
```
📈 Sinyal: EURUSD LONG V2_Dengeli  fiyat=1.15632  SL=1.15132  TP=1.16632
   ATR=0.00333  spread=1.5 pip  tip=pullback
✅ BUY EURUSD 0.05 lot @ 1.15635 (talep=1.15632, kayma=0.3 pip/$0.15,
   spread=1.5 pip/$0.75)  SL=1.15132  TP=1.16632  ticket=12345
🔴 Kapandı: EURUSD LONG V2_Dengeli  brüt=4.50  maliyet=0.90  net=3.60  bakiye=53.60
```

---

## Dosya Yapısı

```
tradebot/
├── data/
│   ├── EURUSD_6m.csv              # 6 aylık H1 fiyat verisi
│   └── XAUUSD_6m.csv              # (fetch_prices.py ile çekilir)
├── results/                       # Backtest ve live sonuçları
│   ├── backtest_*.csv             # İşlem detayları
│   ├── backtest_summary_*.json    # Özet karşılaştırma
│   ├── live_trades_*.csv          # Canlı işlem detayları
│   └── live_summary_*.json        # Canlı test özeti
├── logs/
│   └── live_*.log                 # Bot logları
├── tests/
│   ├── conftest.py
│   ├── test_fetch_prices.py
│   ├── test_strategy.py
│   ├── test_backtest.py
│   └── test_unified.py            # Birleşik sistem testleri
├── strategy.py                    # Strateji + StrategyConfig + 3 preset
├── backtest.py                    # Eski backtest (geriye uyumlu)
├── unified_backtest.py            # Birleşik backtest (3 versiyon × 2 sembol)
├── unified_live.py                # Birleşik canlı test botu
├── mt5_bot.py                     # Eski live bot (geriye uyumlu)
├── fetch_prices.py                # MT5'ten veri çekme
├── requirements.txt
└── REHBER.md                      # Bu dosya
```

---

## Kurulum ve Çalıştırma

### 1. Bağımlılıkları Kur

```bash
pip install -r requirements.txt
```

### 2. Veri Çek (İlk Kez)

```bash
python fetch_prices.py EURUSD XAUUSD
```

### 3. Backtest Çalıştır

```bash
# Tüm versiyonlar, son 30 gün
python unified_backtest.py

# Özel süre
python unified_backtest.py --days 60
```

### 4. Sonuçları İncele

`results/` klasöründeki CSV ve JSON dosyalarını kontrol edin.

### 5. Live Test Başlat

```bash
# Demo hesap ile
python unified_live.py --version V2 --hours 24
```

### 6. Testleri Çalıştır

```bash
python -m pytest tests/ -v
```

---

## Sık Sorulan Sorular

### $50 ile 0.05 lot çok riskli değil mi?

Evet, bu **demo test** amaçlıdır. Gerçek hesapta $50 ile 0.05 lot
kullanmak tavsiye edilmez. Demo hesapta stratejiyi doğrulamak için
uygundur.

### Neden 3 farklı versiyon?

Her trader farklı risk toleransına sahiptir:
- **V1:** Sık işlem, kısa SL/TP, scalp ağırlıklı
- **V2:** Dengeli yaklaşım, standart parametreler
- **V3:** Az işlem, geniş SL/TP, trend ağırlıklı

### XAUUSD verisi yoksa ne olur?

Backtest XAUUSD'yi atlayıp sadece EURUSD ile devam eder.
Veriyi çekmek için: `python fetch_prices.py XAUUSD`

### Live test sırasında bot çökerse?

Açık pozisyonlar MT5 sunucusunda SL/TP ile korunur.
O ana kadarki sonuçlar otomatik kaydedilir.

### Slippage neden sadece SL'de var, TP'de yok?

TP (take-profit) limit emir olarak çalışır → slippage yok.
SL (stop-loss) stop emir olarak çalışır → piyasa koşullarına göre kayma olabilir.

---

## Sorumluluk Reddi

Bu sistem **eğitim ve demo test amaçlıdır**. Forex/CFD piyasasında işlem
yapmak yüksek risk taşır. Geçmiş performans gelecekteki sonuçları garanti
etmez. Gerçek para ile işlem yapmadan önce:
- Demo hesapta yeterli test yapın
- Risk yönetimi kurallarını uygulayın
- Kaybetmeyi göze alabileceğiniz miktarla işlem yapın
