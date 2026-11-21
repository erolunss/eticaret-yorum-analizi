# ⚡ Sentify AI | Türkçe E-Ticaret Duygu & Şikayet İstihbarat Platformu

[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model%20Hub-yellow)](https://huggingface.co/erolunss/berturk-eticaret-duygu-analizi)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-ff4b4b.svg)](https://streamlit.io/)

**Sentify AI**, Türk e-ticaret ekosistemine (Trendyol, Hepsiburada, Amazon TR) özel olarak optimize edilmiş, müşteri yorumlarını derin öğrenme ile analiz eden yeni nesil bir NLP ve Müşteri Deneyimi platformudur.

---

## 🚀 Özellikler

- **BERTurk Fine-Tuned Çekirdek:** `dbmdz/bert-base-turkish-cased` tabanlı 110M parametreli model, 10.000+ dengeli e-ticaret yorumuyla (`TRSAv1`) fine-tune edilmiştir.
- **Departman Bazlı Kök Neden Analizi:** Olumsuz yorumlar sadece tespit edilmekle kalmaz; Kargo/Lojistik, Ürün Kalitesi, Beden/Kalıp, İade/Müşteri Hizmetleri ve Fiyat kategorilerine otomatik atanır.
- **Toplu Veri İstihbaratı:** Excel (`.xlsx`) ve `.csv` formatında binlerce satırlık müşteri yorumunu saniyeler içinde işler.
- **İnteraktif Yönetici Dashboard'u:** Plotly Dark temalı interaktif duygu dağılımı (Donut) ve departman risk haritası (Bar chart).
- **Yönetici Raporu İndirme:** Analiz sonuçlarını Excel/CSV olarak anında indirme imkanı.
- **Tekil Canlı Test İstasyonu:** Gauge (hız göstergesi) ve olasılık barlarıyla gerçek zamanlı yorum test laboratuvarı.

---

## 🧠 Model Mimarisi & Başarım

| Metrik | Değer |
|---|---|
| **Temel Model** | `dbmdz/bert-base-turkish-cased` |
| **Eğitim Veri Seti** | `maydogan/TRSAv1` (10.000 Yorum - Dengeli) |
| **Test Doğruluğu (Accuracy)** | %98.4 |
| **Hesaplama Altyapısı** | NVIDIA GeForce RTX 4050 GPU (FP16 Hızlandırmalı) |
| **Model Deposu** | [erolunss/berturk-eticaret-duygu-analizi](https://huggingface.co/erolunss/berturk-eticaret-duygu-analizi) |

---

## 💻 Yerel Kurulum (Local Setup)

Projeyi bilgisayarınızda çalıştırmak için:

```bash
# 1. Repoyu klonlayın
git clone https://github.com/erolunss/eticaret-yorum-analizi.git
cd eticaret-yorum-analizi

# 2. Sanal ortamı oluşturun ve aktif edin
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows için

# 3. Bağımlılıkları yükleyin
pip install -r requirements.txt

# 4. Uygulamayı başlatın
streamlit run streamlit_app.py
```

---

## 👨‍💻 Geliştirici

- **Geliştirici:** Erol Ünsal ([@erolunss](https://github.com/erolunss))
- **HuggingFace:** [@erolunss](https://huggingface.co/erolunss)
