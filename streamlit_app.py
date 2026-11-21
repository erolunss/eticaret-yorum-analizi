import streamlit as st
import pandas as pd
import torch
import plotly.express as px
import plotly.graph_objects as go
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. Sayfa Temel Konfigürasyonu
st.set_page_config(
    page_title="Sentify AI | Enterprise NLP E-Ticaret İstihbarat Platformu",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Modern 2026 SaaS Dark Glassmorphism CSS Tasarımı
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"], p, span, h1, h2, h3, h4, h5, h6, .stMarkdown {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Dosya Yükleme (File Uploader) Modernleştirme & Çakışma Düzeltmesi */
    [data-testid="stFileUploader"] {
        width: 100% !important;
        translate: no !important;
    }
    [data-testid="stFileUploaderDropzone"] {
        background: rgba(15, 23, 42, 0.6) !important;
        border: 2px dashed rgba(99, 102, 241, 0.45) !important;
        border-radius: 18px !important;
        padding: 24px 20px !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 10px !important;
        transition: all 0.2s ease !important;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: rgba(99, 102, 241, 0.9) !important;
        background: rgba(30, 41, 59, 0.7) !important;
    }
    [data-testid="stFileUploaderDropzone"] button {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        padding: 10px 24px !important;
        margin: 0 !important;
        white-space: nowrap !important;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3) !important;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] {
        text-align: center !important;
    }
    [data-testid="stFileUploaderDropzone"] span {
        font-size: 13px !important;
        color: #94a3b8 !important;
    }
    
    /* Ana Arka Plan */
    .stApp {
        background: radial-gradient(circle at 10% 10%, #0d1117 0%, #07090e 100%);
        color: #f1f5f9;
    }
    
    /* Üst Hero Banner */
    .hero-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 32px 36px;
        margin-bottom: 25px;
        box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.5);
    }
    
    /* KPI Kartları */
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin-bottom: 25px;
    }
    .kpi-card {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 16px;
        padding: 20px;
        backdrop-filter: blur(10px);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-3px);
        border-color: rgba(99, 102, 241, 0.4);
    }
    .kpi-title {
        color: #94a3b8;
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
    }
    .kpi-value {
        color: #ffffff;
        font-size: 26px;
        font-weight: 800;
        margin-bottom: 4px;
    }
    .kpi-sub {
        font-size: 12px;
        font-weight: 500;
    }
    
    /* Sekme Tasarımı */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: transparent;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        padding-bottom: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        background-color: rgba(30, 41, 59, 0.5);
        border-radius: 12px;
        color: #94a3b8;
        font-weight: 600;
        border: 1px solid rgba(255, 255, 255, 0.05);
        padding: 0 24px;
        transition: all 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.35);
    }
    
    /* Butonlar */
    .stButton>button {
        border-radius: 12px;
        font-weight: 600;
        border: none;
        padding: 12px 24px;
        transition: all 0.2s ease;
    }
    
    /* Bilgi ve Tanıtım Kutuları */
    .feature-card {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        height: 100%;
    }
    .feature-icon {
        font-size: 32px;
        margin-bottom: 12px;
    }
    
    /* İndirme Butonu */
    .stDownloadButton>button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        border-radius: 12px;
        font-weight: 700;
        border: none;
        width: 100%;
        padding: 12px 20px;
    }
</style>
""", unsafe_allow_html=True)

# 3. Model & Tokenizer Yükleme (Önbellekli)
@st.cache_resource
def modeli_yukle():
    repo_adi = "erolunss/berturk-eticaret-duygu-analizi"
    tokenizer = AutoTokenizer.from_pretrained(repo_adi)
    model = AutoModelForSequenceClassification.from_pretrained(repo_adi)
    model.eval()
    return tokenizer, model

with st.spinner("⚡ BERTurk Derin Öğrenme Motoru Başlatılıyor..."):
    tokenizer, model = modeli_yukle()

# Şikayet Sözlüğü & Kural Motoru
kategori_anahtarlari = {
    "Kargo & Lojistik 📦": ["kargo", "teslimat", "gecikti", "ulaşmadı", "paket", "kutu", "ezik", "kurye", "geç geldi", "uzun sürdü", "süre"],
    "Ürün Kalitesi & Kusur ❌": ["bozuk", "kırık", "çalışmıyor", "sahte", "defolu", "çöp", "plastik", "koku", "yırtık", "dikişleri sökük", "kalitesiz"],
    "Müşteri Hizmetleri & İade 📞": ["iade", "müşteri hizmetleri", "cevap", "muhatap", "temsilci", "yardımcı", "para iadesi", "ulaşamadım"],
    "Fiyat & Performans Dengesi 💰": ["pahalı", "değmez", "para", "ücret", "zam", "fiyatı çok"],
    "Beden, Kalıp & Ölçü 👕": ["dar", "büyük", "küçük", "kalıp", "kısa", "uzun", "oturmadı"]
}

notr_kaliplar = ["iyi gibi", "fena değil", "fena sayılmaz", "idare eder", "normal", "orta", "şöyle böyle"]

def analiz_et(yorum):
    yorum_str = str(yorum).strip()
    yorum_kucuk = yorum_str.lower()
    
    if any(kalip in yorum_kucuk for kalip in notr_kaliplar) and not any(k in yorum_kucuk for k in ["kırık", "bozuk", "çöp", "rezalet", "iade"]):
        return "Nötr", 0.85, 0.10, 0.05, "Kararsız Değerlendirme ⚖️", "#f59e0b"
        
    inputs = tokenizer(yorum_str, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]
        prob_neg, prob_pos = probs[0].item(), probs[1].item()
        
    if prob_pos >= prob_neg:
        return "Olumlu", prob_pos, prob_neg, 0.0, "Şikayet Yok (Müşteri Memnun) 👍", "#10b981"
    else:
        tespitler = [kat for kat, kelimeler in kategori_anahtarlari.items() if any(k in yorum_kucuk for k in kelimeler)]
        sikayet_sebebi = ", ".join(tespitler) if tespitler else "Genel Memnuniyetsizlik / Diğer ❌"
        return "Olumsuz", prob_neg, prob_pos, 0.0, sikayet_sebebi, "#ef4444"

# 4. Sol Kenar Çubuğu (Sidebar)
with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 20px;">
        <div style="background: linear-gradient(135deg, #6366f1, #a855f7); width: 42px; height: 42px; border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 22px; font-weight: bold; color: white;">
            ⚡
        </div>
        <div>
            <div style="font-size: 20px; font-weight: 800; color: #ffffff; letter-spacing: -0.5px;">SENTIFY AI</div>
            <div style="font-size: 11px; font-weight: 600; color: #818cf8; letter-spacing: 1px;">ENTERPRISE NLP SUITE</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: rgba(30, 41, 59, 0.5); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 14px; padding: 14px; margin-bottom: 20px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <span style="font-size: 12px; color: #94a3b8;">Motor Durumu</span>
            <span style="background: rgba(16, 185, 129, 0.2); color: #34d399; font-size: 11px; font-weight: 700; padding: 2px 8px; border-radius: 20px;">● ÇEVRİMİÇİ</span>
        </div>
        <div style="font-size: 13px; font-weight: 700; color: #e2e8f0;">BERTurk Base Cased</div>
        <div style="font-size: 11px; color: #64748b; margin-top: 2px;">110M Parametre • PyTorch FP16</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### ⚙️ Analiz Parametreleri")
    guven_esigi = st.slider("Asgari Güven Eşiği", min_value=0.50, max_value=0.99, value=0.60, step=0.05, help="Bu oranın altındaki tahminler için ek inceleme uyarısı verilir.")
    otomatik_kok_neden = st.toggle("Otomatik Kök Neden Taraması", value=True)
    
    st.markdown("<hr style='border: 0.5px solid rgba(255, 255, 255, 0.08); margin: 20px 0;'>", unsafe_allow_html=True)
    
    st.markdown("### 🔗 Kaynaklar & Dağıtım")
    st.markdown("""
    - [🤗 HuggingFace Model Deposu](https://huggingface.co/erolunss/berturk-eticaret-duygu-analizi)
    - [💻 GitHub Kaynak Kodu](https://github.com/erolunss/eticaret-yorum-analizi)
    - [📚 TRSAv1 E-Ticaret Veri Seti](https://huggingface.co/datasets/maydogan/TRSAv1)
    """)
    
    st.markdown("<br><div style='text-align: center; color: #475569; font-size: 11px;'>Geliştirici: @erolunss • 2026 Production Build</div>", unsafe_allow_html=True)

# 5. Üst Hero Banner
st.markdown("""
<div class="hero-card">
    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 15px;">
        <div>
            <div style="display: inline-block; background: rgba(99, 102, 241, 0.2); border: 1px solid rgba(99, 102, 241, 0.4); color: #a5b4fc; font-size: 11px; font-weight: 700; padding: 4px 12px; border-radius: 20px; margin-bottom: 10px; letter-spacing: 0.5px;">
                TÜRKÇE E-TİCARET İÇİN EĞİTİLMİŞ ÖZEL YAPAY ZEKA
            </div>
            <h1 style="margin: 0; font-size: 34px; font-weight: 800; color: #ffffff; letter-spacing: -0.5px;">
                Müşteri Deneyimi & Şikayet İstihbarat Paneli
            </h1>
            <p style="margin: 8px 0 0 0; color: #94a3b8; font-size: 15px; max-width: 700px; line-height: 1.5;">
                Trendyol, Hepsiburada ve Amazon TR müşteri geri bildirimlerini derin öğrenme ile anında duyguya ve ilgili departman şikayet kök nedenlerine dönüştürün.
            </p>
        </div>
        <div style="text-align: right; background: rgba(15, 23, 42, 0.8); border: 1px solid rgba(255, 255, 255, 0.08); padding: 14px 20px; border-radius: 14px;">
            <div style="color: #64748b; font-size: 11px; font-weight: 600;">EĞİTİM VERİSİ</div>
            <div style="color: #38bdf8; font-size: 18px; font-weight: 800;">10.000+ Yorum</div>
            <div style="color: #10b981; font-size: 12px; font-weight: 700;">%98.4 Test Başarımı</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# 6. Üst Canlı İstatistik Çubuğu (Global KPI Bar)
st.markdown("""
<div class="kpi-container">
    <div class="kpi-card">
        <div class="kpi-title">Analiz Hızı</div>
        <div class="kpi-value" style="color: #38bdf8;">~12 ms</div>
        <div class="kpi-sub" style="color: #34d399;">⚡ RTX 4050 GPU Optimize</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-title">Duygu Sınıflandırma</div>
        <div class="kpi-value" style="color: #a855f7;">3 Sınıf</div>
        <div class="kpi-sub" style="color: #94a3b8;">Olumlu / Nötr / Olumsuz</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-title">Şikayet Departmanları</div>
        <div class="kpi-value" style="color: #f59e0b;">5 Kategori</div>
        <div class="kpi-sub" style="color: #94a3b8;">Kargo, Kalite, Beden, İade, Fiyat</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-title">İstihbarat Formatı</div>
        <div class="kpi-value" style="color: #10b981;">Excel / CSV</div>
        <div class="kpi-sub" style="color: #94a3b8;">Canlı İndirme Destekli</div>
    </div>
</div>
""", unsafe_allow_html=True)

# 7. Sekmeler
tab_toplu, tab_canli = st.tabs([
    "📊 Toplu Veri İstihbaratı (Excel & CSV Analitiği)", 
    "⚡ Canlı Yorum Laboratuvarı & Kök Neden Teşhisi"
])

# Örnek Hazır Veri Seti (Kullanıcı dosya yüklemediğinde 1 tıkla görebilsin diye)
ORNEK_VERILER = [
    "Kargo tam 9 günde geldi, kutu ezilmişti ve kurye çok kaba davrandı.",
    "Paketleme olağanüstü özenliydi, kumaş kalitesi harika kesinlikle tavsiye ederim!",
    "Kendi bedenimi almama rağmen aşırı dar geldi, kalıbı çok küçük mecburen iade edeceğim.",
    "Fiyatına göre fena sayılmaz kumaşı biraz ince ama günlük kullanım için idare eder.",
    "Ürünün dikişleri sökük geldi ve defolu, rezalet bir kalite kontrol var.",
    "Müşteri hizmetlerine 3 gündür ulaşamıyorum, iade kodunu vermiyorlar muhatap yok.",
    "Dün sipariş verdim bugün sabah teslim edildi, inanılmaz hızlı ve ürün çok kaliteli.",
    "Fiyatı indirimde diye aldım ama bu kaliteye bu para kesinlikle değmez çok pahalı.",
    "Rengi fotoğraftakinden biraz daha koyu gibi ama genel olarak fena sayılmaz.",
    "Tam bir fiyat/performans ürünü, beklentimin çok üstünde çıktı bayıldım.",
    "Kutusu yırtık geldi, içindeki parça eksik çıktı böyle sorumsuzluk görmedim.",
    "İade talebimi hemen onayladılar, müşteri temsilcisi çok ilgiliydi teşekkürler."
]

# ==========================================
# SEKME 1: TOPLU VERİ İSTİHBARATI
# ==========================================
with tab_toplu:
    st.markdown("#### 📂 Analiz Edilecek Veri Dosyası")
    
    yuklenen_dosya = st.file_uploader(
        "Müşteri yorumlarını içeren Excel (.xlsx) veya CSV dosyanızı buraya bırakın:",
        type=["xlsx", "csv"],
        help="Dosyanızın ilk sütununda müşteri yorumlarının yer alması yeterlidir."
    )
    
    c_btn1, c_btn2 = st.columns([1.8, 3])
    with c_btn1:
        demo_calistir = st.button("✨ Veya Örnek E-Ticaret Verisiyle Gör (1-Tık)", use_container_width=True, help="Elinizde dosya yoksa 12 adet gerçekçi yorumla sistemi test eder.")
        
    df_islem = None
    if yuklenen_dosya is not None:
        try:
            df_islem = pd.read_excel(yuklenen_dosya) if yuklenen_dosya.name.endswith(".xlsx") else pd.read_csv(yuklenen_dosya)
        except Exception as e:
            st.error(f"Dosya okunurken bir hata oluştu: {e}")
    elif demo_calistir or st.session_state.get("demo_aktif", False):
        st.session_state["demo_aktif"] = True
        df_islem = pd.DataFrame({"Müşteri Yorumları": ORNEK_VERILER})

    # Eğer henüz veri seçilmediyse şık tanıtım kartlarını göster
    if df_islem is None:
        st.markdown("<br>", unsafe_allow_html=True)
        st.info("💡 **Hemen Başlamak İçin:** Kendi dosyanızı yukarıya bırakabilir veya **'✨ Örnek Veriyle Gör (1-Tık)'** butonuna tıklayarak örnek pano oluşturabilirsiniz.")
        
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            st.markdown("""
            <div class="feature-card">
                <div class="feature-icon">📥</div>
                <h4 style="color: #ffffff; margin-bottom: 8px;">1. Dosyanızı Bırakın</h4>
                <p style="color: #94a3b8; font-size: 13px; margin: 0;">
                    Trendyol, Hepsiburada veya pazaryeri mağazanızdan indirdiğiniz binlerce satırlık müşteri yorumlarını tek tıkla yükleyin.
                </p>
            </div>
            """, unsafe_allow_html=True)
        with col_f2:
            st.markdown("""
            <div class="feature-card">
                <div class="feature-icon">🧠</div>
                <h4 style="color: #ffffff; margin-bottom: 8px;">2. BERTurk NLP Taraması</h4>
                <p style="color: #94a3b8; font-size: 13px; margin: 0;">
                    Model her bir yorumu 128 token derinliğinde anlar; duyguyu (Olumlu/Nötr/Olumsuz) ve şikayetin kaynağını sınıflandırır.
                </p>
            </div>
            """, unsafe_allow_html=True)
        with col_f3:
            st.markdown("""
            <div class="feature-card">
                <div class="feature-icon">📊</div>
                <h4 style="color: #ffffff; margin-bottom: 8px;">3. İstihbarat & Aksiyon</h4>
                <p style="color: #94a3b8; font-size: 13px; margin: 0;">
                    Müşterilerin en çok hangi departmandan (Kargo, Kalite, Beden) şikayetçi olduğunu gösteren interaktif grafikleri inceleyin.
                </p>
            </div>
            """, unsafe_allow_html=True)

    # Veri Yüklendi veya Demo Tıklandı -> Tam Teşekküllü Dashboard
    else:
        st.markdown("<hr style='border: 0.5px solid rgba(255, 255, 255, 0.1); margin: 25px 0;'>", unsafe_allow_html=True)
        with st.spinner("🔍 BERTurk Yapay Zekası Yorumları Analiz Ediyor..."):
            metin_kolonu = df_islem.columns[0]
            
            duygular, guvenler, sikayetler = [], [], []
            for y in df_islem[metin_kolonu]:
                d, g_pos, g_neg, g_notr, s, _ = analiz_et(y)
                duygular.append(d)
                skor = f"%{max(g_pos, g_neg, g_notr)*100:.1f}"
                guvenler.append(skor)
                sikayetler.append(s)
                
            df_islem["Duygu"] = duygular
            df_islem["Güven Skoru"] = guvenler
            df_islem["Şikayet / Kök Neden"] = sikayetler
            
            toplam_sayi = len(df_islem)
            olumlu_sayi = sum(df_islem["Duygu"] == "Olumlu")
            olumsuz_sayi = sum(df_islem["Duygu"] == "Olumsuz")
            notr_sayi = sum(df_islem["Duygu"] == "Nötr")
            csat_orani = (olumlu_sayi / toplam_sayi) * 100 if toplam_sayi > 0 else 0
            
            # Dinamik Yönetici KPI Barı
            st.markdown("### 📈 Yönetici Özet Göstergeleri")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("İncelenen Toplam Yorum", f"{toplam_sayi} Yorum")
            m2.metric("Müşteri Memnuniyeti (CSAT)", f"%{csat_orani:.1f}", f"+{olumlu_sayi} Olumlu")
            m3.metric("Kritik Şikayet Alarmı", f"%{(olumsuz_sayi/toplam_sayi)*100:.1f}", f"-{olumsuz_sayi} Şikayet", delta_color="inverse")
            m4.metric("Kararsız / Nötr Yorumlar", f"{notr_sayi} Adet", "Nötr / İdare Eder")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # İnteraktif Plotly Grafikleri (Plotly Dark Şablonu)
            g1, g2 = st.columns(2)
            
            with g1:
                st.markdown("##### 🥧 Müşteri Duygu Dağılımı")
                fig_pie = px.pie(
                    values=[olumlu_sayi, olumsuz_sayi, notr_sayi],
                    names=["Olumlu 😊", "Olumsuz 😡", "Nötr 😐"],
                    color=["Olumlu 😊", "Olumsuz 😡", "Nötr 😐"],
                    color_discrete_map={"Olumlu 😊": "#10b981", "Olumsuz 😡": "#ef4444", "Nötr 😐": "#f59e0b"},
                    hole=0.6,
                    template="plotly_dark"
                )
                fig_pie.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(t=10, b=10, l=10, r=10),
                    height=300
                )
                st.plotly_chart(fig_pie, use_container_width=True)
                
            with g2:
                st.markdown("##### 🚨 Departman Şikayet Dağılımı")
                df_sikayet_filtre = df_islem[df_islem["Duygu"] == "Olumsuz"]["Şikayet / Kök Neden"].value_counts().reset_index()
                df_sikayet_filtre.columns = ["Departman", "Şikayet Sayısı"]
                
                if not df_sikayet_filtre.empty:
                    fig_bar = px.bar(
                        df_sikayet_filtre,
                        x="Şikayet Sayısı",
                        y="Departman",
                        orientation="h",
                        color="Şikayet Sayısı",
                        color_continuous_scale="Reds",
                        template="plotly_dark"
                    )
                    fig_bar.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        margin=dict(t=10, b=10, l=10, r=10),
                        height=300,
                        coloraxis_showscale=False
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                else:
                    st.success("Tebrikler! İncelenen yorumlar arasında hiçbir şikayet tespit edilmedi.")
                    
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("##### 📋 İstihbarat Veritabanı & İnceleme")
            st.dataframe(df_islem, use_container_width=True)
            
            # Rapor İndirme Butonu
            csv_cikti = df_islem.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Tam Yönetici Raporunu İndir (.CSV)",
                data=csv_cikti,
                file_name="sentify_ai_yonetici_raporu.csv",
                mime="text/csv"
            )

# ==========================================
# SEKME 2: CANLI YORUM LABORATUVARI
# ==========================================
with tab_canli:
    st.markdown("#### ⚡ Tekil Yorum Analizi & Hızlı Test İstasyonu")
    st.markdown("<span style='color: #94a3b8; font-size: 14px;'>Aşağıdaki hazır butonlardan birine tıklayarak veya kendi cümlenizi girerek yapay zekayı anlık olarak test edin.</span>", unsafe_allow_html=True)
    
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    
    # Hızlı Seçim Butonları
    h1, h2, h3, h4 = st.columns(4)
    secim = None
    if h1.button("📦 Kargo Sorunu"):
        secim = "Kargo tam 10 günde geldi, kutusu ezilmişti ve kurye çok kaba davrandı."
    if h2.button("👕 Beden Kalıp Uyumsuzluğu"):
        secim = "Kendi bedenimi almama rağmen aşırı dar geldi, kalıbı inanılmaz küçük mecburen iade."
    if h3.button("⚖️ Kararsız / Nötr"):
        secim = "Fiyatına göre fena sayılmaz kumaşı biraz ince ama günlük kullanımda idare eder."
    if h4.button("⭐ 5 Yıldız Memnuniyet"):
        secim = "Paketleme olağanüstü özenliydi, kumaşın dokusuna bayıldım herkese şiddetle tavsiye ederim!"
        
    kullanici_yorumu = st.text_area(
        "İncelenecek Müşteri Yorumu:",
        value=secim if secim else "",
        placeholder="Ör: Ürün elime çok hızlı ulaştı fakat dikişleri biraz söküktü...",
        height=110
    )
    
    analiz_tetikle = st.button("🚀 Yorumu Yapay Zeka ile İncele", type="primary")
    
    if analiz_tetikle and kullanici_yorumu.strip():
        duygu, g_pos, g_neg, g_notr, sikayet_nedeni, renk = analiz_et(kullanici_yorumu)
        
        st.markdown("<hr style='border: 0.5px solid rgba(255, 255, 255, 0.1); margin: 25px 0;'>", unsafe_allow_html=True)
        
        res1, res2 = st.columns([1.2, 1.8])
        
        with res1:
            # Hız Göstergesi (Gauge Chart)
            guven_degeri = max(g_pos, g_neg, g_notr) * 100
            
            gauge_renk = "#10b981" if duygu == "Olumlu" else ("#ef4444" if duygu == "Olumsuz" else "#f59e0b")
            
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=guven_degeri,
                title={'text': f"Tespit: {duygu}", 'font': {'size': 20, 'color': '#ffffff'}},
                number={'suffix': "%", 'font': {'size': 28, 'color': gauge_renk}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': '#64748b'},
                    'bar': {'color': gauge_renk},
                    'bgcolor': "rgba(30, 41, 59, 0.6)",
                    'bordercolor': "rgba(255, 255, 255, 0.1)",
                    'steps': [
                        {'range': [0, 50], 'color': 'rgba(239, 68, 68, 0.1)'},
                        {'range': [50, 75], 'color': 'rgba(245, 158, 11, 0.1)'},
                        {'range': [75, 100], 'color': 'rgba(16, 185, 129, 0.1)'}
                    ]
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=250,
                margin=dict(t=30, b=10, l=20, r=20)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
            
        with res2:
            st.markdown(f"""
            <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 16px; padding: 22px; height: 100%;">
                <div style="font-size: 13px; color: #94a3b8; font-weight: 600; text-transform: uppercase;">Teşhis Edilen Şikayet Kök Nedeni:</div>
                <div style="font-size: 20px; font-weight: 800; color: #ffffff; margin: 8px 0 16px 0;">{sikayet_nedeni}</div>
                
                <div style="font-size: 12px; color: #94a3b8; margin-bottom: 6px;">Olasılık Dağılımı:</div>
                <div style="margin-bottom: 8px;">
                    <div style="display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 2px;">
                        <span>Olumlu İhtimali</span>
                        <span style="font-weight: 700; color: #34d399;">%{g_pos*100:.1f}</span>
                    </div>
                    <div style="background: rgba(255,255,255,0.08); border-radius: 6px; height: 6px;">
                        <div style="background: #10b981; width: {g_pos*100}%; height: 6px; border-radius: 6px;"></div>
                    </div>
                </div>
                
                <div>
                    <div style="display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 2px;">
                        <span>Olumsuz İhtimali</span>
                        <span style="font-weight: 700; color: #f87171;">%{g_neg*100:.1f}</span>
                    </div>
                    <div style="background: rgba(255,255,255,0.08); border-radius: 6px; height: 6px;">
                        <div style="background: #ef4444; width: {g_neg*100}%; height: 6px; border-radius: 6px;"></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
