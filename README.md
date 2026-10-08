# Analisis-Sentimen-Ulasan-Aplikas-Jakarta-Kini-JAKI-
Sentiment analysis of 4,934 Google Play reviews of JAKI, Jakarta's citizen-services app: scraping, Indonesian text preprocessing, TF-IDF, Naive Bayes / SVM / Logistic Regression / Random Forest (~84% accuracy), IndoBERT label comparison, and a Gradio demo.

# Analisis Sentimen Ulasan Aplikasi JAKI (Jakarta Kini)

Analisis sentimen terhadap 4.934 ulasan Google Play Store aplikasi **JAKI (Jakarta Kini)**, aplikasi layanan warga milik Pemprov DKI Jakarta (`id.go.jakarta.smartcity.jaki`). Alur lengkap: scraping, EDA, preprocessing teks bahasa Indonesia, pelabelan, TF-IDF, perbandingan empat model klasifikasi, perbandingan dengan label IndoBERT, hingga demo web sederhana dengan Gradio.

> **Catatan penting:** label sentimen diambil dari **rating pengguna** (1-3 = negatif, 4-5 = positif), bukan dari penilaian isi teks. Jadi model memprediksi sentimen berbasis rating.

## 🌟 Fitur Utama

- **Scraping**: 4.934 ulasan dengan `google-play-scraper`, rentang 27 September 2019 - 7 September 2026. Kolom: `id`, `teks`, `rating`, `tanggal`, `sumber` (tanpa nama pengguna).
- **EDA**: distribusi rating, panjang ulasan, dan frekuensi teks identik.
- **Preprocessing bertahap** (setiap tahap disimpan sebagai kolom terpisah agar mudah dibandingkan sebelum-sesudah): case folding, cleaning (URL, angka, tanda baca, emoji), normalisasi huruf berulang, normalisasi kata slang, tokenizing (NLTK), stop word removal dan stemming (Sastrawi).
- **Pelabelan**: label final dari rating. Sebagai pembanding, ulasan juga dilabeli dengan `w11wo/indonesian-roberta-base-sentiment-classifier` (Hugging Face).
- **Pembagian data**: 80:20 dengan `GroupShuffleSplit` (`random_state=42`), dikelompokkan berdasarkan teks hasil stemming. Teks yang sama selalu berada di satu sisi, sehingga akurasi tidak menggelembung karena model "menghafal" ulasan pendek seperti "oke" atau "bagus".
- **Klasifikasi**: TF-IDF (maks. 5.000 fitur) dengan Naive Bayes, SVM linear, Logistic Regression, dan Random Forest.
- **Visualisasi**: WordCloud sentimen positif dan negatif, confusion matrix.
- **Demo**: antarmuka Gradio yang menerima ulasan, lalu menampilkan prediksi, teks setelah preprocessing, dan probabilitas.

## 📊 Hasil

Data latih 4.046 baris, data uji 852 baris (457 negatif, 395 positif).

| Model | Akurasi | Macro F1 |
|---|---|---|
| Logistic Regression | 84,74% | 0,845 |
| SVM (linear) | 84,39% | 0,842 |
| Naive Bayes | 83,45% | 0,832 |
| Random Forest | 83,22% | 0,830 |

Selisih antar model kecil (di bawah 1,6 poin), sedangkan data uji hanya 852 baris sehingga margin ketidakpastiannya sekitar ±2,5 poin. Dengan kata lain, keempat model setara. SVM dipilih untuk demo dan disimpan sebagai `svm_model.pkl`.

Confusion matrix SVM (baris = aktual, kolom = prediksi):

| | Prediksi negatif | Prediksi positif |
|---|---|---|
| **Aktual negatif** | 407 | 50 |
| **Aktual positif** | 83 | 312 |

Model lebih baik menangkap ulasan negatif (recall 0,89) dibanding positif (recall 0,79).

## 🔍 Temuan

1. **Rating sangat terpolarisasi.** 80,8% ulasan berating 1 atau 5 (1★: 1.717, 5★: 2.270), sedangkan rating 2-4 hanya sekitar 19%.
2. **Ulasan sangat singkat.** Median hanya 7 kata. Teks yang sama sering muncul dari pengguna berbeda: "oke" (201 ulasan), "bagus" (152), "good" (102), "mantap" (91).
3. **Kata dominan.** Positif: *aplikasi, sangat, bagus, bantu, oke*. Negatif: *aplikasi, vaksin, daftar, mau, lapor*. Kata "vaksin" muncul di kedua kelas (788 kali di negatif, 200 di positif).
4. **IndoBERT kurang selaras dengan rating.** Kecocokannya hanya 79,2%, dan cenderung memprediksi negatif (3.166 negatif vs 1.768 positif, padahal berdasarkan rating 2.350 negatif vs 2.584 positif). Dari 1.028 ketidakcocokan, 922 adalah ulasan berating positif yang dinilai negatif oleh IndoBERT, terutama pada teks pendek dan informal.

## 🛠️ Teknologi & Perpustakaan

- **Bahasa & lingkungan**: Python 3.10+ (Google Colab)
- **Scraping**: google-play-scraper
- **NLP & ML**: pandas, NLTK, PySastrawi, scikit-learn, transformers (label IndoBERT), joblib
- **Visualisasi**: matplotlib, wordcloud
- **Demo**: Gradio

## 📋 Struktur Direktori

```
sentiment-analysis-jaki/
├── notebooks/
│   ├── Analisis_Sentimen.ipynb              # scraping sampai modeling
│   └── Abadillah_Deploy.ipynb               # demo Gradio
├── data/
│   ├── scraped_Jaki.csv                     # hasil scraping mentah
│   ├── dataset_fixrev.csv                   # data dengan semua tahap preprocessing dan label
│   └── perbandingan_label.csv               # 1.028 ulasan yang labelnya berbeda antara rating dan IndoBERT
├── models/
│   ├── svm_model.pkl
│   └── tfidf_vectorizer.pkl
├── images/                                  # WordCloud dan confusion matrix
├── requirements.txt
└── README.md
```


## ⚠️ Keterbatasan

- Label berasal dari rating, bukan isi teks. Ulasan yang sama bisa berating berbeda (21 kelompok teks identik memiliki label tidak konsisten, misalnya "oke" dengan rating 5 dan 3), dan rating 3 (307 ulasan) dikelompokkan sebagai negatif.
- Ulasan sangat singkat dan generik sehingga informasi per teks terbatas.
- Selisih antar model tidak signifikan secara statistik pada 852 data uji.
- Hanya dua kelas (positif dan negatif), tanpa netral.
- Model tidak menangani sarkasme, dan kamus normalisasi slang bersifat terbatas.
