# Tugas 03 & 04: Modeling, Eksperimen Perbaikan, & Evaluasi

## 1. Perbandingan Konfigurasi Model

| Aspek                    | Model Baseline (`baseline_model.keras`) | Model Improved (`improved_model.keras`)                                     |
| :----------------------- | :-------------------------------------- | :-------------------------------------------------------------------------- |
| **Arsitektur Dasar**     | MobileNetV2 (ImageNet, Frozen)          | MobileNetV2 (Warm-up + Fine-Tuning 30 layer)                                |
| **Data Augmentation**    | Tidak ada                               | RandomFlip, Rotation (0.2), Zoom (0.15), Contrast (0.15)                    |
| **Penanganan Imbalance** | Tidak ada (bobot seragam)               | Balanced Class Weight (`trash` = 3,10x)                                     |
| **Classifier Head**      | GAP + Dropout (0.2) + Dense (6)         | GAP + BatchNorm + Dropout (0.35) + Dense (128) + Dropout (0.25) + Dense (6) |

---

## 2. Perbandingan Metrik Evaluasi (Test Set - 383 Citra)

| Metrik Evaluasi            | Model Baseline | Model Improved | Perubahan  |
| :------------------------- | :------------: | :------------: | :--------: |
| **Test Accuracy**          |   **81,46%**   |     79,63%     |   -1,83%   |
| **Macro Average F1**       |   **0,7873**   |     0,7696     |  -0,0177   |
| **Weighted Average F1**    |   **0,8154**   |     0,8012     |  -0,0142   |
| **Recall Kelas `trash`**   |     66,67%     |   **71,43%**   | **+4,76%** |
| **F1-Score Kelas `trash`** |   **0,5833**   |     0,5263     |  -0,0570   |

_(Grafik kurva pelatihan dan Confusion Matrix masing-masing model tersimpan di folder `docs/`)._

---

## 3. Analisis Hasil & Pemilihan Model Akhir

1. **Dampak Class Weighting:** Pemberian bobot kelas sebesar `3,10` pada kelas minoritas `trash` berhasil meningkatkan _recall_ kelas `trash` dari **66,67% menjadi 71,43%** (lebih banyak sampah residu yang berhasil tertangkap), namun menurunkan _precision_ karena beberapa sampel `plastic` dan `paper` ikut terprediksi sebagai `trash`.
2. **Penyebab Penurunan Akurasi pada Model Improved:** Kombinasi augmentasi spasial agresif (_flip vertikal + rotasi + kontras_) dengan pembukaan 30 layer teratas MobileNetV2 (_fine-tuning_) pada dataset berukuran terbatas (1.766 gambar latih) menyebabkan _underfitting_ (akurasi latih tertahan di ~74–78%).
3. **Keputusan Deployment:** Berdasarkan evaluasi objektif pada _Test Set_, **`baseline_model.keras` dipilih sebagai model utama** pada aplikasi web karena memiliki akurasi tertinggi (**81,46%**, melampaui target minimal 80%) dan keseimbangan presisi antar-kelas yang lebih stabil.
