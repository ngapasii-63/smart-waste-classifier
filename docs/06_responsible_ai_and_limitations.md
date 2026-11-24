# Tugas 06: Responsible AI, Analisis Limitasi, & Etika Sistem

## 1. Analisis Limitasi Teknis (Failure Modes)

Berdasarkan evaluasi pada dataset uji (`Test Set`) serta pengujian foto dunia nyata, sistem memiliki beberapa keterbatasan yang perlu dipahami pengguna:

1. **Bias Latar Belakang (_Background Bias_):**  
   Dataset TrashNet didominasi oleh foto objek tunggal dengan latar belakang putih polos/cerah. Saat diuji pada foto lingkungan nyata yang ramai (misalnya sampah di atas meja berpola, lantai bertekstur, atau di dalam tempat sampah penuh), tingkat keyakinan (_confidence_) model dapat menurun.
2. **Kemiripan Visual Antar-Kelas (_Visual Ambiguity_):**
   - **Plastik Bening vs. Kaca (_Plastic_ vs. _Glass_):** Botol plastik transparan yang memantulkan cahaya sering menyerupai botol kaca.
   - **Kertas vs. Kardus (_Paper_ vs. _Cardboard_):** Kertas cokelat tebal atau karton tipis memiliki tekstur dan warna yang berdekatan.
   - **Kelas `trash` (Residu):** Memiliki jumlah data paling sedikit (5,4% dari dataset) dan bentuk yang sangat heterogen, sehingga memiliki presisi terendah dibandingkan kelas lainnya.
3. **Keterbatasan Satu Objek (_Single-Object Assumption_):**  
   Model klasifikasi citra (_Image Classification_) dirancang untuk memprediksi **satu objek dominan** per foto. Jika pengguna memotret tumpukan berbagai jenis sampah sekaligus, model hanya akan memilih satu kategori yang paling menonjol.
4. **Tidak Mendeteksi Sampah Organik & B3 Secara Spesifik:**  
   Dataset TrashNet berfokus pada 6 kategori anorganik/residu standar sehingga belum mencakup kategori sampah basah/organik (sisa makanan, daun) maupun limbah B3 (baterai, lampu neon, elektronik).

---

## 2. Mitigasi Risiko & Prinsip Responsible AI

| Prinsip Responsible AI                             | Implementasi pada Sistem _Smart Waste Classifier_                                                                                                                                                         |
| :------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Transparansi & Keterjelasan (_Explainability_)** | Aplikasi menampilkan persentase _confidence score_ secara terbuka beserta grafik distribusi probabilitas seluruh 6 kelas, bukan sekadar label tunggal tanpa konteks.                                      |
| **Human-in-the-Loop & Ambang Keamanan**            | Sistem menerapkan batas minimum keyakinan (_confidence threshold_ 60%). Jika prediksi di bawah 60%, sistem memberikan peringatan agar pengguna memfoto ulang dan tetap mengandalkan pertimbangan manusia. |
| **Privasi Pengguna (_Privacy-by-Design_)**         | Pemrosesan gambar dilakukan secara lokal di memori (_in-memory processing_ via OpenCV/NumPy) tanpa menyimpan foto unggahan atau tangkapan kamera pengguna ke server maupun database eksternal.            |
| **Keberlanjutan & Dampak Sosial**                  | Sistem dirancang sebagai alat bantu edukasi pemilahan rumah tangga di Indonesia untuk meningkatkan kualitas sampah setor ke Bank Sampah dan mengurangi pencemaran sampah tercampur.                       |

---

## 3. Rencana Pengembangan Lanjutan (_Future Work_)

1. **Pengayaan Dataset Lokal:** Menambahkan dataset foto sampah kemasan produk Indonesia dengan latar belakang alami (_in-the-wild_) serta menambah kategori **Organik** dan **Limbah B3**.
2. **Peningkatan ke Object Detection (YOLOv8):** Mengembangkan sistem dari klasifikasi gambar tunggal menjadi deteksi multi-objek agar mampu mengenali beberapa jenis sampah dalam satu bingkai kamera secara _real-time_.
3. **Integrasi Lokasi Bank Sampah:** Menghubungkan hasil klasifikasi dengan peta lokasi Bank Sampah atau titik drop-box daur ulang terdekat di wilayah pengguna.
