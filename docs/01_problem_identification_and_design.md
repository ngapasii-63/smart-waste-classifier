# Tugas 01: Problem Identification & AI Solution Design

**Proyek:** Smart Waste Classification (Sistem Klasifikasi Sampah Rumah Tangga Berbasis Visi Komputer)  
**Mata Kuliah:** Kecerdasan Artifisial (2026)

---

## 1. Identifikasi Masalah (5 Pertanyaan Utama)

1. **Apa masalah utama yang ingin diselesaikan?**  
   Rendahnya tingkat pemilahan sampah dari sumbernya (rumah tangga) di wilayah perkotaan Indonesia. Masyarakat sering mencampur sampah kemasan (plastik, kertas, kardus, kaca, logam) dengan sampah residu karena bingung menentukan kategori daur ulang yang tepat.
2. **Siapa pengguna utama sistem ini?**  
   Masyarakat umum / anggota rumah tangga (usia remaja hingga dewasa) yang ingin memilah sampah sebelum membuangnya ke tempat sampah terpilah atau menyetorkannya ke Bank Sampah terdekat.
3. **Bagaimana proses saat ini berjalan dan apa kelemahannya?**  
   Saat ini warga memilah sampah secara manual berdasarkan tebakan masing-masing atau melihat label kecil pada kemasan. Kelemahannya: prosesnya memakan waktu, sering salah kategori (misalnya mencampur kertas kotor/berminyak ke kertas daur ulang, atau bingung membedakan plastik keras dan logam tipis), sehingga nilai jual sampah di Bank Sampah turun dan beban TPA meningkat.
4. **Mengapa masalah ini membutuhkan Kecerdasan Artifisial (AI)?**  
   Bentuk, warna, tekstur, dan kondisi sampah di dunia nyata sangat bervariasi (botol penyok, kertas kusut, pencahayaan berbeda). Aturan pemrograman konvensional (`if-else`) tidak mungkin mendefinisikan semua variasi visual tersebut. Pendekatan _Deep Learning_ (Convolutional Neural Network / Transfer Learning) mampu mengekstrak pola visual secara otomatis dari citra sampah.
5. **Apa target keberhasilan (Success Metrics) proyek ini?**
   - **Target Teknis:** Model mampu mengklasifikasikan 6 jenis sampah (_cardboard, glass, metal, paper, plastic, trash_) dengan akurasi validasi minimal **80%** dan waktu inferensi di bawah **2 detik** per gambar pada CPU standar.
   - **Target Pengguna:** Pengguna mendapatkan hasil jenis sampah beserta panduan singkat pemilahannya melalui antarmuka web sederhana (Streamlit).

---

## 2. Problem Statement & Kebutuhan Pengguna

### Problem Statement

Warga rumah tangga di kawasan perkotaan Indonesia kesulitan memilah sampah anorganik dan residu secara akurat akibat kurangnya pengetahuan praktis mengenai kategori material sampah. Dibutuhkan sistem klasifikasi visual otomatis berbasis AI yang cepat, ringan, dan mudah diakses melalui peramban web untuk membantu warga menentukan kategori sampah dan langkah pemilahannya secara tepat.

### Kebutuhan Pengguna (User Requirements)

- **Fungsional:**
  1. Pengguna dapat mengunggah foto sampah (`.jpg`/`.png`) atau mengambil gambar langsung melalui kamera.
  2. Sistem menampilkan prediksi kategori sampah dari 6 kelas (_Cardboard, Glass, Metal, Paper, Plastic, Trash_) beserta tingkat keyakinan (_confidence score_ dalam persen).
  3. Sistem memberikan rekomendasi tindakan pemilahan (misalnya: _"Masukkan ke tempat sampah anorganik / layak setor ke Bank Sampah"_ atau _"Buang ke tempat sampah residu"_).
  4. Sistem menampilkan peringatan jika tingkat keyakinan model rendah (< 60%) agar pengguna mengambil ulang foto dengan pencahayaan lebih jelas.
- **Non-Fungsional:**
  1. Antarmuka bersih, berbahasa Indonesia, dan responsif dibuka di laptop maupun ponsel.
  2. Proses prediksi berjalan ringan di CPU tanpa memerlukan GPU khusus.

---

## 3. Arsitektur Sistem

Sistem dibangun dengan pendekatan _Transfer Learning_ menggunakan arsitektur **MobileNetV2** (yang telah dilatih pada ImageNet) sebagai _feature extractor_, dikombinasikan dengan _OpenCV_ untuk prapemrosesan citra dan _Streamlit_ sebagai antarmuka web.

---

## 4. Alur Interaksi Pengguna (User Flow)

1. **Mulai:** Pengguna membuka aplikasi web _Smart Waste Classifier_ di browser.
2. **Input Citra:** Pengguna memilih metode input (unggah foto dari galeri atau jepret lewat kamera) untuk satu objek sampah.
3. **Prapemrosesan:** Sistem memvalidasi format file, mengubah ukuran gambar menjadi `224x224` piksel, dan menormalisasi nilai piksel.
4. **Inferensi Model:** Model CNN memproses citra dan menghasilkan probabilitas untuk masing-masing dari 6 kelas sampah.
5. **Evaluasi & Output:**
   - Jika _confidence_ $\ge 60\%$: Sistem menampilkan nama kelas sampah, persentase keyakinan, ikon warna tempat sampah, dan saran pengelolaan (daur ulang / residu).
   - Jika _confidence_ $< 60\%$: Sistem menampilkan pesan _"Objek kurang jelas, silakan foto ulang dengan latar belakang yang lebih terang/kontras"_ namun tetap menampilkan prediksi teratas sebagai referensi.
6. **Selesai / Ulang:** Pengguna dapat mengunggah foto sampah berikutnya.

---

## 5. Spesifikasi Dataset & Pertimbangan Awal Responsible AI

- **Dataset:** TrashNet (~2.527 citra, 6 kelas: _cardboard, glass, metal, paper, plastic, trash_).
- **Pembagian Data:** 70% Training, 15% Validation, 15% Testing.
- **Batasan Awal (Limitasi):** Dataset TrashNet didominasi foto dengan latar belakang putih polos dan pencahayaan studio, serta memiliki jumlah data kelas `trash` yang jauh lebih sedikit dibanding kelas lainnya. Oleh karena itu, model akan diuji menggunakan foto sampah nyata dengan latar belakang beragam serta diterapkan teknik _data augmentation_ dan _class weighting_ pada tahap eksperimen.
