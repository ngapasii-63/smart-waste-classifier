# Tugas 02: Dataset Preparation & Exploratory Data Analysis (EDA)

## 1. Sumber & Karakteristik Dataset

- **Nama Dataset:** TrashNet (Gary Thung & Mindy Yang)
- **Total Citra Valid:** 2.527 gambar (0 file rusak/corrupt)
- **Jumlah Kelas:** 6 kategori material sampah

| Kelas       | Jumlah Gambar | Persentase | Karakteristik Visual                            |
| :---------- | :-----------: | :--------: | :---------------------------------------------- |
| `paper`     |      594      |   23,5%    | Lembaran kertas, koran, tisu/kertas bekas       |
| `glass`     |      501      |   19,8%    | Botol kaca bening/berwarna, toples              |
| `plastic`   |      482      |   19,1%    | Botol plastik, gelas kemasan, kantong plastik   |
| `metal`     |      410      |   16,2%    | Kaleng minuman, kaleng makanan, tutup logam     |
| `cardboard` |      403      |   15,9%    | Kardus kemasan, karton cokelat                  |
| `trash`     |      137      |    5,4%    | Sampah residu campuran/tidak dapat didaur ulang |

_(Grafik visual distribusi kelas tersimpan di `docs/eda_class_distribution.png`)._

## 2. Temuan Utama EDA (Data Imbalance)

Terdapat ketidakseimbangan kelas (_class imbalance_) yang cukup mencolok: kelas mayoritas (`paper`, 594 gambar) memiliki jumlah data **4,3 kali lipat** lebih banyak dibandingkan kelas minoritas (`trash`, 137 gambar). Kondisi ini berpotensi membuat model awal (_baseline_) bias terhadap kelas mayoritas dan kurang sensitif dalam mendeteksi kelas `trash`.

## 3. Pipeline Preprocessing & Data Splitting

Seluruh citra diproses menggunakan skrip `src/prepare_data.py` dengan tahapan:

1. **Validasi Citra:** Pembacaan menggunakan OpenCV untuk memastikan tidak ada citra yang korup.
2. **Stratified Splitting:** Pembagian data secara proporsional pada setiap kelas dengan rasio **70% Train** (1.766 gambar), **15% Validation** (378 gambar), dan **15% Test** (383 gambar) menggunakan `random_state=42`.
3. **Standardisasi Ukuran:** Seluruh gambar diubah ukurannya (_resize_) menjadi `224 x 224` piksel (RGB) menggunakan interpolasi `INTER_AREA` agar sesuai dengan dimensi input standar arsitektur MobileNetV2.
