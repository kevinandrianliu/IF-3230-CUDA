# Praktikum 3 IF3230 - CUDA Radix Sort

Cara Penggunaan Program
-----
1. Pull/Clone dari repository

2. Menggunakan perintah make untuk mencompile program di src

3. Menjalankan program dengan perintah <code>./radix_sort <banyak_elemen></code>


Pembagian Tugas
-----
- Kevin Fernaldy (13516070): Pembuatan source code
    - mengerjakan main program(implementasi MPI pada main)

- Kevin Andrian Liwinata (13516118): Pembuatan source code
    - Testing, ide program, optimisasi
 

Deskripsi Solusi Paralel
-----
Radix sort merupakan algoritma <I>sorting</I> yang menggunakan digit-digit sebagai <I>increment</I>. Kami membagi menjadi blok-blok berukuran 32 x 32. Untuk setiap kali <I>increment</I>, kami membagi menjadi <I>array</I> shared_d_sub_number_array untuk menampung <I>array</I> yang akan dishare dalam sebuah blok, dan shared_d_digit_array yang akan menampung digit-digit hasil <I>Prefix Sum</I> pada setiap <I>shared array</I> tersebut. Terakhir, kami mempropagasikan setiap shared_d_digit_array tersebut ke global agar dapat di-sort sesuai dengan yang diinginkan. Berikut ini adalah gambar dari solusi yang kami buat:


Analisis Solusi
-----
Blok yang dibuat adalah ukuran 32 x 32, sehingga ada 1024 <I>thread</I> yang bekerja secara paralel dalam sebuah blok. Namun, solusi kami memiliki banyak <I>critical section</I> yang membuat hanya sebagian <I>thread</I> yang dapat bekerja secara paralel dalam suatu waktu, seperti pencarian nilai digit, <I>prefix sum</I>, dan total dari <I>prefix sum</I>(global). Semua bagian tersebut hanya bisa dikerjakan oleh 10 <I>thread</I>. Selama menjalankan total dari <I>prefix sum</I> tersebut, <I>thread</I> lainnya menunggu sampai selesai(<I>busy waiting</I>).

Pengukuran Kinerja
-----
Kami menguji pada server 167.205.32.104 dengan paralel dan mendapatkan hasil sebagai berikut:.
- N = 5000: 115943 mikrosekon, 115903 mikrosekon, 105885 mikrosekon
- N = 50000: 720618 mikrosekon, 728292 mikrosekon, 728471 mikrosekon
- N = 100000: 2664038 mikrosekon, 2643123 mikrosekon, 2895249 mikrosekon
- N = 200000: 10501679 mikrosekon, 10529893 mikrosekon, 10491956 mikrosekon 
- N = 400000: 36584886 mikrosekon, 35782088 mikrosekon, 35780913 mikrosekon

Untuk waktu <I>radix sort serial</I> yang dapat menghasilkan nilai-nilai berikut:
- N = 5000: 1.949 mikrosekon, 2.088 mikrosekon, 1.272 mikrosekon
- N = 50000: 13.318 mikrosekon, 21.635 mikrosekon, 13.779 mikrosekon
- N = 100000: 36.934 mikrosekon, 27.078 mikrosekon, 27.339 mikrosekon
- N = 200000: 55.188 mikrosekon, 60.575 mikrosekon, 56.118 mikrosekon
- N = 400000: 169.037 mikrosekon, 112.246 mikrosekon, 111.766 mikrosekon

Analisis Kinerja
-----
Dari hasil yang didapatkan, dapat disimpulkan bahwa proses secara paralel menghasilkan waktu yang jauh lebih cepat daripada secara serial. Hal ini disebabkan sorting yang biasanya menggunakan 1 </I>thread</I> menjadi berbagai blok dengan total <I>thread</I> yang banyak 

