# High Performance Image Processing Pipeline (HPC) 🚀

Bu proje, büyük boyutlu görüntülerin (4K ve üzeri) işlenmesi sırasında **Serial (Seri)**, **OpenMP (Paylaşımlı Bellek)** ve **MPI (Dağıtık Bellek)** yöntemlerinin performanslarını karşılaştıran bir C++ uygulamasıdır.

Proje kapsamında Sobel Kenar Tespiti (Sobel Edge Detection) algoritması kullanılarak 3 aşamalı bir görüntü işleme hattı (pipeline) kurulmuştur.

## 🛠️ Kullanılan Teknolojiler
* **Dil:** C++17
* **Kütüphaneler:** OpenCV (Görüntü İşleme), OpenMP, MS-MPI
* **Derleme:** CMake & MinGW64

## 📂 Proje Yapısı
* `serial/`: Tek çekirdekli referans implementasyon.
* `openmp/`: Multi-threading (OpenMP) implementasyonu.
* `mpi/`: Multi-processing (MPI) implementasyonu (Domain Decomposition).

## 🚀 Kurulum ve Derleme

Bu projeyi çalıştırmak için bilgisayarınızda OpenCV ve MS-MPI yüklü olmalıdır.

# Build klasörü oluşturun
mkdir build
cd build

# CMake ile derleyin
cmake ..
cmake --build .
