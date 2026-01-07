#include <opencv2/opencv.hpp>
#include <mpi.h>   
#include <iostream>
#include <vector>
#include <cmath>

// 2. AŞAMA: PROCESS 
void applySobelKernel(const uchar* inputBuffer, uchar* outputBuffer, int rows, int cols) {
    int Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int Gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            // Sınır Kontrolü:
            if (i == 0 || i == rows - 1 || j == 0 || j == cols - 1) {
                outputBuffer[i * cols + j] = 0;
                continue;
            }

            int sumX = 0; 
            int sumY = 0;

            // 3x3 Konvolüsyon İşlemi
            for (int k = -1; k <= 1; ++k) {
                for (int l = -1; l <= 1; ++l) {
                    // Tek boyutlu dizide 2D indeksleme mantığı: (satır * genişlik) + sütun
                    uchar val = inputBuffer[(i + k) * cols + (j + l)];
                    sumX += val * Gx[k + 1][l + 1];
                    sumY += val * Gy[k + 1][l + 1];
                }
            }

            int magnitude = static_cast<int>(std::sqrt(sumX * sumX + sumY * sumY));
            outputBuffer[i * cols + j] = (magnitude > 255) ? 255 : static_cast<uchar>(magnitude);
        }
    }
}

// 3. AŞAMA: POSTPROCESS 
void applyThresholdKernel(uchar* buffer, int size, int threshold) {
    // Her piksel için basit karşılaştırma
    for (int i = 0; i < size; ++i) {
        if (buffer[i] > threshold) buffer[i] = 255;
        else buffer[i] = 0;
    }
}

int main(int argc, char** argv) {
    // MPI Ortamını Başlat
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    MPI_Comm_size(MPI_COMM_WORLD, &size); 

    int imgRows = 0, imgCols = 0;
    std::vector<uchar> fullImageBuffer; 
    std::vector<uchar> finalImageBuffer; 

    // MPI Zaman Ölçümü Başlangıcı
    double t_start_total = MPI_Wtime();
    double t_pre_start = 0, t_pre_end = 0;

    // 1. AŞAMA: PREPROCESS 
    // Dosya okuma işlemi genellikle tek bir işlemci tarafından yapılır ve dağıtılır.
    if (rank == 0) {
        t_pre_start = MPI_Wtime();
        
        cv::Mat img = cv::imread("../input.jpg");
        if (img.empty()) {
            std::cerr << "Hata: Goruntu yuklenemedi!" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Renkliden Griye Çevir (OpenCV kullanıldı)
        cv::Mat grayImg;
        cv::cvtColor(img, grayImg, cv::COLOR_BGR2GRAY);
        
        imgRows = grayImg.rows;
        imgCols = grayImg.cols;

        // Görüntü verisini tek boyutlu (1D) bir vektöre kopyala.
        // MPI fonksiyonları genellikle ham veri (array) ile çalışır.
        if (grayImg.isContinuous()) {
            fullImageBuffer.assign(grayImg.data, grayImg.data + grayImg.total());
        } else {
            cv::Mat cont = grayImg.clone();
            fullImageBuffer.assign(cont.data, cont.data + cont.total());
        }
        
        t_pre_end = MPI_Wtime();
        std::cout << "--- MPI Modu (Detayli Sure Olcumu) ---" << std::endl;
        std::cout << "Process Sayisi: " << size << std::endl;
    }

    // VERİ PAYLAŞIMI 
    MPI_Bcast(&imgRows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&imgCols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // SCATTER HAZIRLIĞI 
    // Hangi process kaç satır alacak hesaplaması.
    int rowsPerProcess = imgRows / size;
    int remainder = imgRows % size; 

    std::vector<int> sendCounts(size);   
    std::vector<int> displacements(size); 
    int currentDisplacement = 0;

    for (int i = 0; i < size; ++i) {
        // Kalan satırları ilk processlere dağıt 
        int r = rowsPerProcess + (i < remainder ? 1 : 0);
        sendCounts[i] = r * imgCols; 
        displacements[i] = currentDisplacement;
        currentDisplacement += sendCounts[i];
    }

    int myRows = sendCounts[rank] / imgCols; 
    int myDataSize = sendCounts[rank];       

    // Gelen veriyi tutmak için yerel bellek ayır
    std::vector<uchar> localBuffer(myDataSize);
    std::vector<uchar> localResult(myDataSize);

    // VERİYİ DAĞIT
    MPI_Scatterv(rank == 0 ? fullImageBuffer.data() : nullptr, sendCounts.data(), displacements.data(), MPI_UNSIGNED_CHAR,
                 localBuffer.data(), myDataSize, MPI_UNSIGNED_CHAR,
                 0, MPI_COMM_WORLD);

    // 2. AŞAMA: PROCESS 
    MPI_Barrier(MPI_COMM_WORLD);
    double t_proc_start = MPI_Wtime();

    // Sobel işlemini kendi küçük parçam üzerinde yapıyorum
    applySobelKernel(localBuffer.data(), localResult.data(), myRows, imgCols);
    
    double t_proc_end = MPI_Wtime();
    double my_proc_time = t_proc_end - t_proc_start;

    // 3. AŞAMA: POSTPROCESS 
    double t_post_start = MPI_Wtime();
    
    // Eşikleme işlemini kendi parçam üzerinde yapıyorum
    applyThresholdKernel(localResult.data(), myDataSize, 100);
    
    double t_post_end = MPI_Wtime();
    double my_post_time = t_post_end - t_post_start;

    // SONUÇLARI TOPLA 
    // Herkesin 'localResult'ını alıp Rank 0'da 'finalImageBuffer'da birleştir.
    if (rank == 0) {
        finalImageBuffer.resize(imgRows * imgCols);
    }

    MPI_Gatherv(localResult.data(), myDataSize, MPI_UNSIGNED_CHAR,
                rank == 0 ? finalImageBuffer.data() : nullptr, sendCounts.data(), displacements.data(), MPI_UNSIGNED_CHAR,
                0, MPI_COMM_WORLD);

    double t_end_total = MPI_Wtime();

    // SÜRELERİ RAPORLA (REDUCE - MAX)
    // Paralel programlamada "Süre", en yavaş çalışanın bitirme süresidir.
    // Herkesin süresini alıp, MPI_MAX ile en büyüğünü buluyoruz.
    double max_proc_time, max_post_time;
    MPI_Reduce(&my_proc_time, &max_proc_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&my_post_time, &max_post_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double preprocess_time_ms = (t_pre_end - t_pre_start) * 1000.0;
        double process_time_ms = max_proc_time * 1000.0;
        double postprocess_time_ms = max_post_time * 1000.0;
        double total_time_ms = (t_end_total - t_start_total) * 1000.0;
        
        // İletişim maliyeti = Toplam Süre 
        // Bu değer, veriyi Scatter ve Gather yaparken kaybedilen zamandır.
        double calc_time = preprocess_time_ms + process_time_ms + postprocess_time_ms;
        double comm_overhead = total_time_ms - calc_time;

        std::cout << "--------------------------------------" << std::endl;
        std::cout << "Preprocess Suresi : " << preprocess_time_ms << " ms" << std::endl;
        std::cout << "Process Suresi    : " << process_time_ms << " ms (Hesaplama)" << std::endl;
        std::cout << "Postprocess Suresi: " << postprocess_time_ms << " ms (Hesaplama)" << std::endl;
        std::cout << "--------------------------------------" << std::endl;
        std::cout << "TOPLAM SURE       : " << total_time_ms << " ms (Iletisim Dahil)" << std::endl;
        std::cout << "Iletisim Maliyeti : " << comm_overhead << " ms (Yaklasik)" << std::endl;

        // Sonucu kaydet
        cv::Mat resultMat(imgRows, imgCols, CV_8UC1, finalImageBuffer.data());
        cv::imwrite("mpi_final.jpg", resultMat);
        std::cout << "mpi_final.jpg kaydedildi." << std::endl;
    }

    MPI_Finalize(); // MPI'ı kapat
    return 0;
}