#include <opencv2/opencv.hpp>
#include <iostream>
#include <omp.h>   
#include <cmath>   
#include <chrono>  // Zaman ölçümü 

// 1. AŞAMA: PREPROCESS 
void toGrayscaleOMP(const cv::Mat& input, cv::Mat& output) {
    output.create(input.size(), CV_8UC1);

    // #pragma omp parallel for: Döngüyü threadlere dağıtır.
    // collapse(2): İç içe olan (i ve j) döngüleri tek bir büyük döngü gibi birleştirip dağıtır.
    // schedule(static): İş yükü her pikselde aynı olduğu için (sadece çarpma/toplama), 
    // işleri threadlere eşit parçalar halinde baştan dağıtmak en hızlı yöntemdir.
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < input.rows; ++i) {
        for (int j = 0; j < input.cols; ++j) {
            // Pikseli al (OpenCV'de varsayılan sıra BGR'dir)
            cv::Vec3b pixel = input.at<cv::Vec3b>(i, j);
            
            // Griye dönüşüm formülü: 0.299*R + 0.587*G + 0.114*B
            // İnsan gözünün renklere hassasiyetine göre ağırlıklandırılmıştır.
            output.at<uchar>(i, j) = static_cast<uchar>(0.299 * pixel[2] + 0.587 * pixel[1] + 0.114 * pixel[0]);
        }
    }
}

// 2. AŞAMA: PROCESS 
void applySobelOMP(const cv::Mat& input, cv::Mat& output) {
    output.create(input.size(), CV_8UC1);

    // Sobel Kernelleri (Yatay ve Dikey değişim matrisleri)
    int Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int Gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    // schedule(dynamic): Konvolüsyon işlemi bellek erişimi açısından yoğun olduğu için
    // bazı threadler işini erken bitirebilir. Dynamic çizelgeleme, işi bitene yeni iş vererek dengeyi sağlar.
    // Kenar pikselleri (sınır kontrolü yapmamak için) 1'den başlatıp sondan 1 eksik bitiriyoruz.
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = 1; i < input.rows - 1; ++i) {
        for (int j = 1; j < input.cols - 1; ++j) {
            int sumX = 0; int sumY = 0;

            // 3x3 Kernel ile görüntüyü çarpıp topluyoruz 
            for (int k = -1; k <= 1; ++k) {
                for (int l = -1; l <= 1; ++l) {
                    uchar val = input.at<uchar>(i + k, j + l);
                    sumX += val * Gx[k + 1][l + 1];
                    sumY += val * Gy[k + 1][l + 1];
                }
            }
            
            // Gradyan büyüklüğü (Hipotenüs hesaplama): sqrt(Gx^2 + Gy^2)
            int magnitude = static_cast<int>(std::sqrt(sumX * sumX + sumY * sumY));
            
            // Değer 255'i aşarsa 255'e sabitle (Clamping), yoksa bozulma olur.
            output.at<uchar>(i, j) = (magnitude > 255) ? 255 : static_cast<uchar>(magnitude);
        }
    }
}

// 3. AŞAMA: POSTPROCESS
long applyThresholdOMP(const cv::Mat& input, cv::Mat& output, int thresholdValue) {
    output.create(input.size(), CV_8UC1);
    long whitePixelCount = 0; // Beyaz piksel sayacı

    // reduction(+:whitePixelCount): En kritik kısım burası.
    // Paralel çalışırken her thread kendi 'whitePixelCount' değişkenini tutar.
    // Döngü sonunda tüm threadlerin sayacı toplanarak ana değişkene yazılır.
    // Bu işlem Race Condition hatasını önler.
    #pragma omp parallel for collapse(2) reduction(+:whitePixelCount)
    for (int i = 0; i < input.rows; ++i) {
        for (int j = 0; j < input.cols; ++j) {
            // Eşik değerinden büyükse Beyaz (255), küçükse Siyah (0) yap
            if (input.at<uchar>(i, j) > thresholdValue) {
                output.at<uchar>(i, j) = 255;
                whitePixelCount++; // Sayacı artır
            } else {
                output.at<uchar>(i, j) = 0;
            }
        }
    }
    return whitePixelCount;
}

int main() {
    // İşlemcinin desteklediği maksimum thread sayısını al ve yazdır
    int max_threads = omp_get_max_threads();
    std::cout << "OpenMP Thread Sayisi: " << max_threads << std::endl;

    cv::Mat img = cv::imread("../input.jpg");
    if (img.empty()) { 
        std::cerr << "Hata: Resim bulunamadi!" << std::endl; 
        return -1; 
    }

    cv::Mat grayImg, sobelImg, resultImg;

    // GENEL ZAMANLAYICI BAŞLAT 
    auto t_start = std::chrono::high_resolution_clock::now();

    // 1. AŞAMA: Grayscale 
    toGrayscaleOMP(img, grayImg);
    auto t_pre = std::chrono::high_resolution_clock::now(); // Preprocess bitiş zamanı

    // 2. AŞAMA: Process 
    applySobelOMP(grayImg, sobelImg);
    auto t_proc = std::chrono::high_resolution_clock::now(); // Process bitiş zamanı

    // 3. AŞAMA: Postprocess 
    applyThresholdOMP(sobelImg, resultImg, 100);
    auto t_post = std::chrono::high_resolution_clock::now(); // Postprocess bitiş zamanı

    // SÜRE HESAPLAMALARI 
    std::chrono::duration<double, std::milli> d_pre = t_pre - t_start;
    std::chrono::duration<double, std::milli> d_proc = t_proc - t_pre;
    std::chrono::duration<double, std::milli> d_post = t_post - t_proc;
    std::chrono::duration<double, std::milli> d_total = t_post - t_start;

    // Sonuçları Ekrana Yazdır
    std::cout << "--------------------------------------" << std::endl;
    std::cout << "Preprocess Suresi : " << d_pre.count() << " ms" << std::endl;
    std::cout << "Process Suresi    : " << d_proc.count() << " ms" << std::endl;
    std::cout << "Postprocess Suresi: " << d_post.count() << " ms" << std::endl;
    std::cout << "--------------------------------------" << std::endl;
    std::cout << "TOPLAM SURE       : " << d_total.count() << " ms" << std::endl;

    // Sonuç görüntüsünü kaydet
    cv::imwrite("omp_final.jpg", resultImg);
    
    return 0;
}