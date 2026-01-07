#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono> // Zaman ölçümü için

// --- 1. AŞAMA: PREPROCESS (Renkli -> Gri Seviye) ---
// Yöntem: Weighted Method (0.299*R + 0.587*G + 0.114*B)
void toGrayscale(const cv::Mat& input, cv::Mat& output) {
    output.create(input.size(), CV_8UC1); // Tek kanallı çıktı
    int rows = input.rows;
    int cols = input.cols;

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            // OpenCV'de renk sırası BGR'dir
            cv::Vec3b pixel = input.at<cv::Vec3b>(i, j);
            uchar b = pixel[0];
            uchar g = pixel[1];
            uchar r = pixel[2];

            // Gri formülü
            output.at<uchar>(i, j) = static_cast<uchar>(0.299 * r + 0.587 * g + 0.114 * b);
        }
    }
}

// 2. AŞAMA: PROCESS 
// Şartname: "Matris çarpımı içeren bir işlem"
void applySobel(const cv::Mat& input, cv::Mat& output) {
    output.create(input.size(), CV_8UC1);
    int rows = input.rows;
    int cols = input.cols;

    // Sobel Kernelleri (3x3 Matrisler)
    int Gx[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };

    int Gy[3][3] = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };

    // Kenarlarda işlem yapmamak için döngüyü 1'den başlatıp sondan 1 eksik bitiriyoruz
    for (int i = 1; i < rows - 1; ++i) {
        for (int j = 1; j < cols - 1; ++j) {
            int sumX = 0;
            int sumY = 0;

            // 3x3 Matris Çarpımı 
            for (int k = -1; k <= 1; ++k) {
                for (int l = -1; l <= 1; ++l) {
                    uchar val = input.at<uchar>(i + k, j + l);
                    sumX += val * Gx[k + 1][l + 1];
                    sumY += val * Gy[k + 1][l + 1];
                }
            }

            // Gradyan büyüklüğü: sqrt(Gx^2 + Gy^2)
            int magnitude = static_cast<int>(std::sqrt(sumX * sumX + sumY * sumY));
            
            // 255 sınırını aşarsa kırp 
            if (magnitude > 255) magnitude = 255;
            output.at<uchar>(i, j) = static_cast<uchar>(magnitude);
        }
    }
}

// 3. AŞAMA: POSTPROCESS 
// Amaç: Görüntüyü siyah-beyaz (binary) hale getirmek
void applyThreshold(const cv::Mat& input, cv::Mat& output, int thresholdValue) {
    output.create(input.size(), CV_8UC1);
    int rows = input.rows;
    int cols = input.cols;

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            uchar val = input.at<uchar>(i, j);
            if (val > thresholdValue)
                output.at<uchar>(i, j) = 255; // Beyaz
            else
                output.at<uchar>(i, j) = 0;   // Siyah
        }
    }
}

int main() {
    // 1. Görüntü Yükleme
    std::string imagePath = "../input.jpg"; 
    cv::Mat img = cv::imread(imagePath);

    if (img.empty()) {
        std::cerr << "Hata: Goruntu bulunamadi! Yol: " << imagePath << std::endl;
        return -1;
    }

    std::cout << "Islem basladi: " << img.cols << "x" << img.rows << " piksel." << std::endl;

    cv::Mat grayImg, sobelImg, resultImg;

    // ZAMANLAYICI BAŞLAT 
    auto start = std::chrono::high_resolution_clock::now();

    // Pipeline Uygulama
    toGrayscale(img, grayImg);      
    applySobel(grayImg, sobelImg);    
    applyThreshold(sobelImg, resultImg, 100); 

    // ZAMANLAYICI DURDUR 
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    std::cout << "Seri Islem Suresi: " << duration.count() << " ms" << std::endl;

    // Sonuçları Kaydet
    cv::imwrite("1_gray.jpg", grayImg);
    cv::imwrite("2_sobel.jpg", sobelImg);
    cv::imwrite("3_final.jpg", resultImg);

    std::cout << "Dosyalar kaydedildi." << std::endl;

    return 0;
}