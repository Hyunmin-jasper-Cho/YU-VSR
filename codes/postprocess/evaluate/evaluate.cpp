/**
 * @file evaluate_PSNR/MSSIM.cpp
 * @author hyunmin cho (cho.hyun@icloud.com)
 * @brief compare PSNR/MSSIM with two model's result, based on gnu++20
 * @version 0.1
 * @date 2022-07-25
 *
 * @copyright Copyright (c) 2022
 *
 */

#include <iostream> // for standard I/O
#include <filesystem>
#include <fstream>
#include <vector>

#include <opencv2/core.hpp>     // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/imgproc.hpp>  // Gaussian Blur
#include <opencv2/imgcodecs.hpp>

using namespace cv;
namespace fs = std::filesystem;

typedef std::vector<std::string> str_vec;

double
getPSNR(const Mat &I1, const Mat &I2);

Scalar
getMSSIM(const Mat &I1, const Mat &I2);

str_vec
get_titles_of_each_png(fs::path& path);

static void
get_psnr_mssim(str_vec &titles, const std::string& text_name, fs::path &ref_root, fs::path &use_root);

static void
help() {
    std::cout
            << "------------------------------------------------------------------------------" << std::endl
            << "This program shows how to get a PSNR & MSSIM from two image file with OpenCV."
            << "It tests the similarity of two input image first with PSNR, and for the frames "
            << "below a PSNR trigger value, also with MSSIM." << std::endl
            << "Usage:" << std::endl
            << "./imageSet-psnr-ssim <dir referenceImageSet> <dir useCaseImageSet> <refImageSetName>" << std::endl
            << "--------------------------------------------------------------------------" << std::endl
            << std::endl;
}


int
main(int argc, char * argv[]) {

    help();

    /*
        Target ImageSet folder is in the same level of the cpp file

        |-- referenceImageSet
            |--0000.png
            |--0001.png
            ...
        |-- useCaseImageSet
            |--0000.png
            |--0001.png
            ...

        |-- evaluate.cpp
    */

    if (argc != 4) {std::cout << "Not enough parameters\n" << std::endl; return EXIT_FAILURE; }

    fs::path ref_root(argv[1]);
    fs::path use_root(argv[2]);

    // ref/use_case whatever you want.
    // the file name should be the same.
    str_vec titles = get_titles_of_each_png(ref_root);

    // iterate each file and get PSNR and MSSIM
    get_psnr_mssim(titles, std::string(argv[3]), ref_root, use_root);


    return EXIT_SUCCESS;
}


// ![get-psnr]
double
getPSNR(const Mat &I1, const Mat &I2) {
    Mat s1;
    absdiff(I1, I2, s1);             // |I1 - I2|
    s1.convertTo(s1, CV_32F);             // cannot make a square on 8 bits
    s1 = s1.mul(s1);                            // |I1 - I2|^2
    Scalar s = sum(s1);                        // sum elements per channel
    double sse = s.val[0] + s.val[1] + s.val[2];   // sum channels
    if (sse <= 1e-10) {                            // for small values return zero
        return 0;
    } else {
        double mse = sse / (double) (I1.channels() * I1.total());
        double psnr = 10.0 * log10((255 * 255) / mse);
        return psnr;
    }
}

Scalar
getMSSIM(const Mat &i1, const Mat &i2) {
    const double C1 = 6.5025, C2 = 58.5225;
    /***************************** INITS **********************************/
    int d = CV_32F;
    Mat I1, I2;
    i1.convertTo(I1, d);            // cannot calculate on one byte large values
    i2.convertTo(I2, d);
    Mat I2_2 = I2.mul(I2);        // I2^2
    Mat I1_2 = I1.mul(I1);        // I1^2
    Mat I1_I2 = I1.mul(I2);        // I1 * I2
    /*************************** END INITS **********************************/
    Mat mu1, mu2;                   // PRELIMINARY COMPUTING
    GaussianBlur(I1, mu1, Size(11, 11), 1.5);
    GaussianBlur(I2, mu2, Size(11, 11), 1.5);
    Mat mu1_2 = mu1.mul(mu1);
    Mat mu2_2 = mu2.mul(mu2);
    Mat mu1_mu2 = mu1.mul(mu2);
    Mat sigma1_2, sigma2_2, sigma12;
    GaussianBlur(I1_2, sigma1_2, Size(11, 11), 1.5);
    sigma1_2 -= mu1_2;
    GaussianBlur(I2_2, sigma2_2, Size(11, 11), 1.5);
    sigma2_2 -= mu2_2;
    GaussianBlur(I1_I2, sigma12, Size(11, 11), 1.5);
    sigma12 -= mu1_mu2;
    Mat t1, t2, t3;
    t1 = 2 * mu1_mu2 + C1;
    t2 = 2 * sigma12 + C2;
    t3 = t1.mul(t2);                 // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))
    t1 = mu1_2 + mu2_2 + C1;
    t2 = sigma1_2 + sigma2_2 + C2;
    t1 = t1.mul(t2);                 // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))
    Mat ssim_map;
    divide(t3, t1, ssim_map);        // ssim_map =  t3./t1;
    Scalar mssim = mean(ssim_map);   // mssim = average of ssim map
    return mssim;
}

str_vec
get_titles_of_each_png(fs::path& path) {
    str_vec titles;

    fs::directory_iterator itr(path);
    while (itr != fs::end(itr)) {
        const fs::directory_entry& entry = *itr;
        const std::string str_path = entry.path().generic_string();

        std::istringstream iss(str_path);
        std::string token;
        int token_itr = 0;
        while (std::getline(iss, token, '/')) {
            token_itr++;
            if (token_itr == 3) {
                titles.emplace_back(token);
            }
        } itr++;
    }

    return titles;
}

static void
get_psnr_mssim(str_vec &titles, const std::string& text_name, fs::path &ref_root, fs::path &use_root) {

    fs::path ref_temp = ref_root, use_temp = use_root;

    for (auto& title: titles) {
        Mat ref = cv::imread((ref_temp /= (title + ".png")).generic_string());
        Mat use = cv::imread((use_temp /= (title + ".png")).generic_string());


        fs::path textfile("./");
        textfile /= ( text_name + ".txt");

        fs::create_directory(textfile.parent_path());

        std::ofstream ofs(textfile);
        ofs << "PSNR OF " << title << ": " << getPSNR(ref, use) << ".\n";
        ofs << "SSIM OF " << title << ": " << getMSSIM(ref, use) << ".\n\n";
        ofs.close();

        ref_temp = ref_root;
        use_temp = use_root;
    }
}