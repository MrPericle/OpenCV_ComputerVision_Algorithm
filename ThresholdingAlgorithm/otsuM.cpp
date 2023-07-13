#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdlib.h>
#include <iostream>
#include <vector>

using std::cout;
using std::endl;
using std::vector;
using namespace cv;

vector<float> normalizedHist(const Mat& src){
    const int NORM_FACTOR = src.rows*src.cols;
    vector<float> hist(256,0.0f);
    for(int y = 0; y < src.rows; y++){
        for(int x = 0; x < src.cols; x++){
            hist.at(src.at<uchar>(y,x))++;
        }
    }
    for(int i = 0; i<256; i++)
        hist.at(i) /= NORM_FACTOR; 
    return hist;
}

vector<int> otsuM(const Mat& src){
    Mat blur;
    medianBlur(src,blur,5);
    vector<float> hist = normalizedHist(blur);

    vector<int> kStar(2,0);

    vector<float>probK(3,0.0f);
    vector<float> cumMean(3,0.0f);
    float globMean = 0.0f;
    float interclassVarianceK = 0.0f;
    float maxInterC = 0.0f;

    for(int i = 0; i < 256; i++)
        globMean += (i+1)*hist.at(i);

    for(int i = 0; i < 256 - 2; i++){
        probK.at(0) += hist.at(i);
        cumMean.at(0) += (i+1)*hist.at(i);
        for(int j = i+1; j < 256 - 1; j++){
            probK.at(1) += hist.at(j);
            cumMean.at(1) += (j+1)*hist.at(j);
            for(int k = j+1; k < 256; k++){
                probK.at(2) += hist.at(k);
                cumMean.at(2) += (k+1)*hist.at(k);
                for(int w = 0; w<3; w++){
                    if(probK.at(w)){
                        interclassVarianceK += probK.at(w)*pow(cumMean.at(w)/probK.at(w)-globMean,2);
                    }
                }
                if(interclassVarianceK > maxInterC){
                    maxInterC = interclassVarianceK;
                    kStar.at(0) = i;
                    kStar.at(1) = j;
                }
                interclassVarianceK = 0.0f;
            }
            probK.at(2) = cumMean.at(2) = 0.0f;
        }
        probK.at(1) = cumMean.at(1) = 0.0f;
    }
    return kStar;
}

void otsuTh(const Mat& src, Mat& out){
    vector<int> kStar = otsuM(src);
    out = Mat::zeros(src.size(),CV_8U);
    for(int y = 0; y < src.rows; y++){
        for(int x = 0; x < src.cols; x++){
            if(src.at<uchar>(y,x) > kStar.at(1)) 
                out.at<uchar>(y,x) = 255;
            else if(src.at<uchar>(y,x) >= kStar.at(0) && src.at<uchar>(y,x) <= kStar.at(1))
                out.at<uchar>(y,x) = 127; 
            else out.at<uchar>(y,x) = 0;
        }
    }
}

int main(int argc, char const* argv[])
{
    if (argc < 2)
    {
        cout << "usage: " << argv[0] << " image_name" << endl;
        exit(0);
    }

    Mat src = imread(argv[1], IMREAD_GRAYSCALE);
    Mat dest;
    if (src.cols > 500 || src.rows > 500) {
        cv::resize(src, src, cv::Size(0, 0), 0.5, 0.5); // resize for speed
    }

    imshow("src", src);
    otsuTh(src,dest);
    imshow("dest",dest);
    waitKey();

    return 0;
}