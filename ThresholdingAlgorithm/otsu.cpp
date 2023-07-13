#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdlib.h>
#include <iostream>
#include <vector>

using std::cout;
using std::endl;
using std::vector;
using namespace cv;

vector<double> normalizedHistogram(const Mat& img){
    vector<double> hist(256,0.0f);
    for(int y = 0; y < img.rows; y++){
        for(int x = 0; x < img.cols; x++){
            hist.at(img.at<uchar>(y,x))++;
        }
    }
    for(int i = 0; i < 256; i++){
        hist.at(i) /= (img.rows*img.cols);
    }
    return hist;
}

int otsuS1(const Mat& src){
    Mat temp;
    GaussianBlur(src,temp,Size(3,3),0,0);
    vector<double> normHist = normalizedHistogram(temp);
    double globalMean = 0.0f;
    double kProb = 0.0f;
    double cumKMean = 0.0f;
    double interclassVariance = 0.0f;
    double maxVar = 0.0f;
    int th = 0;

    for(int i = 0; i < 256; i++){
        globalMean += (i+1)*normHist.at(i);
    }

    for(int k = 0; k < 256; k++ ){
        kProb += normHist.at(k);
        cumKMean += (k+1)*normHist.at(k);

        double interclassVarNum = pow((globalMean*kProb)-cumKMean,2);
        double interclassVarDen = kProb*(1-kProb);
        
        if(interclassVarDen != 0) 
            interclassVariance = interclassVarNum/interclassVarDen;
        else 
            interclassVariance = 0;
        if(interclassVariance > maxVar){
            maxVar = interclassVariance;
            th = k;
        }
    }
    return th;
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
    threshold(src,dest,otsuS1(src),255,THRESH_BINARY);
    imshow("segmentation",dest);
    waitKey();


    return 0;
}
