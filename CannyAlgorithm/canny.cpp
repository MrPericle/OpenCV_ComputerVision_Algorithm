#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdlib.h>
#include <iostream>

using std::cout;
using std::endl;
using namespace cv;

int lth = 80;
int hth = 120;

void caclculateMagAndAng(const Mat& src, Mat& mag, Mat& ang){
    Mat dx,dy;
    Sobel(src,dx,CV_32F,1,0);
    Sobel(src,dy,CV_32F,0,1);
    //mag = abs(dx) + abs(dy);
    magnitude(dx,dy,mag);
    // dx.convertTo(dx,CV_32F);
    // dy.convertTo(dy,CV_32F);

    phase(dx,dy,ang,true);

}

void NMS(const Mat& mag,const Mat& ang, Mat& nmsOut){
    copyMakeBorder(mag,nmsOut,1,1,1,1,BORDER_CONSTANT,Scalar(0));
    for(int y = 1; y < mag.rows; y++){
        for(int x = 1; x < mag.cols; x++){
            float angVal = ang.at<float>(y-1,x-1) > 180 ? ang.at<float>(y,x)-180 : ang.at<float>(y,x);
            if(angVal >=0 && angVal <= 22.5 || angVal <= 180 && angVal > 157.5 ){
                if(nmsOut.at<float>(y,x) < nmsOut.at<float>(y,x - 1) || nmsOut.at<float>(y,x) < nmsOut.at<float>(y,x + 1) ){
                    nmsOut.at<float>(y,x) = 0;
                }
            }
            else if(angVal > 22.5 && angVal <= 67.5){
                if(nmsOut.at<float>(y,x) < nmsOut.at<float>(y+1,x+1) || nmsOut.at<float>(y,x) < nmsOut.at<float>(y-1,x-1) ){
                    nmsOut.at<float>(y,x) = 0;
                }
            } 
            else if(angVal > 67.5 && angVal <= 112.5){
                if(nmsOut.at<float>(y,x) < nmsOut.at<float>(y-1,x) || nmsOut.at<float>(y,x) < nmsOut.at<float>(y+1,x) ){
                    nmsOut.at<float>(y,x) = 0;
                }
            } 
            else if(angVal > 112.5 && angVal <= 157.5 ){
                if(nmsOut.at<float>(y,x) < nmsOut.at<float>(y+1,x-1) || nmsOut.at<float>(y,x) < nmsOut.at<float>(y-1,x+1) ){
                    nmsOut.at<float>(y,x) = 0;
                }
            }  
        }
    }
}

void HThreshold(Mat& nmsOut, Mat& dest){
    dest = Mat::zeros(nmsOut.rows-2,nmsOut.cols-2,CV_8U);
    for(int y = 1; y < nmsOut.rows-1; y++){
        for(int x = 1; x < nmsOut.cols;x++){
            if(nmsOut.at<float>(y,x) > hth) dest.at<uchar>(y-1,x-1) = 255;
            else if(nmsOut.at<float>(y,x)< lth) dest.at<uchar>(y-1,x-1) = 0;
            else{
                bool strongN = false;
                for(int j = -1; j >= 1; j++){
                    for(int i = -1; i >= 1; i++){
                        if(nmsOut.at<float>(y + j,x + i) > hth) strongN = true;
                    }
                }
                if(strongN) dest.at<uchar>(y-1,x-1) = 255;
                else dest.at<uchar>(y-1,x-1) = 0;
            }
        }
    }
}

void myCanny(const Mat& src, Mat& dest){
    Mat blur,mag,ang,nmsOut;
    GaussianBlur(src,blur,Size(5,5),0,0);
    caclculateMagAndAng(src,mag,ang);
    NMS(mag,ang,nmsOut);
    HThreshold(nmsOut,dest);
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
    // if (src.cols > 500 || src.rows > 500) {
    //     cv::resize(src, src, cv::Size(0, 0), 0.5, 0.5); // resize for speed
    // }

    imshow("src", src);
    myCanny(src,dest);
    imshow("dest",dest);
    waitKey();

    return 0;
}
