#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdlib.h>
#include <iostream>

using std::cout;
using std::endl;
using namespace cv;

int th = 200;

void calculateGradient(const Mat& src, Mat&C00, Mat&C11,Mat&C10){
    Mat dx,dy,dxy;
    Sobel(src,dx,CV_32F,1,0);
    Sobel(src,dy,CV_32F,0,1);
    multiply(dx,dy,dxy);
    pow(dx,2,dx);
    pow(dy,2,dy);

    GaussianBlur(dx,C00,Size(5,5),0,0);
    GaussianBlur(dy,C11,Size(5,5),0,0);
    GaussianBlur(dxy,C10,Size(5,5),0,0);
}

void calculateDetAndTrack(const Mat& C00,const Mat& C11, const Mat& C10, Mat& det, Mat& track){
    Mat diag1,diag2;
    multiply(C00,C11,diag1);
    multiply(C10,C10,diag2);
    det = diag1-diag2;

    track = C00 + C11;
    pow(track,2,track);
    track = track*0.04;
}

void calculateR(const Mat& det,const Mat& track,Mat& R){
    R = det - track;
    normalize(R,R,0,255,NORM_MINMAX);
}

void detectCorner(const Mat& R, Mat& out){
    convertScaleAbs(R,out);
    cvtColor(out,out,COLOR_GRAY2BGR);
    for(int y = 0; y < R.rows; y++){
        for(int x = 0; x < R.cols; x++){
            if(R.at<float>(y,x)>th)
                circle(out,Point(x,y),4,Scalar(0,0,255));
        }
    }
}

void harris(const Mat& src, Mat& dest){
    Mat C00,C11,C1O,det,track,R,blur;
    GaussianBlur(src,blur,Size(5,5),0,0);
    calculateGradient(blur,C00,C11,C1O);
    calculateDetAndTrack(C00,C11,C1O,det,track);
    calculateR(det,track,R);
    detectCorner(R,dest);
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
    harris(src,dest);
    imshow("dest",dest);
    waitKey();

    return 0;
}
