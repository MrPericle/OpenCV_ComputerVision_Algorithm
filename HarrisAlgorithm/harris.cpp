#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdlib.h>
#include <iostream>

using namespace cv;
using namespace std;

int th;
Mat src, dest;

// Function to perform the Harris corner detection algorithm
void myHarris(const Mat& src, Mat& dest, int th);

// Function to apply a threshold to the Harris response and display the detected corners
void HarrisThreshold(int, void*);

int main(int argc, char const *argv[])
{
    if (argc < 2){
        cout << "usage: " << argv[0] << " image_name" << endl;
        exit(0);
    }

    src = imread(argv[1], IMREAD_GRAYSCALE);
    imshow("src", src);
    cv::createTrackbar("Trackbar th", "src", &th, 255, HarrisThreshold);
    HarrisThreshold(0, 0);

    waitKey(0);
    return 0;
}

void myHarris(const Mat& src, Mat& dest, int th)
{
    Mat Dx, Dx2, Dy, Dy2, Dxy, C00, C11, C10, det, track, R, out1, out2;
    double k = 0.05; // k in [0.04, 0.06]

    Size kernelS(3, 3);

    // STEP 1: Calculate Dx and Dy using Sobel operator
    Sobel(src, Dx, CV_32FC1, 1, 0);
    Sobel(src, Dy, CV_32FC1, 0, 1);

    // STEP 2: Calculate Dx^2, Dy^2, and Dx*Dy
    pow(Dx, 2, Dx2);
    pow(Dy, 2, Dy2);
    multiply(Dx, Dy, Dxy);

    // STEP 3 + 4: Apply Gaussian blur to Dx^2, Dy^2, and Dx*Dy
    // and obtain C00, C11, and C10
    copyMakeBorder(Dx2, Dx2, 1, 1, 1, 1, BORDER_REFLECT);
    GaussianBlur(Dx2, C00, kernelS, 5);
    copyMakeBorder(Dy2, Dy2, 1, 1, 1, 1, BORDER_REFLECT);
    GaussianBlur(Dy2, C11, kernelS, 5);
    copyMakeBorder(Dxy, Dxy, 1, 1, 1, 1, BORDER_REFLECT);
    GaussianBlur(Dxy, C10, kernelS, 5);

    // STEP 5: Calculate the determinant and trace of the structure tensor
    multiply(C00, C11, out1);
    multiply(C10, C10, out2);
    det = out1 - out2;
    track = C00 + C11;

    pow(track, 2, track);
    track = k * track;

    // STEP 6: Calculate the Harris response
    R = det - track;

    // STEP 7: Normalize the Harris response in the range [0, 255]
    normalize(R, dest, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
}

void HarrisThreshold(int, void*)
{
    myHarris(src, dest, th);
    Mat dest_scale;
    normalize(dest, dest_scale, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
    convertScaleAbs(dest, dest_scale);

    // Display the detected corners using circles
    for (int i = 1; i < dest.rows - 1; i++)
    {
        for (int j = 1; j < dest.cols - 1; j++)
        {
            if (dest.at<float>(i, j) > th)
            {
                circle(dest_scale, Point(j, i), 5, Scalar(0), 2, 8, 0);
            }
        }
    }

    namedWindow("Corners detected");
    imshow("Corners detected", dest_scale);
}
