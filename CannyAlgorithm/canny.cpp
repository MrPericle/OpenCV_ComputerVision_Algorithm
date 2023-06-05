#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdlib.h>
#include <iostream>

using namespace cv;
using namespace std;

Mat src, dest, edges;
int ht = 100, lt = 30;

// Function to perform Gaussian blur on the source image
void applyGaussianBlur(const Mat& src, Mat& dest)
{
    Size kernelS(3, 3);
    GaussianBlur(src, dest, kernelS, 0);
}

// Function to calculate gradient magnitude and angle
void calculateGradient(const Mat& src, Mat& sobelX, Mat& sobelY, Mat& mag, Mat& ang)
{
    Sobel(src, sobelX, src.type(), 1, 0);
    Sobel(src, sobelY, src.type(), 0, 1);
    mag = abs(sobelX) + abs(sobelY);
    sobelX.convertTo(sobelX, CV_32FC1);
    sobelY.convertTo(sobelY, CV_32FC1);
    phase(sobelX, sobelY, ang, true);
}

// Function to perform non-maximum suppression on the gradient magnitude image
void performNonMaxSuppression(Mat& mag, Mat& ang)
{
    copyMakeBorder(mag, mag, 1, 1, 1, 1, BORDER_CONSTANT, Scalar(0));

    for (int i = 1; i < mag.rows - 1; ++i)
    {
        for (int j = 1; j < mag.cols - 1; j++)
        {
            float ang_val = ang.at<float>(i, j) > 180 ? ang.at<float>(i, j) - 360 : ang.at<float>(i, j);

            if ((ang_val <= -157.5 && ang_val > 157.5) || (ang_val > -22.5 && ang_val <= 22.5))
            {
                if (mag.at<uchar>(i, j) < mag.at<uchar>(i, j - 1) || mag.at<uchar>(i, j) < mag.at<uchar>(i, j + 1))
                    mag.at<uchar>(i, j) = 0;
            }
            else if ((ang_val <= -112.5 && ang_val > -157.5) || (ang_val > 22.5 && ang_val <= 67.5))
            {
                if (mag.at<uchar>(i, j) < mag.at<uchar>(i + 1, j + 1) || mag.at<uchar>(i, j) < mag.at<uchar>(i - 1, j - 1))
                    mag.at<uchar>(i, j) = 0;
            }
            else if ((ang_val <= -67.5 && ang_val > -112.5) || (ang_val > 67.5 && ang_val <= 112.5))
            {
                if (mag.at<uchar>(i, j) < mag.at<uchar>(i + 1, j) || mag.at<uchar>(i, j) < mag.at<uchar>(i - 1, j))
                    mag.at<uchar>(i, j) = 0;
            }
            else if ((ang_val <= -22.5 && ang_val > -67.5) || (ang_val > 112.5 && ang_val <= 157.5))
            {
                if (mag.at<uchar>(i, j) < mag.at<uchar>(i + 1, j - 1) || mag.at<uchar>(i, j) < mag.at<uchar>(i - 1, j + 1))
                    mag.at<uchar>(i, j) = 0;
            }
        }
    }
}

// Function to perform hysteresis thresholding on the gradient magnitude image
void performHysteresisThresholding(Mat& mag, Mat& dest, int low_th, int high_th)
{
    dest = Mat::zeros(mag.rows - 1, mag.cols - 1, CV_8UC1);

    for (int i = 1; i < mag.rows - 1; ++i)
    {
        for (int j = 1; j < mag.cols - 1; j++)
        {
            if (mag.at<uchar>(i, j) > high_th)
                dest.at<uchar>(i, j) = 255;
            else if (mag.at<uchar>(i, j) < low_th)
                dest.at<uchar>(i, j) = 0;
            else if (mag.at<uchar>(i, j) <= high_th && mag.at<uchar>(i, j) >= low_th)
            {
                bool strong_n = false;

                for (int x = -1; x <= 1 && !strong_n; x++)
                {
                    for (int y = -1; y <= 1 && !strong_n; y++)
                    {
                        if (mag.at<uchar>(i + x, j + y) > high_th)
                            strong_n = true;
                    }
                }

                if (strong_n)
                    dest.at<uchar>(i - 1, j - 1) = 255;
            }
        }
    }
}

// Callback function for the Canny threshold trackbars
void CannyThreshold(int, void*)
{
    applyGaussianBlur(src, dest);
    Mat sobelX, sobelY, mag, ang;
    calculateGradient(dest, sobelX, sobelY, mag, ang);
    performNonMaxSuppression(mag, ang);
    performHysteresisThresholding(mag, dest, lt, ht);
    cv::Canny(src, edges, lt, ht, 3, false);

    imshow("cvCanny", edges);
    imshow("Canny", dest);
}

int main(int argc, char const* argv[])
{
    if (argc < 2)
    {
        cout << "usage: " << argv[0] << " image_name" << endl;
        exit(0);
    }

    src = imread(argv[1], IMREAD_GRAYSCALE);

    imshow("src", src);
    namedWindow("Canny");
    cv::createTrackbar("Trackbar th", "src", &ht, 255, CannyThreshold);
    cv::createTrackbar("Trackbar lh", "src", &lt, 255, CannyThreshold);
    CannyThreshold(0, 0);

    waitKey(0);

    return 0;
}
