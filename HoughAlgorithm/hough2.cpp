#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdlib.h>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

Mat src; // Source image variable
float th = 0.05; // Threshold for circle voting

class Circle {
private:
    float a, b, radius; // Circle center coordinates and radius

public:
    Circle(float a, float b, float radius) {
        this->a = a;
        this->b = b;
        this->radius = radius;
    }

    float getA() { return a; };
    float getB() { return b; };
    float getRadius() { return radius; };
};

void myHoughCircle(const Mat& edgeSource, Mat& dest, int minRadius, int maxRadius, float th) {
    float radConver = CV_PI / 180; // Conversion factor from degrees to radians
    int voters = 0; // Total number of voters (pixels with value 255)
    int a, b;

    int dimensions[] = { edgeSource.rows, edgeSource.cols, (maxRadius - minRadius + 1) };
    Mat votingSpace3D = Mat(3, dimensions, CV_32FC1, Scalar(0)); // 3D voting space for circles

    // Loop through all pixels in the edgeSource image
    for (int x = 0; x < edgeSource.rows; x++) {
        for (int y = 0; y < edgeSource.cols; y++) {
            if (edgeSource.at<uchar>(x, y) == 255) { // If the pixel is an edge pixel
                voters++; // Increment the number of voters
                // Loop through all possible radii and angles
                for (int r = minRadius; r < maxRadius; r++) {
                    for (int angIndex = 0; angIndex < 360; angIndex++) {
                        float theta = angIndex * radConver; // Convert angle to radians
                        float cosTheta = cos(theta);
                        float sinTheta = sin(theta);

                        int a = cvRound(x - r * sinTheta); // Calculate circle center coordinates
                        int b = cvRound(y - r * cosTheta);

                        if (a > 0 && a < edgeSource.rows && b > 0 && b < edgeSource.cols) {
                            int rIndex = r - minRadius; // Calculate the index for the radius in votingSpace3D
                            votingSpace3D.at<float>(a, b, rIndex)++; // Increment the voting value at the corresponding position
                        }
                    }
                }
            }
        }
    }

    float threshold = voters * th; // Calculate the threshold for circle detection
    vector<Circle> foundCircle; // Vector to store detected circles

    // Loop through the voting space to find circles above the threshold
    for (int x = 0; x < edgeSource.rows; x++) {
        for (int y = 0; y < edgeSource.cols; y++) {
            for (int rIndex = 0; rIndex < maxRadius - minRadius; rIndex++) {
                if (votingSpace3D.at<float>(x, y, rIndex) > threshold) {
                    Circle myCircle(x, y, rIndex + minRadius); // Create a Circle object
                    foundCircle.push_back(myCircle); // Add the detected circle to the vector
                }
            }
        }
    }

    edgeSource.copyTo(dest); // Copy the edgeSource image to the destination image
    cvtColor(edgeSource, dest, cv::COLOR_GRAY2BGR); // Convert the destination image to color

    // Draw the detected circles on the destination image
    for (auto aCircle : foundCircle) {
        Point center(aCircle.getB(), aCircle.getA()); // Get the center coordinates of the circle
        int radius = aCircle.getRadius(); // Get the radius of the circle

        // Draw circles on the destination image
        circle(dest, center, 2, Scalar(255, 0, 0)); // Draw a small circle at the center
        circle(dest, center, radius, Scalar(0, 0, 255), 2); // Draw the circle boundary
    }
}

int main(int argc, char const* argv[]) {
    if (argc < 2) {
        cout << "usage: " << argv[0] << " image_name" << endl;
        exit(0);
    }

    src = imread(argv[1], IMREAD_GRAYSCALE); // Read the source image in grayscale
    Mat edges, circles; // Variables for storing edges and detected circles

    Canny(src, edges, 150, 170); // Apply Canny edge detection

    myHoughCircle(edges, circles, 20, 60, th); // Detect circles using custom Hough transform

    imshow("Edge", edges); // Display the edge image

    // Convert the source image to color and display it
    imshow("base img", src);

    imshow("Circle Detection", circles); // Display the detected circles
    waitKey(0);

    return 0;
}
