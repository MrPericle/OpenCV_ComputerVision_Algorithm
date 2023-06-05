#include <opencv2/imgproc/imgproc.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <stdlib.h>  
#include <iostream>  
#include <vector>  

using namespace cv;
using namespace std;

Mat src;  // Input image
float th = 0.008;  // Threshold value for rectangle detection

// Class representing a line segment
class Rette{
    private:
        Point p1, p2;  // Start and end points of the line
    public:
        Rette(Point p1, Point p2){
            this->p1 = p1;
            this->p2 = p2;
        }
        Point getStart(){return p1;};
        Point getEnd(){return p2;};
};

// Function to perform rectangle detection using the Hough transform on an edge image
void myRectHought(const Mat& srcEdge, Mat& out, float th, vector<Rette>& myRect){
    if (th < 0 || th > 1)
        throw out_of_range("Threshold value out of [0,1] range");

    int voters = 0;  // Counter for edge pixels
    float rad_conv = CV_PI/180;  // Conversion factor from degrees to radians

    int dist = hypot(srcEdge.cols, srcEdge.rows);  // Maximum possible distance for rho value
    Mat votingSpace = Mat::zeros(2*dist, 180, CV_32FC1);  // Initialize the voting space

    // Loop through all pixels in the edge image
    for (int x = 0; x < srcEdge.rows; x++){
        for (int y = 0; y < srcEdge.cols; y++){
            if (srcEdge.at<uchar>(x, y) == 255){  // Edge pixel found
                voters++;
                // Loop through all theta values (angles)
                for (int thetaIndex = 0; thetaIndex < 180; thetaIndex++){
                    float theta = (thetaIndex - 90) * rad_conv;
                    float rhoIndex = cvRound(y*cos(theta) + x*sin(theta)) + dist;
                    votingSpace.at<float>(rhoIndex, thetaIndex) += 1;
                }
            }
        }
    }

    cout << "voters = " << voters << endl;
    float threshold = voters * th;  // Calculate the threshold based on the number of edge pixels

    // Loop through all values in the voting space
    for (int rhoIndex = 0; rhoIndex < votingSpace.rows; rhoIndex++){
        for (int thetaIndex = 0; thetaIndex < votingSpace.cols; thetaIndex++){
            // Check if the voting value at the current (rhoIndex, thetaIndex) exceeds the threshold
            if (votingSpace.at<float>(rhoIndex, thetaIndex) > threshold){
                Point beg, end;
                float x0, y0;

                // Calculate the actual rho value based on the offset and the distance
                int rho = rhoIndex - dist;
                // Convert the theta index to radians and shift it by 90 degrees
                float theta = (thetaIndex - 90) * rad_conv;

                // Calculate the cosine and sine of the theta value
                float cosTheta = cos(theta);
                float sinTheta = sin(theta);
                
                // Calculate the coordinates of the line segment's start point
                x0 = cvRound(rho * cosTheta);
                y0 = cvRound(rho * sinTheta);

                // Calculate the coordinates of the line segment's end point by offsetting from the start point
                beg.x = cvRound(x0 + 1000 * (-sinTheta));
                beg.y = cvRound(y0 + 1000 * (cosTheta));

                // Calculate the coordinates of the line segment's end point by offsetting from the start point
                end.x = cvRound(x0 - 1000 * (-sinTheta));
                end.y = cvRound(y0 - 1000 * (cosTheta));

                Rette r(beg, end);
                myRect.push_back(r);  // Store the detected rectangle
            }
        }
    }

    cvtColor(srcEdge, out, cv::COLOR_GRAY2BGR);  // Convert the edge image to color for visualization

    // Draw the detected rectangles on the output image
    for (auto r : myRect){
        line(out, r.getStart(), r.getEnd(), Scalar(0, 0, 255), 2);
    }
}

int main(int argc, char const *argv[]){
    if (argc < 2){
        cout << "usage: " << argv[0] << " image_name" << endl;
        exit(0);
    }

    vector<Rette> rect;  // Vector to store the detected rectangles
    src = imread(argv[1], IMREAD_GRAYSCALE);  // Read the input image as grayscale

    Mat edges, lines;  // Intermediate images
    Canny(src, edges, 150, 170);  // Apply Canny edge detection algorithm

    try{
        myRectHought(edges, lines, th, rect);  // Perform rectangle detection
    } catch(const out_of_range& ex){
        cerr << "Exception: " << ex.what() << endl;
        exit(EXIT_FAILURE);
    }

    imshow("Edge", edges);  // Display the edge image
    cout << "rect found = " << rect.size() << endl;

    cvtColor(src, src, COLOR_GRAY2BGR);  // Convert the input image to color for visualization
    imshow("base img", src);  // Display the input image

    imshow("rect Detection", lines);  // Display the output image with detected rectangles
    waitKey(0);

    return 0;
}
