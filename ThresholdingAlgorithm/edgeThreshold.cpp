#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdlib.h>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

void applyEdgeDetection(const Mat& source, Mat& edgeBinary, int th){
    Mat laplacian,edge,blurSource;

    GaussianBlur(source,blurSource,Size(5,5),0);
    Laplacian(blurSource, laplacian, CV_16S);
    // Convert the result to the appropriate range
    convertScaleAbs(laplacian, edge);

    threshold(edge,edgeBinary,th,255,THRESH_BINARY);
}

unsigned int getEdgePoint(const Mat& edgeBinary, vector<Point>& edgePoints){
    unsigned int edgePointNumber = 0;
    for(int i = 0; i<edgeBinary.rows;i++){
        for(int j = 0; j<edgeBinary.cols;j++){
            if(edgeBinary.at<uchar>(i,j) == 255){
                edgePointNumber++;
                Point edgePoint;
                edgePoint.x = i;
                edgePoint.y = j;
                edgePoints.push_back(edgePoint);
            }
        }
    }
    return edgePointNumber;
}

vector<float> calculateEdgeHistogram(const Mat& source, const vector<Point>& edgePoints,unsigned int edgePointNumber){
    vector<float> normHistogram(256,0);
    for(auto edgePoint : edgePoints){
        normHistogram.at(source.at<uchar>(edgePoint.x,edgePoint.y))++;
    }

    for(int i = 0; i<normHistogram.size(); i++){
        normHistogram.at(i) = (edgePointNumber > 0)? normHistogram.at(i)/edgePointNumber : 0;
    }
    return normHistogram;
}

vector<float> pixelProbability(const vector<float>& normHist) {
    vector<float> pK(256, 0); // Cumulative pixel probability vector

    // Loop through all pixel values
    for (int k = 0; k < 256; k++) {
        // Calculate the cumulative probability for each pixel value
        for (int i = 0; i <= k; i++) {
            pK.at(k) += normHist.at(i);
        }
    }

    return pK;
}

// Function to calculate the cumulative intensity average
vector<float> cumulativeIntensityAvg(vector<float> normHist) {
    vector<float> m(256, 0); // Cumulative intensity average vector
    m.at(0) = normHist.at(0);

    // Loop through all pixel values
    for (int k = 1; k < 256; k++) {
        // Calculate the cumulative intensity average for each pixel value
        for (int i = 0; i <= k; i++) {
            m.at(k) += (i) * normHist.at(i);
        }
    }

    return m;
}

// Function to calculate the global intensity (mean)
float globalIntensity(vector<float> normHist) {
    float mG = normHist.at(0);

    // Loop through all pixel values
    for (int i = 1; i < 256; i++) {
        mG += (i) * normHist.at(i);
    }

    return mG;
}

// Function to calculate the interclass variance
vector<float> interclassVariance(float globalInt, vector<float> cumIntensityAvg, vector<float> probability) {
    vector<float> sigma2B(256, 0); // Interclass variance vector
    float num, num2, den;

    // Loop through all pixel values
    for (int k = 0; k < 256; k++) {
        num = globalInt * probability.at(k) - cumIntensityAvg.at(k);
        num2 = pow(num, 2);

        den = probability.at(k) * (1 - probability.at(k));

        // Calculate the interclass variance for each pixel value
        sigma2B.at(k) = den == 0 ? 0 : num2 / den;
    }

    return sigma2B;
}

// Function to calculate the global variance
float globalVariance(float globalInt, vector<float> probability) {
    float sigmaG2 = 0;

    // Loop through all pixel values
    for (int i = 0; i < 256; i++) {
        float val = i - globalInt;
        float val2 = pow(val, 2);

        // Calculate the global variance
        sigmaG2 += (val2 * probability.at(i));
    }

    return sigmaG2;
}

// Function to find the optimal threshold value (kStar)
int findKStar(vector<float> intClassVariance) {
    float maxVal = intClassVariance.at(0);
    int kStar = 0;

    // Loop through all pixel values
    for (int i = 1; i < 256; i++) {
        // Find the pixel value with the maximum interclass variance
        if (intClassVariance.at(i) > maxVal) {
            maxVal = intClassVariance.at(i);
            kStar = i;
        }
    }

    return kStar;
}

// Function to calculate the separability value
float separabilityValue(int kStar, vector<float> intClassVariance, float globalVariance) {
    return intClassVariance.at(kStar) / globalVariance;
}

// Function implementing Otsu's thresholding algorithm
int edgeThreshold(const Mat& src,int th) {
    unsigned int edgePointNumber;
    vector<Point> edgePoints;
    Mat blurSrc,edge;
    GaussianBlur(src, blurSrc, Size(5, 5), 0); // Apply Gaussian blur to the source image
    applyEdgeDetection(src,edge,th);
    edgePointNumber = getEdgePoint(edge,edgePoints);
    vector<float> normHisto = calculateEdgeHistogram(src,edgePoints,edgePointNumber); // Create a normalized histogram using only edge points
    vector<float> P = pixelProbability(normHisto); // Calculate pixel probabilities
    vector<float> m = cumulativeIntensityAvg(normHisto); // Calculate cumulative intensity average
    float mG = globalIntensity(normHisto); // Calculate global intensity

    vector<float> sigma2B = interclassVariance(mG, m, P); // Calculate interclass variances
    float sigmaG2 = globalVariance(mG, P); // Calculate global variance

    int kStar = findKStar(sigma2B); // Find the optimal threshold value

    return kStar;
}


int main(int argc, char const* argv[]) {
    if (argc < 2) {
        cout << "usage: " << argv[0] << " image_name" << endl;
        exit(0);
    }

    Mat src = imread(argv[1], IMREAD_GRAYSCALE); // Read the source image in grayscale
    Mat dest;

    threshold(src, dest, edgeThreshold(src,15), 255, THRESH_BINARY); // Apply Otsu's thresholding based on edge

    imshow("src", src); // Display the source image
    imshow("otsu seg", dest); // Display the Otsu segmented image
    waitKey(0);

    return 0;
}