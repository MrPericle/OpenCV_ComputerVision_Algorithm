#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdlib.h>
#include <iostream>
#include <vector>

using std::cout;
using std::endl;
using std::vector;
using namespace cv;

vector<float> normalizedEdgeHist(const Mat& src){
    Mat edge,tmp;
    vector<float> normHist(256,0);
    int edgePixel = 0;
    medianBlur(src,tmp,5);
    Canny(tmp,edge,130,90);
    for(auto y = 0; y < edge.rows; y++){
        for(auto x = 0; x < edge.cols; x++){
            if(edge.at<uchar>(y,x) == 255){
                normHist.at(src.at<uchar>(y,x))++;
                edgePixel++;
            }
        }
    }
    for(auto&x : normHist){
        x/=edgePixel;
    }
    return normHist;
}

int otsuEdge(const Mat& src){
    vector<float> histEdge = normalizedEdgeHist(src);
    float probK = 0.0f;
    float cumMeanK = 0.0f;
    float globMean = 0.0f;
    float interclassVarianceK = 0.0f;
    float maxItClassVar = 0.0f;
    int kStar = 0;

    for(auto i = 0; i<256; i++){
        globMean += (i+1)*histEdge.at(i);
    }
    for(auto i = 0; i < 255; i++){
        probK += histEdge.at(i);
        cumMeanK += (i+1)*histEdge.at(i);
        float num = pow(probK*globMean - cumMeanK,2);
        float den = probK * (1-probK);
        interclassVarianceK = den != 0 ? num/den : 0;
        if(interclassVarianceK > maxItClassVar){
            maxItClassVar = interclassVarianceK;
            kStar = i;
        }
    }
    return kStar;
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
    threshold(src,dest,otsuEdge(src),255,NORM_MINMAX);
    imshow("dest",dest);
    waitKey();

    return 0;
}
