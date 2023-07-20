#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <tuple>

using namespace std;
using namespace cv;

int th = 180, rMin = 25, rMax = 90;
const double RADC = CV_PI/180;

void createVotingSpace(const Mat& src, Mat& votingSpace){
    int dim[] = {src.rows,src.cols, rMax-rMin + 1};
    votingSpace = Mat(3,dim,CV_8U,Scalar(0));
}

void voteCircle(const Mat& edge, Mat& votingSpace){
    for(auto y = 0; y < edge.rows; y++){
        for(auto x = 0; x < edge.cols; x++){
            if(edge.at<uchar>(y,x) == 255){
                for(auto r = rMin; r <= rMax; r++){
                    for(int thetaIndex = 0; thetaIndex <= 360; thetaIndex++){
                        double theta = thetaIndex*RADC;
                        double sint = sin(theta);
                        double cost = cos(theta);
                        int a = cvRound(x - r*cost);
                        int b = cvRound(y - r*sint);
                        if(a>=0 && a < edge.cols && b >= 0 && b < edge.rows)
                            votingSpace.at<uchar>(b,a,r-rMin)++;
                    }
                }
            }
        }
    }
}

vector<tuple<int,int,int>> detectCircle(const Mat& src,const Mat& votingSpace){
    vector<tuple<int,int,int>> detected;
    for(auto b = 0; b < src.rows; b++){
        for(auto a = 0; a < src.cols; a++){
            for(auto r = 0; r <= rMax - rMin; r++){
                if(votingSpace.at<uchar>(b,a,r) > th)
                    detected.push_back(make_tuple(b,a,r+rMin));
            }
        } 
    }
    return detected;
}

void drowCircle(Mat& out, vector<tuple<int,int,int>> detected){
    for(auto circ : detected){
        circle(out,Point(get<1>(circ),get<0>(circ)),get<2>(circ),Scalar(0,0,255),2);
        circle(out,Point(get<1>(circ),get<0>(circ)),1,Scalar(0,0,255));
    }
}

void houghCircle(const Mat& src, Mat& out){
    Mat blur,edge,votingSpace;
    src.copyTo(out);
    GaussianBlur(src,blur,Size(5,5),0);
    cvtColor(blur,blur,COLOR_BGR2GRAY);
    Canny(blur,edge,120,110);
    createVotingSpace(src,votingSpace);
    voteCircle(edge,votingSpace);
    vector<tuple<int,int,int>> detected = detectCircle(src,votingSpace);
    drowCircle(out,detected);
}

int main(int argc, char const* argv[])
{
    if (argc < 2)
    {
        cout << "usage: " << argv[0] << " image_name" << endl;
        exit(0);
    }

    Mat src = imread(argv[1], IMREAD_COLOR);
    Mat dest;
    // if (src.cols > 500 || src.rows > 500) {
    //     cv::resize(src, src, cv::Size(0, 0), 0.5, 0.5); // resize for speed
    // }

    imshow("src", src);
    houghCircle(src,dest);
    imshow("dest",dest);
    waitKey();
    return 0;
}
