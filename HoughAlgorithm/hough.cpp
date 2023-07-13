#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdlib.h>
#include <iostream>
#include <vector>

using std::cout;
using std::endl;
using std::vector;
using namespace cv;

const float RADC = CV_PI/180;
const int th = 190;

class rect{
        Point beg,end;
    public:
        rect(Point beg, Point end) :beg(beg),end(end){};
        inline Point getB(){return beg;};
        inline Point getE(){return end;};
};

void createVotingSpace(const Mat& src, Mat&votingSpace){
    int dist = hypot(src.rows,src.cols);
    votingSpace = Mat::zeros(2*dist, 181,CV_32F);
}

void votation(const Mat& edge, Mat& votingSpace){
    int dist = votingSpace.rows/2;
    for(int y = 0; y < edge.rows; y++){
        for(int x = 0; x < edge.cols; x++){
            if(edge.at<uchar>(y,x) == 255){
                for(int thetaIndex = 0; thetaIndex <= 180; thetaIndex++){
                    float theta = (thetaIndex - 90)*RADC;
                    float sint = sin(theta);
                    float cost = cos(theta);
                    int rhoIndex = cvRound(x*cost + y*sint) + dist;
                    votingSpace.at<float>(rhoIndex,thetaIndex)++;
                }
            }
        }
    }
}

void detectRect(const Mat& votingSpace, vector<rect>& detected){
    int dist = votingSpace.rows/2;
    for(int rhoIndex = 0; rhoIndex < votingSpace.rows; rhoIndex++){
        for(int thetaIndex = 0; thetaIndex <= 180; thetaIndex++){
            if(votingSpace.at<float>(rhoIndex,thetaIndex) > th){
                int rho = rhoIndex - dist;
                float theta = (thetaIndex - 90) * RADC;
                float cost = cos(theta);
                float sint = sin(theta);
                int x0 = rho*cost;
                int y0 = rho*sint;

                Point begin, end;

                begin.x = cvRound(x0 - 1000 * -sint);
                begin.y = cvRound(y0 - 1000 *cost);

                end.x = cvRound(x0 + 1000 * -sint);
                end.y = cvRound(y0 + 1000 * cost);

                detected.push_back(rect(begin,end));

            }
        }
    }
}

void drowRect(const vector<rect>& detected, Mat& out){
    for(auto r : detected){
        line(out,r.getB(),r.getE(),Scalar(0,0,255));
    }
}

void houghRette(const Mat& src, Mat& out){
    Mat blur,edge,votingSpace;
    vector<rect> detected;
    src.copyTo(out);
    GaussianBlur(src,blur,Size(5,5),0,0);
    cvtColor(blur,blur,COLOR_BGR2GRAY);
    createVotingSpace(blur,votingSpace);
    Canny(blur,edge,120,140);
    votation(edge,votingSpace);
    detectRect(votingSpace,detected);
    drowRect(detected,out);
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
    if (src.cols > 500 || src.rows > 500) {
        cv::resize(src, src, cv::Size(0, 0), 0.5, 0.5); // resize for speed
    }

    imshow("src", src);
    houghRette(src,dest);
    imshow("dest",dest);
    waitKey();

    return 0;
}
