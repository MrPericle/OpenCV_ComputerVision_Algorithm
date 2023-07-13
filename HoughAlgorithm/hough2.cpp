#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdlib.h>
#include <iostream>
#include <vector>

using std::cout;
using std::endl;
using std::vector;
using namespace cv;

class rect{
        Point beg,end;
    public:
        rect(Point b, Point e): beg(b),end(e){};
        Point getB(){return beg;};
        Point getE(){return end;};
};

void createVotingSpace(const Mat& src, Mat& votingSpace){
    int dist = hypot(src.rows,src.cols);
    votingSpace = Mat::zeros(2*dist, 180,CV_32F);
}

void vote(const Mat& edge,Mat& votingSpace){
    const float RAD_CONVERT = CV_PI/180;
    int dist = votingSpace.rows/2;
    for(int x = 0; x < edge.rows;x++){
        for(int y = 0; y < edge.cols; y++){
            if(edge.at<uchar>(x,y) == 255){
                for(int thetaIndex = 0; thetaIndex < 180; thetaIndex++){
                    float theta = (thetaIndex - 90) * RAD_CONVERT;
                    float cost = cos(theta);
                    float sint = sin(theta);
                    int rho = cvRound(x*sint + y*cost) + dist;

                    votingSpace.at<float>(rho,thetaIndex)++;
                }
            }
        }
    }
}

void detectRect(const Mat& edge,Mat& votingSpace,int th,vector<rect>& detected){
    const float RAD_CONVERT = CV_PI/180;
    int dist = votingSpace.rows/2;
    for(int rhoIndex = 0; rhoIndex<votingSpace.rows; rhoIndex++){
        for(int thetaIndex = 0; thetaIndex<180; thetaIndex++){
            if(votingSpace.at<float>(rhoIndex,thetaIndex) > th){
                float theta = (thetaIndex - 90) * RAD_CONVERT;
                float sint = sin(theta);
                float cost = cos(theta);

                int rho = rhoIndex - dist;

                int x0 = cvRound(rho * cost);
                int y0 = cvRound(rho * sint);

                Point beg,end;

                beg.x = cvRound(x0 + 1000*(-sint));
                beg.y = cvRound(y0 + 1000*cost);

                end.x = cvRound(x0 - 1000*(-sint));
                end.y = cvRound(y0 - 1000*cost);

                rect r(beg,end);
                detected.push_back(r);
            }
        }
    }
}
void drowRect(const vector<rect>& detected, Mat& out){
    for(auto r : detected){
        line(out,r.getB(),r.getE(),Scalar(0,0,255),2);
    }
}

void houghRect(const Mat& src,Mat& out,int th = 180){
    Mat edge,votingSpace;
    vector<rect> detected;

    src.copyTo(edge);
    src.copyTo(out);

    cvtColor(edge,edge,COLOR_BGR2GRAY);
    GaussianBlur(edge,edge,Size(5,5),0);
    Canny(edge,edge,110,140);

    createVotingSpace(edge,votingSpace);
    vote(edge,votingSpace);

    detectRect(edge,votingSpace,th,detected);
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
    imshow("src", src);
    houghRect(src,dest);
    imshow("rect",dest);
    waitKey(0);

    return 0;
}
