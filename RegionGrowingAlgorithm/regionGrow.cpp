#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdlib.h>
#include <iostream>
#include <stack>

using std::cout;
using std::endl;
using std::stack;
using namespace cv;

Point shift8[8]{
    Point(-1,-1),
    Point(-1, 0),
    Point(-1, 1),
    Point(0, -1),
    Point(0 , 1),
    Point(1, -1),
    Point(1,  0),
    Point(1,  1)
};

int th = 200;
int regionNumber = 100;
float minAreaFactor = 0.01;

void grow(const Mat& src, const Mat& dest, Mat& mask, Point seed){
    stack<Point> front;
    front.push(seed);

    while(!front.empty()){
        Point center = front.top();
        mask.at<uchar>(center) = 1;
        front.pop();

        for(int i = 0; i < 8; i++){
            Point neigh = center + shift8[i];
            if(neigh.x < 0 || neigh.x >= src.cols || neigh.y < 0 || neigh.y >= src.rows) continue;
            else{
                int delta = cvRound(pow(src.at<Vec3b>(center)[0] - src.at<Vec3b>(neigh)[0],2 ) +
                                    pow(src.at<Vec3b>(center)[1] - src.at<Vec3b>(neigh)[1],2)+
                                    pow(src.at<Vec3b>(center)[2] - src.at<Vec3b>(neigh)[2],2));
                if(delta < th && !dest.at<uchar>(neigh) && !mask.at<uchar>(neigh))
                    front.push(neigh);
            }  
        }
    }
}

void regionGrow(const Mat& src, Mat& dest){
    dest = Mat::zeros(src.rows,src.cols,CV_8UC1);
    Mat mask = Mat::zeros(src.rows,src.cols,CV_8UC1);
    int minAreaRegion = cvRound(src.rows*src.cols*minAreaFactor);
    uchar padding = 1;
    for(int y = 0; y < src.rows; y++){
        for(int x = 0; x < src.cols; x++){
            if(dest.at<uchar>(Point(x,y)) == 0){
                grow(src,dest,mask,Point(x,y));
                if(sum(mask).val[0]>minAreaRegion){
                    imshow("region",mask*255);
                    waitKey();
                    dest = dest+mask*padding;
                    if(++padding > regionNumber){
                        cout<<"Error : max num region reached"<<endl;
                        exit(EXIT_FAILURE);
                    }
                }
                else dest = dest + mask*255;
                mask = mask - mask;
            }
        }
    }
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
    regionGrow(src,dest);
    imshow("dest",dest);
    waitKey();

    return 0;
}