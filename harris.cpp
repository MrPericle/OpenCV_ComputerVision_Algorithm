#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdlib.h>
#include <iostream>

using namespace cv;
using namespace std;

/*
*STEP 1: CALCOLARE Dx E Dy
*STEP 2: CALCOLARE Dx^2, Dy^2 e Dx*Dy
*STEP 3: APPLICARE FILTRO GAUSSIANO A Dx^2, Dy^2 e Dx*Dy 
*STEP 4: OTTENERE C_00,C_11;C_10
*STEP 5: CALCOLARE R
*STEP 6: NORMALIZZARE R IN [0,255]
*STEP 7: SOGLIARE R
*/

int th;
Mat src,dest;

void myHarris(const Mat& src, Mat& dest,int th){
	Mat Dx,Dx2,Dy,Dy2,Dxy,C00,C11,C10,det,track,R,out1,out2;
	double k = 0.05; //k in[0.04,0.06]

	Size kernelS(3,3);

//STEP 1
	Sobel(src,Dx,CV_32FC1,1,0);
	Sobel(src,Dy,CV_32FC1,0,1);
//STEP 2
	pow(Dx,2,Dx2);
	pow(Dy,2,Dy2);
	multiply(Dx,Dy,Dxy);
//STEP 3 + 4
	copyMakeBorder(Dx2,Dx2,1,1,1,1,BORDER_REFLECT);
	GaussianBlur(Dx2,C00,kernelS,5);
	copyMakeBorder(Dy2,Dy2,1,1,1,1,BORDER_REFLECT);
	GaussianBlur(Dy2,C11,kernelS,5);
	copyMakeBorder(Dxy,Dxy,1,1,1,1,BORDER_REFLECT);
	GaussianBlur(Dxy,C10,kernelS,5);
//STEP5
	multiply(C00,C11,out1);
	multiply(C10,C10,out2);
	det = out1 - out2;
	track = C00 + C11;

	pow(track,2,track);
	track = k * track;

	R = det - track;
//STEP 6
	normalize(R,dest,0,255,NORM_MINMAX, CV_32FC1, Mat() );
//STEP 7
	//threshold(dest,dest,th,255,THRESH_BINARY);

}

void HarrisThreshold(int,void*){
	myHarris(src,dest,th);
	Mat dest_scale;
	//dest.copyTo(dest_scale);
	normalize(dest,dest_scale, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	convertScaleAbs(dest,dest_scale);
	for( int i = 1; i < dest.rows-1 ; i++ )
    {
        for( int j = 1; j < dest.cols-1; j++ )
        {
            if( dest.at<float>(i,j) > th )
            {
                circle( dest_scale, Point(j,i), 5,  Scalar(0), 2, 8, 0 );
            }
        }
    }
 	namedWindow( "Corners detected" );
    imshow( "Corners detected", dest_scale );
}

int main(int argc, char const *argv[])
{
	if( argc < 2){
	    cout<<"usage: "<<argv[0]<<" image_name"<<endl;
	    exit(0);
    }
    src = imread(argv[1],IMREAD_GRAYSCALE);
    imshow("src",src);
    cv::createTrackbar("Trackbar th", "src", &th, 255, HarrisThreshold);
    HarrisThreshold(0,0);

    waitKey(0);
	return 0;
}