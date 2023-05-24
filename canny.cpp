#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdlib.h>
#include <iostream>

using namespace cv;
using namespace std;

Mat src,dest;
int ht = 198,lt = 144;


void myCanny(const Mat& src, Mat& dest, int low_th, int high_th){

	Mat bdest,sobelX, sobelY,mag,ang;
	float ang_val;
	Size kernelS(3,3);
/*
*	STEP 1: Gaussian blurr;
*	STEP 2: Calcola mamagitudo e angolo;
*	STEP 3: Non-Maxima-Suppression;
*	STEP 4: Threshold con isteresi
*/
// GAUSSIAN BLUR
	copyMakeBorder(src,bdest,1,1,1,1,BORDER_REFLECT);
	GaussianBlur(bdest,bdest,kernelS,5);

	bdest.convertTo(bdest,CV_32FC1);

// CALCOLO GRADIENTE

	Sobel(bdest,sobelX,bdest.type(),1,0);
	Sobel(bdest,sobelY,bdest.type(),0,1);

// CALCOLO MAGNITUDINE

	mag = abs(sobelX) + abs(sobelY);

// CALCOLO ANGOLO

	phase(sobelX,sobelY,ang,true);
	normalize(mag,mag,0,255,NORM_MINMAX);
	mag.convertTo(mag,CV_8UC1);
	
	normalize(ang,ang,-180,180,NORM_MINMAX);

//	NON_MAXIMA_SUPPRESSION

	for (int i = 1; i < mag.rows-1; ++i)
	{
		for(int j = 1; j < mag.cols-1; j++){
			ang_val = ang.at<float>(i,j);
			
			if ((ang_val <= -157.5 && ang_val > 157.5) || (ang_val > -22.5 && ang_val <= 22.5) ){ 
				if(mag.at<uchar>(i,j) < mag.at<uchar>(i,j-1) || mag.at<uchar>(i,j) < mag.at<uchar>(i,j+1))
					mag.at<uchar>(i,j) = 0;
			}
			else if ((ang_val <= -112.5 && ang_val > -157.5) || (ang_val > 22.5 && ang_val <= 67.5) ){
				if(mag.at<uchar>(i,j) < mag.at<uchar>(i-1,j-1) || mag.at<uchar>(i,j) < mag.at<uchar>(i+1,j+1))
					mag.at<uchar>(i,j) = 0;
			}

			else if ((ang_val <= -67.5 && ang_val > -112.5) || (ang_val > 67.5 && ang_val <= 112.5) ){
				if(mag.at<uchar>(i,j) < mag.at<uchar>(i+1,j) || mag.at<uchar>(i,j) < mag.at<uchar>(i-1,j))
					mag.at<uchar>(i,j) = 0; 	
			}
			else if ((ang_val <= -22.5 && ang_val > -67.5) || (ang_val > 112.5 && ang_val <= 157.5) ){
				if(mag.at<uchar>(i,j) < mag.at<uchar>(i-1,j+1) || mag.at<uchar>(i,j) < mag.at<uchar>(i+1,j-1))
					mag.at<uchar>(i,j) = 0;	
			}
		}
	}

//G_n THRESHOLE
	for (int i = 1; i < mag.rows-1; ++i)
	{
		for(int j = 1; j<mag.cols-1; j++){
			if(mag.at<uchar>(i,j) > high_th) mag.at<uchar>(i,j) = 255;

			else if(mag.at<uchar>(i,j) < low_th) mag.at<uchar>(i,j) = 0;

			else if(mag.at<uchar>(i,j) <= high_th && mag.at<uchar>(i,j) >= low_th){
				bool strong_n = false;
				for(int x = -1; x <= 1 && !strong_n ; x++){
					for(int y = -1; y <= 1 && !strong_n ; y++){
						{
							if(mag.at<uchar>(i+x,j+y) > high_th) strong_n = true;
						}
					}
				}
				if(strong_n) mag.at<uchar>(i,j) = 255;
				else mag.at<uchar>(i,j) = 0;
			}
		}
	}
	int x = 1;
	int y = 1;
	int height = mag.rows -2;
	int width = mag.cols -2;
	Rect roiRect(x, y, width, height);
	dest = mag(roiRect);

	//mag.copyTo(dest);

}


void CannyThreshold(int, void*){
	    myCanny(src, dest,ht,lt);

	    imshow("Canny",dest);

}

int main(int argc, char const *argv[])
{
	if( argc < 2){
        cout<<"usage: "<<argv[0]<<" image_name"<<endl;
        exit(0);
    }
    src = imread(argv[1],IMREAD_GRAYSCALE);
    Mat edges;
    imshow("src",src);
    namedWindow("Canny");
    cv::createTrackbar("Trackbar th", "Canny", &ht, 255, CannyThreshold);
    cv::createTrackbar("Trackbar lh", "Canny", &lt, 255, CannyThreshold);
    CannyThreshold(0,0);
    /*myCanny(src, dest,ht,lt);

    imshow("Canny",dest);*/
    waitKey(0);

   cv::Canny(src, edges, 144, 198, 3, false);
    imshow("cvCanny",edges);
    waitKey(0);

    return 0;

}