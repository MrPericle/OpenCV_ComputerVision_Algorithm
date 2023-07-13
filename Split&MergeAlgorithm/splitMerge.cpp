#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdlib.h>
#include <iostream>
#include <vector>

using std::cout;
using std::endl;
using std::vector;
using namespace cv;

int minWidth;
float stdDevTH;

class TNode{
    private:
        Rect region;
        TNode *UL, *UR, *LR, *LL;
        vector<TNode*> merged;
        vector<bool> mergedB = vector<bool>(4,false);
        float mean,stdDev;
    public:
        TNode(Rect R): region(R){};

        void addRegion(TNode* R){merged.push_back(R);};
        void setUL(TNode* N){UL = N;};
        void setUR(TNode* N){UR = N;};
        void setLR(TNode* N){LR = N;};
        void setLL(TNode* N){LL = N;};
        void setMergedB(int i){mergedB.at(i) = true;};
        void setMean(float m){mean = m;};
        void setStdDev(float s){stdDev = s;};

        Rect& getRegion(){return region;};
        TNode* getUL(){return UL;};
        TNode* getUR(){return UR;};
        TNode* getLR(){return LR;};
        TNode* getLL(){return LL;};
        vector<TNode*>& getMerged(){return merged;};
        bool isMerged(int i){return mergedB.at(i);};
        float getMean(){return mean;};
        float getStdDev(){return stdDev;};
};

TNode* Split(Mat& img, Rect R){
    TNode* root = new TNode(R);
    Scalar mean,stddev;
    meanStdDev(img(R),mean,stddev);
    root->setMean(mean[0]);
    root->setStdDev(stddev[0]);

    if(R.width > minWidth && root->getStdDev() > stdDevTH){
        Rect ul = Rect(R.x, R.y, R.height/2, R.width/2);
        root->setUL(Split(img,ul));

        Rect ur = Rect(R.x, R.y + R.width/2, R.height/2, R.width/2);
        root->setUR(Split(img,ur));

        Rect lr = Rect(R.x + R.height/2, R.y + R.width/2, R.height/2, R.width/2);
        root->setLR(Split(img,lr));

        Rect ll = Rect(R.x + R.height/2 ,R.y, R.height/2, R.width/2);
        root->setLL(Split(img,ll));
    }
    rectangle(img,R,Scalar(0));
    return root;

}

void merge(TNode* root){
    if(root->getRegion().width > minWidth && root->getStdDev() > stdDevTH){
        if(root->getUL()->getStdDev() <= stdDevTH && root->getUR()->getStdDev() <= stdDevTH ){
            root->addRegion(root->getUL()); root->setMergedB(0);
            root->addRegion(root->getUR()); root->setMergedB(1);
            if(root->getLR()->getStdDev() <= stdDevTH && root->getLL()->getStdDev() <= stdDevTH ){
                root->addRegion(root->getLR()); root->setMergedB(2);
                root->addRegion(root->getLL()); root->setMergedB(3);
            }
            else{
                merge(root->getLR());
                merge(root->getLL());
            }
        }
        else if(root->getLR()->getStdDev() <= stdDevTH && root->getLL()->getStdDev() <= stdDevTH ){
            root->addRegion(root->getLR()); root->setMergedB(2);
            root->addRegion(root->getLL()); root->setMergedB(3);
            if(root->getUL()->getStdDev() <= stdDevTH && root->getUR()->getStdDev() <= stdDevTH){
                root->addRegion(root->getUL()); root->setMergedB(0);
                root->addRegion(root->getUR()); root->setMergedB(1);
            }
            else{
                merge(root->getUL());
                merge(root->getUR());
            }
        }
        else if(root->getUL()->getStdDev() <= stdDevTH && root->getLL()->getStdDev() <= stdDevTH ){
            root->addRegion(root->getUL()); root->setMergedB(0);
            root->addRegion(root->getLL()); root->setMergedB(3);
            if(root->getUR()->getStdDev() <= stdDevTH && root->getLR()->getStdDev() <= stdDevTH ){
                root->addRegion(root->getUR()); root->setMergedB(1);
                root->addRegion(root->getLR()); root->setMergedB(2);
            }
            else{
                merge(root->getUR());
                merge(root->getLR());
            }
        }
        else if(root->getUR()->getStdDev() <= stdDevTH && root->getLR()->getStdDev() <= stdDevTH ){
            root->addRegion(root->getUR()); root->setMergedB(1);
            root->addRegion(root->getLR()); root->setMergedB(2);
            if(root->getUL()->getStdDev() <= stdDevTH && root->getLL()->getStdDev() <= stdDevTH ){
                root->addRegion(root->getUL()); root->setMergedB(0);
                root->addRegion(root->getLL()); root->setMergedB(3);
            }
            else{
                merge(root->getUL());
                merge(root->getLL());
            }
        }
        else{
            merge(root->getUL());
            merge(root->getUR());
            merge(root->getLR());
            merge(root->getLL());
        }


    }else{
        root->addRegion(root);
        root->setMergedB(0);root->setMergedB(1);root->setMergedB(2);root->setMergedB(3);
    }
}

void segment(TNode* root, Mat& dest){
    vector<TNode*> tmp = root->getMerged();
    if(!tmp.size()){
        segment(root->getUL(),dest);
        segment(root->getUR(),dest);
        segment(root->getLR(),dest);
        segment(root->getLL(),dest);
    }else{
        float val = 0.0f;
        for(auto m : tmp){
            val+=m->getMean();
        }
        val/=tmp.size();
        for(auto r : tmp){
            dest(r->getRegion()) = cvRound(val);
        }
        if(tmp.size()>1){
            if(!root->isMerged(0))
                segment(root->getUL(),dest);
            if(!root->isMerged(1))
                segment(root->getUR(),dest);
            if(!root->isMerged(2))
                segment(root->getLR(),dest);
            if(!root->isMerged(3))
                segment(root->getLL(),dest);
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

    Mat src = imread(argv[1], IMREAD_GRAYSCALE);
    minWidth = atoi(argv[2]);
    stdDevTH = atof(argv[3]);
    GaussianBlur(src,src,Size(5,5),0,0);
    Mat dest;
    // if (src.cols > 500 || src.rows > 500) {
    //     cv::resize(src, src, cv::Size(0, 0), 0.5, 0.5); // resize for speed
    // }
    int exp = log(min(src.cols,src.rows)) / log(2);
    int s = pow(2.0, float(exp));
    Rect square = Rect(0,0,s,s);
    src = src(square).clone();
    src.copyTo(dest);
    TNode* root = Split(src,Rect(0,0,src.rows,src.cols));
    merge(root);
    segment(root,dest);

    imshow("src", src);
    imshow("dest",dest);
    waitKey();

    return 0;
}