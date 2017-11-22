#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/features2d/features2d.hpp>
#include<opencv2/nonfree/nonfree.hpp>
#include<opencv2/legacy/legacy.hpp>
#include <iostream>
#include "Saliency.h"
#include "Image.h"
using namespace cv;
using namespace std;


int main()
{
	//getImage("../picture/12.jpg");
	//Mat img = imread("../picture/1.jpg");
	//resize(img,img,cv::Size(480,270), (0, 0), (0, 0), cv::INTER_LINEAR);
	//GetSaliencyImg(img);
	Image image("../picture/5.jpg");
	image.CutImage();
	Mat imgCut = image.GetImgCut();

	//sift特征检测  
    SiftFeatureDetector siftdtc;  
    vector<KeyPoint>kp;  
    siftdtc.detect(imgCut, kp);//到这里其实只是提取到了特征点的具体位置和角度，其保存在kp中  
    Mat outimg;  
    drawKeypoints(imgCut, kp, outimg);  
    imshow("image keypoints", outimg);//之前看别的人博客，到这就可以看到特征点画在图上了，一直以为那些点就是自己要用的那个，其实真正要用的在下边  
  
    SiftDescriptorExtractor extractor;//特征提取器  
    Mat descriptor; //这个描述符才是做实验要用的特征  
    extractor.compute(imgCut, kp, descriptor);  
	waitKey(0);
	system("pause");
	return 0;
}
