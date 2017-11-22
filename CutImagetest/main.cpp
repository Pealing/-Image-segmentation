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

	//sift�������  
    SiftFeatureDetector siftdtc;  
    vector<KeyPoint>kp;  
    siftdtc.detect(imgCut, kp);//��������ʵֻ����ȡ����������ľ���λ�úͽǶȣ��䱣����kp��  
    Mat outimg;  
    drawKeypoints(imgCut, kp, outimg);  
    imshow("image keypoints", outimg);//֮ǰ������˲��ͣ�����Ϳ��Կ��������㻭��ͼ���ˣ�һֱ��Ϊ��Щ������Լ�Ҫ�õ��Ǹ�����ʵ����Ҫ�õ����±�  
  
    SiftDescriptorExtractor extractor;//������ȡ��  
    Mat descriptor; //���������������ʵ��Ҫ�õ�����  
    extractor.compute(imgCut, kp, descriptor);  
	waitKey(0);
	system("pause");
	return 0;
}
