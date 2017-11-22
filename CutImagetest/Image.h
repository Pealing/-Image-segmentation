#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include "Saliency.h"
using namespace cv;
using namespace std;

class Image
{
protected:
	Mat imgRaw;
	Mat imgSaliency;
	Mat imgGrab;
	Mat imgBin;
	Mat imgCut;//�ü����ͼƬ
	Rect rectGrab;
	Rect rectImg;//����λ�þ���

public:
	Image(string imgDir);
	void GetSaliency();//��ȡ������ͼ��
	void getBinMask(const Mat& comMask, Mat& binMask);//Grabcut���躯��
	void GrabCut();//GrabCut������ȡͼƬǰ��
	int GetMaxAreaIndex(vector<std::vector<cv::Point>> contours);//��ȡ�����������±�
	void GetGrabRect();
	void CutImage();//�ü�������λ��
	Mat GetImgCut();


};