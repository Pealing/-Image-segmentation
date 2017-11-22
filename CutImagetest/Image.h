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
	Mat imgCut;//裁剪结果图片
	Rect rectGrab;
	Rect rectImg;//海报位置矩形

public:
	Image(string imgDir);
	void GetSaliency();//获取显著性图像
	void getBinMask(const Mat& comMask, Mat& binMask);//Grabcut所需函数
	void GrabCut();//GrabCut方法获取图片前景
	int GetMaxAreaIndex(vector<std::vector<cv::Point>> contours);//获取面积最大轮廓下标
	void GetGrabRect();
	void CutImage();//裁剪出海报位置
	Mat GetImgCut();


};