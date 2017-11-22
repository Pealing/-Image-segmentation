#include "Image.h"
#include "Saliency.h"

Image::Image(string imgDir)
{
	imgRaw = imread(imgDir);
	if(imgRaw.empty())
		cout <<"img error"<<endl;
	resize(imgRaw, imgRaw, cv::Size(imgRaw.cols/4,imgRaw.rows/4), (0, 0), (0, 0), cv::INTER_LINEAR);

}
void Image::GetSaliency()//获取显著性图像
{
	Saliency sal;

    vector<unsigned int >imgInput;
    vector<double> imgSal;

    //Mat to vector
    int nr = imgRaw.rows; // number of rows  
    int nc = imgRaw.cols; // total number of elements per line  
    if (imgRaw.isContinuous()) {
        // then no padded pixels  
        nc = nc*nr;
        nr = 1;  // it is now a 1D array  
    }

    for (int j = 0; j<nr; j++) {
        uchar* data = imgRaw.ptr<uchar>(j);
        for (int i = 0; i<nc; i++) {
            unsigned int t = 0;
            t += *data++;
            t <<= 8;
            t += *data++;
            t <<= 8;
            t += *data++;
            imgInput.push_back(t);

        }                
    }

    sal.GetSaliencyMap(imgInput, imgRaw.cols, imgRaw.rows, imgSal);

    //vector to Mat
    int index0 = 0;
    Mat imgout(imgRaw.size(), CV_64FC1);
    for (int h = 0; h < imgRaw.rows; h++) {
        double* p = imgout.ptr<double>(h);
        for (int w = 0; w < imgRaw.cols; w++) {
            *p++ = imgSal[index0++];
        }
    }
    normalize(imgout, imgout, 0, 1, NORM_MINMAX);
	imgout.convertTo(imgSaliency,CV_8UC1,1*255.0);
}
void Image::getBinMask( const Mat& comMask, Mat& binMask )  //Grabcut所需函数
{  
    binMask.create( comMask.size(), CV_8UC1 );  
    binMask = comMask & 1;  
} 
void Image::GrabCut()//GrabCut方法获取图片前景
{
	Mat bg;Mat fg;   
    Mat mask,res;  
	Mat img = imgRaw.clone();
	mask.create( img.size(), CV_8UC1);
    grabCut( img, mask, rectGrab, bg, fg, 1, 0 );
	Mat binMask;
	getBinMask( mask, binMask );
	img.copyTo( imgGrab, binMask );
}
int Image::GetMaxAreaIndex(vector<std::vector<cv::Point>> contours)//获取面积最大轮廓下标
{
	 
	double maxarea = -1;
	int maxindex = -1;
	for(int i=0;i<contours.size();i++)
	{
		double area = contourArea(contours[i]);
		if(area > maxarea)
		{
			maxarea = contourArea(contours[i]);
			maxindex = i;
		}
	}
	return maxindex;
}
void Image::GetGrabRect()//获取图片中最大的外接矩形，用于GrabCut
{
	vector<vector<Point>> contours;
	Mat imb = imgBin.clone();
	findContours(imb, contours, CV_RETR_LIST,  CV_CHAIN_APPROX_NONE); 
	Point leftTop(1e10,1e10);
	Point rightBottom(-1,-1);
	for(int i=0;i<contours.size();i++)
	{
		//将图中的几个长方形区域组合成一个大的长方形区域
		Rect boundRect;
		boundRect = boundingRect(Mat(contours[i])); 
		if(boundRect.x < leftTop.x)
			leftTop.x = boundRect.x;
		if(boundRect.y < leftTop.y)
			leftTop.y = boundRect.y;
		if(boundRect.x + boundRect.width >  rightBottom.x)
			rightBottom.x = boundRect.x + boundRect.width;
		if(boundRect.y + boundRect.height > rightBottom.y)
			rightBottom.y = boundRect.y + boundRect.height;
	}
	Rect resultrect(leftTop.x,leftTop.y,rightBottom.x-leftTop.x,rightBottom.y-leftTop.y);
	rectGrab = resultrect;
}
void Image::CutImage()//裁剪出海报位置
{
	GetSaliency();
	threshold(imgSaliency,imgBin,30,255,THRESH_BINARY);//获取显著性后的二值图
	erode(imgBin,imgBin,Mat(3,3,CV_8U),Point(-1,-1));
	dilate(imgBin,imgBin,Mat(13,13,CV_8U),Point(-1,-1));

	//*****************************************************************
	//先删减部分小区域杂乱区域
	Mat imb = imgBin.clone();
	std::vector<std::vector<cv::Point>> contours; 
    cv::findContours(imb, contours, CV_RETR_LIST,  CV_CHAIN_APPROX_NONE); 
	double maxarea = -1;
	int maxindex = -1;
	double SizeArea = imgRaw.cols * imgRaw.rows;
	Mat imb_new = Mat::zeros(imgRaw.rows, imgRaw.cols, CV_8UC1);
	for(int i=0;i<contours.size();i++)
	{
		double area = contourArea(contours[i]);
		if(area >= SizeArea/32)
		{
			drawContours(imb_new,contours,i,Scalar(255,255,255), -1,8);
			if(area > maxarea)
			{
				maxarea = contourArea(contours[i]);
				maxindex = i;
			}
		}
	}
	imgBin = imb_new.clone();//删减后的图复制回成员变量二值图中；
	//*******************************************************************
	GetGrabRect();
	GrabCut();

	//*******************************************************************
	//获取GrabCut之后的二值图
	Mat hsv;
	Mat imb2;
	Mat _imgBin;
	cvtColor(imgGrab,hsv,CV_BGR2HSV);
	vector<Mat> b;
	split(hsv,b);
	Mat h = b[0];
	Mat s = b[1];
	Mat v = b[2];

	inRange(h,0,180,b[0]);
	inRange(s,0,255,b[1]);
	inRange(v,0,46,b[2]);
	bitwise_and(b[0],b[1],imb2);
	bitwise_and(imb2,b[2],_imgBin);
	bitwise_not(_imgBin,imgBin);
	imshow("imgBin",imgBin);
	//***************************************************
	contours.clear();
	imb = imgBin.clone();
    cv::findContours(imb, contours, CV_RETR_LIST,  CV_CHAIN_APPROX_NONE);
	maxindex = GetMaxAreaIndex(contours);

	rectImg = boundingRect(Mat(contours[maxindex])); 
	double width = rectImg.width;  
    double height = rectImg.height;  
    double x = rectImg.x;  
    double y = rectImg.y; 

	RotatedRect rect=minAreaRect(contours[maxindex]);  
	Point2f P[4];  
    rect.points(P);   

	Mat Cut = imgRaw(rectImg);

	//************************************************
	//找到矩阵的4个点位置
	Point2f canvas[4];
	int leftTop = -1;
	int rightTop = -1;
	int leftBottom = -1;
	int rightBottom = -1;
	//左上
	for(int j=0;j<4;j++)
	{
		int xnum = 0;
		int ynum = 0;
		for(int i=0;i<4;i++)
		{
			if(i == j)
				continue;
			if(P[j].x < P[i].x)
				xnum++;
			if(P[j].y < P[i].y)
				ynum ++;
		}
		if(xnum >= 2 && ynum >= 2)
		{
			leftTop = j;
			break;
		}
	}
	cout << leftTop<<endl;
	canvas[0] = P[leftTop];
	//右上
	for(int j=0;j<4;j++)
	{
		if(leftTop == j)
			continue;
		if(P[j].x > P[leftTop].x)
		{
			int ynum = 0;
			for(int i=0;i<4;i++)
			{
				if(leftTop == i)
					continue;
				if(P[j].y < P[i].y)
					ynum++;
			}
			if(ynum >= 2)
			{
				rightTop = j;
				break;
			}
		}
	}
	canvas[1] = P[rightTop];
	//左下
	for(int j=0;j<4;j++)
	{
		if(leftTop == j || rightTop == j)
			continue;
		for(int i=0;i<4;i++)
		{
			if(leftTop == i || rightTop == i)
				continue;
			if(P[j].x < P[i].x)
				leftBottom = j;
		}
	}
	canvas[2] = P[leftBottom];
	//右下
	for(int j=0;j<4;j++)
	{
		if(leftTop == j || rightTop == j || leftBottom == j)
			continue;
		rightBottom = j;
	}
	canvas[3] = P[rightBottom];


	Point2f corners[4];
	corners[0] = Point2f(0,0);  
	corners[1] = Point2f(Cut.cols,0);  
    corners[2] = Point2f(0,Cut.rows);  
    corners[3] = Point2f(Cut.cols,Cut.rows);

	imgCut = Mat::zeros( Cut.rows, Cut.cols,CV_8UC3);;
	Mat M = getPerspectiveTransform(canvas,corners);
	warpPerspective(imgRaw, imgCut, M,imgCut.size());

	for(int j=0;j<=3;j++)  
    {  
            line(imgRaw,P[j],P[(j+1)%4],Scalar(255),2);  
    }  

	imshow("imgRaw",imgRaw);
      
}

Mat Image::GetImgCut()
{
	return imgCut;
}

