#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

const size_t center_window_interval = 500;
const Size windowSize(600, 600);

//void recogniseStickersByThreshold(Mat &frame ,vector<vector<Point>> &vLine)
void recogniseStickersByThreshold(Mat &frame ,Point &QRCenter)
{

/* 
    Mat grad_x,grad_y;
    Mat abs_grad_x,abs_grad_y;
    
    GaussianBlur(frame,frame,Size(3,3),0,0,BORDER_DEFAULT);

    Sobel(frame, grad_x,CV_32F,1,0,-1);
    Sobel(frame, grad_y,CV_32F,0,1,-1);

    convertScaleAbs(grad_x,abs_grad_x);
    convertScaleAbs(grad_x,abs_grad_y);

    addWeighted(abs_grad_x,0.5,abs_grad_y,0.5,0,frame);
*/
// Вовин вариант

    cv::Mat mask(frame.size(),CV_8U);
    cvtColor(frame, mask, COLOR_BGR2GRAY); 
    threshold(mask, mask, 180, 255, THRESH_BINARY);

/*  Возможное (но это не точно) решение для 3-го (проблемного) видео 
    (выделить черную часть кода и совместить с белой чтобы отличить код от белых штор)

    cv::Mat mask2(frame.size(),CV_8U);
    cvtColor(frame, mask2, COLOR_BGR2GRAY); 
    threshold(mask2, mask2, 75, 255, THRESH_BINARY_INV);
*/

    vector<vector<Point>> contours; 
    vector<Vec4i> hierarchy;
    findContours(mask, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    size_t biggest_contour = -1;
    size_t max_area = 0;
    for(size_t i = 0; i < contours.size(); i++)
    {
        size_t cur_area = contourArea(contours[i]);
        if (cur_area > max_area) 
        {
            biggest_contour = i;
            max_area = cur_area;
        }
    }
    if (biggest_contour != -1)
    {
        Rect rect = boundingRect(contours[biggest_contour]);
        rectangle(frame,rect, Scalar(0,255,255),2);
        QRCenter.x = rect.x + rect.width/2;
        QRCenter.y = rect.y + rect.height/2;
    }
}

void sendConrolSignal(const Point &QRCenter, const Size windowSize)
{

}

int main() 
{
    VideoCapture cap("videos/4.mp4"); // open the video file for reading
    if (!cap.isOpened()) return -1;
    vector<vector<Point>> vLine;
    vector<Point> x;
    for (int i=0;i<2;i++)
    {
        vLine.push_back(x);
    }

    namedWindow("MyVideo",WINDOW_AUTOSIZE); //create a window called "MyVideo"
    while(1) 
    {        
        Mat frame;
        bool flag =cap.read(frame);
        if (!flag) break;
        Point QRCenter;
        recogniseStickersByThreshold(frame,QRCenter);
        cout << "x: " << QRCenter.x << " y: " << QRCenter.y << endl;
        sendConrolSignal(QRCenter, windowSize);
        resize(frame,frame,windowSize);
        imshow("MyVideo", frame); //show the frame in "MyVideo" window
        if(waitKey(30) == 27)  break;
    }
    return 0;
}
