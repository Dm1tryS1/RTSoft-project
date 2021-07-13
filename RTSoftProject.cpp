#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

void recogniseStickersByThreshold(Mat &frame ,vector<vector<Point>> &vLine)
{
    Mat edges;
    vector<vector<Point>> contours; 
    cvtColor(frame, edges, COLOR_BGR2GRAY);    
    Mat tmp(frame.size(),CV_8U); 

    Mat grad_x,grad_y;
    Mat abs_grad_x,abs_grad_y;
    
    GaussianBlur(frame,frame,Size(3,3),0,0,BORDER_DEFAULT);

    Sobel(frame, grad_x,CV_32F,1,0,-1);
    Sobel(frame, grad_y,CV_32F,0,1,-1);

    convertScaleAbs(grad_x,abs_grad_x);
    convertScaleAbs(grad_x,abs_grad_y);

    addWeighted(abs_grad_x,0.5,abs_grad_y,0.5,0,frame);

    


}

int main() 
{
    VideoCapture cap("1.mp4"); // open the video file for reading
    if ( !cap.isOpened() ) return -1;
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

        recogniseStickersByThreshold(frame,vLine);
        
        resize(frame,frame,Size(600,600));
        imshow("MyVideo", frame); //show the frame in "MyVideo" window
        if(waitKey(30) == 27)  break;
    }
    return 0;
}
