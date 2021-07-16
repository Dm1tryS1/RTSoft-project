#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

//QR Square lenght = 78mm

const size_t center_window_interval = 500;
const Size windowSize(600, 600);

void recogniseStickersByThreshold(Mat &frame ,Rect &QRRect)
{

    Mat k = getStructuringElement(MORPH_RECT, Size(7,7));

    cv::Mat mask1(frame.size(),CV_8U);
    cvtColor(frame, mask1, COLOR_BGR2GRAY); 
    threshold(mask1, mask1, 210, 255, THRESH_BINARY);
    morphologyEx(mask1, mask1, MORPH_CLOSE, k, Point(-1, -1), 2);

    cv::Mat mask2(frame.size(),CV_8U);
    cvtColor(frame, mask2, COLOR_BGR2GRAY); 
    //Canny(mask2, mask2, 60, 200);
    GaussianBlur(mask2, mask2, Size(9,9), 1.0);
    adaptiveThreshold(mask2, mask2, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 11,5);
   // threshold(mask2, mask2, 100, 255, THRESH_BINARY_INV);
    morphologyEx(mask2, mask2, MORPH_CLOSE, k);

    cv::Mat mask(frame.size(),CV_8U);
    bitwise_and(mask1, mask2, mask);

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
        QRRect = boundingRect(contours[biggest_contour]);
        //rectangle(frame,QRRect, Scalar(0,255,255),2);
        //drawContours(frame, contours, biggest_contour, Scalar(0 ,0, 255), 2);
    }
}

void sendConrolSignal(const Rect &QRRect, const Size windowSize)
{
    Point QRCenter (0, 0);
    QRCenter.x = QRRect.x + QRRect.width/2;
    QRCenter.y = QRRect.y + QRRect.height/2;
    cout << "x: " << QRCenter.x << " y: " << QRCenter.y << endl;
}

void getWorldCoordinates(Mat &frame ,Rect &QRRect)
{
    Mat intrinsics ((Mat1d(3, 3) << 656.7382002, 0, 302.2583313, 
                                    0, 646.92684674, 247.13654762, 
                                    0, 0, 1));
    Mat distCoeffs ((Mat1d(1, 5) << 0.31157188, -1.34319836, -0.0042151,  -0.00592606,  1.28977951));

    std::vector<cv::Point2f> imagePoints;
    std::vector<cv::Point3f> objectPoints;
    //img points are green dots in the picture
    /*
    QRCodeDetector detector;
    detector.detect(frame, imagePoints);
    for (auto e : imagePoints)
    {
        circle(frame, e, 3, Scalar(0,0,255), FILLED);
    }
    if (!imagePoints.empty())
        circle(frame, imagePoints[0], 3, Scalar(255,0,0), FILLED);
        */
    imagePoints.push_back(cv::Point2f(QRRect.x, QRRect.y));
    imagePoints.push_back(cv::Point2f(QRRect.x + QRRect.width, QRRect.y));
    imagePoints.push_back(cv::Point2f(QRRect.x + QRRect.width, QRRect.y + QRRect.height));
    imagePoints.push_back(cv::Point2f(QRRect.x, QRRect.y + QRRect.height));
    for (auto e : imagePoints)
    {
        circle(frame, e, 3, Scalar(0,0,255), FILLED);
    }
    //object points are measured in millimeters because calibration is done in mm also
    objectPoints.push_back(cv::Point3f(0., 0., 0.));
    objectPoints.push_back(cv::Point3f(78.,0.,0.));
    objectPoints.push_back(cv::Point3f(78.,78.,0.));
    objectPoints.push_back(cv::Point3f(0.,78.,0.));


    cv::Mat rvec(1,3,cv::DataType<double>::type);
    cv::Mat tvec(1,3,cv::DataType<double>::type);
    cv::Mat rotationMatrix(3,3,cv::DataType<double>::type);

    if (imagePoints.size() == 4){
        cv::solvePnP(objectPoints, imagePoints, intrinsics, distCoeffs, rvec, tvec);
        cv::Rodrigues(rvec,rotationMatrix);

        cv::Mat uvPoint = cv::Mat::ones(3,1,cv::DataType<double>::type); //u,v,1

        // image point
    //uvPoint.at<double>(0,0) = QRRect.x + QRRect.width/2; //got this point using mouse callback
    // uvPoint.at<double>(1,0) = QRRect.y + QRRect.height/2;

        uvPoint.at<double>(0,0) =   320; //got this point using mouse callback
        uvPoint.at<double>(1,0) = 480;
        circle(frame, Point(320,480), 4, Scalar(0,255,0), FILLED);

        cv::Mat tempMat, tempMat2;
        double s, zConst = 0;
        tempMat = rotationMatrix.inv() * intrinsics.inv() * uvPoint;
        tempMat2 = rotationMatrix.inv() * tvec;
        s = zConst + tempMat2.at<double>(2,0);
        s /= tempMat.at<double>(2,0);
        cv::Mat wcPoint = rotationMatrix.inv() * (s * intrinsics.inv() * uvPoint - tvec);

        Point3f realPoint(wcPoint.at<double>(0, 0), wcPoint.at<double>(1, 0), wcPoint.at<double>(2, 0)); 
        cout << realPoint << endl;
    }
    
}

int main() 
{
    VideoCapture cap(0); // open the video file for reading
    if (!cap.isOpened()) return -1;
    
    namedWindow("MyVideo",WINDOW_AUTOSIZE); //create a window called "MyVideo"
    while(1) 
    {        
        Mat frame;
        bool flag =cap.read(frame);
        if (!flag) break;
        Rect QRRect;
        recogniseStickersByThreshold(frame,QRRect);
      //  sendConrolSignal(QRRect, windowSize);
        getWorldCoordinates(frame, QRRect);
        imshow("MyVideo", frame); //show the frame in "MyVideo" window
        if(waitKey(30) == 27)  break;
    }
    return 0;
}
