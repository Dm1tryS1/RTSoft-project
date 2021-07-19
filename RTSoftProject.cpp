#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

//QR Square lenght = 78mm

const Size windowSize(480, 640);

vector<Point2d> trajectory_points;
vector<Point> qrPlain; 
RotatedRect QRRect;
Point3f realPoint;

void recogniseStickersByThreshold(Mat &frame)
{

    Mat k = getStructuringElement(MORPH_RECT, Size(7,7));

    cv::Mat mask1(frame.size(),CV_8U);
    cvtColor(frame, mask1, COLOR_BGR2GRAY); 
    threshold(mask1, mask1, 220, 255, THRESH_BINARY);
    morphologyEx(mask1, mask1, MORPH_CLOSE, k, Point(-1, -1), 2);

    cv::Mat mask2(frame.size(),CV_8U);
    cvtColor(frame, mask2, COLOR_BGR2GRAY); 
    GaussianBlur(mask2, mask2, Size(9,9), 1.0);
    //Canny(mask2, mask2, 70, 200);
    //cornerHarris(mask2, mask2, 3,13, 0.07);
    // threshold(mask2, mask2, 100, 255, THRESH_BINARY_INV);
    adaptiveThreshold(mask2, mask2, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 11,5);
    morphologyEx(mask2, mask2, MORPH_CLOSE, k);

    cv::Mat mask(frame.size(),CV_8U);
    bitwise_and(mask1, mask2, mask);
    mask = mask1;
    
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

        QRRect = minAreaRect(contours[biggest_contour]);
        //rectangle(frame,r, Scalar(0,255,255),2);
        double epsilon = 0.01*arcLength(contours[biggest_contour], true);
        approxPolyDP(contours[biggest_contour], contours[biggest_contour], epsilon, true);
        //drawContours(frame, contours, biggest_contour, Scalar(0 ,255, 0), 2);
        qrPlain = contours[biggest_contour];
        
    }
    //frame = mask;
}

void homo(Mat &frame)
{
    namedWindow("qr",WINDOW_AUTOSIZE);

    Mat qr = imread("./calibrate/qr_front.png");
    std::vector<cv::Point2f> imagePoints;
    std::vector<cv::Point2f> objectPoints;
   
    if (qrPlain.empty())
        return;
    imagePoints.push_back(qrPlain[0]);
    imagePoints.push_back(qrPlain[1]);
    imagePoints.push_back(qrPlain[2]);
    imagePoints.push_back(qrPlain[3]);
    for (auto e : imagePoints)
    {
        circle(frame, e, 3, Scalar(0,0,255), FILLED);
    }
    circle(frame, imagePoints[0], 3, Scalar(255,0,0), FILLED);
    objectPoints.push_back(Point2f(400, 0));
    objectPoints.push_back(Point2f(400, 400*sqrt(2)));
    objectPoints.push_back(Point2f(0, 400*sqrt(2)));
    objectPoints.push_back(Point2f(0, 0));
    Mat h = findHomography(imagePoints, objectPoints, RANSAC);
    Mat img_perspective(frame.size(), CV_8UC3);
    warpPerspective(frame, img_perspective,h, frame.size());
    //Mat k ((Mat1d(3, 3) << -1, -1, -1, -1, 9, -1, -1, -1, -1));
    //cvtColor(img_perspective, img_perspective, COLOR_BGR2GRAY);
    //filter2D(img_perspective, img_perspective, -1, k);
    //cv::GaussianBlur(img_perspective, img_perspective, cv::Size(0, 0), 11);
   // Canny(img_perspective, img_perspective, 40, 120);
  /*  std::vector<cv::Point2f> qrPoints;
    QRCodeDetector detector;
    detector.detect(img_perspective, qrPoints);
    for (auto e : qrPoints)
    {
        circle(img_perspective, e, 3, Scalar(0,0,255), FILLED);
    }
    if (qrPoints.empty())
        return;
    circle(img_perspective, qrPoints[0], 3, Scalar(255,0,0), FILLED);
*/
    imshow("qr", img_perspective);

}

void sendConrolSignal(const Rect &QRRect, const Size windowSize)
{
    Point QRCenter (0, 0);
    QRCenter.x = QRRect.x + QRRect.width/2;
    QRCenter.y = QRRect.y + QRRect.height/2;
    cout << "x: " << QRCenter.x << " y: " << QRCenter.y << endl;
}

void getWorldCoordinates(Mat &frame)
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
    cv::Point2f imgPoints[4] = {Point2f(0, 0), Point2f(0, 0), Point2f(0, 0), Point2f(0, 0)};
    QRRect.points(imgPoints);
    if (imgPoints[0] == Point2f(0, 0))
        return;
    imagePoints.push_back(imgPoints[0]);
    imagePoints.push_back(imgPoints[1]);
    imagePoints.push_back(imgPoints[2]);
    imagePoints.push_back(imgPoints[3]);
/*    imagePoints.push_back(cv::Point2f(QRRect.x, QRRect.y));
    imagePoints.push_back(cv::Point2f(QRRect.x + QRRect.width, QRRect.y));
    imagePoints.push_back(cv::Point2f(QRRect.x + QRRect.width, QRRect.y + QRRect.height));
    imagePoints.push_back(cv::Point2f(QRRect.x, QRRect.y + QRRect.height));*/
    for (auto e : imagePoints)
    {
        circle(frame, e, 3, Scalar(0,0,255), FILLED);
    }
    circle(frame, imagePoints[0], 3, Scalar(255,0,0), FILLED);
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

        uvPoint.at<double>(0,0) = 320; 
        uvPoint.at<double>(1,0) = 480;
        circle(frame, Point(320,480), 4, Scalar(0,255,0), FILLED);

        cv::Mat tempMat, tempMat2;
        double s, zConst = 0;
        tempMat = rotationMatrix.inv() * intrinsics.inv() * uvPoint;
        tempMat2 = rotationMatrix.inv() * tvec;
        s = zConst + tempMat2.at<double>(2,0);
        s /= tempMat.at<double>(2,0);
        cv::Mat wcPoint = rotationMatrix.inv() * (s * intrinsics.inv() * uvPoint - tvec);

        realPoint.x = wcPoint.at<double>(0, 0);
        realPoint.y = wcPoint.at<double>(1, 0);
        realPoint.z = 0;
        cout << realPoint << endl;
    }
    
}

void drawPlainTrajectory()
{
    Mat plain (600, 600, CV_8UC3, Scalar(0,0,0));
    namedWindow("Plain",WINDOW_AUTOSIZE); 
    rectangle(plain, Rect(Point(280,280), Point(320,320)), Scalar(255,255,255), FILLED);

    if (trajectory_points.size() >= 2)
    {
        for (size_t i = 2; i < trajectory_points.size()-1; i++)
        {
            //line(plain, trajectory_points[i-1], trajectory_points[i],Scalar(0,255,0), 2);
        }
    }
    imshow("Plain", plain);
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
        recogniseStickersByThreshold(frame);
        homo(frame);

        //sendConrolSignal(QRRect, windowSize);
       //getWorldCoordinates(frame);




        //drawPlainTrajectory();
        imshow("MyVideo", frame); //show the frame in "MyVideo" window
        if(waitKey(30) == 27)  break;
    }
    return 0;
}
