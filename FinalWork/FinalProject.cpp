#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

const double QRSideLength = 105.0;
Size windowSize;

vector<Point> trajectory_points;
vector<Point2f> imagePoints;

Mat intrinsics ((Mat1d(3, 3) << 1509.713490619002, 0, 699.3177375561561, 
                                0, 646.92684674, 247.13654762, 
                                0, 0, 1));
Mat distCoeffs ((Mat1d(1, 5) << 0.1486271150215541, -0.6491785880834027, 1.710689337182047e-05, -0.005221251049022714, -0.1356788841394454));

void drawPoints(Mat& frame, vector<Point2f> imgPoints);
void recogniseStickersByThreshold(Mat &frame);
void homo(Mat &frame);
void getWorldCoordinates(Mat &frame);
void drawPlainTrajectory(Point3f realPoint);
bool goodPoint(Point camera_pos);

void calibration()
{
    int CHECKERBOARD[2] = {7,5};
    // Создание вектора для хранения векторов 3D точек для каждого изображения шахматной доски
    vector<vector<Point3f>> objpoints; 
    // Создание вектора для хранения векторов 2D точек для каждого изображения шахматной доски
    vector<vector<Point2f>> imgpoints;

    // Определение мировых координат для 3D точек
    vector<Point3f> objp;

    for (int i=0;i<CHECKERBOARD[1];i++)
        for (int j=0;j<CHECKERBOARD[0];j++)
            objp.push_back(Point3f(j,i,0));
            
    Mat frame,gray;
    vector<Point2f> corner_pts;
    bool success;

    vector<String> images;
    string path = "./images/";
    glob(path, images);

    for(int i=0; i<images.size(); i++)
    {
        frame = imread(images[i]);
        cvtColor(frame,gray,COLOR_BGR2GRAY);
       
        success = findChessboardCorners(gray, Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);
        
        if(success)
        {
            TermCriteria criteria(TermCriteria::EPS | TermCriteria::COUNT, 30, 0.001);
            cornerSubPix(gray,corner_pts,Size(11,11), Size(-1,-1),criteria);
            objpoints.push_back(objp);
            imgpoints.push_back(corner_pts);
        }
        
    }
    Mat R,T;
    calibrateCamera(objpoints, imgpoints, Size(gray.rows,gray.cols), intrinsics, distCoeffs, R, T);
    cout << "Matrix: " << intrinsics << endl;
    cout << "distCoeffs: " << distCoeffs << endl;
}

void drawPoints(Mat& frame,vector<Point2f> imgPoints)
{
    for (auto e : imgPoints)
        circle(frame, e, 3, Scalar(0,0,255), FILLED);
    circle(frame, imgPoints[0], 3, Scalar(255,0,0), FILLED);
}

void recogniseStickersByThreshold(Mat &frame)
{
    Mat k = getStructuringElement(MORPH_RECT, Size(7,7));
  
    Mat mask1(frame.size(),CV_8U);
    cvtColor(frame, mask1, COLOR_BGR2GRAY); 
    threshold(mask1, mask1, 180, 255, THRESH_BINARY);
    morphologyEx(mask1, mask1, MORPH_CLOSE, k, Point(-1, -1), 3);

    Mat mask2(frame.size(),CV_8U);
    cvtColor(frame, mask2, COLOR_BGR2GRAY); 
    GaussianBlur(mask2, mask2, Size(9,9), 1.0);
    adaptiveThreshold(mask2, mask2, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 11,5);
    morphologyEx(mask2, mask2, MORPH_CLOSE, k, Point(-1, -1), 3);

    Mat mask(frame.size(),CV_8U);
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
        double epsilon = 0.05*arcLength(contours[biggest_contour], true);
        approxPolyDP(contours[biggest_contour], contours[biggest_contour], epsilon, true);

        imagePoints.clear();
        if (contours[biggest_contour].size() != 4) return;

        imagePoints.push_back(contours[biggest_contour][0]);
        imagePoints.push_back(contours[biggest_contour][1]);
        imagePoints.push_back(contours[biggest_contour][2]);
        imagePoints.push_back(contours[biggest_contour][3]);
        //drawPoints(frame , imagePoints);
        homo(frame);
    }
}

void homo(Mat &frame)
{
    Size homoWindowSize(200, 200);
    namedWindow("QRHomo",WINDOW_AUTOSIZE);

    vector<Point2f> objectPoints;
    double border = 30;

    objectPoints.push_back(Point2f(border, border));
    objectPoints.push_back(Point2f(homoWindowSize.width-border, border));
    objectPoints.push_back(Point2f(homoWindowSize.width-border, homoWindowSize.height-border));
    objectPoints.push_back(Point2f(border, homoWindowSize.height-border));

    Mat h = findHomography(imagePoints, objectPoints, RANSAC);
    Mat img_perspective(homoWindowSize, CV_8UC3);
    warpPerspective(frame, img_perspective, h, homoWindowSize);
    cvtColor(img_perspective, img_perspective, COLOR_BGR2GRAY);
    threshold(img_perspective, img_perspective, 100, 255, THRESH_BINARY);

    Mat k ((Mat1d(3, 3) << -1, -1, -1, -1, 9, -1, -1, -1, -1));
    filter2D(img_perspective, img_perspective, -1, k);

    vector<Point2f> qrPoints;
    QRCodeDetector detector;
    detector.detect(img_perspective, qrPoints);

    if (h.empty() || qrPoints.empty()) return;

    drawPoints(img_perspective, qrPoints);

    for (size_t i = 0; i < qrPoints.size(); i++)
    {
        Mat pt1 = (Mat_<double>(3,1) << qrPoints[i].x, qrPoints[i].y, 1);
        Mat pt2 = h.inv() * pt1;
        pt2 /= pt2.at<double>(2);
        imagePoints[i].x = pt2.at<double>(0);
        imagePoints[i].y = pt2.at<double>(1);
    }

    drawPoints(frame,imagePoints);
    
    resizeWindow("qr", homoWindowSize);
    imshow("QRHomo", img_perspective);

    getWorldCoordinates(frame);
}

void getWorldCoordinates(Mat &frame)
{
    vector<Point3f> objectPoints;
    Point3f realPoint;

    //object points are measured in millimeters because calibration is done in mm also
    objectPoints.push_back(Point3f(0., 0., 0.));
    objectPoints.push_back(Point3f(QRSideLength,0.,0.));
    objectPoints.push_back(Point3f(QRSideLength,QRSideLength,0.));
    objectPoints.push_back(Point3f(0.,QRSideLength,0.));


    Mat rvec(1,3,DataType<double>::type);
    Mat tvec(1,3,DataType<double>::type);
    Mat rotationMatrix(3,3,DataType<double>::type);

    if (imagePoints.size() == 4){
        solvePnP(objectPoints, imagePoints, intrinsics, distCoeffs, rvec, tvec);
        Rodrigues(rvec,rotationMatrix);

        Mat uvPoint = Mat::ones(3,1,DataType<double>::type); //u,v,1

        uvPoint.at<double>(0,0) = windowSize.width/2; 
        uvPoint.at<double>(1,0) = windowSize.height;
        circle(frame, Point(windowSize.width/2,windowSize.height), 4, Scalar(0,255,0), FILLED);

        Mat tempMat, tempMat2;
        double s, zConst = 0;
        tempMat = rotationMatrix.inv() * intrinsics.inv() * uvPoint;
        tempMat2 = rotationMatrix.inv() * tvec;
        s = zConst + tempMat2.at<double>(2,0);
        s /= tempMat.at<double>(2,0);
        Mat wcPoint = rotationMatrix.inv() * (s * intrinsics.inv() * uvPoint - tvec);

        realPoint.x = wcPoint.at<double>(0, 0);
        realPoint.y = wcPoint.at<double>(1, 0);
        realPoint.z = 0;

        drawPlainTrajectory(realPoint);
    }   
}

bool goodPoint(Point camera_pos)
{
    if (camera_pos.x < 600 && camera_pos.y < 600 && camera_pos.x > 0 &&camera_pos.y > 0)
    {
        if (trajectory_points.empty())
            return true;

        Point last = trajectory_points.back();
        double dst = norm(camera_pos - last);
        if (dst > 10 && dst < 100)
            return true;
    }
    return false;
}

void drawPlainTrajectory(Point3f realPoint)
{
    Point QRCentre(280,280);
    size_t scale = 2;
    Mat plain (600, 600, CV_8UC3, Scalar(0,0,0));
    namedWindow("Plain",WINDOW_AUTOSIZE); 

    rectangle(plain, Rect(QRCentre, Point(QRCentre.x+QRSideLength/scale,QRCentre.y+QRSideLength/scale)), Scalar(255,255,255), FILLED);
    line(plain, Point(10, 580), Point(10 + QRSideLength/scale, 580), Scalar(255,255,255), 2);
    line(plain, Point(10 + QRSideLength/scale, 580), Point(10 + QRSideLength/scale, 575), Scalar(255,255,255), 2);
    putText(plain, "105 mm", Point(5 + QRSideLength/scale, 570), FONT_HERSHEY_PLAIN , 1, Scalar(255,255,255), 1);

    Point camera_pos (realPoint.y, realPoint.x);
    camera_pos.x = QRCentre.x+camera_pos.x/scale;
    camera_pos.y = QRCentre.y+camera_pos.y/scale;
    cout<< camera_pos << endl;

    if (goodPoint(camera_pos))
    {
        trajectory_points.push_back(camera_pos);
        //avg_pts(trajectory_points, 1);
        circle(plain, camera_pos, 4, Scalar(0, 255, 0), FILLED);
    }
    
    if (trajectory_points.size() >= 2)
        for (size_t i = 2; i < trajectory_points.size()-1; i++)
        {
            line(plain, trajectory_points[i-1], trajectory_points[i],Scalar(255,0,0), 1);
            circle(plain, trajectory_points.back(), 4, Scalar(0, 255, 0), FILLED);
        }

    imshow("Plain", plain);
}

int main()
{
    calibration();
    VideoCapture cap("/home/rudakov/Downloads/sample3.mp4");

    if (!cap.isOpened()) return -1;

    Mat frame;
    bool flag = cap.read(frame);

    if (flag) windowSize = frame.size();

    namedWindow("MyVideo",WINDOW_AUTOSIZE); //create a window called "MyVideo"
    while(1) 
    {        
        flag = cap.read(frame);
        if (!flag) break;

        recogniseStickersByThreshold(frame);

        imshow("MyVideo", frame); //show the frame in "MyVideo" window
        if(waitKey(30) == 27)  break;
    }
    return 0;
}
{
    calibration();
    VideoCapture cap("/home/rudakov/Downloads/sample3.mp4");

    if (!cap.isOpened()) return -1;

    Mat frame;
    bool flag = cap.read(frame);

    if (flag) windowSize = frame.size();

    namedWindow("MyVideo",WINDOW_AUTOSIZE); //create a window called "MyVideo"
    while(1) 
    {        
        flag = cap.read(frame);
        if (!flag) break;

        recogniseStickersByThreshold(frame);

        imshow("MyVideo", frame); //show the frame in "MyVideo" window
        if(waitKey(30) == 27)  break;
    }
    return 0;
}