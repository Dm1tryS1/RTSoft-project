#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

const Size windowSize(480, 640);

vector<Point2d> trajectory_points;
vector<Point> qrPlain; 
RotatedRect QRRect;
Point3f realPoint;
Mat intrinsics, distCoeffs;

void Calibration()
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

    /*
    Matrix: [1509.713490619002, 0, 699.3177375561561;
             0, 1512.261884355301, 952.0531126357746;
             0, 0, 1]
    distCoeffs: [0.1486271150215541, -0.6491785880834027, 1.710689337182047e-05, -0.005221251049022714, -0.1356788841394454]
    */
}

void findCoord(Mat &frame)
{
    size_t min_dist = sqrt(qrPlain[0].x * qrPlain[0].x + qrPlain[0].y * qrPlain[0].y);
    size_t min_ind = 0;
    for (size_t i = 1; i < 4; i++)
    {
        size_t dist = sqrt(qrPlain[i].x * qrPlain[i].x + qrPlain[i].y * qrPlain[i].y);
        if (dist < min_dist)
        {
            min_dist = dist;
            min_ind = i;
        }
    }

    for (size_t i = 0; i < min_ind; i++)
        for (size_t j = 0; j < 3; j++)        
            swap(qrPlain[j], qrPlain[j+1]);

    for (auto e : qrPlain)
        circle(frame, e, 3, Scalar(0,0,255), FILLED);
        
    circle(frame, qrPlain[0], 3, Scalar(255,0,0), FILLED);
}

void recogniseStickersByThreshold(Mat &frame)
{
    Mat k = getStructuringElement(MORPH_RECT, Size(7,7));
  
    Mat mask1(frame.size(),CV_8U);
    cvtColor(frame, mask1, COLOR_BGR2GRAY); 
    threshold(mask1, mask1, 200, 255, THRESH_BINARY);
    morphologyEx(mask1, mask1, MORPH_CLOSE, k, Point(-1, -1), 2);

    Mat mask2(frame.size(),CV_8U);
    cvtColor(frame, mask2, COLOR_BGR2GRAY); 
    GaussianBlur(mask2, mask2, Size(9,9), 1.0);
    adaptiveThreshold(mask2, mask2, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 11,5);
    morphologyEx(mask2, mask2, MORPH_CLOSE, k);

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
        QRRect = minAreaRect(contours[biggest_contour]);
        double epsilon = 0.05*arcLength(contours[biggest_contour], true);
        approxPolyDP(contours[biggest_contour], contours[biggest_contour], epsilon, true);
        qrPlain = contours[biggest_contour];
        if (qrPlain.size() != 4)
        {
            qrPlain.clear();
            return;
        }
        //findCoord(frame);
    }
}

void homo(Mat &frame)
{
    namedWindow("qr",WINDOW_AUTOSIZE);

    vector<Point2f> imagePoints;
    vector<Point2f> objectPoints;
   
    if (qrPlain.empty())
        return;

    imagePoints.push_back(qrPlain[0]);
    imagePoints.push_back(qrPlain[1]);
    imagePoints.push_back(qrPlain[2]);
    imagePoints.push_back(qrPlain[3]);

    objectPoints.push_back(Point2f(50, 50));
    objectPoints.push_back(Point2f(50, 250));
    objectPoints.push_back(Point2f(250, 250));
    objectPoints.push_back(Point2f(250, 50));

    Mat h = findHomography(imagePoints, objectPoints, RANSAC);
    Mat img_perspective(frame.size(), CV_8UC3);
    warpPerspective(frame, img_perspective,h, frame.size());

    Mat k ((Mat1d(3, 3) << -1, -1, -1, -1, 9, -1, -1, -1, -1));
    filter2D(img_perspective, img_perspective, -1, k);

    vector<Point2f> qrPoints;
    QRCodeDetector detector;
    detector.detect(img_perspective, qrPoints);
    if (!qrPoints.empty())
    {
        for (auto e : qrPoints)
            circle(img_perspective, e, 3, Scalar(0,0,255), FILLED);
        circle(img_perspective, qrPoints[0], 3, Scalar(255,0,0), FILLED);
    }
    else
    cout<<"No";
    resizeWindow("qr", 800, 800);
    imshow("qr", img_perspective);
}

void getWorldCoordinates(Mat &frame)
{
    Mat intrinsics ((Mat1d(3, 3) << 1509.713490619002, 0, 699.3177375561561, 
                                    0, 646.92684674, 247.13654762, 
                                    0, 0, 1));
    Mat distCoeffs ((Mat1d(1, 5) << 0.1486271150215541, -0.6491785880834027, 1.710689337182047e-05, -0.005221251049022714, -0.1356788841394454));

    vector<Point2f> imagePoints;
    vector<Point3f> objectPoints;
      
    if (qrPlain.empty())
        return;
    imagePoints.push_back(qrPlain[0]);
    imagePoints.push_back(qrPlain[1]);
    imagePoints.push_back(qrPlain[2]);
    imagePoints.push_back(qrPlain[3]);

    for (auto e : imagePoints)
        circle(frame, e, 3, Scalar(0,0,255), FILLED);
    circle(frame, imagePoints[0], 3, Scalar(255,0,0), FILLED);

    //object points are measured in millimeters because calibration is done in mm also
    objectPoints.push_back(Point3f(0., 0., 0.));
    objectPoints.push_back(Point3f(55.,0.,0.));
    objectPoints.push_back(Point3f(55.,55.,0.));
    objectPoints.push_back(Point3f(0.,55.,0.));


    Mat rvec(1,3,DataType<double>::type);
    Mat tvec(1,3,DataType<double>::type);
    Mat rotationMatrix(3,3,DataType<double>::type);

    if (imagePoints.size() == 4){
        solvePnP(objectPoints, imagePoints, intrinsics, distCoeffs, rvec, tvec);
        Rodrigues(rvec,rotationMatrix);

        Mat uvPoint = Mat::ones(3,1,DataType<double>::type); //u,v,1

        uvPoint.at<double>(0,0) = frame.cols/2; 
        uvPoint.at<double>(1,0) = frame.rows;
        circle(frame, Point(frame.cols/2,frame.rows), 4, Scalar(0,255,0), FILLED);

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
        if ((dst > 20 && dst < 100))
            return true;        
    }
    return false;
}

void drawPlainTrajectory()
{
    Mat plain (600, 600, CV_8UC3, Scalar(0,0,0));
    namedWindow("Plain",WINDOW_AUTOSIZE); 
    rectangle(plain, Rect(Point(280,280), Point(320,320)), Scalar(255,255,255), FILLED);
    Point camera_pos (realPoint.y, realPoint.x);
    //camera_pos.x = 300 + camera_pos.x/1.5;
    //camera_pos.y = 300 + camera_pos.y/1.5;
    cout<< camera_pos << endl;
    circle(plain, camera_pos, 4, Scalar(0, 255, 0), FILLED);

    if (goodPoint(camera_pos))
        trajectory_points.push_back(camera_pos);
    
    if (trajectory_points.size() > 1)
        for (size_t i = 2; i < trajectory_points.size()-1; i++)
            line(plain, trajectory_points[i-1], trajectory_points[i],Scalar(255,0,0), 2);

    imshow("Plain", plain);
}





int main()
{
    Calibration();
    VideoCapture cap("test.mp4"); // open the video file for reading
    if (!cap.isOpened()) return -1; 
    namedWindow("MyVideo",WINDOW_AUTOSIZE); //create a window called "MyVideo"
    while(1) 
    {        
        Mat frame;
        bool flag =cap.read(frame);
        if (!flag) break;
        recogniseStickersByThreshold(frame);
        homo(frame);
        getWorldCoordinates(frame);
        drawPlainTrajectory();
        imshow("MyVideo", frame); //show the frame in "MyVideo" window
        if(waitKey(30) == 27)  break;
    }
    return 0;
}