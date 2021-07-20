#include "opencv2/opencv.hpp"
#include <iostream>
#include "opencv2/xfeatures2d.hpp"

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;
//QR Square lenght = 78mm

const Size windowSize(480, 640);

vector<Point> trajectory_points;
vector<Point> qrPlain; 
RotatedRect QRRect;
Point3f realPoint;

void calibration()
{
    int CHECKERBOARD[2] = {6,9};
    //int criteria[3] = {TERM_CRITERIA_EPS + TERM_CRITERIA_MAX_ITER, 30, 0.001};
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
    Mat cameraMatrix,distCoeffs,R,T;
    calibrateCamera(objpoints, imgpoints, Size(gray.rows,gray.cols), cameraMatrix, distCoeffs, R, T);
    cout << "Matrix: " << cameraMatrix << endl;
    cout << "distCoeffs: " << distCoeffs << endl;
    cout << "Rotation vector: " << R << endl;
    cout << "Translation vector: " << T << endl;
    /*
     Mat intrinsics ((Mat1d(3, 3) << 656.7382002, 0, 302.2583313, 
                                    0, 646.92684674, 247.13654762, 
                                    0, 0, 1));
    Mat distCoeffs ((Mat1d(1, 5) << 0.31157188, -1.34319836, -0.0042151,  -0.00592606,  1.28977951));
    */

}

void recogniseStickersByThreshold(Mat &frame)
{

    Mat k = getStructuringElement(MORPH_RECT, Size(7,7));

    cv::Mat mask1(frame.size(),CV_8U);
    cvtColor(frame, mask1, COLOR_BGR2GRAY); 
    threshold(mask1, mask1, 200, 255, THRESH_BINARY);
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
    //mask = mask1;

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
        double epsilon = 0.05*arcLength(contours[biggest_contour], true);
        approxPolyDP(contours[biggest_contour], contours[biggest_contour], epsilon, true);
        qrPlain = contours[biggest_contour];
        if (qrPlain.size() != 4)
        {
            qrPlain.clear();
            return;
        }
        //drawContours(frame, contours, biggest_contour, Scalar(0 ,255, 0), 2);

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
        {
            for (size_t j = 0; j < 3; j++)
            {   
                swap(qrPlain[j], qrPlain[j+1]);
            }
        }

        for (auto e : qrPlain)
        {
            circle(frame, e, 3, Scalar(0,0,255), FILLED);
        }
        circle(frame, qrPlain[0], 3, Scalar(255,0,0), FILLED);

    }
    //frame = mask1;
}

void homo(Mat &frame)
{
    namedWindow("qr",WINDOW_AUTOSIZE);

    std::vector<cv::Point2f> imagePoints;
    std::vector<cv::Point2f> objectPoints;

    if (qrPlain.empty())
        return;
    imagePoints.push_back(qrPlain[0]);
    imagePoints.push_back(qrPlain[1]);
    imagePoints.push_back(qrPlain[2]);
    imagePoints.push_back(qrPlain[3]);
   /* for (auto e : imagePoints)
    {
        //circle(frame, e, 3, Scalar(0,0,255), FILLED);
    }
    //circle(frame, imagePoints[0], 3, Scalar(255,0,0), FILLED);*/

    objectPoints.push_back(Point2f(400, 0));
    objectPoints.push_back(Point2f(400, 400));
    objectPoints.push_back(Point2f(0, 400));
    objectPoints.push_back(Point2f(0, 0));

    Mat h = findHomography(imagePoints, objectPoints, RANSAC);
    Mat img_perspective(Size(600, 600), CV_8UC3);
    warpPerspective(frame, img_perspective, h, Size(600, 600));

    Mat k ((Mat1d(3, 3) << -1, -1, -1, -1, 9, -1, -1, -1, -1));
    filter2D(img_perspective, img_perspective, -1, k);
    //cvtColor(img_perspective, img_perspective, COLOR_BGR2GRAY);
    //cv::GaussianBlur(img_perspective, img_perspective, cv::Size(0, 0), 1);
    //cv::addWeighted(img_perspective, 1.5, img_perspective, -0.5, 1, img_perspective);

   // Canny(img_perspective, img_perspective, 40, 120);
    std::vector<cv::Point2f> qrPoints;
    QRCodeDetector detector;
    detector.detect(img_perspective, qrPoints);
    if (!qrPoints.empty())
    {
        for (auto e : qrPoints)
        {
        circle(img_perspective, e, 3, Scalar(0,0,255), FILLED);
        }
        circle(img_perspective, qrPoints[0], 3, Scalar(255,0,0), FILLED);
    }

    //threshold(img_perspective, img_perspective, 200, 225, THRESH_BINARY_INV);
    resizeWindow("qr", 600, 600);
    imshow("qr", img_perspective);

}

void getWorldCoordinates(Mat &frame)
{
    Mat intrinsics ((Mat1d(3, 3) << 656.7382002, 0, 302.2583313, 
                                    0, 646.92684674, 247.13654762, 
                                    0, 0, 1));
    Mat distCoeffs ((Mat1d(1, 5) << 0.31157188, -1.34319836, -0.0042151,  -0.00592606,  1.28977951));

    std::vector<cv::Point2f> imagePoints;
    std::vector<cv::Point3f> objectPoints;

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
        //cout << realPoint << endl;
    }

}

bool goodPoint(Point camera_pos)
{
    if (camera_pos.x < 600 && camera_pos.y < 600 && camera_pos.x > 0 &&camera_pos.y > 0)
    {
        if (trajectory_points.empty())
        {
            return true;
        }
        Point last = trajectory_points.back();
        double dst = norm(camera_pos - last);
        if (dst > 10 && dst < 100)
        {
            return true;
        }
    }
    return false;
}

void drawPlainTrajectory()
{
    Mat plain (600, 600, CV_8UC3, Scalar(0,0,0));
    namedWindow("Plain",WINDOW_AUTOSIZE); 
    rectangle(plain, Rect(Point(280,180), Point(320,220)), Scalar(255,255,255), FILLED);
    Point camera_pos (realPoint.y, realPoint.x);
    camera_pos.x = 300 + camera_pos.x / 1.5;
    camera_pos.y = 220 + camera_pos.y / 1.5;
    cout<< camera_pos << endl;
    circle(plain, camera_pos, 4, Scalar(0, 255, 0), FILLED);

    if (goodPoint(camera_pos))
    {
        trajectory_points.push_back(camera_pos);
    }
    
    if (trajectory_points.size() >= 2)

    {
        for (size_t i = 2; i < trajectory_points.size()-1; i++)
        {
            line(plain, trajectory_points[i-1], trajectory_points[i],Scalar(255,0,0), 1);
        }
    }
    imshow("Plain", plain);
}

void advancedHomo(Mat &frame)
{
    Mat img_object = imread( "./calibrate/qr_squish.png", IMREAD_GRAYSCALE );
    Mat img_scene = frame;
    if ( img_object.empty() || img_scene.empty() )
    {
        cout << "Could not open or find the image!\n" << endl;
        return;
    }
    //-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
    int minHessian = 400;
    Ptr<SIFT> detector = SIFT::create( minHessian );
    std::vector<KeyPoint> keypoints_object, keypoints_scene;
    Mat descriptors_object, descriptors_scene;
    detector->detectAndCompute( img_object, noArray(), keypoints_object, descriptors_object );
    detector->detectAndCompute( img_scene, noArray(), keypoints_scene, descriptors_scene );
    //-- Step 2: Matching descriptor vectors with a FLANN based matcher
    // Since SURF is a floating-point descriptor NORM_L2 is used
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    std::vector< std::vector<DMatch> > knn_matches;
    matcher->knnMatch( descriptors_object, descriptors_scene, knn_matches, 2 );
    //-- Filter matches using the Lowe's ratio test
    const float ratio_thresh = 0.75f;
    std::vector<DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
        {
            good_matches.push_back(knn_matches[i][0]);
        }
    }
    //-- Draw matches
    Mat img_matches;
    drawMatches( img_object, keypoints_object, img_scene, keypoints_scene, good_matches, img_matches, Scalar::all(-1),
                 Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    //-- Localize the object
    std::vector<Point2f> obj;
    std::vector<Point2f> scene;
    for( size_t i = 0; i < good_matches.size(); i++ )
    {
        //-- Get the keypoints from the good matches
        obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
        scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
    }
    if (scene.size() < 4)
    {
        return;
    }
    Mat H = findHomography( obj, scene, RANSAC );
    //-- Get the corners from the image_1 ( the object to be "detected" )
    std::vector<Point2f> obj_corners(4);
    obj_corners[0] = Point2f(0, 0);
    obj_corners[1] = Point2f( (float)img_object.cols, 0 );
    obj_corners[2] = Point2f( (float)img_object.cols, (float)img_object.rows );
    obj_corners[3] = Point2f( 0, (float)img_object.rows );
    std::vector<Point2f> scene_corners(4);
    if (H.empty())
    {
        return;
    }
    perspectiveTransform( obj_corners, scene_corners, H);
    //-- Draw lines between the corners (the mapped object in the scene - image_2 )
    line( img_matches, scene_corners[0] + Point2f((float)img_object.cols, 0),
          scene_corners[1] + Point2f((float)img_object.cols, 0), Scalar(0, 255, 0), 4 );
    line( img_matches, scene_corners[1] + Point2f((float)img_object.cols, 0),
          scene_corners[2] + Point2f((float)img_object.cols, 0), Scalar( 0, 255, 0), 4 );
    line( img_matches, scene_corners[2] + Point2f((float)img_object.cols, 0),
          scene_corners[3] + Point2f((float)img_object.cols, 0), Scalar( 0, 255, 0), 4 );
    line( img_matches, scene_corners[3] + Point2f((float)img_object.cols, 0),
          scene_corners[0] + Point2f((float)img_object.cols, 0), Scalar( 0, 255, 0), 4 );
    //-- Show detected matches
    imshow("Good Matches & Object detection", img_matches );
}


int main()
{
    //calibration();
    VideoCapture cap(0); // open the video file for reading
    if (!cap.isOpened()) return -1;
    namedWindow("MyVideo",WINDOW_AUTOSIZE); //create a window called "MyVideo"
    while(1) 
    {        
        Mat frame;
        bool flag =cap.read(frame);
        if (!flag) break;
        recogniseStickersByThreshold(frame);
        //advancedHomo(frame);
        //homo(frame);
        getWorldCoordinates(frame);

        drawPlainTrajectory();
        imshow("MyVideo", frame); //show the frame in "MyVideo" window
        if(waitKey(30) == 27)  break;
    }
    return 0;
}
