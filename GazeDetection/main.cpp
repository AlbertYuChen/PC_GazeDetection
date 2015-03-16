//
//  Created by Chen Yu on 02/09/15.
//  Copyright (c) 2015 Chen Yu. All rights reserved.
//

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"

#include <iostream>
#include <stdio.h>
#include <fstream>
#include <sys/time.h>


#include "KalmanFilter.h"

using namespace std;
using namespace cv;

void detectAndDisplay( Mat frame );
Mat get_template(CascadeClassifier clasificator, Rect area, int size, bool left_right);
void match_eye(Rect area, Mat mTemplate, int type, bool left_right);

string work_path = "/Users/chenyu/Workspace/Xcode/GazeDetection/";

//String face_cascade_name = work_path + "haarcascades/haarcascade_frontalface_alt.xml";
String face_cascade_name = work_path + "lbpcascades/lbpcascade_frontalface.xml";
String eyes_cascade_name = work_path + "haarcascades/haarcascade_eye_tree_eyeglasses.xml";

bool gaze_control = true;
CascadeClassifier face_cascade;
CascadeClassifier mJavaDetectorEye;
string window_name = "Capture - Face detection";
RNG rng(12345);

Mat mRgba;
Mat mGray;
int learn_frames = 0;
Mat teplateR;
Mat teplateL;
float mRelativeFaceSize = 0.3;
int mAbsoluteFaceSize = 0;
int method = TM_CCOEFF_NORMED; //TM_CCORR_NORMED //TM_CCORR //TM_CCOEFF_NORMED //TM_CCOEFF //TM_SQDIFF // TM_SQDIFF_NORMED
bool LEFT_EYE = true;
bool RIGHT_EYE = false;
Point left_eye_position;
Point right_eye_position;
Point ground_left_eye_position;
Point ground_right_eye_position;
int eyes_on_off;
long start_millis;
bool take_record = false;
KalmanFilter kalmanfilter_x;
KalmanFilter kalmanfilter_y;



ofstream data_file;
ofstream eyes_file;


int main( int argc, const char** argv )
{
    timeval start_time;
    gettimeofday(&start_time, NULL);
    start_millis = (start_time.tv_sec * 1000) + (start_time.tv_usec / 1000);
    
    
    
    CvCapture* capture;
    Mat frame;
    namedWindow( window_name, WINDOW_AUTOSIZE );
    moveWindow(window_name, 400, 0);
    
    // Load the cascades training set
    if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
    if( !mJavaDetectorEye.load( eyes_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
    
    // Read the video stream from webcame
    capture = cvCaptureFromCAM( -1 );
    cvSetCaptureProperty( capture, CV_CAP_PROP_FRAME_WIDTH, 640 );
    cvSetCaptureProperty( capture, CV_CAP_PROP_FRAME_HEIGHT, 480 );
    
    data_file.open (work_path + "eyes_distance.csv");
    eyes_file.open (work_path + "eyes.csv");
    
    if( capture )
    {
        while( true )
        {
            mRgba = cvQueryFrame( capture );
            // mirror image
            flip(mRgba,mRgba,1);
            cvtColor( mRgba, mGray, CV_BGR2GRAY );
            equalizeHist( mGray, mGray );
            
            // Apply the classifier to the frame
            if( !mRgba.empty() ) {
                detectAndDisplay( mRgba );
            }
            else { printf(" --(!) No captured frame -- Break!"); break; }
            int c = waitKey(10);
            if( (char)c == 'q' ){
                break;
            }
        }
    }
    
    data_file.close();
    eyes_file.close();
    return 0;
}

//main detection function
void detectAndDisplay( Mat mRgba )
{
    std::vector<Rect> faces;
    
    if (mAbsoluteFaceSize == 0) {
        int height = mGray.rows;
        if (height * mRelativeFaceSize > 0) {
            mAbsoluteFaceSize = height * mRelativeFaceSize;
        }
    }
    
    // Detect faces, in green box
    face_cascade.detectMultiScale( mGray, faces, 1.1,
                                  2, 2, // objdetect.CV_HAAR_SCALE_IMAGE
                                  Size(30, 30) );
    
    int eyes_area_blue_box_width = 0;
    Rect eyearea_right;
    Rect eyearea_left;
    
    for( size_t i = 0; i < faces.size(); i++){
        Point face_start( faces[i].x , faces[i].y );
        Point face_end( faces[i].x + faces[i].width, faces[i].y + faces[i].height );
        rectangle(mRgba, face_start, face_end, Scalar(0, 255, 0), 2);
        
        
        Rect r = faces[i];
        
        eyes_area_blue_box_width = r.width;
        // calculate the eyes area, in blue box
        eyearea_right = Rect(r.x + r.width / 10, (r.y + (r.height / 3.5)),
                             (r.width - 2 * r.width / 10) / 2, (r.height / 4.0));
        
        eyearea_left = Rect(r.x + r.width / 10 + (r.width - 2 * r.width / 10) / 2, (r.y + (r.height / 3.5)),
                            (r.width - 2 * r.width / 10) / 2, (r.height / 4.0));
        
        rectangle(mRgba, eyearea_left.tl(), eyearea_left.br(),
                  Scalar(0, 0, 205, 255), 1);
        rectangle(mRgba, eyearea_right.tl(), eyearea_right.br(),
                  Scalar(0, 0, 205, 255), 1);
        
        match_eye(eyearea_right, teplateR, method, RIGHT_EYE);
        match_eye(eyearea_left, teplateL, method, LEFT_EYE);
        
    }
    
    putText(mRgba, "init left:" + to_string(ground_left_eye_position.x) + ","
            + to_string(ground_left_eye_position.y), cvPoint(50,30),
            FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(250,0,0), 1, CV_AA);
    
    putText(mRgba, "init right:" + to_string(ground_right_eye_position.x) + ","
            + to_string(ground_right_eye_position.y), cvPoint(50,60),
            FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(250,0,0), 1, CV_AA);
    
    
    int left_delta_x = left_eye_position.x - ground_left_eye_position.x;
    int left_delta_y = left_eye_position.y - ground_left_eye_position.y;
    
    int right_delta_x = right_eye_position.x - ground_right_eye_position.x;
    int right_delta_y = right_eye_position.y - ground_right_eye_position.y;
    
    int abs_delta_x = abs(right_delta_x + left_delta_x);
    int abs_delta_y = abs(right_delta_y + right_delta_y);
    
    // kalman filter add here
    kalmanfilter_x.step(abs_delta_x);
    kalmanfilter_y.step(abs_delta_y);
    
    int kal_abs_delta_x = kalmanfilter_x.getcurrentstate();
    int kal_abs_delta_y = kalmanfilter_y.getcurrentstate();
    
    data_file << to_string( abs_delta_x ) + "," + to_string( abs_delta_y ) + ","
    + to_string(kal_abs_delta_x) + "," + to_string(kal_abs_delta_y)<< endl;
    
    
    // the face detection judger
    if (faces.size() == 0) {
        eyes_on_off = 0;
        putText(mRgba, "No face", Point(100, 400), FONT_ITALIC, 1, Scalar(255, 0, 255, 255), 1, CV_AA);
    } else if(faces.size() > 1){
        eyes_on_off = 0;
        putText(mRgba, "So many faces", Point(100, 400), FONT_ITALIC, 1, Scalar(255, 0, 255, 255), 1, CV_AA);
    }else if (kal_abs_delta_x > eyes_area_blue_box_width / 25 ||
              kal_abs_delta_y > eyes_area_blue_box_width / 25) {
        eyes_on_off = 0;
        putText(mRgba, "EYES OFF", Point(200, 400), FONT_ITALIC, 1, Scalar(0, 0, 255, 255), 2, CV_AA);
    } else {
        eyes_on_off = 1;
        putText(mRgba, "EYES ON", Point(200, 400), FONT_ITALIC, 1, Scalar(0, 255, 0, 255), 2, CV_AA);
        
    }
    
    
    putText(mRgba,  "delta_left x:" + to_string(left_delta_x)  + " y:" + to_string(left_delta_y) ,
            Point(100, 100), FONT_HERSHEY_SIMPLEX, 1,  Scalar(0, 0, 255, 255), 1, CV_AA);
    
    putText(mRgba, "delta_right x:" + to_string(right_delta_x) + " y:" + to_string(right_delta_y) ,
            Point(100, 150), FONT_HERSHEY_SIMPLEX, 1,  Scalar(0, 0, 255, 255), 1, CV_AA);
    
    putText(mRgba, "delta x:" + to_string(abs(right_delta_x + left_delta_x)) + " y:" + to_string( abs(right_delta_y + right_delta_y)),
            Point(100, 200), FONT_HERSHEY_SIMPLEX, 1,  Scalar(0, 0, 255, 255), 1, CV_AA);
    
    
    if (take_record) {
        putText(mRgba, "Taking Record",
                Point(350, 400), FONT_HERSHEY_SIMPLEX, 1,  Scalar(0, 0, 255, 255), 1, CV_AA);
        
        // get the system time and record eyes on or off
        timeval time;
        gettimeofday(&time, NULL);
        long millis = (time.tv_sec * 1000) + (time.tv_usec / 1000);
        //    cout << millis - start_millis <<endl;
        
        eyes_file << to_string(millis - start_millis)+ "," + to_string(eyes_on_off) <<endl;

    }else{
        putText(mRgba, "Not Taking Record",
                Point(350, 400), FONT_HERSHEY_SIMPLEX, 1,  Scalar(0, 0, 255, 255), 1, CV_AA);
    }
    
    int c = waitKey(10);
    if ((char)c == 'f' ){
        //            teplateL = imread(work_path + "teplateL.jpg",CV_LOAD_IMAGE_GRAYSCALE);
        //            teplateR = imread(work_path + "teplateR.jpg",CV_LOAD_IMAGE_GRAYSCALE);
        
        teplateR = get_template(mJavaDetectorEye, eyearea_right, 25, false);
        teplateL = get_template(mJavaDetectorEye, eyearea_left, 25, true);
        
        if (teplateR.cols > 0 && teplateL.rows > 0){
            imshow( "left eye", teplateL );
            moveWindow("left eye", 170, 10);
            
            imshow( "right eye", teplateR );
            moveWindow("right eye", 170, 100);
            //            imwrite( work_path + "teplateR.jpg" , teplateR );
            //            imwrite( work_path + "teplateL.jpg" , teplateL );
        }
    }else if((char)c == ' ' ){
        take_record = !take_record;
        
        
        
    }
    
    imshow( window_name, mRgba );
}

Mat get_template(CascadeClassifier clasificator, Rect area, int size, bool left_right) {
    Mat temp;
    Mat mROI = mGray(area);
    std::vector<Rect> eyes;
    Point iris = Point();
    Rect eye_template = Rect();
    clasificator.detectMultiScale(mROI, eyes, 1.15, 2,
                                  CASCADE_FIND_BIGGEST_OBJECT | CASCADE_SCALE_IMAGE,
                                  Size(30, 30), Size());
    
    for (int i = 0; i < eyes.size(); i++) {
        Rect e = eyes[i];
        e.x = area.x + e.x;
        e.y = area.y + e.y;
        Rect eye_only_rectangle = Rect(e.tl().x, (e.tl().y + e.height * 0.4), e.width, (e.height * 0.6));
        
        mROI = mGray(eye_only_rectangle);
        Mat vyrez = mRgba(eye_only_rectangle);
        
        double minVal; double maxVal; Point minLoc; Point maxLoc;
        
        minMaxLoc( mROI, &minVal, &maxVal, &minLoc, &maxLoc, Mat() );
        
        circle(vyrez, minLoc, 2, Scalar(255, 255, 255, 255), 2);
        iris.x = minLoc.x + eye_only_rectangle.x;
        iris.y = minLoc.y + eye_only_rectangle.y;
        
        eye_template = Rect(iris.x - size / 2, iris.y - size / 2, size, size);
        
        rectangle(mRgba, eye_template.tl(), eye_template.br(), Scalar(255, 0, 0, 255), 3);
        temp = mGray(eye_template);
        
        if (left_right) {
            ground_left_eye_position = * new Point(eye_template.tl().x - area.x, eye_template.tl().y - area.y);
        }else{
            ground_right_eye_position = * new Point(eye_template.tl().x - area.x, eye_template.tl().y - area.y);
        }
    }
    
    return temp.clone();// not temp, because Mat is a pointer.
}

void match_eye(Rect area, Mat mTemplate, int type, bool left_right) {
    Point matchLoc;
    Mat mROI = mGray(area);
    
    int result_cols = mROI.cols - mTemplate.cols + 1;
    int result_rows = mROI.rows - mTemplate.rows + 1;
    
    //    if (mTemplate.cols == 0 || mTemplate.rows == 0) {
    //        return;
    //    }
    
    if (mTemplate.empty()) {
        return;
    }
    
    if (left_right) {
        //        putText(mRgba, "left:" + to_string(area.x) + "," + to_string(area.y), cvPoint(30,30),
        //                FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200,200,250), 1, CV_AA);
        imshow( "left eye", mTemplate );
    }else{
        //        putText(mRgba, "right:" + to_string(area.x) + "," + to_string(area.y), cvPoint(30,60),
        //                FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200,200,250), 1, CV_AA);
        imshow( "right eye", mTemplate );
    }
    
    if (result_cols <= 0 || result_rows <= 0) {
        return;
    }
    
    Mat mResult;
    mResult.create( result_cols, result_rows, CV_32FC4 );
    
    
    matchTemplate(mROI, mTemplate, mResult, type);
    normalize( mResult, mResult, 0, 1, NORM_MINMAX, -1, Mat() );
    
    double minVal; double maxVal; Point minLoc; Point maxLoc;
    
    minMaxLoc( mResult, &minVal, &maxVal, &minLoc, &maxLoc, Mat() );
    
    // there is difference in matching methods - best match is max/min value
    if (type == TM_SQDIFF || type == TM_SQDIFF_NORMED) {
        matchLoc = minLoc;
    } else {
        matchLoc = maxLoc;
    }
    
    Point matchLoc_tx = Point(matchLoc.x + area.x, matchLoc.y + area.y);
    
    Point matchLoc_ty = Point(matchLoc.x + mTemplate.cols + area.x, matchLoc.y + mTemplate.rows + area.y);
    
    if (left_right == LEFT_EYE) {
        left_eye_position = matchLoc;
        putText(mRgba, "left:" + to_string(left_eye_position.x) + ","
                + to_string(left_eye_position.y), cvPoint(30,90),
                FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(250,0,0), 1, CV_AA);
    } else {
        right_eye_position = matchLoc;
        putText(mRgba, "right:" + to_string(right_eye_position.x) + ","
                + to_string(right_eye_position.y), cvPoint(30,120),
                FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(250,0,0), 1, CV_AA);
    }
    
    rectangle(mRgba, matchLoc_tx, matchLoc_ty, Scalar(255, 255, 0,255),1);
    
}













