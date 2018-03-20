#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;

int main(int argc, char** argv){
  if(argc != 2){
    cout<<" Usage: "<<endl;
    return -1;
  }

  Mat inputImage;
  inputImage = imread(argv[1]);

  if(!inputImage.data){
    cout<<"Could not read image or find image"<<endl;
    return -1;
  }

  Mat final = inputImage.clone();
  Mat grayImage,grayBlur,grayImageOrg;
  cvtColor(inputImage,grayImageOrg,COLOR_BGR2GRAY);
  // blur(grayImageOrg,grayImageOrg,Size(7,7));
  cv::GaussianBlur(grayImageOrg, grayBlur, cv::Size(0, 0), 3);
  cv::addWeighted(grayImageOrg, 1.5, grayBlur, -0.5, 0, grayImage);

  int morph_size = 1;
  Mat element = getStructuringElement( MORPH_RECT, Size( 2*morph_size + 1, 2*morph_size+1 ), Point( morph_size, morph_size ) );
  // blur(grayImage,grayImage,Size(7,7));
  // morphologyEx(grayImage,grayImage,MORPH_CLOSE,element,Point(-1,-1),1);
  // erode(grayImage,grayImage,element,Point(-1,-1),2,1,1);
  blur(grayImage,grayImage,Size(3,3));
  // erode(grayImage,grayImage,element,Point(-1,-1),2,1,1);
  // blur(grayImage,grayImage,Size(3,3));
  // blur(grayImage,grayImage,Size(3,3));
  // blur(grayImage,grayImage,Size(3,3));
  // erode(grayImage,grayImage,element,Point(-1,-1),2,1,1);

  vector<vector<Point> > contours;
  vector<Vec4i> hierarchy;
  int sum =0;

  //// Adapted from https://www.learnopencv.com/blob-detection-using-opencv-python-c/
  SimpleBlobDetector::Params params;
  // Filter by Area.
  params.filterByArea = true;
  params.minArea = 50;

  // Filter by Circularity
  params.filterByCircularity = true;
  params.minCircularity = 0.6;
  Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);

  Mat cannyOutput;
  Canny(grayImage,cannyOutput, 100, 200, 3);
  findContours(cannyOutput, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0,0));
  Mat allcnts = Mat::zeros(cannyOutput.size(), CV_8UC3);

  drawContours(allcnts, contours, -1, Scalar(0,255,0), 2,8,hierarchy);
  imshow("allcnts",allcnts);

  Mat cnts = Mat::zeros(cannyOutput.size(), CV_8UC3);
  int j=0;
  for(size_t i=0; i<contours.size();i++){
    double cArea = contourArea(contours[i]);
    if(cArea >= 800){
      drawContours(cnts, contours,(int)i, Scalar(0,255,0), 2,8,hierarchy,0,Point());
      imshow("cnts",cnts);


      j++;
      cout<<"Loop: "<<j<<endl;
      Mat mask = Mat::zeros(cannyOutput.size(), CV_8UC3);
      fillConvexPoly(mask,contours[i],Scalar(255,255,255));
      imshow("mask",mask);
      Mat diceImage = Mat::zeros(cannyOutput.size(), CV_8UC3);
      bitwise_and(mask,inputImage,diceImage);
      imshow("Dice",diceImage);

      vector<KeyPoint> keypoints;
      detector->detect( diceImage, keypoints );
      // Mat im_with_keypoints;
      drawKeypoints( final, keypoints, final, Scalar(0,255,0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
      int number = keypoints.size();
      sum += number;
      char Text[10];
      sprintf(Text,"%d",number);
      putText(final, Text, Point(contours[i].front().x+50,contours[i].front().y),FONT_HERSHEY_SIMPLEX, 1, cvScalar(0,255,0), 2, CV_AA);
      drawContours(final, contours, (int)i, Scalar(0,255,0), 2,8,hierarchy,0,Point());
    }
  }
  char sumFinal[30];
  sprintf(sumFinal,"Sum: %d",sum);
  putText(final, sumFinal, Point(20,50),FONT_HERSHEY_SIMPLEX, 1, cvScalar(0,255,0), 2, CV_AA);
  imshow("Output Image",final);

  waitKey(0);
  return 0;
}
