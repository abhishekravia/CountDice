#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>

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

  // Mat grayImage;
  // cvtColor(inputImage,grayImage,COLOR_BGR2GRAY);
  //// Adapted from https://www.learnopencv.com/blob-detection-using-opencv-python-c/
  SimpleBlobDetector::Params params;
  // Change thresholds
  params.minThreshold = 10;
  params.maxThreshold = 200;

  // Filter by Area.
  params.filterByArea = true;
  params.minArea = 50;

  // Filter by Circularity
  params.filterByCircularity = true;
  params.minCircularity = 0.8;

  // Filter by Convexity
  params.filterByConvexity = true;
  params.minConvexity = 0.87;

  // Filter by Inertia
  params.filterByInertia = true;
  params.minInertiaRatio = 0.01;
  Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);
  vector<KeyPoint> keypoints;
  detector->detect( inputImage, keypoints );


  // DrawMatchesFlags::DRAW_RICH_KEYPOINTS flag ensures the size of the circle corresponds to the size of blob
  Mat im_with_keypoints;
  drawKeypoints( inputImage, keypoints, im_with_keypoints, Scalar(0,255,0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );

  // Show blobs
  imshow("keypoints", im_with_keypoints );
  //

  waitKey(0);
  return 0;
}
