//
// Created by zwpdbh on 18/03/2018.
//

#include <flann/flann.hpp>
#include <flann/io/hdf5.h>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>

#include "kNN.hpp"

using namespace std;
using namespace cv;

int main(int argc, char* argv[]) {

    // step 1, get the features from pictures

    Mat img01 = imread("/Users/zw/code/resources/opencv_test_data/"
                       "practical-instance-recognition-2018a/data/oxbuild_lite/"
                       "all_souls_000002.jpg",
                       IMREAD_COLOR);
    Mat img02 = imread("/Users/zw/code/resources/opencv_test_data/"
                       "practical-instance-recognition-2018a/data/oxbuild_lite/"
                       "all_souls_000013.jpg",
                       IMREAD_COLOR);

    Ptr<xfeatures2d::SIFT> featureDetector =
        xfeatures2d::SIFT::create(0, 3, 0.04, 10, 1.6);

    vector<KeyPoint> keypoints_of_img01, keypoints_of_img02;
    Mat              descriptors_of_img01, descriptor_of_img02;

    featureDetector->detectAndCompute(
        img01, Mat(), keypoints_of_img01, descriptors_of_img01);
    featureDetector->detectAndCompute(
        img02, Mat(), keypoints_of_img02, descriptor_of_img02);

    cout << "from img01, we extract " << keypoints_of_img01.size()
         << " KeyPoint, and each of them corresponds to a feature descriptor"
         << endl;
    cout << "the number of features in img01 is: " << descriptors_of_img01.rows
         << endl;
    cout << "each of those feature's dimensionality is: "
         << descriptors_of_img01.cols << endl;

    cout << "the number of features in img02 is: " << descriptor_of_img02.rows
         << endl;

    KNN kNN(descriptors_of_img01, descriptor_of_img02, 10);
    kNN.bruteForceKNN();
    kNN.printRecordsIntoFile("log.txt");

    return 0;
}