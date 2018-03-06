//
// Created by zwpdbh on 13/02/2018.
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

using namespace std;
using namespace cv;
// using namespace cv::FlannBasedMatcher;

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

    featureDetector->detectAndCompute(img01, Mat(), keypoints_of_img01,
                                      descriptors_of_img01);
    featureDetector->detectAndCompute(img02, Mat(), keypoints_of_img02,
                                      descriptor_of_img02);

    Mat result_img01, result_img02;
    cout << "number of keypoints_of_img01 are : " << keypoints_of_img01.size()
         << endl;
    cout << "number of keypoints_of_img02 are : " << keypoints_of_img02.size()
         << endl;

    drawKeypoints(img01, keypoints_of_img01, result_img01, Scalar::all(0),
                  DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    drawKeypoints(img02, keypoints_of_img02, result_img02, Scalar::all(0),
                  DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    //    imshow("SIFT result on image 01", result_img01);
    //    imshow("SIFT result on image 02", result_img02);

    // use opencv flann to to the match
    cv::FlannBasedMatcher matcher;
    std::vector<DMatch>   matches;
    matcher.cv::FlannBasedMatcher::match(descriptors_of_img01,
                                         descriptor_of_img02, matches);

    double max_dist = 0;
    double min_dist = 100;
    for (int i = 0; i < descriptors_of_img01.rows; i++) {
        double dist = matches[i].distance;
        if (dist < min_dist)
            min_dist = dist;
        if (dist > max_dist)
            max_dist = dist;
    }

    std::vector<DMatch> good_matches;
    for (int i = 0; i < descriptors_of_img01.rows; i++) {
        if (matches[i].distance <= max(2 * min_dist, 0.02)) {
            good_matches.push_back(matches[i]);
        }
    }

    Mat img_matches;
    cv::drawMatches(img01, keypoints_of_img01, img02, keypoints_of_img02,
                    good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                    vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    imshow("Good matches", img_matches);
    for (int i = 0; i < (int)good_matches.size(); i++) {
        printf("-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n", i,
               good_matches[i].queryIdx, good_matches[i].trainIdx);
    }

    waitKey(0);

    return 0;
}
