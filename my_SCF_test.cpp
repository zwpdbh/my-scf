
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace std;
using namespace cv;

void test_descriptor_from_opencv();
void test_example_from_SCF_paper();

int main() {

    test_descriptor_from_opencv();

    return 0;
}

/**Test the functionality of extract image features from opencv.The features
 * extracted is a Mat which is 128 dimension * the number of features.*/
void test_descriptor_from_opencv() {
    Mat img01 = imread("/Users/zw/Downloads/project_NUMA_KNN/"
                       "source_code_mac_osx/resources/all_souls_000002.jpg",
                       cv::IMREAD_COLOR);
    Ptr<xfeatures2d::SIFT> featureDetector =
        xfeatures2d::SIFT::create(0, 3, 0.04, 10, 1.6);

    vector<KeyPoint> keypoints_of_img01;
    Mat              descriptors_of_img01;

    featureDetector->detectAndCompute(img01, Mat(), keypoints_of_img01,
                                      descriptors_of_img01);

    cout << "number of features of img01: " << keypoints_of_img01.size()
         << endl;

    cout << "size of descriptors of img01: " << descriptors_of_img01.size()
         << endl;

    cout << "the first descriptor is: " << descriptors_of_img01.row(0).size()
         << endl;
}