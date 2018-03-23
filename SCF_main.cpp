
#include <iostream>
#include <opencv2/core.hpp>
#include <random>
#include <typeinfo>

using namespace std;
using namespace cv;

void match(const Mat_<int>& features, const Mat_<int>& q);
void do_clustering(const Mat_<int>& features, Mat_<int> c1, Mat_<int> c2);

int main(int argc, char* argv[]) {
    Mat_<int> features(4, 4);

    // feature_a = -4, 2, 3, 4
    features(0, 0) = -4;
    features(0, 1) = 2, features(0, 2) = 3;
    features(0, 3) = 4;

    // features_b = -2, 2, 1, 0
    features(1, 0) = -2;
    features(1, 1) = 2;
    features(1, 2) = 1;
    features(1, 3) = 0;

    // feature_c = 2, -3 , 0 , 1
    features(2, 0) = 2;
    features(2, 1) = -3;
    features(2, 2) = 0;
    features(2, 3) = 1;

    // feature_d = 4, -3, 4, 3
    features(3, 0) = 4;
    features(3, 1) = -3;
    features(3, 2) = 4;
    features(3, 3) = 3;
    cout << "features are : \n" << features << endl;

    Mat_<int> queries(1, 4);
    queries(0, 0) = 0;
    queries(0, 1) = 0;
    queries(0, 2) = 0;
    queries(0, 3) = 0;
    cout << "query features = " << queries << endl;

    match(features, queries.row(0));

    return 0;
}

void match(const Mat_<int>& features, const Mat_<int>& q) {
    cout << "q = " << q << endl;
    cout << "if using clustering to estimate distances directly: " << endl;

    Mat_<int> c1;
    Mat_<int> c2;

    do_clustering(features, c1, c2);
}

void do_clustering(const Mat_<int>& features,
                   Mat_<int>        cluster1,
                   Mat_<int>        cluster2) {
    // randomly pick 2 features as centroids
    random_device              rd;
    mt19937                    gen(rd());
    uniform_int_distribution<> dis(0, features.rows);

    int c1 = dis(gen);
    int c2 = dis(gen);

    cout << "c1 = " << c1 << endl;
    cout << "c2 = " << c2 << endl;
    
}