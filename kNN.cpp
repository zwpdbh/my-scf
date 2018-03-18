#include "kNN.hpp"
#include <fstream>
#include <iostream>
#include <math.h>

using namespace std;

KNN::KNN(const cv::Mat& descriptors, const cv::Mat& queryDescriptors, int k) {
    _descriptors      = descriptors;
    _queryDescriptors = queryDescriptors;

    _k = k;
}

void KNN::bruteForceKNN() {
    for (int i = 0; i < _descriptors.rows; i++) {
        for (int j = 0; j < _queryDescriptors.rows; j++) {
            double d = sqrt(_descriptors.row(i).dot(_queryDescriptors.row(j)));
            Record r(i, j, d);
            _priorityQueue.push(r);
        }
    }
}

vector<Record> KNN::getKNNRecords() {
    priority_queue<Record> tmpQ(_priorityQueue);
    vector<Record>         kNNRecords;
    int                    k = 0;
    while (k < _k || !tmpQ.empty()) {
        kNNRecords.push_back(tmpQ.top());
        tmpQ.pop();
        k++;
    }

    return kNNRecords;
}

void KNN::printRecordsIntoFile(const std::string& filename) {
    // copy constructor
    ofstream ofs;
    ofs.open(filename);
    cout << "writing brute search results into file: " << filename << "..."
         << endl;
    priority_queue<Record> tmpQ(_priorityQueue);
    int                    i = 0;
    while (!tmpQ.empty()) {
        ofs << tmpQ.top();
        // cout << tmpQ.top();
        tmpQ.pop();
        i++;
        if (i % 2000 == 0) {
            cout << "Have writon " << i << "records into it... " << endl;
        }
    }
    cout << "Done. total records =  " << i << endl;
}