#ifndef parallel_KNN_HPP
#define parallel_KNN_HPP

#include <iostream>
#include <opencv2/core.hpp>
#include <queue>
#include <string>
#include <tuple>
#include <vector>
class Record {
  private:
    double               _priority_value;
    std::tuple<int, int> _indexToIndex;

  public:
    Record(int index1, int index2, double v) {
        _indexToIndex   = std::make_tuple(index1, index2);
        _priority_value = v;
    }

    /***/
    bool operator<(const Record& another) const {
        return _priority_value < another._priority_value;
    }
    friend std::ostream& operator<<(std::ostream& o, const Record& r) {
        double i, j;
        std::tie(i, j) = r._indexToIndex;
        return o << "<" << i << ", " << j << ">"
                 << " : " << r._priority_value << std::endl;
    }
};

class KNN {
  private:
    cv::Mat                     _descriptors;
    cv::Mat                     _queryDescriptors;
    int                         _k;
    std::priority_queue<Record> _priorityQueue;

  public:
    KNN(const cv::Mat& descriptors, const cv::Mat& queryDescriptors, int k);
    void                bruteForceKNN();
    std::vector<Record> getKNNRecords();
    void                printRecordsIntoFile(const std::string& filename);
};

#endif // !parallel_KNN_HPP