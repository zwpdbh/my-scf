#ifndef parallel_KNN_HPP
#define parallel_KNN_HPP

#include <iostream>
#include <queue>
#include <string>
#include <tuple>
#include <vector>
#include <iterator>


/**Record represent one of k-NN search result. Such as, the distance between 
 * reference feature i and query feature j, and their distance. It is used by the priority queue.
 */
class Record {
  private:
    double               _priority_value;
    std::tuple<int, int> _indexToIndex;

  public:
    Record(int index1, int index2, double v) {
        _indexToIndex   = std::make_tuple(index1, index2);
        _priority_value = v;
    }

    std::tuple<int, int> getIndexToIndex() {
        return _indexToIndex;
    }

    /***/
    bool operator<(const Record& another) const {
        return _priority_value > another._priority_value;
    }
    friend std::ostream& operator<<(std::ostream& o, const Record& r) {
        double i, j;
        std::tie(i, j) = r._indexToIndex;
        return o << "<" << i << ", " << j << ">"
                 << " : " << r._priority_value << std::endl;
    }
};

/**used to print out the vector<T>*/
template <typename T>
std::ostream& operator<<(std::ostream& out, const std::vector<T>& v) {
  if (!v.empty()) {
    out << '[';
    std::copy(v.begin(), v.end(), std::ostream_iterator<T>(out, ", "));
    out << "\b\b]";
  }
  return out;
}


class KNN {
  private:
    std::vector<std::vector<float> > _descriptors;
    std::vector<std::vector<float> > _queryDescriptors;
    int                             _k;
    std::priority_queue<Record>     _priorityQueue;

  public:
    KNN(const std::vector<std::vector<float> >& descriptors,
        const std::vector<std::vector<float> >& queryDescriptors,
        int                                    k);

    void                bruteForceKNN();
    std::vector<Record> getKNNRecords();
    void                printRecordsIntoFile(const std::string& filename);
  void scfKNN(int numOfSubspaces, int numOfClusters, int timeOfLoops);
};


/**SubFeature is stores the mapping. It shows for each reference feature, which subfeature part
 * belongs to which cluster in that subspace
 */
class SubFeature{
public:
  SubFeature(int whichCluster, int whichSubspace) {
    _whichSubspace = whichSubspace;
    _whichCluster = whichCluster;
  }

  int _whichSubspace;
  int _whichCluster;
  
  friend std::ostream& operator<<(std::ostream& o, const SubFeature& sf) {
    return o << "<at_subspace: " << sf._whichSubspace << ", at_cluster: " << sf._whichCluster << ">";
  }
};


/**Its internal data structure store the mapping to guid for each query feature's corresponding
 * subspace part during the process of comparing it with each reference feature's corresponding
 * subspace part, it needs to go where to get the corresponding representative cluster center
 * to compare the distance
 * 
 * subSpaceFeatureIndex guid the road
 *
 * distancesIndex stores the distance between the query feature's each subspace part and all
 * subspace cluster centers
 *
 * subspaceClusters stores the cluster centers to represent the belonging features in that subspace part
 */
class SCFIndex {
public:
  int _numOfSubspaces;
  int _numOfClusters;
  int _numOfFeatures;
  int _timeOfLoops;
  /**a matrix: num_features by num_subspaces */
  std::vector<std::vector<SubFeature> > subspaceFeatureIndex;

  /**a matrix: num_clusters by num_subspaces*/
  std::vector<std::vector<std::vector<float> > > subspaceClusters;

  /**a matrix: num_clusters by num_subspaces */
  std::vector<std::vector<float> > distancesIndex;

  SCFIndex(int numOfSubspaces, int numOfClusters, const std::vector<std::vector<float>>& features , const int timeOfLoops);
  void buildClustersIndexInSubspaces(const std::vector<std::vector<float>>& subFeatures, int whichSubspace);
};

#endif // !parallel_KNN_HPP
