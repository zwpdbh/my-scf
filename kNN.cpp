#include <math.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include "kNN.hpp"

using namespace std;

KNN::KNN(const vector<vector<float>>& descriptors,
         const vector<vector<float>>& queryDescriptors,
         int                          k) {
    _descriptors      = descriptors;
    _queryDescriptors = queryDescriptors;

    _k = k;
}

void KNN::scfKNN(int numOfSubspaces, int numOfClusters, int timeOfLoops) {
    SCFIndex scfindex(numOfSubspaces, numOfClusters, _descriptors, timeOfLoops);

    // cout << "\nAfter finished building scfindex.." << endl;

    // d is the dimensionality of each subspace
    int d = _descriptors[0].size() / numOfSubspaces;
    // cout << "dimension of subspace d = " << d << endl;

    for (int i = 0; i < _descriptors.size(); i++) {
        for (int j = 0; j < _queryDescriptors.size(); j++) {
            // the distance between query feature and reference feature == sum
            // of distance from subspace
            float         total = 0.0;
            vector<float> qf    = _queryDescriptors[j];
            // cout << "query feature = " << qf << endl;
            for (int s = 0; s < numOfSubspaces; s++) {
                int from = s * d;
                int to   = (s + 1) * d;
                // sqf is the the partial query feature in subspace
                vector<float> sqf(&qf[from], &qf[to]);
                // cout << "query feature in subspace " << s << ", is : " << sqf
                // << endl; use subspace feature index to speedup the searching
                // process check in current subspace, the subspace part of
                // reference feature belong to which cluster
                SubFeature sf = scfindex.subspaceFeatureIndex[i][s];
                // cout << "subspaceFeatureIndex[" << i << "][" << s << "] = "
                // << sf  << endl;
                if (scfindex.distancesIndex[sf._whichCluster][s] < 0) {
                    // if there is no previous distance record, compute it or
                    // later use
                    float         dist = 0.0;
                    vector<float> scentroid =
                        scfindex.subspaceClusters[sf._whichCluster]
                                                 [sf._whichSubspace];
                    for (int t = 0; t < d; t++) {
                        dist +=
                            ((sqf[t] - scentroid[t]) * (sqf[t] - scentroid[t]));
                    }
                    scfindex.distancesIndex[sf._whichCluster][s] = dist;
                }
                total += scfindex.distancesIndex[sf._whichCluster][s];
            }
            Record r(i, j, total);
            _priorityQueue.push(r);
        }
    }
}

void KNN::bruteForceKNN() {
    for (int i = 0; i < _descriptors.size(); i++) {
        for (int j = 0; j < _queryDescriptors.size(); j++) {
            double d = sqrt(inner_product(_descriptors[i].begin(),
                                          _descriptors[i].end(),
                                          _queryDescriptors[j].begin(),
                                          0));
            Record r(i, j, d);
            _priorityQueue.push(r);
        }
    }
}

vector<Record> KNN::getKNNRecords() {
    priority_queue<Record> tmpQ(_priorityQueue);
    vector<Record>         kNNRecords;
    int                    k = 0;
    while (k < _k && !tmpQ.empty()) {
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

/**create a simple demo to show the idea of SCF*/
SCFIndex::SCFIndex(int                          numOfSubspaces,
                   int                          numOfClusters,
                   const vector<vector<float>>& features,
                   const int                    timeOfLoops) {
    _numOfClusters  = numOfClusters;
    _numOfSubspaces = numOfSubspaces;
    _numOfFeatures  = features.size();
    _timeOfLoops    = timeOfLoops;

    if (features.size() == 0) {
        cout << "features are empty" << endl;
        cout << "exit.." << endl;
        exit(1);
    }

    int d = features[0].size();

    if (d < _numOfSubspaces) {
        cout << "the number of subspaces must <= dimension of feature" << endl;
        cout << "exit.." << endl;
        exit(1);
    }

    if (d % _numOfSubspaces != 0) {
        cout << "dimension of feature / num of subspaces should be == zero"
             << endl;
        cout << "exit.." << endl;
        exit(1);
    }

    int s = d / _numOfSubspaces;

    // initialize data fields
    for (int i = 0; i < _numOfFeatures; i++) {
        vector<SubFeature> eachRow;
        for (int j = 0; j < _numOfSubspaces; j++) {
            SubFeature sf(-1, -1);
            eachRow.push_back(sf);
        }
        this->subspaceFeatureIndex.push_back(eachRow);
    }

    for (int i = 0; i < _numOfClusters; i++) {
        vector<vector<float>> eachRow;
        for (int j = 0; j < _numOfSubspaces; j++) {
            vector<float> subspaceClusterCenter(d, -0.1);
            eachRow.push_back(subspaceClusterCenter);
        }
        this->subspaceClusters.push_back(eachRow);
    }

    for (int i = 0; i < _numOfClusters; i++) {
        vector<float> eachRow;
        for (int j = 0; j < _numOfSubspaces; j++) {
            eachRow.push_back(-0.1);
        }
        this->distancesIndex.push_back(eachRow);
    }

    for (int i = 0; i < _numOfSubspaces; i++) {
        // divide the feature space into _numOfSubspaces which each of it has s
        // dimensions
        vector<vector<float>> featuresInSubspaces;
        int                   from = i * s;
        int                   to   = (i + 1) * s;
        // cout << "from = " << from << ", "
        //      << "to = " << to << endl;
        for (int index = 0; index < _numOfFeatures; index++) {
            vector<float> sf(&features[index][from], &features[index][to]);
            // cout << "In subspace[" << i << "]"
            //      << ", feature[" << index << "]"
            //      << " = " << features[index] << ", its subspace part is: " <<
            //      sf
            //      << endl;
            featuresInSubspaces.push_back(sf);
        }

        // apply clustering method in those subspaces
        buildClustersIndexInSubspaces(featuresInSubspaces, i);
    }  // end of loop for each subspace
       // cout << "Check the content of subspaceFeatureIndex" << endl;
       // for (int n = 0; n < subspaceFeatureIndex.size(); n++) {
       //  for(int s = 0; s < _numOfSubspaces; s++){
       //    cout << "subspaceFeatureIndex[" << n << "][" << s << "] = "<<
       //    subspaceFeatureIndex[n][s] << endl;
       //  }
       // }
}

/**used for recording the distance between feature and centroid during the
 * process of clustering
 */
struct distance_to_centroid {
    int   centroid_id;
    float distance;
};

/**This method apply k-means clustering on subFeatures
 * In this process, it updates the data field subSpaceFeatureIndex
 */
void SCFIndex::buildClustersIndexInSubspaces(
    const std::vector<std::vector<float>>& subFeatures,
    int                                    whichSubspace) {
    vector<int> seed;
    for (int i = 0; i < _numOfFeatures; i++) {
        seed.push_back(i);
    }

    std::random_device rd;
    std::mt19937       g(rd());

    // randomly pick _numOfClusters features as initial centroid for clustering
    std::shuffle(seed.begin(), seed.end(), g);
    vector<vector<float>> centroids;
    for (int i = 0; i < _numOfClusters; i++) {
        vector<float> c = subFeatures[seed[i]];
        centroids.push_back(c);
    }

    // do k-means clusterting
    // 1) create k vectors to store the clustered subFeatures' index
    vector<vector<int>> clusters;
    for (int i = 0; i < _numOfClusters; i++) {
        vector<int> eachCluster;  // stores the index of features in subFeatures
        clusters.push_back(eachCluster);
    }

    // cout << "\n===Compute the clusters using k-means with loop = " <<
    // _timeOfLoops << "===" << endl;
    for (int t = 0; t < _timeOfLoops; t++) {
        // for every feature, compare it with each cluster's centroid, compute
        // their distance and add the feature index into corresponding cluster
        // cout << "Update centroid at loop : " << t << " : " << endl;
        for (int i = 0; i < _numOfFeatures; i++) {
            // initialize metrics, for each sub-feature comparing with multiple
            // clusters
            vector<distance_to_centroid> metrics;

            for (int k = 0; k < _numOfClusters; k++) {
                // cout << "\nthe distance between subFeatures[" << i << "] = "
                // << subFeatures[i] << "; and centroids[" << k << "] = " <<
                // centroids[k] << " is: " << endl;
                distance_to_centroid d;
                d.centroid_id  = k;
                float distance = 0.0;
                for (int d = 0; d < subFeatures[0].size(); d++) {
                    float dist = subFeatures[i][d] - centroids[k][d];
                    // cout << "dist = " << subFeatures[i][d] << " - " <<
                    // centroids[k][d] << " = " << dist;
                    distance += (dist * dist);
                    // cout << ", distance += (dist * dist) = " << distance <<
                    // endl;
                }

                d.distance = distance;
                metrics.push_back(d);
            }

            // cout << "It is time to decide subFeature[i] belongs to which
            // cluster" << endl;
            int   id      = metrics[0].centroid_id;
            float minDist = metrics[0].distance;
            // cout << "The initial id = " << id << ", minDist = " << minDist <<
            // endl;
            for (int k = 0; k < _numOfClusters; k++) {
                if (metrics[k].distance < minDist) {
                    minDist = metrics[k].distance;
                    id      = metrics[k].centroid_id;
                }
            }
            // after we see the current subfeature is near to which cluster
            // center
            clusters[id].push_back(i);
        }

        // cout << "\n==update centroids..==" << endl;
        // update centroids
        for (int i = 0; i < _numOfClusters; i++) {
            // cout << "for cluster: " << i << endl;
            vector<float> mean(subFeatures[0].size(), 0.0);
            // cout << "initial mean = " << mean << endl;

            for (int n = 0; n < clusters[i].size(); n++) {
                for (int d = 0; d < subFeatures[0].size(); d++) {
                    mean[d] += (subFeatures[clusters[i][n]][d]);
                }
            }

            // devide by the total number of features in that cluster to get
            // mean
            for (int d = 0; d < subFeatures[0].size(); d++) {
                if (clusters[i].size() != 0) {
                    mean[d] = mean[d] / clusters[i].size();
                }
            }
            // cout << "updated mean = " << mean << endl;
            centroids[i] = mean;
            // cout << ">> centroids[" << i << "] = " << centroids[i] << endl;
        }
    }  // end of _timeOfLoop

    // loop through each clusters, use the record in each cluster to update data
    // fields
    for (int i = 0; i < _numOfClusters; i++) {
        for (int n = 0; n < clusters[i].size(); n++) {
            SubFeature sf(i, whichSubspace);
            // clusters stores multiple clusters' records which is the
            // subFeature' index
            subspaceFeatureIndex[clusters[i][n]][whichSubspace] = sf;
        }
        subspaceClusters[i][whichSubspace] = centroids[i];
    }
}
