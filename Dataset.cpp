#include "Dataset.hpp"
#include "kNN.hpp"
#include <fstream>
#include <sstream>


using namespace std;

Dataset::Dataset(const std::string& descriptors01,
		 const std::string& descriptors02) {

    ifstream referencefs(descriptors01);
    ifstream queryfs(descriptors02);

    if (!referencefs.good()) {
        cout << "can not load reference descriptors records: " << descriptors01
             << endl;
        exit(1);
    }

    if (!queryfs.good()) {
        cout << "can not load query descriptors records: " << descriptors02
             << endl;
    }

    for (string line; getline(referencefs, line);) {
        stringstream ss;
        ss << line;
        string        token;
        vector<float> eachFeature;
        while (ss >> token) {
            eachFeature.push_back(stof(token));
        }

        this->referenceFeatures.push_back(eachFeature);
    }

    for (string line; getline(queryfs, line);) {
        stringstream ss;
        ss << line;
        string        token;
        vector<float> eachFeature;
        while (ss >> token) {
            eachFeature.push_back(stof(token));
        }
        this->queryFeatures.push_back(eachFeature);
    }
}

const vector<vector<float>>& Dataset::getReferenceFeatures() const {
    return this->referenceFeatures;
}

const vector<vector<float>>& Dataset::getQueryFeatures() const {
    return this->queryFeatures;
}



const void Dataset::checkDataset() const {
    cout << "Dataset: " << endl;
    cout << "reference features: " << this->referenceFeatures.size() << " * "
         << this->referenceFeatures[0].size() << endl;
    cout << "qeury features: " << this->queryFeatures.size() << " * "
         << this->queryFeatures[0].size() << endl;
}

void Dataset::checkRecord(Record& record) {
    int referenceFeatureIndex;
    int queryFeatureIndex;
    tie(referenceFeatureIndex, queryFeatureIndex) = record.getIndexToIndex();

    cout << "reference feature: " << referenceFeatureIndex << " = "
         << this->referenceFeatures[referenceFeatureIndex] << endl;

    cout << "query features: " << queryFeatureIndex << " = "
         << this->queryFeatures[queryFeatureIndex] << endl;
}
