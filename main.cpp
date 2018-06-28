//
// Created by zwpdbh on 18/03/2018.
//
#include <sys/time.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include "Dataset.hpp"
#include "kNN.hpp"

using namespace std;

void scfDemo();
void scfDemoOnRealData(const std::string& rfFile, const std::string& qfFile);

/**usage: ./bin/<exe> <feature1.txt> <feature2.txt>*/
int main(int argc, char* argv[]) {
    auto started = std::chrono::high_resolution_clock::now();

    // if (argc != 3) {
    //   cout << "argc should be == 3" << endl;
    //   cout << "<main> <reference_features.txt> <query_features.txt>" << endl;
    //   cout << "Exit.." << endl;
    //   return 1;
    // }
    //  scfDemoOnRealData(argv[1], argv[2]);
    scfDemo();

    auto done = std::chrono::high_resolution_clock::now();
    std::cout << "From initialization to finished, use: "
              << std::chrono::duration_cast<std::chrono::nanoseconds>(done -
                                                                      started)
                     .count()
              << "ns\n"
              << endl;
    return 0;
}

void scfDemoOnRealData(const std::string& rfFile, const std::string& qfFile) {
    Dataset dataset(rfFile, qfFile);
    dataset.checkDataset();

    KNN kNN(dataset.getReferenceFeatures(), dataset.getReferenceFeatures(), 10);
    //  kNN.bruteForceKNN();
    kNN.scfKNN(64, 128, 20);

    vector<Record> kNNRecords = kNN.getKNNRecords();
    for (int i = 0; i < kNNRecords.size(); i++) {
        cout << kNNRecords[i] << endl;
        dataset.checkRecord(kNNRecords[i]);
    }

    kNN.printRecordsIntoFile("log.txt");
}

void scfDemo() {
    vector<vector<float> > features;
    vector<float>         fa = {-4, 2, 3, 4};
    vector<float>         fb = {-2, 2, 1, 0};
    vector<float>         fc = {2, -3, 0, 1};
    vector<float>         fd = {4, -3, 4, 3};

    features.push_back(fa);
    features.push_back(fb);
    features.push_back(fc);
    features.push_back(fd);

    vector<float>         q = {0, 0, 0, 0};
    vector<vector<float>> queryFeatures;

    queryFeatures.push_back(q);

    cout << "The features are:" << endl;
    for (int i = 0; i < features.size(); i++) {
        cout << features[i] << endl;
    }
    cout << "The query feature is: " << endl;
    cout << q << endl;

    KNN kNN(features, queryFeatures, 4);

    kNN.scfKNN(2, 2, 20);

    vector<Record> records = kNN.getKNNRecords();
    cout << "\nkNN result: " << endl;
    cout << "<rf_index, qf_index> : distance" << endl;
    for (int i = 0; i < records.size(); i++) {
        cout << records[i] << endl;
    }

}
