#ifndef dataset_hpp
#define dataset_hpp

#include <sstream>
#include <string>
#include <vector>
class Record;
class Dataset {
  private:
    std::vector<std::vector<float>> referenceFeatures;
    std::vector<std::vector<float>> queryFeatures;

  public:
    Dataset(const std::string& descriptors01, const std::string& descriptors02);
    const std::vector<std::vector<float>>& getReferenceFeatures() const;
    const std::vector<std::vector<float>>& getQueryFeatures() const;
    const void                             checkDataset() const;
    void checkRecord(Record& record);
};

#endif // !dataset_hpp
