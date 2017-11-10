#ifndef CSEGMENTATOR_H
#define CSEGMENTATOR_H


#include <caffe/caffe.hpp>
#include <caffe/proto/caffe.pb.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>


#include <typeinfo>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <boost/array.hpp>
#include <time.h>
#include <limits.h>

using namespace caffe;  // NOLINT(build/namespaces)
using std::string;
using namespace std;

/* Pair (label, confidence) representing a prediction. */
typedef std::pair<string, float> Prediction;

class C_SEGMENTATOR {
 public:
    C_SEGMENTATOR () {}
    void Net_initialize(const string& model_file,
          const string& trained_file,
          const string& label_index_file, double _cf_thres);
    cv::Mat Classify(const cv::Mat& img);

    private:
    cv::Mat Predict(const cv::Mat& img);

    void WrapInputLayer(std::vector<cv::Mat>* input_channels);

    void Preprocess(const cv::Mat& img,
                  std::vector<cv::Mat>* input_channels);

 private:
    caffe::shared_ptr<Net<float> > net_;
    cv::Size input_geometry_;
    int num_channels_;
    cv::Mat mean_;
    std::vector<string> labels_;
    double cf_thres = 0.0;
    vector<pair<cv::Point3d,string> > label_rgb_data_list;
};

#endif // CSEGMENTATOR_H
