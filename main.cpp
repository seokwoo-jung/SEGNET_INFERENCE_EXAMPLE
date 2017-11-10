#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "c_segmentator.h"

using namespace std;

int main()
{
    // Load img
    cv::Mat ori_img = cv::imread("../SEGNET_INFERENCE_EXAMPLE/test.png");

    // Create Segmentator Object
    C_SEGMENTATOR c_segmentator_obj;

    // Get $HOME folder
    string home_env = getenv("HOME");

    // Setting Trained model ,weight and mean file
    string model_file = home_env + "/caffe-segnet-cudnn6/examples/CamVid_example/deploy.prototxt";
    string weights_file = home_env + "/caffe-segnet-cudnn6/examples/CamVid_example/Inference/test_weights.caffemodel";
    string label_index_file = home_env + "/caffe-segnet-cudnn6/data/CamVid_example/label_color_index.txt";

    double cf_thres = 0.0;

    // Deep Learning Network Initialize
    c_segmentator_obj.Net_initialize(model_file,weights_file,label_index_file,cf_thres);

    // Segmentation Image
    cv::Mat segment_img;

    segment_img = c_segmentator_obj.Classify(ori_img);

    cv::Mat segment_img_resized;

    cv::resize(segment_img,segment_img_resized,ori_img.size());

    cv::imshow("ori_img",ori_img);
    cv::imshow("segmentation_img",segment_img_resized);
    cv::waitKey();

    return 0;
}
