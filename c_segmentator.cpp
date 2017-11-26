#include "c_segmentator.h"

void C_SEGMENTATOR::Net_initialize(const string& model_file,
                       const string& trained_file, const string& label_index_file, double _cf_thres)
{

  Caffe::set_mode(Caffe::GPU);
  /* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(trained_file);

  Blob<float>* input_layer = net_->input_blobs()[0];

  num_channels_ = input_layer->channels();
  input_geometry_.width = input_layer->width();
  input_geometry_.height = input_layer->height();

  /* PARSING LABEL_COLOR_INDEX_FILE *******************
   *
   * */
  string label_rgb_index_file_path(label_index_file);
  ifstream label_rgb_index_file(label_rgb_index_file_path.c_str());
  string line_str;
  while(getline(label_rgb_index_file,line_str))
  {
      istringstream iss(line_str);

      pair<cv::Point3d,string> label_rgb_data;
      string parsed_str;

      uint index = 0;
      while(getline(iss,parsed_str,','))
      {
          switch(index)
          {
          case 0:
              label_rgb_data.first.x = std::atoi(parsed_str.c_str());
              break;
          case 1:
              label_rgb_data.first.y = std::atoi(parsed_str.c_str());
              break;
          case 2:
              label_rgb_data.first.z = std::atoi(parsed_str.c_str());
              break;
          case 3:
              label_rgb_data.second = parsed_str;
              break;
          }
          index++;
      }
      label_rgb_data_list.push_back(label_rgb_data);
  }
  label_rgb_index_file.close();
  /******************************************************/

  cf_thres = _cf_thres;
}

/* Return the top N predictions. */
cv::Mat C_SEGMENTATOR::Classify(const cv::Mat& img) {
  return Predict(img);
}

cv::Mat C_SEGMENTATOR::Predict(const cv::Mat& img) {
    Blob<float>* input_layer = net_->input_blobs()[0];
    input_layer->Reshape(1, num_channels_,
                           input_geometry_.height, input_geometry_.width);
    /* Forward dimension change to all layers. */
    net_->Reshape();
    std::vector<cv::Mat> input_channels;
    WrapInputLayer(&input_channels);
    Preprocess(img, &input_channels);
    Caffe::set_mode(Caffe::GPU);
    net_->Forward();

    float* data_addr = input_layer->mutable_cpu_data();
    vector<cv::Mat> img_RGB_channel;
    for(int channel=0;channel< num_channels_;channel++)
    {
      cv::Mat img_f(input_geometry_.height,input_geometry_.width,CV_32FC1,data_addr);
      cv::Mat img_single_channel;
      img_f.convertTo(img_single_channel,CV_8UC3);
      img_RGB_channel.push_back(img_single_channel);
      data_addr += input_geometry_.width*input_geometry_.height;
    }

    cv::Mat color_img;

    merge(img_RGB_channel,color_img);

    const caffe::shared_ptr<Blob<float> > prob = net_->blob_by_name("prob");
    float* prob_addr = prob->mutable_cpu_data();

    vector<cv::Vec3b> label;

    for(uint class_index = 0; class_index < label_rgb_data_list.size(); class_index++)
    {
        cv::Vec3b class_rgb_color;
        class_rgb_color[0] = (int)label_rgb_data_list.at(class_index).first.x;
        class_rgb_color[1] = (int)label_rgb_data_list.at(class_index).first.y;
        class_rgb_color[2] = (int)label_rgb_data_list.at(class_index).first.z;
        label.push_back(class_rgb_color);
    }

    cv::Mat result_img = cv::Mat::zeros(input_geometry_.height,input_geometry_.width,CV_8UC3);
    cv::Mat prob_max = cv::Mat::zeros(input_geometry_.height,input_geometry_.width,CV_32FC2);
    for(int channel=0;channel<prob->channels();channel++)
    {
        cv::Mat prob_f(input_geometry_.height,input_geometry_.width,CV_32FC1,prob_addr);
        for(int x=0;x<prob_f.size().width;x++)
        {

          for(int y=0;y<prob_f.size().height;y++)
          {
              if (prob_f.at<float>(y,x) > prob_max.at<cv::Vec2f>(y,x)[0])
              {
                  prob_max.at<cv::Vec2f>(y,x)[0] = prob_f.at<float>(y,x);
                  prob_max.at<cv::Vec2f>(y,x)[1] = channel;
              }
              if(channel == prob->channels()-1)
              {
                  if(prob_max.at<cv::Vec2f>(y,x)[0] >= cf_thres)
                  {
                      result_img.at<cv::Vec3b>(y,x)[0] = label.at(prob_max.at<cv::Vec2f>(y,x)[1])[2];
                      result_img.at<cv::Vec3b>(y,x)[1] = label.at(prob_max.at<cv::Vec2f>(y,x)[1])[1];
                      result_img.at<cv::Vec3b>(y,x)[2] = label.at(prob_max.at<cv::Vec2f>(y,x)[1])[0];

                  }
              }
          }

        }
        prob_addr += input_geometry_.width*input_geometry_.height;
    }

    return result_img;
}


void C_SEGMENTATOR::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
      Blob<float>* input_layer = net_->input_blobs()[0];

      int width = input_layer->width();
      int height = input_layer->height();
      float* input_data = input_layer->mutable_cpu_data();
      for (int i = 0; i < input_layer->channels(); ++i) {
        cv::Mat channel(height, width, CV_32FC1, input_data);
        input_channels->push_back(channel);
        input_data += width * height;
      }


}

void C_SEGMENTATOR::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels) {
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else
    sample = img;

  cv::Mat sample_resized;
  if (sample.size() != input_geometry_)
    cv::resize(sample, sample_resized, input_geometry_);
  else
    sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);


  cv::split(sample_float, *input_channels);

  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}
