// This file is part of ScaViSLAM.
//
// Copyright 2011 Hauke Strasdat (Imperial College London)
//
// ScaViSLAM is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published
// by the Free Software Foundation, either version 3 of the License, or
// any later version.
//
// ScaViSLAM is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with ScaViSLAM.  If not, see <http://www.gnu.org/licenses/>.

#include <iostream>
#include <tr1/unordered_set>
#include <tr1/unordered_map>

#include <boost/filesystem.hpp>
#include <boost/regex.hpp>
#include <boost/dynamic_bitset.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/flann/flann.hpp>


// Thanks a lot to Adrien Angeli for all help and discussion concerning
// place recognition using "bag of words".

using namespace std;

list<string> preprocessFiles(const boost::filesystem::path & directory)
{
  list<string> name_list;

  if( exists( directory ) )
  {
    boost::filesystem::directory_iterator end ;
    for(boost::filesystem::directory_iterator iter(directory);
        iter != end ; ++iter )
    {
      if (is_directory( *iter)==false )
      {
        boost::filesystem3::path  name = iter->path().filename();
        name_list.push_back(name.string());
      }
    }
  }
  return name_list;
}

cv::Mat
loadImage(const string & img_name)
{
  int MAX_WIDTH = 640;
  int MAX_HEIGHT = 480;
  int MAX_AREA = MAX_WIDTH*MAX_HEIGHT;

  cout << "load image: " << img_name << endl;
  cv::Mat img = cv::imread(img_name, 1);

  while(img.size().area()>MAX_AREA)
  {
    cv::pyrDown(img, img);
    cout << "Downsample!" << endl;
  }
  cout << "Final size: " << img.size().width <<" "<< img.size().height << endl;
  cv::Mat img_mono;
  cv::cvtColor(img, img_mono, cv::COLOR_BGR2GRAY);
  cout << endl;
  return img_mono;
}

bool
computeKeypoints(const cv::Mat & img_mono,
                 vector<cv::KeyPoint>  * key_points)
{
  double thr = 300;
  bool failed = false;
  int trials = 0;

  cout << "Detect SURF features:" << endl;

  while(true)
  {
    cv::SurfFeatureDetector surf(thr, 2);
    surf.detect(img_mono, *key_points);
    int num_keypoints = key_points->size();

    cout << "Trial: " << trials << "; threshold: " << thr;
    cout << "; no features: " << num_keypoints << endl;

    if (num_keypoints>2000)
    {
      if (num_keypoints>10000)
      {
        failed = true;
        break;
      }
      thr += 200;
    }
    else if (num_keypoints<500)
    {
      thr -=50;
    }
    else
    {
      break;
    }
    if (trials>=5)
    {
      failed = true;
      break;
    }
    ++trials;

    if (failed)
    {
      return false;
    }
  }
  cout << endl;
  return true;
}

void
computeDescriptors(const cv::Mat & img_mono,
                   vector<cv::KeyPoint> * key_points,
                   cv::Mat * descriptors)
{
  cv::Mat desc;
  cv::SurfDescriptorExtractor surf_extr(2, 4, 2, false);
  surf_extr.compute(img_mono, *key_points, desc);
  for (int row=0; row<desc.rows; ++row)
  {
    descriptors->push_back(desc.row(row));
  }
}

void
calculateWordsAndSaveThem(int TARGET_NUM_WORDS,
                          const cv::Mat & descriptors)
{
  cout << "Creating up to " << TARGET_NUM_WORDS << " clusters/words..." << endl;
  cout << "... " << endl;
  cvflann::KMeansIndexParams kmeans(32, 11, cvflann::FLANN_CENTERS_KMEANSPP);
  cv::Mat centers(TARGET_NUM_WORDS, descriptors.cols, CV_32F);
  typedef cv::flann::L2<float> distance;
  typedef distance::ResultType DistanceType;
  typedef distance::ElementType ElementType;
  cvflann::Matrix<ElementType>
      flann_features((ElementType*)descriptors.ptr<ElementType>(0),
                     descriptors.rows, descriptors.cols);
  cvflann::Matrix<DistanceType>
      flann_centers((DistanceType*)centers.ptr<DistanceType>(0),
                    centers.rows, centers.cols);
  int num_centers
      = ::cvflann::hierarchicalClustering<distance>(flann_features,
                                                    flann_centers,
                                                    kmeans,
                                                    distance());
  cout << "Done: dictionary of " << num_centers << " words created!" << endl;
  assert(sizeof(float)==4);
  cv::Mat centers_float_as_four_uint8(num_centers,
                                      descriptors.cols*4,
                                      CV_8U,
                                      centers.data);
  stringstream str_stream;
  str_stream<< "surfwords" << num_centers << ".png";
  cv::imwrite(str_stream.str(), centers_float_as_four_uint8);

  cout << "Saved as file: " << str_stream.str() << endl;
}

void
createDictionary(const string & base_str,
                 int MAX_NUM_IMAGES,
                 int TARGET_NUM_WORDS)
{
  list<string> name_list
      = preprocessFiles(base_str);

  int num_processed_images = 0;
  cv::Mat descriptors;
  for (list<string>::iterator it = name_list.begin(); it!=name_list.end(); ++it)
  {
    stringstream sst;
    sst << base_str << *it;
    cv::Mat img_mono = loadImage(sst.str());
    vector<cv::KeyPoint> key_points;

    bool success = computeKeypoints(img_mono, &key_points);

    if (success==false)
    {
      cout << "abort!" << endl;
      cout << endl;
      continue;
    }
    computeDescriptors(img_mono, &key_points, &descriptors);

    cout << "Image processed: " << num_processed_images;
    cout << " of max. " << MAX_NUM_IMAGES << endl;
    cout << "Number of features: " << descriptors.rows;
    cout << " (TARGET_NUM_WORDS: " << TARGET_NUM_WORDS << ")" << endl;
    cout << endl;
    cout << endl;

    ++num_processed_images;
    if(num_processed_images>MAX_NUM_IMAGES)
      break;
  }

  if (descriptors.rows<TARGET_NUM_WORDS*10)
  {
    cout << "ERROR: By far not enough features detected to calculate "
         << TARGET_NUM_WORDS << " words/clusters!" << endl;
    exit(0);
  }
  calculateWordsAndSaveThem(TARGET_NUM_WORDS,
                            descriptors);
}


int
main(int argc, const char* argv[])
{
  if (argc<2)
  {
    cout << "USAGE: create_dictionary FOLDER_WITH_IMAGES "
         << "[MAX_NUM_IMAGES] [TARGET_NUM_WORDS]"
         << endl;
    cout << endl;
    exit(0);
  }

  string base_str(argv[1]);


  int MAX_NUM_IMAGES = 150;
  if (argc>=3)
    MAX_NUM_IMAGES = atoi(argv[2]);
  int TARGET_NUM_WORDS = max(1000, MAX_NUM_IMAGES*10);
  if (argc>=4)
    TARGET_NUM_WORDS = atoi(argv[3]);

  cout << endl;
  cout << "MAX_NUM_IMAGES: " << MAX_NUM_IMAGES << endl;
  cout << "TARGET_NUM_WORDS: " << TARGET_NUM_WORDS << endl;
  cout << endl;

  createDictionary(base_str, MAX_NUM_IMAGES, TARGET_NUM_WORDS);
}

