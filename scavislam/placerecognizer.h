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

#ifndef SCAVISLAM_PLACERECOGNIZER_H
#define SCAVISLAM_PLACERECOGNIZER_H

#include <boost/thread.hpp>
#include <opencv2/flann/flann.hpp>

#include <sophus/se3.h>

#include "matcher.hpp"
#include "keyframes.h"
#include "stereo_camera.h"


namespace ScaViSLAM
{
using namespace Sophus;

struct PlaceRecognizerData
{
  int keyframe_id;
  Frame keyframe;
  tr1::unordered_set<int>  exclude_set;
  bool do_loop_detection;
};

struct DetectedLoop
{
  int query_keyframe_id;
  int loop_keyframe_id;
  SE3 T_query_from_loop;
};

class PlaceRecognizerMonitor
{
public:
  bool
  getKeyframeDate            (PlaceRecognizerData * data);

  void
  addKeyframeData            (const PlaceRecognizerData & data);

  bool
  getLoop                    (DetectedLoop * loop);

  void
  addLoop                    (const DetectedLoop & loop);

private:
  queue<PlaceRecognizerData> new_keyframe_queue_;
  stack<DetectedLoop> detected_loop_stack_;
  boost::mutex my_mutex_;
};

struct Place
{
  Place() : number_of_words(0) {}
  ALIGNED<Vector3d>::vector uvu_0_vec;
  ALIGNED<Vector3d>::vector xyz_vec;
  cv::Mat descriptors;
  int number_of_words;
  int keyframe_id;
};

class PlaceRecognizer
{
public:
  PlaceRecognizer            (const StereoCamera & stereo_cam_);

  void
  addLocation                (const PlaceRecognizerData & pr_data);

  void
  operator()                 ();

  PlaceRecognizerMonitor monitor;
  bool stop;

private:
  typedef cvflann::L2<float> distance_type;
  typedef cv::flann::GenericIndex< distance_type > generic_index_type;

  void
  geometricCheck             (const Place & query,
                              const Place & train);
  void
  calcLoopStatistics         (int cur_keyframe_id,
                              const tr1::unordered_set<int> & exclude_set,
                              const tr1::unordered_map< int,int>
                              & keyframe_to_wordcount_map,
                              tr1::unordered_map<int,float> & location_stats);

  const StereoCamera & stereo_cam_;
  tr1::shared_ptr<generic_index_type> flann_index_;
  cv::Mat words_;
  tr1::unordered_map<int,Place> location_map_;
  vector<IntTable > inverted_index_;
};
}

#endif
