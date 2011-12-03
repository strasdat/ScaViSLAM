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

#ifndef SCAVISLAM_MATCHER_HPP
#define SCAVISLAM_MATCHER_HPP

#include <stdint.h>
#include <tr1/memory>

#include <opencv2/opencv.hpp>

#include "global.h"
#include "quadtree.h"

namespace Sophus
{
class SE3;
}

namespace ScaViSLAM
{

using namespace Sophus;
template<int>
class IdObs;
template<int>
class CandidatePoint;
class Frame;
class Homography;
class SE3XYZ;

template <int obs_dim>
struct TrackData
{
  void clear()
  {
    obs_list.clear();
    point_list.clear();
    ba2globalptr.clear();
  }

  typename ALIGNED<IdObs<obs_dim> >::list obs_list;
  vector<Vector3d> point_list;
  vector<tr1::shared_ptr<CandidatePoint<obs_dim> > > ba2globalptr;
};

template <class Camera>
class GuidedMatcher
{
public:

  static void
  match                      (const tr1::unordered_map<int,Frame> &
                              keyframe_map,
                              const SE3 & T_cur_from_actkey,
                              const Frame & cur_frame,
                              const ALIGNED<QuadTree<int> >::vector &
                              feature_tree,
                              const typename ALIGNED<Camera>::vector & cam_vec,
                              int actkey_id,
                              const ALIGNED<SE3>::int_hash_map &
                              T_me_from_world_map,
                              const list< tr1::shared_ptr<
                              CandidatePoint<Camera::obs_dim> > > & ap_map,
                              int HALFBOXSIZE,
                              int SEARCHRADIUS,
                              int thr_mean,
                              int thr_std,
                              TrackData<Camera::obs_dim> * track_data);

  static cv::Mat
  warpPatchProjective        (const cv::Mat & frame,
                              const Homography & homo,
                              const Vector3d & xyz_c1,
                              const Vector3d & normal,
                              const Vector2d & key_uv,
                              const Camera & cam,
                              int HALFBOXSIZE);

private:
  struct MatchData
  {
    explicit MatchData(int min_dist)
      : min_dist(min_dist),
        index(-1),
        uv_pyr(0,0)
    {}
    double min_dist;
    int index;
    Vector2i uv_pyr;
  };

  static int
  matchPatchZeroMeanSSD      (const cv::Mat & patch_cur,
                              const cv::Mat & patch_key,
                              int key_pixel_sum,
                              int key_pixel_sum_sq);


  static cv::Mat
  warp2d                     (const cv::Mat & patch_in,
                              const Vector2f & uv_pyr);


  static void
  returnBestMatch            (const cv::Mat & ap_patch,
                              const Frame & cur_frame,
                              const MatchData & match_data,
                              const Vector3d & xyz_actkey,
                              const tr1::shared_ptr<
                              CandidatePoint<Camera::obs_dim> > & ap,
                              TrackData<Camera::obs_dim> * track_data);

  static void
  matchCandidates            (const ALIGNED<QuadTreeElement<int> >::list &
                              candidates,
                              const Frame & cur_frame,
                              const typename ALIGNED<Camera>::vector & cam_vec,
                              const cv::Mat & ap_patch,
                              int pixel_sum,
                              int pixel_sum_square,
                              int level,
                              int HALFBOXSIZE,
                              MatchData * match_data);

  static void
  computePatchScores         (const cv::Mat & patch,
                              int * pixel_sum,
                              int * pixel_sum_square);

  static bool
  computePrediction          (const SE3 & T_w_from_cur,
                              const SE3XYZ & se3xyz,
                              const typename ALIGNED<Camera>::vector & cam_vec,
                              const tr1::shared_ptr<CandidatePoint<Camera::obs_dim> > & ap,
                              const ALIGNED<SE3>::int_hash_map & T_me_from_world_map,
                              int HALFBOXSIZE,
                              Vector2d * uv_pyr,
                              SE3 * T_anchorkey_from_w);

  static bool
  createObervation           (const Vector2f & new_uv_pyr,
                              const cv::Mat & disp,
                              int level,
                              Matrix<double,Camera::obs_dim,1> * obs);

  static bool
  subpixelAccuracy           (const cv::Mat & key_patch_8u,
                              const Frame & cur_frame,
                              const Vector2i & uv_pyr_in,
                              int level,
                              Vector2f * uv_pyr_pout);

  DISALLOW_COPY_AND_ASSIGN(GuidedMatcher)
};
}

#endif
