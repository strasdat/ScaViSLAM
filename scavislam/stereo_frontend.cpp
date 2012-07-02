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

#include "stereo_frontend.h"

#include <stdint.h>

#include <opencv2/features2d/features2d.hpp>

#include <visiontools/accessor_macros.h>
#include <visiontools/performance_monitor.h>
#include <visiontools/stopwatch.h>

#include "backend.h"
#include "homography.h"
#include "matcher.hpp"
#include "maths_utils.h"
#include "quadtree.h"
#include "pose_optimizer.h"

namespace ScaViSLAM
{
using namespace VisionTools;

StereoFrontend
::StereoFrontend(FrameData<StereoCamera> * frame_data,
                 PerformanceMonitor * per_mon)
  : frame_data_(frame_data),
    per_mon_(per_mon),
    neighborhood_(new Neighborhood()),
    se3xyz_stereo_(frame_data->cam),
    unique_id_counter_(-1),
    tracker_(*frame_data_)
{
}


void StereoFrontend
::initialize()
{
  params_.newpoint_clearance =
      pangolin::Var<int>("newpoint_clearance",2);
  params_.covis_thr =
      pangolin::Var<int>("frontend.covis_thr",15);
  params_.num_frames_metric_loop_check
      = pangolin::Var<int>("frontend.num_frames_metric_loop_check",200);
  params_.new_keyframe_pixel_thr
      = pangolin::Var<int>("frontend.new_keyframe_pixel_thr",70);
  params_.new_keyframe_featuerless_corners_thr
      = pangolin::Var<int>("frontend.new_keyframe_featuerless_corners_thr",2);
  params_.save_dense_cloud
      = pangolin::Var<bool>("frontend.save_dense_cloud",true);

  pangolin::Var<size_t> use_n_levels_in_frontent("use_n_levels_in_frontent",2);
  USE_N_LEVELS_FOR_MATCHING = use_n_levels_in_frontent;

  fast_grid_.resize(USE_N_LEVELS_FOR_MATCHING);

  for (int l=0; l<USE_N_LEVELS_FOR_MATCHING; ++l)
  {
    int dim = std::max(3-(int)(l*0.5),1);
    int num_cells = dim*dim;
    double inv_fac = pyrFromZero_d(1.,l);

    int total_num_feat = 2000*inv_fac*inv_fac;
    int num_feat_per_cell = total_num_feat/num_cells;
    int bound = std::max((int)num_feat_per_cell/3,10);

    fast_grid_.at(l) = FastGrid(frame_data_->cam_vec[l].image_size(),
                                num_feat_per_cell,
                                bound,
                                25,
                                cv::Size(dim,dim));
  }
}

void StereoFrontend
::recomputeFastCorners(const Frame & frame,
                       ALIGNED<QuadTree<int> >::vector * feature_tree)
{
  int num_levels = frame.cell_grid2d.size();
  feature_tree->resize(num_levels);

  for (int level=0; level<num_levels; ++level)
  {
    feature_tree->at(level)
        =  QuadTree<int>(Rectangle(0,0,frame_data_->cam_vec.at(level).width(),
                                   frame_data_->cam_vec.at(level).height()),
                         1);
    FastGrid::detect(frame.pyr.at(level),
                     frame.cell_grid2d.at(level),
                     &(feature_tree->at(level)));
  }
}

void StereoFrontend
::processFirstFrame()
{
  draw_data_.clear();
  T_cur_from_actkey_ = SE3();

  ALIGNED<QuadTree<int> >::vector feature_tree;

  per_mon_->start("dense point cloud");

  actkey_id = getNewUniqueId();
  FrontendVertex vf;
  vf.T_me_from_w = T_cur_from_actkey_;
  neighborhood_->vertex_map.insert(make_pair(actkey_id, vf));


  if (frame_data_->have_disp_img)
  {
    vector<cv::Mat> hsv_array(3);
    hsv_array[0] = cv::Mat(frame_data_->disp.size(), CV_8UC1);

    frame_data_->disp.convertTo(hsv_array[0],CV_8UC1, 5.,0.);
    hsv_array[1] = cv::Mat(frame_data_->disp.size(), CV_8UC1, 255);
    hsv_array[2] = cv::Mat(frame_data_->disp.size(), CV_8UC1, 255);

    cv::Mat hsv(frame_data_->disp.size(), CV_8UC3);
    cv::merge(hsv_array, hsv);
    cv::cvtColor(hsv, frame_data_->color_disp, CV_HSV2BGR);
#ifdef SCAVISLAM_CUDA_SUPPORT
    frame_data_->gpu_disp_32f.upload(frame_data_->disp);
#endif
  }
  else
  {
#ifdef SCAVISLAM_CUDA_SUPPORT
    calcDisparityGpu();
#else
    calcDisparityCpu();
#endif
  }

  Frame kf = Frame(frame_data_->cur_left().pyr_uint8,
                   frame_data_->disp).clone();

  per_mon_->start("fast");
  computeFastCorners(5, &feature_tree, &kf.cell_grid2d);
  per_mon_->stop("fast");

  addNewPoints(actkey_id,feature_tree);

  AddToOptimzerPtr to_optimizer(new AddToOptimzer(true));
  to_optimizer->newkey_id = actkey_id;
  to_optimizer->kf = kf;

  assert(to_optimizer->kf.cell_grid2d.size()>0);

  //make sure keyframe is added before pushing to optimiser_stack!!
  keyframe_map.insert(make_pair(actkey_id,kf));
  keyframe_id2num.insert(make_pair(actkey_id, keyframe_id2num.size()));
  keyframe_num2id.push_back(actkey_id);


  to_optimizer_stack.push(to_optimizer);

#ifdef SCAVISLAM_CUDA_SUPPORT
  tracker_.computeDensePointCloudGpu(T_cur_from_actkey_);
#else
  tracker_.computeDensePointCloudCpu(T_cur_from_actkey_);
#endif
  per_mon_->stop("dense point cloud");
}


bool StereoFrontend
::processFrame(bool * is_frame_dropped)
{
  draw_data_.clear();

  const ALIGNED<StereoCamera>::vector & cam_vec = frame_data_->cam_vec;


  per_mon_->start("dense tracking");
#ifdef SCAVISLAM_CUDA_SUPPORT
  tracker_.denseTrackingGpu(&T_cur_from_actkey_);
#else
  tracker_.denseTrackingCpu(&T_cur_from_actkey_);
#endif
  per_mon_->stop("dense tracking");

  per_mon_->start("stereo");

  if (frame_data_->have_disp_img)
  {
    vector<cv::Mat> hsv_array(3);
    hsv_array[0] = cv::Mat(frame_data_->disp.size(), CV_8UC1);

    frame_data_->disp.convertTo(hsv_array[0],CV_8UC1, 5.,0.);
    hsv_array[1] = cv::Mat(frame_data_->disp.size(), CV_8UC1, 255);
    hsv_array[2] = cv::Mat(frame_data_->disp.size(), CV_8UC1, 255);

    cv::Mat hsv(frame_data_->disp.size(), CV_8UC3);
    cv::merge(hsv_array, hsv);
    cv::cvtColor(hsv, frame_data_->color_disp, CV_HSV2BGR);
#ifdef SCAVISLAM_CUDA_SUPPORT
    frame_data_->gpu_disp_32f.upload(frame_data_->disp);
#endif
  }
  else
  {
#ifdef SCAVISLAM_CUDA_SUPPORT
    calcDisparityGpu();
#else
    calcDisparityCpu();
#endif
  }
  per_mon_->stop("stereo");

  per_mon_->start("fast");
  cur_frame_ = Frame(frame_data_->cur_left().pyr_uint8,
                     frame_data_->disp);
  /*cur_frame_.feature_tree*/
  ALIGNED<QuadTree<int> >::vector feature_tree;
  computeFastCorners(6, &feature_tree, &cur_frame_.cell_grid2d);
  per_mon_->stop("fast");

  per_mon_->start("match");
  ScaViSLAM::TrackData<3> track_data;
  int num_new_feat_matched;
  bool matched_enough_features = matchAndTrack(feature_tree,
                                               &track_data,
                                               &num_new_feat_matched);
  per_mon_->stop("match");

  if (matched_enough_features==false)
    return false;
  per_mon_->start("process points");
  PointStatistics point_stats(USE_N_LEVELS_FOR_MATCHING);
  tr1::unordered_set<CandidatePoint3Ptr > matched_new_feat;
  ALIGNED<QuadTree<int> >::vector  point_tree(USE_N_LEVELS_FOR_MATCHING);
  for (int l=0; l<USE_N_LEVELS_FOR_MATCHING; ++l)
  {
    point_tree.at(l)
        = QuadTree<int>(Rectangle(0,0,
                                  cam_vec[l].width(),cam_vec[l].height()),
                        1);
  }
  AddToOptimzerPtr to_optimizer
      = processMatchedPoints(track_data,
                             num_new_feat_matched,
                             &point_tree,
                             &matched_new_feat,
                             &point_stats);
  per_mon_->stop("process points");

  per_mon_->start("drop keyframe");


  int other_id = -1;
  SE3 T_cur_from_other;

  ALIGNED<QuadTree<int> >::vector other_point_tree;
  PointStatistics other_stat(USE_N_LEVELS_FOR_MATCHING);
  if (shallWeSwitchKeyframe(to_optimizer->track_point_list,
                            &other_id, &T_cur_from_other,
                            &other_point_tree,
                            &other_stat))
  {
    actkey_id = other_id;
    T_cur_from_actkey_ = T_cur_from_other;

  }
  else
  {


    *is_frame_dropped = shallWeDropNewKeyframe(point_stats);
    if(*is_frame_dropped)
    {
      addNewKeyframe(feature_tree,
                     to_optimizer,
                     &matched_new_feat,
                     &point_tree,
                     &point_stats);
    }
  }
  per_mon_->stop("drop keyframe");

  per_mon_->start("dense point cloud");
#ifdef SCAVISLAM_CUDA_SUPPORT
  tracker_.computeDensePointCloudGpu(T_cur_from_actkey_);
#else
  tracker_.computeDensePointCloudCpu(T_cur_from_actkey_);
#endif
  per_mon_->stop("dense point cloud");

  return true;
}


void StereoFrontend
::addNewKeyframe(const ALIGNED<QuadTree<int> >::vector & feature_tree,
                 const AddToOptimzerPtr & to_optimizer,
                 tr1::unordered_set<CandidatePoint3Ptr > * matched_new_feat,
                 ALIGNED<QuadTree<int> >::vector * point_tree,
                 PointStatistics * point_stats)
{
  Matrix3i add_flags;
  add_flags.setZero();

  static pangolin::Var<int> ui_min_num_points
      ("ui.min_num_points",25,20,200);

  for (int i = 0; i<3; ++i)
  {
    for (int j = 0; j<3; ++j)
    {
      if(point_stats->num_points_grid3x3(i,j)<=ui_min_num_points)
      {
        add_flags(i,j) = 1;
      }
    }
  }

  int oldkey_id = actkey_id;
  actkey_id = getNewUniqueId();

  const SE3 & T_oldkey_from_w
      = GET_MAP_ELEM(oldkey_id, neighborhood_->vertex_map).T_me_from_w;


  tr1::unordered_map<int,int> num_matches;
  FrontendVertex vf;
  vf.T_me_from_w = T_cur_from_actkey_*T_oldkey_from_w;

  for (tr1::unordered_set<CandidatePoint3Ptr>::const_iterator it
       = matched_new_feat->begin(); it!= matched_new_feat->end(); ++it)
  {
    const CandidatePoint3Ptr & p = *it;
    neighborhood_->point_list.push_back(p);
  }

  for (list<NewTwoViewPoint3Ptr>::const_iterator it
       = to_optimizer->new_point_list.begin();
       it!= to_optimizer->new_point_list.end(); ++it)
  {
    const NewTwoViewPoint3Ptr & p = *it;
    ADD_TO_MAP_ELEM(p->anchor_id, 1, &num_matches);
    vf.feat_map.insert(make_pair(p->point_id, p->feat_newkey));
  }

  tr1::unordered_map<int,list<CandidatePoint3Ptr > >::const_iterator it
      = newpoint_map.begin();
  for (; it!=newpoint_map.end(); ++it)
  {
    newpoint_map[it->first].remove_if(RemoveCondition(*matched_new_feat));
  }


  const FrontendVertex & oldactive_vertex
      = GET_MAP_ELEM(oldkey_id, neighborhood_->vertex_map);


  for (list<TrackPoint3Ptr>::const_iterator itp
       = to_optimizer->track_point_list.begin();
       itp != to_optimizer->track_point_list.end(); ++itp)
  {

    const TrackPoint3Ptr & p = *itp;


    const ImageFeature<3>::Table & old_feat_map
        = GET_MAP_ELEM(oldkey_id, neighborhood_->vertex_map).feat_map;

    if (IS_IN_SET( p->global_id, old_feat_map))
    {
      ADD_TO_MAP_ELEM(oldkey_id, 1, &num_matches);
    }


    for (multimap<int,int>::const_iterator it
         = oldactive_vertex.strength_to_neighbors.begin();
         it!=oldactive_vertex.strength_to_neighbors.end(); ++it)
    {
      int other_pose_id = it->second;
      const ImageFeature<3>::Table & other_feat_map
          = GET_MAP_ELEM(other_pose_id, neighborhood_->vertex_map).feat_map;

      if (IS_IN_SET( p->global_id, other_feat_map))
      {
        ADD_TO_MAP_ELEM(other_pose_id, 1, &num_matches);
      }
    }
    vf.feat_map.insert(make_pair(p->global_id, p->feat));
  }

  for (tr1::unordered_map<int,int>::const_iterator it = num_matches.begin();
       it != num_matches.end(); ++it)
  {
    int pose_id = it->first;
    int num_machtes = it->second;

    if (num_machtes>params_.covis_thr)
    {
      vf.strength_to_neighbors.insert(make_pair(num_machtes,pose_id));
    }
  }

  neighborhood_->vertex_map.insert(
        make_pair(actkey_id, vf));



  addMorePoints(
        actkey_id,
        feature_tree,
        add_flags,
        point_tree,
        &(point_stats->num_matched_points));

  to_optimizer->newkey_id = actkey_id;
  to_optimizer->oldkey_id = oldkey_id;
  to_optimizer->T_newkey_from_oldkey = T_cur_from_actkey_;

  Frame kf = cur_frame_.clone();
  to_optimizer->kf = kf;

  //make sure keyframe is added before pushinh to optimiser_stack!!
  keyframe_map.insert(make_pair(actkey_id,kf));
  keyframe_id2num.insert(make_pair(actkey_id, keyframe_id2num.size()));
  keyframe_num2id.push_back(actkey_id);
  to_optimizer_stack.push(to_optimizer);

  T_cur_from_actkey_ = SE3();
}

bool StereoFrontend
::shallWeSwitchKeyframe(const list<TrackPoint3Ptr> & trackpoint_list,
                        int * other_id,
                        SE3 * T_cur_from_other,
                        ALIGNED<QuadTree<int> >::vector * other_point_tree,
                        PointStatistics * other_stat)
{
  static pangolin::Var<float> ui_parallax_thr
      ("ui.parallax_thr",0.75f,0,2);

  double min_dist = 0.5*ui_parallax_thr;
  int closest = -1;

  const SE3 & T_act_from_w
      = GET_MAP_ELEM(actkey_id,
                     neighborhood_->vertex_map).T_me_from_w;

  ALIGNED<FrontendVertex>::int_hash_map::const_iterator it
      = neighborhood_->vertex_map.begin();
  for (;
       it!=neighborhood_->vertex_map.end();
       ++it)
  {
    int other_id = it->first;

    if (other_id!=actkey_id)
    {
      const SE3 & T_other_from_w = it->second.T_me_from_w;
      SE3 T_diff = T_cur_from_actkey_*T_act_from_w*T_other_from_w.inverse();
      double dist = T_diff.translation().norm();

      if (dist<min_dist)
      {
        *T_cur_from_other = T_diff;
        min_dist = dist;
        closest = other_id;
      }
    }
  }

  if (closest!=-1 )
  {
    ImageFeature<3>::Table feat_table
        = GET_MAP_ELEM(closest, neighborhood_->vertex_map).feat_map;

    int count = 0;

    for (list<TrackPoint3Ptr>::const_iterator it = trackpoint_list.begin();
         it!= trackpoint_list.end(); ++it)
    {
      const TrackPoint3Ptr & p = *it;
      if (IS_IN_SET(p->global_id, feat_table))
      {
        ++count;
      }
    }

    //TODO: add more sophisticated check
    if (count>100)
    {
      *other_id = closest;
      return true;
    }
  }
  return  false;
}

bool StereoFrontend
::shallWeDropNewKeyframe(const PointStatistics & point_stats)
{
  int num_featuerless_corners=0;

  for (int i=0; i<2; ++i)
    for (int j=0; j<2; ++j)
      if (point_stats.num_points_grid2x2(i,j)<15)
        ++num_featuerless_corners;

  static pangolin::Var<float> ui_parallax_thr
      ("ui.parallax_thr",0.75f,0,2);

  return num_featuerless_corners>params_.new_keyframe_featuerless_corners_thr
      || T_cur_from_actkey_.translation().norm()>ui_parallax_thr
      || av_track_length_>75.;
}

#ifdef SCAVISLAM_CUDA_SUPPORT
//TODO: Method too long
void StereoFrontend::
calcDisparityGpu()
{
  static pangolin::Var<int> stereo_method("ui.stereo_method",2,1,4);
  static pangolin::Var<int> num_disp16("ui.num_disp16",2,1,10);
  int num_disparities = num_disp16*16;

  if (stereo_method==1)
  {
    cv::StereoBM stereo_bm;
    stereo_bm.state->preFilterCap = 31;
    stereo_bm.state->SADWindowSize = 7;
    stereo_bm.state->minDisparity = 0;

    stereo_bm.state->textureThreshold = 10;
    stereo_bm.state->uniquenessRatio = 15;
    stereo_bm.state->speckleWindowSize = 100;
    stereo_bm.state->speckleRange = 32;
    stereo_bm.state->disp12MaxDiff = 1;

    stereo_bm.state->numberOfDisparities = num_disparities;

    stereo_bm(frame_data_->cur_left().pyr_uint8[0],
              frame_data_->right.uint8,
              frame_data_->disp,
              CV_32F);

    frame_data_->gpu_disp_32f.upload(frame_data_->disp);
    frame_data_->gpu_disp_32f.convertTo(frame_data_->gpu_disp_16s,CV_16S, 1.);
    cv::gpu::drawColorDisp(frame_data_->gpu_disp_16s,
                           frame_data_->gpu_color_disp,num_disparities);
    frame_data_->gpu_color_disp.download(frame_data_->color_disp);

  }
  else if (stereo_method==2)
  {
    cv::gpu::StereoBM_GPU gpu_stereo_bm(cv::gpu::StereoBM_GPU::PREFILTER_XSOBEL,
                                        num_disparities);

    gpu_stereo_bm(frame_data_->cur_left().gpu_uint8,
                  frame_data_->right.gpu_uint8,
                  frame_data_->gpu_disp_16s);
    cv::gpu::drawColorDisp(frame_data_->gpu_disp_16s,
                           frame_data_->gpu_color_disp,
                           num_disparities);
    frame_data_->gpu_disp_16s.convertTo(frame_data_->gpu_disp_32f,CV_32F,1.);
    frame_data_->gpu_disp_32f.download(frame_data_->disp);
    frame_data_->gpu_color_disp.download(frame_data_->color_disp);
  }
  else if (stereo_method==3)
  {
    cv::gpu::StereoBeliefPropagation gpu_stereo_bm(num_disparities);
    gpu_stereo_bm(frame_data_->cur_left().gpu_uint8,
                  frame_data_->right.gpu_uint8,
                  frame_data_->gpu_disp_16s);
    cv::gpu::drawColorDisp(frame_data_->gpu_disp_16s,
                           frame_data_->gpu_color_disp,
                           num_disparities);
    frame_data_->gpu_disp_16s.convertTo(frame_data_->gpu_disp_32f,CV_32F,1.);
    frame_data_->gpu_disp_32f.download(frame_data_->disp);
    frame_data_->gpu_color_disp.download(frame_data_->color_disp);
  }
  else if (stereo_method==4)
  {
    static pangolin::Var<int> stereo_iters("ui.stereo_iters",4,1,20);
    static pangolin::Var<int> stereo_levels("ui.stereo_levels",4,1,5);
    static pangolin::Var<int> stereo_nr_plane("ui.stereo_nr_plane",1,1,10);
    cv::gpu::StereoConstantSpaceBP gpu_stereo_bm(num_disparities,
                                                 stereo_iters,
                                                 stereo_levels,
                                                 stereo_nr_plane);
    gpu_stereo_bm(frame_data_->cur_left().gpu_uint8,
                  frame_data_->right.gpu_uint8,
                  frame_data_->gpu_disp_16s);

    cv::gpu::drawColorDisp(frame_data_->gpu_disp_16s,
                           frame_data_->gpu_color_disp,
                           num_disparities);
    frame_data_->gpu_disp_16s.convertTo(frame_data_->gpu_disp_32f,CV_32F,1.);
    frame_data_->gpu_disp_32f.download(frame_data_->disp);
    frame_data_->gpu_color_disp.download(frame_data_->color_disp);
  }
}



#else

void StereoFrontend
::calcDisparityCpu()
{
  static pangolin::Var<int> num_disp16("ui.num_disp16",2,1,10);
  int num_disparities = num_disp16*16;
  cv::StereoBM stereo_bm;
  stereo_bm.state->preFilterCap = 31;
  stereo_bm.state->SADWindowSize = 7;
  stereo_bm.state->minDisparity = 0;

  stereo_bm.state->textureThreshold = 10;
  stereo_bm.state->uniquenessRatio = 15;
  stereo_bm.state->speckleWindowSize = 100;
  stereo_bm.state->speckleRange = 32;
  stereo_bm.state->disp12MaxDiff = 1;

  stereo_bm.state->numberOfDisparities = num_disparities;

  stereo_bm(frame_data_->cur_left().pyr_uint8[0],
            frame_data_->right.uint8,
            frame_data_->disp,
            CV_32F);

  vector<cv::Mat> hsv_array(3);
  hsv_array[0] = cv::Mat(frame_data_->disp.size(), CV_8UC1);

  frame_data_->disp.convertTo(hsv_array[0],CV_8UC1, 5.,0.);
  hsv_array[1] = cv::Mat(frame_data_->disp.size(), CV_8UC1, 255);
  hsv_array[2] = cv::Mat(frame_data_->disp.size(), CV_8UC1, 255);

  cv::Mat hsv(frame_data_->disp.size(), CV_8UC3);
  cv::merge(hsv_array, hsv);
  cv::cvtColor(hsv, frame_data_->color_disp, CV_HSV2BGR);
}

#endif

void StereoFrontend
::computeFastCorners(int trials,
                     ALIGNED<QuadTree<int> >::vector * feature_tree,
                     vector<CellGrid2d> * cell_grid_2d)
{
  const ALIGNED<StereoCamera>::vector & cam_vec
      = frame_data_->cam_vec;
  feature_tree->resize(USE_N_LEVELS_FOR_MATCHING);
  cell_grid_2d->resize(USE_N_LEVELS_FOR_MATCHING);

  for (int level=0; level<USE_N_LEVELS_FOR_MATCHING; ++level)
  {
    feature_tree->at(level)
        =  QuadTree<int>(Rectangle(0,0,cam_vec.at(level).width(),
                                   cam_vec.at(level).height()),
                         1);
    fast_grid_.at(level).detectAdaptively(
          frame_data_->cur_left().pyr_uint8.at(level),
          trials,
          &(feature_tree->at(level)));
    cell_grid_2d->at(level) = fast_grid_.at(level).cell_grid2d();
  }
}


void StereoFrontend
::addNewPoints(int new_keyframe_id,
               const ALIGNED<QuadTree<int> >::vector & feature_tree)
{
  vector<int> num_points(USE_N_LEVELS_FOR_MATCHING, 0);
  const ALIGNED<StereoCamera>::vector & cam_vec = frame_data_->cam_vec;

  ALIGNED<QuadTree<int> >::vector  point_tree(USE_N_LEVELS_FOR_MATCHING);

  for (int l=0; l<USE_N_LEVELS_FOR_MATCHING; ++l)
  {
    point_tree.at(l)
        = QuadTree<int>(Rectangle(0,0,
                                  cam_vec[l].width(),cam_vec[l].height()),
                        1);
  }

  Matrix3i add_flags;
  add_flags.setOnes();
  return addMorePoints(new_keyframe_id, feature_tree,
                       add_flags,
                       &point_tree, &num_points);
}

void  StereoFrontend
::addMorePoints(int new_keyframe_id,
                const ALIGNED<QuadTree<int> >::vector & feature_tree,
                const Matrix3i & add_flags,
                ALIGNED<QuadTree<int> >::vector * point_tree,
                vector<int> * num_points)
{
  addMorePointsToOtherFrame(new_keyframe_id,
                            SE3(),
                            feature_tree,
                            add_flags,
                            frame_data_->disp,
                            point_tree,
                            num_points);
}


//TODO: method too long
void  StereoFrontend
::addMorePointsToOtherFrame(int new_keyframe_id,
                            const SE3 & T_newkey_from_cur,
                            const ALIGNED<QuadTree<int> >::vector & feature_tree,
                            const Matrix3i & add_flags,
                            const cv::Mat & disp_img,
                            ALIGNED<QuadTree<int> >::vector * point_tree,
                            vector<int> * num_points)
{
  const StereoCamera & cam = frame_data_->cam;

  int CLEARANCE_RADIUS = params_.newpoint_clearance;
  int CLEARANCE_DIAMETER = CLEARANCE_RADIUS*2+1;

  float third = 1./3.;
  int third_width = cam.width()*third;
  int third_height = cam.height()*third;
  int twothird_width = cam.width()*2*third;
  int twothird_height = cam.height()*2*third;
  pangolin::Var<int> var_num_max_points("ui.num_max_points",300,50,1000);

  for (int level = 0; level<USE_N_LEVELS_FOR_MATCHING;++level)
  {
    DrawItems::Point2dVec new_point_vec;

    int num_max_points = pyrFromZero_i(var_num_max_points, level);
    for (QuadTree<int>::EquiIter iter=feature_tree.at(level).begin_equi();
         !iter.reached_end();
         ++iter)
    {
      Vector2d uv_pyr = iter->pos;
      Vector2i uv_pyri = uv_pyr.cast<int>();

      double disp
          = interpolateDisparity(disp_img,
                                 uv_pyri, level);
      if (disp>0)
      {
        Vector2i uvi = zeroFromPyr_2i(uv_pyri,level);

        if (!cam.isInFrame(uvi,1))
          continue;

        int i = 2;
        int j = 2;
        if (uvi[0]<third_width)
          i = 0;
        else if (uvi[0]<twothird_width)
          i = 1;
        if (uvi[1]<third_height)
          j = 0;
        else if (uvi[1]<twothird_height)
          j = 1;

        if (add_flags(i,j)==0)
          continue;


        Rectangle win(uv_pyr[0]-CLEARANCE_RADIUS,
                      uv_pyr[1]-CLEARANCE_RADIUS,
                      CLEARANCE_DIAMETER,
                      CLEARANCE_DIAMETER);

        if((*point_tree)[level].isWindowEmpty(win))
        {
          new_point_vec.push_back(GlPoint2f(uv_pyr[0],uv_pyr[1]));

          Vector3d uvu_pyr
              = Vector3d(uv_pyr[0],uv_pyr[1],uv_pyr[0]-disp);

          Vector3d uvu_0
              = zeroFromPyr_3d(uvu_pyr,level);

          Vector3d xyz_cur = cam.unmap_uvu(uvu_0);

          (*point_tree)[level].insert(uv_pyr,(*num_points)[level]);
          draw_data_.new_points2d.at(level).push_back(GlPoint2f(uv_pyr));
          draw_data_.new_points3d.at(level).push_back(GlPoint3f(xyz_cur));

          double dist = xyz_cur.norm();
          Vector3d normal = -xyz_cur/dist;

          int new_point_id = getNewUniqueId();

          newpoint_map[new_keyframe_id].push_front(
                CandidatePoint3Ptr(new CandidatePoint<3>(new_point_id,
                                                         T_newkey_from_cur*xyz_cur,
                                                         new_keyframe_id,
                                                         uvu_pyr,
                                                         level,
                                                         normal)));
          ++(*num_points)[level];

          if ((*num_points)[level]>num_max_points)
            break;
        }
      }
    }
  }
}


int StereoFrontend
::getNewUniqueId()
{
  ++unique_id_counter_;
  return unique_id_counter_;
}

//TODO: method too long
AddToOptimzerPtr StereoFrontend
::processMatchedPoints(const TrackData<3> & track_data,
                       int num_new_feat_matched,
                       ALIGNED<QuadTree<int> >::vector * point_tree,
                       tr1::unordered_set<CandidatePoint3Ptr > * matched_new_feat,
                       PointStatistics * stats)
{
  AddToOptimzerPtr to_optimizer(new AddToOptimzer);
  const StereoCamera & cam = frame_data_->cam;
  SE3XYZ se3xyz(cam);

  static  pangolin::Var<float> max_reproj_error
      ("ui.max_reproj_error",2,0,5);

  int half_width = cam.width()*0.5;
  int half_height = cam.height()*0.5;
  float third = 1./3.;
  int third_width = cam.width()*third;
  int third_height = cam.height()*third;
  int twothird_width = cam.width()*2*third;
  int twothird_height = cam.height()*2*third;

  int num_track_points = 0;
  double sum_track_length = 0.f;

  for (list<IdObs<3> >::const_iterator it=track_data.obs_list.begin();
       it!=track_data.obs_list.end(); ++it)
  {
    IdObs<3> id_obs = *it;
    const Vector3d & point = track_data.point_list.at(id_obs.point_id);
    const Vector3d & uvu_pred = se3xyz_stereo_.map(T_cur_from_actkey_,point);
    const Vector3d & uvu = id_obs.obs;
    Vector3d diff = uvu - uvu_pred;
    const CandidatePoint3Ptr & ap
        = track_data.ba2globalptr.at(id_obs.point_id);
    int factor = zeroFromPyr_i(1, ap->anchor_level);

    if (abs(diff[0])<max_reproj_error*factor
        && abs(diff[1])<max_reproj_error*factor
        && abs(diff[2])<3.*max_reproj_error)
    {

      int i = 1;
      int j = 1;
      if (uvu[0]<half_width)
        i = 0;
      if (uvu[1]<half_height)
        j = 0;
      ++(stats->num_points_grid2x2(i,j));

      i = 2;
      j = 2;
      if (uvu[0]<third_width)
        i = 0;
      else if (uvu[0]<twothird_width)
        i = 1;
      if (uvu[1]<third_height)
        j = 0;
      else if (uvu[1]<twothird_height)
        j = 1;
      ++(stats->num_points_grid3x3(i,j));


      ++(stats->num_matched_points)[ap->anchor_level];
      Vector2d curkey_uv_pyr
          = pyrFromZero_2d(se3xyz.map(SE3(),point),
                           ap->anchor_level);
      Vector2d uv_pyr = pyrFromZero_2d(Vector2d(uvu.head(2)),
                                       ap->anchor_level);

      ALIGNED<DrawItems::Point2dVec>::int_hash_map & keymap
          = draw_data_.tracked_anchorpoints2d.at(ap->anchor_level);
      ALIGNED<DrawItems::Point2dVec>::int_hash_map::iterator keymap_it
          = keymap.find(ap->anchor_id);

      if (keymap_it!=keymap.end())
      {
        keymap_it->second.push_back(GlPoint2f(curkey_uv_pyr));
      }
      else
      {
        DrawItems::Point2dVec point_vec;
        point_vec.push_back(GlPoint2f(curkey_uv_pyr));
        keymap.insert(make_pair(ap->anchor_id,point_vec));
      }

      (*point_tree).at(ap->anchor_level).insert(uv_pyr,ap->point_id);

      if (id_obs.point_id<num_new_feat_matched)
      {
        (*matched_new_feat).insert(ap);
        draw_data_.newtracked_points2d.at(ap->anchor_level)
            .push_back(DrawItems::Line2d(uv_pyr,curkey_uv_pyr));

        sum_track_length += (uv_pyr-curkey_uv_pyr).norm();
        ++num_track_points;

        SE3 T_w_from_anchor
            = GET_MAP_ELEM(ap->anchor_id, neighborhood_->vertex_map)
            .T_me_from_w.inverse();
        draw_data_.newtracked_points3d.at(ap->anchor_level)
            .push_back(GlPoint3f(T_w_from_anchor*ap->xyz_anchor));

        ImageFeature<3>feat (uvu,
                             ap->anchor_level);
        NewTwoViewPoint3Ptr np(
              new   NewTwoViewPoint<3>(ap->point_id,
                                       ap->anchor_id,
                                       ap->xyz_anchor,
                                       ap->anchor_obs_pyr,
                                       ap->anchor_level,
                                       ap->normal_anchor,
                                       feat));
        to_optimizer->new_point_list.push_back(np);
      }
      else
      {
        ImageFeature<3> feat(id_obs.obs,
                             ap->anchor_level);
        to_optimizer->track_point_list.push_back(
              TrackPoint3Ptr(new TrackPoint<3> (ap->point_id,
                                                feat)));
        draw_data_.tracked_points2d.at(ap->anchor_level)
            .push_back(DrawItems::Line2d(uv_pyr,curkey_uv_pyr));

        sum_track_length += (uv_pyr-curkey_uv_pyr).norm();
        ++num_track_points;
        SE3 T_w_from_anchor
            = GET_MAP_ELEM(ap->anchor_id, neighborhood_->vertex_map)
            .T_me_from_w.inverse();
        draw_data_.tracked_points3d.at(ap->anchor_level)
            .push_back(GlPoint3f(T_w_from_anchor*ap->xyz_anchor));
      }
    }
  }
  av_track_length_ = sum_track_length/num_track_points;

 // cerr << av_track_length_ << endl;

  return to_optimizer;
}


bool StereoFrontend
::matchAndTrack(const ALIGNED<QuadTree<int> >::vector & feature_tree,
                TrackData<3> * track_data,
                int * num_new_feat_matched)
{

  BA_SE3_XYZ_STEREO ba;
  *num_new_feat_matched  = 0;

  const FrontendVertex & active_vertex
      = GET_MAP_ELEM(actkey_id, neighborhood_->vertex_map);

  GuidedMatcher<StereoCamera>::match(keyframe_map,
                                     T_cur_from_actkey_,
                                     cur_frame_,
                                     feature_tree,
                                     frame_data_->cam_vec,
                                     actkey_id,
                                     neighborhood_->vertex_map,
                                     newpoint_map[actkey_id],
                                 #ifdef SCAVISLAM_CUDA_SUPPORT
                                     4,
                                 #else
                                     8,
                                 #endif
                                     22,
                                     10,
                                     track_data);

  pangolin::Var<int> var_num_max_points("ui.num_max_points",300,50,1000);
  for (multimap<int,int>::const_iterator it
       = active_vertex.strength_to_neighbors.begin();
       it!=active_vertex.strength_to_neighbors.end()
       && (int)(2*track_data->obs_list.size())<var_num_max_points; ++it)
  {
    GuidedMatcher<StereoCamera>::match(keyframe_map,
                                       T_cur_from_actkey_,
                                       cur_frame_,
                                       feature_tree,
                                       frame_data_->cam_vec,
                                       actkey_id,
                                       neighborhood_->vertex_map,
                                       newpoint_map[it->second],
                                   #ifdef SCAVISLAM_CUDA_SUPPORT
                                       4,
                                   #else
                                       8,
                                   #endif
                                       22,
                                       10,
                                       track_data);

  }
  *num_new_feat_matched  += track_data->obs_list.size();




  GuidedMatcher<StereoCamera>::match(keyframe_map,
                                     T_cur_from_actkey_,
                                     cur_frame_,
                                     feature_tree,
                                     frame_data_->cam_vec,
                                     actkey_id,
                                     neighborhood_->vertex_map,
                                     neighborhood_->point_list,
                                   #ifdef SCAVISLAM_CUDA_SUPPORT
                                     4,
                                   #else
                                     8,
                                   #endif
                                     22,
                                     10,
                                     track_data);


  if (track_data->obs_list.size()<20)
  {
    return false;
  }

  OptimizerStatistics opt
      = ba.calcFastMotionOnly(track_data->obs_list,
                              se3xyz_stereo_,
                              PoseOptimizerParams(true,2,15),
                              &T_cur_from_actkey_,
                              &track_data->point_list);
  return true;
}


}
