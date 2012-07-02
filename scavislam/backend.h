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

#ifndef SCAVISLAM_BACKEND_H
#define SCAVISLAM_BACKEND_H

#include "global.h"
#include "placerecognizer.h"
#include "slam_graph.hpp"
#include "stereo_camera.h"
#include "transformations.h"

#include <boost/thread.hpp>

namespace ScaViSLAM
{

typedef SlamGraph<SE3,StereoCamera,SE3XYZ_STEREO,3> StereoGraph;


struct BackendDrawData
{
  StereoGraph::WindowTable double_window;
  tr1::unordered_set<int> active_point_set;
  tr1::unordered_set<int> outer_point_set;
  StereoGraph::VertexTable vertex_table;
  StereoGraph::PointTable  point_table;
  StereoGraph::EdgeTable edge_table;
  StereoGraph::EdgeTable new_edges;
};

typedef tr1::shared_ptr<BackendDrawData> BackendDrawDataPtr;

class BackendMonitor
{
public:

  BackendMonitor() {}

  void
  queryNeighborhood          (int frame_id);

  bool
  getQueryFrameId            (int * frame_id);

  void
  pushNeighborhood           (const NeighborhoodPtr & neighborhood);

  bool
  getNeighborhood            (NeighborhoodPtr * neighborhood);

  void
  pushKeyframe               (const AddToOptimzerPtr & to_optimizer);

  bool
  getKeyframe                (AddToOptimzerPtr * to_optimizer);

  void
  pushDrawData               (const BackendDrawDataPtr & draw_data);

  bool
  getDrawData                (BackendDrawDataPtr * draw_data);

  void
  pushClosedLoop             (const DetectedLoop & loop);

  bool
  getClosedLoop              (DetectedLoop * loop);


private:
  queue<AddToOptimzerPtr> new_keyframe_queue_;
  list<int> query_frame_id_storage_;
  list<NeighborhoodPtr> neighborhood_ptr_storage_;
  list<BackendDrawDataPtr> draw_storage_;
  list<DetectedLoop> loop_storage_;

  boost::mutex my_mutex_;
};


class Backend
{
public:
  Backend(const ALIGNED<StereoCamera>::vector & cam_vec_,
          PlaceRecognizerMonitor * place_reg_monitor);

  void
  operator()();

  BackendMonitor monitor;
  bool stop;

private:
  struct ImageStats
  {
    ImageStats() : num_upper(0), num_lower(0),num_left(0),num_right(0) {}
    int num_upper;
    int num_lower;
    int num_left;
    int num_right;
    list<StereoGraph::MyTrackPointPtr> point_list;
  };

  typedef
  tr1::unordered_map<int, ImageStats>
  ImageStatsTable;



  NeighborhoodPtr
  computeNeighborhood        (int query_frame_id) const;

  void
  addKeyframeToGraph         (const AddToOptimzerPtr & to_optimiser);

  void
  addKeyframeToPlaceRecog    (const AddToOptimzerPtr & to_optimiser);

  void
  findNeighborsOfRoot        (int root_id,
                              int max_num_neighbors,
                              Neighborhood * neighborhood) const;
  void
  addPointsToNeighbors       (Neighborhood * neighborhood) const;

  void
  addAnchorPosesToNeighbors  (Neighborhood * neighborhood) const;

  void
  addPoseToNeighborhood      (int new_pose_id,
                              Neighborhood * neighborhood) const;
  bool
  localRegisterFrame         (int frame_id);

  tr1::unordered_set<int>
  directNeighborsOf          (int frame_id);

  BackendDrawDataPtr
  cloneDrawData              () const;

  void
  recomputeFastCorners       (const Frame & frame,
                              ALIGNED<QuadTree<int> >::vector * feature_tree);
  void
  pointsVisibleInRoot        (const SE3 & T_root_from_world,
                              const tr1::unordered_set<int> &
                              larger_neighborhood,
                              const tr1::unordered_set<int> &
                              direct_neighbors,
                              list< CandidatePoint3Ptr > * point_list,
                              ALIGNED<FrontendVertex>::int_hash_map
                              * vertex_table);
  void
  dumpRegistrationData       (const Frame & root_frame,
                              const IntTable & neighborid_to_strength,
                              const ImageStatsTable & frameid_to_pointlist);
  bool
  matchAndAlign              (const Frame & root_frame,
                              int rootframe_id,
                              const ALIGNED<FrontendVertex>::int_hash_map
                              & vertex_table,
                              const list<CandidatePoint3Ptr>
                              & candidate_point_list,
                              SE3 * T_newroot_from_oldroot,
                              TrackData<3> * track_data);
  void
  keyframesToRegister        (int rootframe_id,
                              const tr1::unordered_set<int> & direct_neighbors,
                              const ALIGNED<FrontendVertex>::int_hash_map
                              & vertex_table,
                              const SE3 & T_newroot_from_oldroot,
                              const TrackData<3> & track_data,
                              ImageStatsTable  * frameid_to_pointlist,
                              list<StereoGraph::MyTrackPointPtr>
                              * trackpoint_list,
                              IntTable * neighborid_to_strength);
  bool
  globalLoopClosure          (const DetectedLoop & loop);

  ALIGNED<StereoCamera>::vector cam_vec_;
  StereoGraph graph_;
  PlaceRecognizerMonitor * place_reg_monitor_;
  stack<int> local_registration_stack;
  tr1::unordered_map<int,Frame>  keyframe_map_;
  SE3XYZ_STEREO se3xyz_stereo_;
  StereoGraph::EdgeTable  new_edges_;
};
}

#endif
