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

#include "backend.h"

#include <pangolin/pangolin.h>

#include <visiontools/accessor_macros.h>

#include "fast_grid.h"
#include "pose_optimizer.h"

namespace ScaViSLAM
{

void BackendMonitor
::queryNeighborhood(int frame_id)
{
  boost::mutex::scoped_lock lock(my_mutex_);
  query_frame_id_storage_.clear();
  query_frame_id_storage_.push_front(frame_id);
}

bool BackendMonitor
::getQueryFrameId(int * frame_id)
{
  boost::mutex::scoped_lock lock(my_mutex_);
  if (query_frame_id_storage_.size()==0)
    return false;

  *frame_id = query_frame_id_storage_.front();
  query_frame_id_storage_.clear();

  return true;
}

void BackendMonitor
::pushNeighborhood(const NeighborhoodPtr & neighborhood)
{
  boost::mutex::scoped_lock lock(my_mutex_);
  neighborhood_ptr_storage_.clear();
  neighborhood_ptr_storage_.push_front(neighborhood);
}

bool BackendMonitor
::getNeighborhood(NeighborhoodPtr * neighborhood)
{
  boost::mutex::scoped_lock lock(my_mutex_);
  if (neighborhood_ptr_storage_.size()==0)
    return false;

  *neighborhood = neighborhood_ptr_storage_.front();
  neighborhood_ptr_storage_.clear();

  return true;
}

void BackendMonitor
::pushKeyframe(const AddToOptimzerPtr & to_optimizer)
{
  boost::mutex::scoped_lock lock(my_mutex_);
  new_keyframe_queue_.push(to_optimizer);
}

bool BackendMonitor
::getKeyframe(AddToOptimzerPtr * to_optimizer)
{
  boost::mutex::scoped_lock lock(my_mutex_);
  if (new_keyframe_queue_.size()==0)
    return false;

  *to_optimizer = new_keyframe_queue_.front();
  new_keyframe_queue_.pop();

  return true;
}

void BackendMonitor::pushDrawData(const BackendDrawDataPtr & draw_data)
{
  boost::mutex::scoped_lock lock(my_mutex_);
  draw_storage_.clear();
  draw_storage_.push_front(draw_data);
}

bool BackendMonitor
::getDrawData(BackendDrawDataPtr * draw_data)
{
  boost::mutex::scoped_lock lock(my_mutex_);
  if (draw_storage_.size()==0)
    return false;

  *draw_data = draw_storage_.front();
  draw_storage_.clear();

  return true;
}

void BackendMonitor::pushClosedLoop(const DetectedLoop & loop)
{
  boost::mutex::scoped_lock lock(my_mutex_);
  loop_storage_.clear();
  loop_storage_.push_front(loop);
}

bool BackendMonitor
::getClosedLoop(DetectedLoop * loop)
{
  boost::mutex::scoped_lock lock(my_mutex_);
  if (loop_storage_.size()==0)
    return false;

  *loop = loop_storage_.front();
  loop_storage_.clear();

  return true;
}


Backend
::Backend(const ALIGNED<StereoCamera>::vector & cam_vec,
          PlaceRecognizerMonitor * place_reg_monitor)
  : cam_vec_(cam_vec),
    graph_(cam_vec.at(0)),
    place_reg_monitor_(place_reg_monitor),
    se3xyz_stereo_(cam_vec.at(0))
{
  int graph_inner_window= pangolin::Var<int>
      ("graph.inner_window",25);
  int graph_outer_window= pangolin::Var<int>
      ("graph.outer_window",200);

  int covis_thr= pangolin::Var<int>
      ("frontend.covis_thr",15);

  graph_.initialize(graph_inner_window,
                    graph_outer_window,
                    covis_thr);

  stop = false;
}


void Backend
::operator()()
{
  AddToOptimzerPtr to_optimiser;

  while(stop==false)
  {
    if (monitor.getKeyframe(&to_optimiser))
    {
      addKeyframeToGraph(to_optimiser);
      addKeyframeToPlaceRecog(to_optimiser);
      keyframe_map_.insert(make_pair(to_optimiser->newkey_id,
                                     to_optimiser->kf));
      continue;
    }

    int query_frame_id = -1;
    if (monitor.getQueryFrameId(&query_frame_id))
    {
      bool do_optimization = graph_.prepareForOptimization(query_frame_id,-1);
      local_registration_stack.push(query_frame_id);

      BackendDrawDataPtr draw_data = cloneDrawData();
      monitor.pushDrawData(draw_data);

      NeighborhoodPtr neighborhood_ptr =
          computeNeighborhood(query_frame_id);

      monitor.pushNeighborhood(neighborhood_ptr);
      if (do_optimization)
        graph_.optimize(OptParams(2,true,3));
    }

    if (local_registration_stack.size()>0)
    {
      int frame_id = local_registration_stack.top();
      local_registration_stack.pop();
      if(localRegisterFrame(frame_id))
      {
        if(graph_.prepareForOptimization(frame_id,-1))
          graph_.optimize(OptParams(2,true,3));
      }
    }

    DetectedLoop loop;
    if (place_reg_monitor_->getLoop(&loop))
    {
      if (graph_.edge_table().orderd_find(loop.loop_keyframe_id,
                                          loop.query_keyframe_id)
          == graph_.edge_table().end()
          && (IS_IN_SET(loop.loop_keyframe_id,
                        graph_.double_window())==false
              || GET_MAP_ELEM(loop.loop_keyframe_id,
                              graph_.double_window()))==StereoGraph::OUTER)
      {
        if(globalLoopClosure(loop))
        {
          monitor.pushClosedLoop(loop);
          if(graph_.prepareForOptimization(loop.query_keyframe_id,
                                           loop.loop_keyframe_id))
            graph_.optimize(OptParams(2,true,3));
        }
      }
    }

    boost::this_thread::sleep(boost::posix_time::milliseconds(1));
  }
}


BackendDrawDataPtr Backend
::cloneDrawData() const
{
  BackendDrawDataPtr draw_data(new BackendDrawData);

  draw_data->active_point_set = graph_.active_point_set();
  draw_data->outer_point_set = graph_.outer_point_set();
  draw_data->double_window = graph_.double_window();
  draw_data->point_table = graph_.point_table();
  draw_data->vertex_table = graph_.vertex_table();
  draw_data->edge_table = graph_.edge_table();
  draw_data->new_edges = new_edges_;

  return draw_data;
}


NeighborhoodPtr Backend
::computeNeighborhood(int root_id) const
{
  assert(IS_IN_SET(root_id, graph_.vertex_table()));
  assert(IS_IN_SET(root_id, graph_.double_window())
         && GET_MAP_ELEM(root_id,graph_.double_window())
         ==StereoGraph::INNER);
  NeighborhoodPtr neighborhood_ptr(new Neighborhood());

  addPoseToNeighborhood(root_id,
                        neighborhood_ptr.get());
  findNeighborsOfRoot(root_id, 10,
                      neighborhood_ptr.get());

  addPointsToNeighbors(neighborhood_ptr.get());
  addAnchorPosesToNeighbors(neighborhood_ptr.get());

  int row = 0;
  for(ALIGNED<FrontendVertex>::int_hash_map::iterator
      it = neighborhood_ptr->vertex_map.begin();
      it!=neighborhood_ptr->vertex_map.end(); ++it, ++row)
  {
    int id1 = it->first;
    int col = 0;
    for(ALIGNED<FrontendVertex>::int_hash_map::iterator
        it2 = neighborhood_ptr->vertex_map.begin();
        col<row; ++it2, ++col)
    {
      int id2 = it2->first;
      StereoGraph::StdEdgeTable::const_iterator edge_it
          = graph_.edge_table().orderd_find(id1, id2);
      if (edge_it!=graph_.edge_table().end())
      {
        int strength = edge_it->second->strength;
        it->second.strength_to_neighbors.insert(make_pair(strength,id2));
        it2->second.strength_to_neighbors.insert(make_pair(strength,id1));
      }
    }
  }

  return neighborhood_ptr;
}

void Backend
::findNeighborsOfRoot(int root_id,
                      int max_num_neighbors,
                      Neighborhood * neighborhood) const
{
  const StereoGraph::Vertex & root
      = GET_MAP_ELEM(root_id, graph_.vertex_table());
  int count = 0;
  for (multimap<int,int>::const_iterator
       it = root.neighbor_ids_ordered_by_strength.begin();
       it!=root.neighbor_ids_ordered_by_strength.end();++it)
  {
    int frame_id = it->second;
    if (IS_IN_SET(frame_id, graph_.double_window()))
    {
      addPoseToNeighborhood(frame_id,
                            neighborhood);
      if (count>=max_num_neighbors)
        return;
      ++count;
    }
  }
}

void Backend
::addPointsToNeighbors(
    Neighborhood * neighborhood) const
{
  tr1::unordered_set<int> added_point_set;
  for (ALIGNED<FrontendVertex>::int_hash_map::const_iterator
       it = neighborhood->vertex_map.begin();
       it!=neighborhood->vertex_map.end(); ++it)
  {
    int pose_id = it->first;

    const StereoGraph::Vertex & v = GET_MAP_ELEM(pose_id,
                                                 graph_.vertex_table());
    for (ImageFeature<3>::Table::const_iterator
         it_feat = v.feature_table.begin(); it_feat!=v.feature_table.end();
         ++it_feat)
    {
      int point_id = it_feat->first;
      if(IS_IN_SET(point_id,added_point_set))
        continue;
      added_point_set.insert(point_id);

      const StereoGraph::Point & p
          = GET_MAP_ELEM(point_id, graph_.point_table());

      CandidatePoint3Ptr ap(new CandidatePoint<3>(point_id,
                                                  p.xyz_anchor,
                                                  p.anchorframe_id,
                                                  p.anchor_obs_pyr,
                                                  p.anchor_level,
                                                  p.normal_anchor));
      neighborhood->point_list.push_back(ap);
    }
  }
}

void Backend
::addPoseToNeighborhood(int new_pose_id,
                        Neighborhood * neighborhood) const
{
  const StereoGraph::Vertex & v = GET_MAP_ELEM(new_pose_id,
                                               graph_.vertex_table());
  FrontendVertex vf;
  vf.feat_map = v.feature_table;

  if (graph_.double_window().find(new_pose_id)
      != graph_.double_window().end())
  {
    vf.T_me_from_w = v.T_me_from_world;
  }
  else
  {
    vf.T_me_from_w = graph_.computeAbsolutePose(new_pose_id);
  }
  neighborhood->vertex_map.insert(
        make_pair(new_pose_id, vf));
//  neighborhood->feat_map
//      .insert(make_pair(new_pose_id, v.feature_table));
}

void Backend
::addAnchorPosesToNeighbors(Neighborhood * neighborhood) const
{
  for (list<CandidatePoint3Ptr>::const_iterator
       it = neighborhood->point_list.begin();
       it!=neighborhood->point_list.end(); ++it)
  {
    const CandidatePoint3Ptr & ap = *it;
    int new_pose_id = ap->anchor_id;
    if (IS_IN_SET(new_pose_id,neighborhood->vertex_map))
      continue;

    addPoseToNeighborhood(new_pose_id, neighborhood);
  }

}


void Backend
::addKeyframeToGraph(const AddToOptimzerPtr & to_optimiser)
{
  if(to_optimiser->first_frame)
  {
    graph_.addFirstKeyframe(to_optimiser->newkey_id);
  }
  else
  {
    graph_.addKeyframe(to_optimiser->oldkey_id,
                       to_optimiser->newkey_id,
                       to_optimiser->T_newkey_from_oldkey,
                       to_optimiser->new_point_list,
                       to_optimiser->track_point_list);
  }
}


void Backend
::addKeyframeToPlaceRecog(const AddToOptimzerPtr & to_optimiser)
{
  const StereoGraph::Vertex & v_new
      = GET_MAP_ELEM(to_optimiser->newkey_id, graph_.vertex_table());

  PlaceRecognizerData pr_data;
  pr_data.exclude_set.insert(to_optimiser->newkey_id);

  for (multimap<int,int>::const_iterator
       it=v_new.neighbor_ids_ordered_by_strength.begin();
       it!=v_new.neighbor_ids_ordered_by_strength.end(); ++it)
  {
    int neighborid = it->second;
    pr_data.exclude_set.insert(neighborid);
  }

  pr_data.do_loop_detection = pr_data.exclude_set.size()
      < graph_.vertex_table().size();
  pr_data.keyframe = to_optimiser->kf;
  pr_data.keyframe_id = to_optimiser->newkey_id;

  place_reg_monitor_->addKeyframeData(pr_data);
}


tr1::unordered_set<int> Backend
::directNeighborsOf(int frame_id)
{
  tr1::unordered_set<int> neighbors;
  neighbors.insert(frame_id);

  const StereoGraph::Vertex & v
      = GET_MAP_ELEM(frame_id, graph_.vertex_table());
  for (multimap<int,int>::const_iterator it
       = v.neighbor_ids_ordered_by_strength.begin();
       it!=v.neighbor_ids_ordered_by_strength.end(); ++it)
  {
    neighbors.insert(it->second);
  }

  return neighbors;
}


void Backend
::recomputeFastCorners(const Frame & frame,
                       ALIGNED<QuadTree<int> >::vector * feature_tree)
{
  int num_levels = frame.cell_grid2d.size();
  feature_tree->resize(num_levels);

  for (int level=0; level<num_levels; ++level)
  {
    feature_tree->at(level)
        =  QuadTree<int>(Rectangle(0,0,cam_vec_.at(level).width(),
                                   cam_vec_.at(level).height()),
                         1);
    FastGrid::detect(frame.pyr.at(level),
                     frame.cell_grid2d.at(level),
                     &(feature_tree->at(level)));
  }
}


void Backend
::pointsVisibleInRoot(const SE3 & T_root_from_world,
                      const tr1::unordered_set<int> &
                      larger_neighborhood,
                      const tr1::unordered_set<int> & direct_neighbors,
                      list< CandidatePoint3Ptr > * candidate_point_list,
                      ALIGNED<FrontendVertex>::int_hash_map * vertex_table)
{
  tr1::unordered_set<int> point_set;
  for (tr1::unordered_set<int>::const_iterator it
       = larger_neighborhood.begin(); it!=larger_neighborhood.end(); ++it)
  {
    int keyframe_id = *it;
    if (IS_IN_SET(keyframe_id, direct_neighbors))
      continue;

    const StereoGraph::Vertex & v
        = GET_MAP_ELEM(keyframe_id, graph_.vertex_table());
    for (ImageFeature<3>::Table::const_iterator feat_it
         = v.feature_table.begin(); feat_it!=v.feature_table.end(); ++feat_it)
    {
      int point_id = feat_it->first;
      if (IS_IN_SET(point_id, point_set))
        continue;

      point_set.insert(point_id);


      const StereoGraph::Point & p
          = GET_MAP_ELEM(point_id, graph_.point_table());



      //TODO: include anchor verteces not in double window
      if (IS_IN_SET(p.anchorframe_id, graph_.double_window())==false)
      {
        continue;
      }

      const StereoGraph::Vertex & v_anchor =
          GET_MAP_ELEM(p.anchorframe_id, graph_.vertex_table());

      SE3 T_world_from_anchor = v_anchor.T_me_from_world.inverse();

      const StereoCamera & cam_pyr = cam_vec_.at(p.anchor_level);
      Vector3d xyz_root = T_root_from_world*T_world_from_anchor*p.xyz_anchor;




      Vector2d uv_pyr = cam_pyr.map(project2d(xyz_root));


      if (cam_pyr.isInFrame(uv_pyr.cast<int>(), 0)==false)
        continue;


      candidate_point_list->push_back(
            CandidatePoint3Ptr(
              new CandidatePoint<3>(point_id,
                                    p.xyz_anchor,
                                    p.anchorframe_id,
                                    p.anchor_obs_pyr,
                                    p.anchor_level,
                                    p.normal_anchor)));
      if (IS_IN_SET(p.anchorframe_id, *vertex_table) == false)
      {
        FrontendVertex vf;
        vf.T_me_from_w = v_anchor.T_me_from_world;

        vertex_table->insert(make_pair(p.anchorframe_id, vf));
      }
    }
  }
}


bool Backend
::localRegisterFrame(int rootframe_id)
{
  int NUM_FRAMES_TO_CHECK_FOR_REGISTRATION = 40;
  size_t COVIS_THR = graph_.covis_thr();

  tr1::unordered_set<int> direct_neighbors = directNeighborsOf(rootframe_id);

  int num_potential_frames
      = direct_neighbors.size() + NUM_FRAMES_TO_CHECK_FOR_REGISTRATION;

  tr1::unordered_set<int> larger_neighborhood =
      graph_.framesInNeighborhood(rootframe_id, num_potential_frames);

  const Frame & root_frame = GET_MAP_ELEM(rootframe_id,keyframe_map_);
  assert(root_frame.cell_grid2d.size()>0);
  const StereoGraph::Vertex & v_root
      = GET_MAP_ELEM(rootframe_id, graph_.vertex_table());
  ALIGNED<FrontendVertex>::int_hash_map vertex_table;
  list<CandidatePoint3Ptr> candidate_point_list;
  FrontendVertex vf;
  vf.T_me_from_w = v_root.T_me_from_world;
  vertex_table.insert(make_pair(rootframe_id, vf));

  pointsVisibleInRoot(v_root.T_me_from_world, larger_neighborhood,
                      direct_neighbors,
                      &candidate_point_list, &vertex_table);

  if (candidate_point_list.size()<COVIS_THR)
    return false;

  TrackData<3> track_data;
  SE3 T_newroot_from_oldroot;

  if (matchAndAlign(root_frame, rootframe_id, vertex_table,
                    candidate_point_list,
                    &T_newroot_from_oldroot, &track_data) == false)
    return false;

  ImageStatsTable frameid_to_pointlist;
  list<StereoGraph::MyTrackPointPtr> trackpoint_list;
  IntTable neighborid_to_strength;

  keyframesToRegister(rootframe_id, direct_neighbors, vertex_table,
                      T_newroot_from_oldroot, track_data,
                      &frameid_to_pointlist, &trackpoint_list,
                      &neighborid_to_strength);


  if (neighborid_to_strength.size()<=0)
    return false;

  SE3 T_newroot_from_w
      = T_newroot_from_oldroot*v_root.T_me_from_world;
  graph_.registerKeyframes(rootframe_id, T_newroot_from_w,
                           neighborid_to_strength, trackpoint_list);

//  dumpRegistrationData(root_frame,
//                       neighborid_to_strength,
//                       frameid_to_pointlist);

  return true;
}


//TODO: method too long
void Backend
::keyframesToRegister(int rootframe_id,
                      const tr1::unordered_set<int> & direct_neighbors,
                      const ALIGNED<FrontendVertex>::int_hash_map & vertex_table,
                      const SE3 & T_newroot_from_oldroot,
                      const TrackData<3> & track_data,
                      ImageStatsTable * frameid_to_pointlist,
                      list<StereoGraph::MyTrackPointPtr> * trackpoint_list,
                      IntTable * neighborid_to_strength)
{
  double REPROJ_THR = 2.0;
  int COVIS_THR = graph_.covis_thr();

  for (list<IdObs<3> >::const_iterator it = track_data.obs_list.begin();
       it!=track_data.obs_list.end(); ++it)
  {
    IdObs<3> id_obs = *it;
    const Vector3d & point = track_data.point_list.at(id_obs.point_id);
    const Vector3d & uvu_pred
        = se3xyz_stereo_.map(T_newroot_from_oldroot,point);
    const Vector3d & uvu = id_obs.obs;
    Vector3d diff = uvu - uvu_pred;
    int image_width = cam_vec_.at(0).width();
    int image_height = cam_vec_.at(0).height();

    StereoGraph::MyActivePointPtr point_ptr
        = GET_VEC_VAL(id_obs.point_id, track_data.ba2globalptr);
    int factor = zeroFromPyr_i(1, point_ptr->anchor_level);

    if (abs(diff[0])<REPROJ_THR*factor
        && abs(diff[1])<REPROJ_THR*factor
        && abs(diff[2])<REPROJ_THR*3)
    {
      int global_point_id = point_ptr->point_id;

      for (ALIGNED<FrontendVertex>::int_hash_map::const_iterator it_poses
           = vertex_table.begin(); it_poses!=vertex_table.end();
           ++it_poses)
      {
        int pose_id = it_poses->first;
        if (IS_IN_SET(pose_id, direct_neighbors))
          continue;

        const StereoGraph::Vertex & v
            = GET_MAP_ELEM(pose_id, graph_.vertex_table());

        if (IS_IN_SET(global_point_id, v.feature_table))
        {
          ImageFeature<3> feat(uvu,
                               point_ptr->anchor_level);
          StereoGraph::MyTrackPointPtr trackpoint(
                new StereoGraph::MyTrackPoint(global_point_id,
                                              feat));

          double u = uvu[0];
          double v = uvu[1];

          ImageStats * stats = NULL;

          ImageStatsTable::iterator find_it
              = frameid_to_pointlist->find(pose_id);
          if (find_it!=frameid_to_pointlist->end())
          {

            stats = &(find_it->second);
          }
          else
          {
            frameid_to_pointlist->insert(make_pair(pose_id, ImageStats()));
            stats = &(GET_MAP_ELEM_REF(pose_id, frameid_to_pointlist));
          }
          assert(stats!=NULL);
          stats->point_list.push_back(trackpoint);
          if (u>image_width*0.5)
            ++stats->num_left;
          else
            ++stats->num_right;
          if (v>image_height*0.5)
            ++stats->num_lower;
          else
            ++stats->num_upper;
        }
      }
    }
  }

  new_edges_.clear();
  for (ImageStatsTable::const_iterator it = frameid_to_pointlist->begin();
       it!=frameid_to_pointlist->end(); ++it)
  {
    int neighbor_id = it->first;
    int strength = it->second.point_list.size();
    if (strength>=COVIS_THR
        && it->second.num_left>=COVIS_THR/2
        && it->second.num_right>=COVIS_THR/2
        && it->second.num_upper>=COVIS_THR/2
        && it->second.num_lower>=COVIS_THR/2)
    {
      neighborid_to_strength->insert(make_pair(neighbor_id, strength));
      list<StereoGraph::MyTrackPointPtr> pointlist_for_neighbor
          = GET_MAP_ELEM(neighbor_id, *frameid_to_pointlist).point_list;
      trackpoint_list->insert(trackpoint_list->begin(),
                              pointlist_for_neighbor.begin(),
                              pointlist_for_neighbor.end());
      new_edges_.insertEdge(rootframe_id, neighbor_id, strength, StereoGraph::METRIC);
    }
  }
}


bool Backend
::matchAndAlign(const Frame & root_frame,
                int rootframe_id,
                const ALIGNED<FrontendVertex>::int_hash_map & vertex_table,
                const list<CandidatePoint3Ptr> & candidate_point_list,
                SE3 * T_newroot_from_oldroot,
                TrackData<3> * track_data)
{
  size_t COVIS_THR = graph_.covis_thr();
  ALIGNED<QuadTree<int> >::vector feature_tree;
  recomputeFastCorners(root_frame, &feature_tree);
  int BOXSIZE = 8;
  int HALFBOXSIZE = BOXSIZE/2;
  GuidedMatcher<StereoCamera>::match(keyframe_map_,
                                     *T_newroot_from_oldroot,
                                     root_frame,
                                     feature_tree,
                                     cam_vec_,
                                     rootframe_id,
                                     vertex_table,
                                     candidate_point_list,
                                     10,
                                     22,
                                     10,
                                     track_data);

  if (track_data->obs_list.size()<COVIS_THR)
    return false;

  BA_SE3_XYZ_STEREO ba;
  OptimizerStatistics opt
      = ba.calcFastMotionOnly(track_data->obs_list,
                              se3xyz_stereo_,
                              PoseOptimizerParams(true,2,25),
                              T_newroot_from_oldroot,
                              &(track_data->point_list));
  track_data->clear();

  GuidedMatcher<StereoCamera>::match(keyframe_map_,
                                     *T_newroot_from_oldroot,
                                     root_frame,
                                     feature_tree,
                                     cam_vec_,
                                     rootframe_id,
                                     vertex_table,
                                     candidate_point_list,
                                     4,
                                     22,
                                     10,
                                     track_data);
  opt = ba.calcFastMotionOnly(track_data->obs_list,
                              se3xyz_stereo_,
                              PoseOptimizerParams(true,2,15),
                              T_newroot_from_oldroot,
                              &(track_data->point_list));
  if (track_data->obs_list.size()<COVIS_THR)
    return false;

  return true;
}

void Backend
::dumpRegistrationData(const Frame & root_frame,
                       const IntTable & neighborid_to_strength,
                       const ImageStatsTable & frameid_to_pointlist)
{
  size_t COVIS_THR = graph_.covis_thr();
  static int counter = 0;
  char name[80];
  cv::Mat left = root_frame.pyr.at(0).clone();
  for (IntTable::const_iterator it = neighborid_to_strength.begin();
       it!=neighborid_to_strength.end(); ++it)
  {
    sprintf(name,"r%04d_%04d.png", counter, it->first);
    cv::Mat img = GET_MAP_ELEM(it->first, keyframe_map_).pyr.at(0).clone();
    img.push_back(left);
    list<StereoGraph::MyTrackPointPtr> pointlist_for_neighbor
        = GET_MAP_ELEM(it->first, frameid_to_pointlist).point_list;
    assert(pointlist_for_neighbor.size()>=COVIS_THR);

    const StereoGraph::Vertex & v_other
        = GET_MAP_ELEM(it->first, graph_.vertex_table());

    for (list<StereoGraph::MyTrackPointPtr>::const_iterator it
         = pointlist_for_neighbor.begin();
         it!=pointlist_for_neighbor.end(); ++it)
    {
      const StereoGraph::MyTrackPointPtr & track_point = *it;
      ImageFeature<3> feat_root = track_point->feat;
      const ImageFeature<3> & feat_other
          = GET_MAP_ELEM(track_point->global_id, v_other.feature_table);

      cv::circle(img, cv::Point2d(feat_other.center.x(), feat_other.center.y()),
                 2, cv::Scalar(1,0,0,1),3);
      cv::line(img, cv::Point2d(feat_other.center.x(), feat_other.center.y()),
               cv::Point2d(feat_root.center.x(),
                           feat_root.center.y()+left.size().height),
               cv::Scalar(1,0,0,1),1,CV_AA);
    }
    cv::imwrite(name, img);
  }
  ++counter;
}

// TODO: method way to long!!
bool Backend
::globalLoopClosure(const DetectedLoop & loop)
{

  const Frame & loop_frame
      = GET_MAP_ELEM(loop.loop_keyframe_id, keyframe_map_);


  const StereoGraph::Vertex & v_query
      = GET_MAP_ELEM(loop.query_keyframe_id, graph_.vertex_table());

  const StereoGraph::Vertex & v_loop
      = GET_MAP_ELEM(loop.loop_keyframe_id, graph_.vertex_table());

  SE3 T_loop_from_world
      = loop.T_query_from_loop.inverse()*v_query.T_me_from_world;

  list< CandidatePoint3Ptr > candidate_point_list;
  ALIGNED<FrontendVertex>::int_hash_map vertex_table;
  FrontendVertex vf;
  vf.T_me_from_w = T_loop_from_world;
  vertex_table.insert(make_pair(loop.loop_keyframe_id,vf));

  for (ImageFeature<3>::Table::const_iterator it
       = v_query.feature_table.begin(); it!=v_query.feature_table.end(); ++it)
  {
    int point_id = it->first;
    const StereoGraph::Point & p
        = GET_MAP_ELEM(point_id, graph_.point_table());

    //TODO: include anchor verteces not in double window??
    if (IS_IN_SET(p.anchorframe_id, graph_.double_window())==false)
    {
      continue;
    }

    const StereoGraph::Vertex & v_anchor =
        GET_MAP_ELEM(p.anchorframe_id, graph_.vertex_table());

    SE3 T_world_from_anchor = v_anchor.T_me_from_world.inverse();

    const StereoCamera & cam_pyr = cam_vec_.at(p.anchor_level);
    Vector3d xyz_loop = T_loop_from_world*T_world_from_anchor*p.xyz_anchor;

    Vector2d uv_pyr = cam_pyr.map(project2d(xyz_loop));


    if (cam_pyr.isInFrame(uv_pyr.cast<int>(), 0)==false)
      continue;

    candidate_point_list.push_back(CandidatePoint3Ptr(
                                     new CandidatePoint<3>(point_id,
                                                           p.xyz_anchor,
                                                           p.anchorframe_id,
                                                           p.anchor_obs_pyr,
                                                           p.anchor_level,
                                                           p.normal_anchor)));
    if (IS_IN_SET(p.anchorframe_id, vertex_table) == false)
    {
      FrontendVertex vf;
      vf.T_me_from_w = v_anchor.T_me_from_world;
      vertex_table.insert(make_pair(p.anchorframe_id, vf));
    }
  }


  TrackData<3> track_data;

  SE3 T_newloop_from_oldloop;
  if (matchAndAlign(loop_frame, loop.loop_keyframe_id, vertex_table,
                    candidate_point_list,
                    &T_newloop_from_oldloop, &track_data) == false)
    return false;

  double REPROJ_THR = 2.0;
  size_t COVIS_THR = graph_.covis_thr();
  list<StereoGraph::MyTrackPointPtr> trackpoint_list;
  size_t num_lower = 0;
  size_t num_upper = 0;
  size_t num_left = 0;
  size_t num_right = 0;
  int img_width = loop_frame.pyr.at(0).size().width;
  int img_height = loop_frame.pyr.at(0).size().height;

  for (list<IdObs<3> >::const_iterator it = track_data.obs_list.begin();
       it!=track_data.obs_list.end(); ++it)
  {
    IdObs<3> id_obs = *it;
    const Vector3d & point = track_data.point_list.at(id_obs.point_id);
    const Vector3d & uvu_pred
        = se3xyz_stereo_.map(T_newloop_from_oldloop, point);
    const Vector3d & uvu = id_obs.obs;
    Vector3d diff = uvu - uvu_pred;

    StereoGraph::MyActivePointPtr point_ptr
        = GET_VEC_VAL(id_obs.point_id, track_data.ba2globalptr);
    int factor = zeroFromPyr_i(1, point_ptr->anchor_level);

    if (abs(diff[0])<REPROJ_THR*factor
        && abs(diff[1])<REPROJ_THR*factor
        && abs(diff[2])<REPROJ_THR*3)
    {
      int global_point_id = point_ptr->point_id;

      double u = uvu[0];
      double v = uvu[1];
      if (u>img_width*0.5)
        num_right += 1;
      else
        num_left += 1;
      if (v>img_height*0.5)
        num_lower += 1;
      else
        num_upper += 1;

      ImageFeature<3> feat(uvu,
                           point_ptr->anchor_level);
      StereoGraph::MyTrackPointPtr trackpoint(
            new StereoGraph::MyTrackPoint(global_point_id,
                                          feat));
      trackpoint_list.push_back(trackpoint);
    }
  }
  if (trackpoint_list.size()<COVIS_THR)
    return false;

  // Make sure that enough point are found in all parts of the image.
  // (Thus make sure we found also points in the foreground so that the
  // relative loop closure constraint is accurately estimated.)
  if (num_lower<COVIS_THR/2 || num_upper<COVIS_THR/2
      || num_left<COVIS_THR/2 || num_right<COVIS_THR/2)
    return false;

  //calculate loop closure vertex in the metrical neighborhood around v_query
  SE3 T_newloop_from_w
      = T_newloop_from_oldloop*loop.T_query_from_loop.inverse()
      *v_query.T_me_from_world;

  graph_.addLoopClosure(loop.query_keyframe_id,
                        loop.loop_keyframe_id,
                        T_newloop_from_w,
                        trackpoint_list);

//  // Dump loop closure to hard disk
//  char name[80];
//  static int counter = 0;
//  sprintf(name,"LOOP%04d_%04d.png", counter, loop.loop_keyframe_id);
//  cv::Mat img
//      = GET_MAP_ELEM(loop.query_keyframe_id, keyframe_map_).pyr.at(0).clone();
//  img.push_back(loop_frame.pyr.at(0));

//  for (list<StereoGraph::MyTrackPointPtr>::const_iterator it
//       = trackpoint_list.begin();
//       it!=trackpoint_list.end(); ++it)
//  {
//    const StereoGraph::MyTrackPointPtr & track_point = *it;
//    ImageFeature<3> feat_loop = track_point->feat;
//    const ImageFeature<3> & feat_query
//        = GET_MAP_ELEM(track_point->global_id, v_query.feature_table);

//    cv::circle(img, cv::Point2d(feat_query.center.x(), feat_query.center.y()),
//               2, cv::Scalar(1,0,0,1),3);
//    cv::line(img, cv::Point2d(feat_query.center.x(), feat_query.center.y()),
//             cv::Point2d(feat_loop.center.x(),
//                         feat_loop.center.y() +img_height),
//             cv::Scalar(1,0,0,1),1,CV_AA);
//  }
//  cv::imwrite(name, img);
//  ++counter;

  return true;
}


}
