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

#include "slam_graph.hpp"

#include <queue>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>



#include <visiontools/accessor_macros.h>
#include <visiontools/stopwatch.h>

#include "maths_utils.h"

namespace ScaViSLAM
{
namespace SlamGraphMethods
{

template <class Pose, class Cam, class Proj, int ObsDim>
bool
restorePoseFromG2o(const g2o::HyperGraph::Vertex * g2o_vertex,
                   typename SlamGraph<Pose,Cam,Proj,ObsDim>::VertexTable
                   * vertex_map)
{
  const G2oVertexSE3 * v_se3 = dynamic_cast<const G2oVertexSE3*>(g2o_vertex);
  if (v_se3!=NULL)
  {
    int frame_id =  v_se3->id();
    typename SlamGraph<Pose,Cam,Proj,ObsDim>::Vertex & frame_vertex
        = GET_MAP_ELEM_REF(frame_id, vertex_map);
    frame_vertex.T_me_from_world = v_se3->estimate();
    return true;
  }
  return false;
}

template<typename T>
typename T::LinearSolverType *
allocateLinearSolver()
{
  return (new g2o::LinearSolverCSparse<typename T::PoseMatrixType>);
}

}

template <typename Pose, typename Cam, typename Proj, int ObsDim>
bool SlamGraph<Pose,Cam,Proj,ObsDim>
::shortestPathToWindow(int root_id,
                       list<int> * path) const
{
  queue<PathTraversalNode> bfs_queue;
  bfs_queue.push(PathTraversalNode(root_id,list<int>()));

  tr1::unordered_set<int> vertex_set;

  while(bfs_queue.size()!=0)
  {
    PathTraversalNode  leaf_v = bfs_queue.front();
    bfs_queue.pop();

    if (IS_IN_SET(leaf_v.own_id, double_window()))
    {
      *path = leaf_v.path_to_me;
      return true;
    }

    //Avoid cycles!
    if (vertex_set.find(leaf_v.own_id)!=vertex_set.end())
      continue;

    vertex_set.insert(leaf_v.own_id);

    const Vertex & v = GET_MAP_ELEM(leaf_v.own_id, vertex_table_);

    for (map<int,int>::const_reverse_iterator it
         = v.neighbor_ids_ordered_by_strength.rbegin();
         it!=v.neighbor_ids_ordered_by_strength.rend();
         ++it)
    {
      bfs_queue.push(PathTraversalNode(it->second, leaf_v.path_to_me));
    }
  }

  return false;
}

template <typename Pose, typename Cam, typename Proj, int ObsDim>
tr1::unordered_set<int>  SlamGraph<Pose,Cam,Proj,ObsDim>
::framesInNeighborhood(int root_id,
                       size_t size_of_neighborhood) const
{
  queue<int> bfs_queue;
  bfs_queue.push(root_id);

  tr1::unordered_set<int> vertex_set;

  while(bfs_queue.size()!=0 && vertex_set.size() < size_of_neighborhood)
  {
    int leaf_v = bfs_queue.front();
    bfs_queue.pop();

    //Avoid cycles!
    if (vertex_set.find(leaf_v)!=vertex_set.end())
      continue;

    if (IS_IN_SET(leaf_v, double_window_)==false)
      continue;

    vertex_set.insert(leaf_v);

    const Vertex & v = GET_MAP_ELEM(leaf_v, vertex_table_);

    for (map<int,int>::const_reverse_iterator it
         = v.neighbor_ids_ordered_by_strength.rbegin();
         it!=v.neighbor_ids_ordered_by_strength.rend();
         ++it)
    {
      bfs_queue.push(it->second);
    }
  }
  return vertex_set;
}


template <typename Pose, typename Cam, typename Proj, int ObsDim>
void SlamGraph<Pose,Cam,Proj,ObsDim>
::addKeyframe(int oldkey_id,
              int newkey_id,
              const Pose & T_newkey_from_oldkey,
              const list<MyNewTwoViewPointPtr> & newpoint_list,
              const list<MyTrackPointPtr> & trackpoint_list)
{
  const Pose & T_oldkey_from_world
      = GET_MAP_ELEM(oldkey_id, vertex_table_).T_me_from_world;

  VertexPtr v_newkey(new Vertex());
  v_newkey->own_id = newkey_id;
  v_newkey->T_me_from_world
      = T_newkey_from_oldkey*T_oldkey_from_world;

  IntTable neighborid_to_strength;

  computeStrength(newpoint_list,
                  trackpoint_list,
                  &neighborid_to_strength);

  assert(neighborid_to_strength.find(oldkey_id)!=neighborid_to_strength.end());

  int & strength_to_oldkey
      = GET_MAP_ELEM_REF(oldkey_id, &neighborid_to_strength);

  //Make sure the strength between oldkey and newkey is strong enough
  // TODO: THIS  SHOULD BE CHECKED INSIDE OF THE FRONTEND
    if (strength_to_oldkey<covis_thr_)
  {
    strength_to_oldkey = covis_thr_;
  }

  addNewPointsToMap(newpoint_list, neighborid_to_strength, v_newkey.get());

  addNewObsToOldPoints(trackpoint_list, v_newkey.get());

  bool inserted =
      vertex_table_.insert(make_pair(v_newkey->own_id, v_newkey)).second;
  assert(inserted);

  addNewEdges(neighborid_to_strength, LOCAL, v_newkey.get());
}

template <typename Pose, typename Cam, typename Proj, int ObsDim>
void SlamGraph<Pose,Cam,Proj,ObsDim>
::registerKeyframes(int root_id,
                    const Pose & T_newroot_from_w,
                    const IntTable & neighborid_to_strength,
                    const list<MyTrackPointPtr> & trackpoint_list)
{
  Vertex & v_root = GET_MAP_ELEM_REF(root_id, &vertex_table_);
  //bring root vertex temporarilly to new neighborhood
  Pose T_rootold_from_world = v_root.T_me_from_world;
  v_root.T_me_from_world = T_newroot_from_w;

  addNewObsToOldPoints(trackpoint_list, &v_root);
  addNewEdges(neighborid_to_strength, METRIC, &v_root);

  //restore old pose
  v_root.T_me_from_world = T_rootold_from_world;
}

template <typename Pose, typename Cam, typename Proj, int ObsDim>
void SlamGraph<Pose,Cam,Proj,ObsDim>
::addLoopClosure(int root_id,
                 int loop_id,
                 const Pose & T_newloop_from_w,
                 const list<MyTrackPointPtr> & trackpoint_list)
{
  int strength = trackpoint_list.size();
  assert(strength>=covis_thr());
  Vertex & v_root = GET_MAP_ELEM_REF(root_id, &vertex_table_);
  Vertex & v_loop = GET_MAP_ELEM_REF(loop_id, &vertex_table_);
  //v_loop.reinitialize_pose = true;

  addNewObsToOldPoints(trackpoint_list, &v_loop);

  v_loop.neighbor_ids_ordered_by_strength.insert(
        make_pair(strength, root_id));
  v_root.neighbor_ids_ordered_by_strength.insert(
        make_pair(strength, loop_id));

  edge_table_.insertEdge(root_id, loop_id, strength, APPREARANCE);


  Matrix<double, Pose::DoF, Pose::DoF> Lambda;


  Pose T_oldloop_from_w = v_loop.T_me_from_world;
  //bring v_loop in metrical neighboorhood around v_root
  v_loop.T_me_from_world = T_newloop_from_w;

  Pose T_loop_from_root;
  computeConstraint(v_loop,
                    v_root,
                    &T_loop_from_root,
                    &Lambda);

  //restore v_loop
  v_loop.T_me_from_world = T_oldloop_from_w;

  edge_table_.setConstraint(v_loop.own_id,
                            v_root.own_id,
                            T_loop_from_root,
                            Lambda,
                            Lambda);
}



template <typename Pose, typename Cam, typename Proj, int ObsDim>
void SlamGraph<Pose,Cam,Proj,ObsDim>
::addFirstKeyframe(int newkey_id)
{
  assert(vertex_table_.size()==0);
  assert(point_table_.size()==0);

  VertexPtr v(new Vertex);
  v->T_me_from_world = Pose();
  v->own_id = newkey_id;

  bool inserted = vertex_table_.insert(make_pair(v->own_id, v)).second;
  assert(inserted);
}


template <typename Pose, typename Cam, typename Proj, int ObsDim>
Pose SlamGraph<Pose,Cam,Proj,ObsDim>
::getRelativePose_1_from_2(int id1, int id2) const
{
  assert(id1!=id2);
  Pose T_1_from_2;

  if (edge_table_.getConstraint_id1_from_id2(id1, id2, &T_1_from_2))
    return T_1_from_2;

  const Pose & T_1_from_w = GET_MAP_ELEM(id1, vertex_table_).T_me_from_world;
  const Pose & T_2_from_w = GET_MAP_ELEM(id2, vertex_table_).T_me_from_world;

  T_1_from_2 = T_1_from_w*T_2_from_w.inverse();
  return T_1_from_2;
}

template <typename Frame, typename Cam, typename Proj, int ObsDim>
bool SlamGraph<Frame,Cam,Proj,ObsDim>
::prepareForOptimization(int root_id, int loop_id)
{
  WindowTable old_window = double_window_;
  double_window_.clear();
  active_point_set_.clear();
  outer_point_set_.clear();

  computeInitialDoubleWin(root_id, inner_window_size_, double_window_size_);
  computeActivePointsAndExtendOuterWindow();
  reinitializePoses(root_id,
                    old_window,
                    loop_id);

  if(double_window_.size()<2)
    return false;

  unmargPosesEnteringInnerW();
  margPosesLeftInnerWindow(old_window);

  return true;
}

template <typename Frame, typename Cam, typename Proj, int ObsDim>
void SlamGraph<Frame,Cam,Proj,ObsDim>
::optimize(const OptParams & opt_params)
{
  optimize(opt_params, NULL);
}

template <typename Frame, typename Cam, typename Proj, int ObsDim>
void SlamGraph<Frame,Cam,Proj,ObsDim>
::optimize(const OptParams & opt_params,
           Statistics * stats)
{
  g2o::SparseOptimizer optimizer;
  G2oCameraParameters  * g2o_cam
      = new G2oCameraParameters(cam_.principal_point(),
                                cam_.focal_length(),
                                cam_.baseline());
  g2o_cam->setId(0);


  setupG2o(g2o_cam, &optimizer);

  copyDataToG2o(opt_params, &optimizer);

  optimizer.initializeOptimization();

  double static lambda = 50.;
  g2o::OptimizationAlgorithmLevenberg * lm
      = static_cast<g2o::OptimizationAlgorithmLevenberg *>(optimizer.solver());

  lm->setUserLambdaInit(lambda);

  StopWatch sw;
  sw.start();
  optimizer.optimize(opt_params.num_iters);
  sw.stop();

  if (stats!=NULL)
  {
    stats->calc_time = sw.get_stopped_time();
  }

  restoreDataFromG2o(optimizer);
}


template <typename Pose, typename Cam, typename Proj, int ObsDim>
void SlamGraph<Pose,Cam,Proj,ObsDim>
::addNewPointsToMap(const list<MyNewTwoViewPointPtr> & newpoint_list,
                    const IntTable & neighborid_to_strength,
                    Vertex * v_newkey)
{
  for (typename list<MyNewTwoViewPointPtr>::const_iterator
       it=newpoint_list.begin(); it!=newpoint_list.end();++it)
  {
    const MyNewTwoViewPointPtr & np = *it;

    if (GET_MAP_ELEM(np->anchor_id, neighborid_to_strength)<covis_thr())
      continue;

    Vertex & v_anchor
        = GET_MAP_ELEM_REF(np->anchor_id, &vertex_table_);

    tr1::unordered_set<int> vis_set;
    vis_set.insert(v_newkey->own_id);
    vis_set.insert(np->anchor_id);

    PointPtr p (new Point(np->xyz_anchor,
                          vis_set,
                          np->anchor_id,
                          np->anchor_obs_pyr,
                          np->anchor_level,
                          np->normal_anchor));

    // TODO: replace pow(2,l) with function call
    ImageFeature<ObsDim>
        feat_anchor(np->anchor_obs_pyr * (pow(2.,np->anchor_level)),
                    np->anchor_level);

    v_newkey->feature_table.insert(make_pair(np->point_id,np->feat_newkey));
    v_anchor.feature_table.insert(make_pair(np->point_id,feat_anchor));

    point_table_.insert(make_pair(np->point_id,p));

  }
}


template <typename Pose, typename Cam, typename Proj, int ObsDim>
void SlamGraph<Pose,Cam,Proj,ObsDim>
::addNewObsToOldPoints(const list<MyTrackPointPtr> & trackpoint_list,
                       Vertex * v_newkey)
{
  for (typename list<MyTrackPointPtr>::const_iterator
       it=trackpoint_list.begin(); it!=trackpoint_list.end();++it)
  {
    const MyTrackPointPtr & tp = *it;

    //test whether point acutally exists!
    typename PointTable::iterator find_it
        = point_table_.find(tp->global_id);
    if (find_it==point_table_.end())
      continue;

    PointPtr & p = find_it->second;
    v_newkey->feature_table.insert(make_pair(tp->global_id,tp->feat));
    p->vis_set.insert(v_newkey->own_id);
  }
}


template <typename Pose, typename Cam, typename Proj, int ObsDim>
void SlamGraph<Pose,Cam,Proj,ObsDim>
::addNewEdges(const IntTable & neighborid_to_strength,
              EdgeType edge_type,
              Vertex * v_newkey)
{
  for (IntTable::const_iterator
       it=neighborid_to_strength.begin();
       it!=neighborid_to_strength.end(); ++it)
  {
    int strength = it->second;
    if (strength>=covis_thr_)
    {
      int other_id = it->first;

      Vertex & v_other = GET_MAP_ELEM_REF(other_id, &vertex_table_);
      //      if (is_loop_closure_edge)
      //        v_other.reinitialize_pose = true;

      v_other.neighbor_ids_ordered_by_strength.insert(
            make_pair(strength,v_newkey->own_id));
      v_newkey->neighbor_ids_ordered_by_strength.insert(
            make_pair(strength,other_id));

      edge_table_.insertEdge(other_id, v_newkey->own_id, strength, edge_type);

      Pose T_other_from_new;
      Matrix<double, Pose::DoF, Pose::DoF> Lambda;

      computeConstraint(v_other,
                        *v_newkey,
                        &T_other_from_new,
                        &Lambda);

      edge_table_.setConstraint(v_other.own_id,
                                v_newkey->own_id,
                                T_other_from_new,
                                Lambda,
                                Lambda);
    }
  }
}


template <typename Pose, typename Cam, typename Proj, int ObsDim>
void SlamGraph<Pose,Cam,Proj,ObsDim>
::computeStrength(const list<MyNewTwoViewPointPtr> & newpoint_list,
                  const list<MyTrackPointPtr> & trackpoint_list,
                  IntTable *
                  neighborid_to_strength)
{
  set<int> recent_keyframe;
  IntTable num_top;
  IntTable num_bottom;
  IntTable num_left;
  IntTable num_right;

  int half_width = cam_.width()*0.5;
  int half_height = cam_.height()*0.5;

  //calculate strength using new points
  for (typename list<MyNewTwoViewPointPtr>::const_iterator
       it=newpoint_list.begin(); it!=newpoint_list.end();++it)
  {
    const MyNewTwoViewPointPtr & np = *it;

    ADD_TO_MAP_ELEM(np->anchor_id,1,neighborid_to_strength);
    recent_keyframe.insert(np->anchor_id);
  }

  //and tracked points
  for (typename list<MyTrackPointPtr>::const_iterator
       it=trackpoint_list.begin();
       it!=trackpoint_list.end();++it)
  {
    const MyTrackPointPtr & tp = *it;

    //test whether point acutally exists!
    typename PointTable::const_iterator find_it
        = point_table_.find(tp->global_id);
    if (find_it==point_table_.end())
      continue;

    const PointPtr & p = find_it->second;

    for (tr1::unordered_set<int>::const_iterator it=p->vis_set.begin();
         it!=p->vis_set.end();++it)
    {
      ADD_TO_MAP_ELEM(*it, 1, neighborid_to_strength);
      double u = tp->feat.center[0];
      double v = tp->feat.center[1];

      if (u<half_width)
        ADD_TO_MAP_ELEM(*it, 1, &num_left);
      else
        ADD_TO_MAP_ELEM(*it, 1, &num_right);
      if (v<half_height)
        ADD_TO_MAP_ELEM(*it, 1, &num_top);
      else
        ADD_TO_MAP_ELEM(*it, 1, &num_bottom);
    }

    //dont add link to very distant keyframe
    //Thus, only add a link to keyframe if
    // - there are enough covisible points
    // - AND
    //   * if new features are to be initialized in the keyframe
    //     (which implies the keyframe is recent in time)
    //   * or if features are matched in all parts of the image.
    for (IntTable::iterator it = neighborid_to_strength->begin();
         it!=neighborid_to_strength->end(); ++it)
    {
      int frame_id = it->first;
//      if (it->second<covis_thr())
//        continue;
//      if (IS_IN_SET(frame_id, recent_keyframe))
//        continue;
      if (IS_IN_SET(frame_id, num_top)
          && GET_MAP_ELEM(frame_id, num_top)>=covis_thr_/2
          && IS_IN_SET(frame_id, num_bottom)
          && GET_MAP_ELEM(frame_id, num_bottom)>=covis_thr_/2
          && IS_IN_SET(frame_id, num_left)
          && GET_MAP_ELEM(frame_id, num_left)>=covis_thr_/2
          && IS_IN_SET(frame_id, num_right)
          && GET_MAP_ELEM(frame_id, num_right)>=covis_thr_/2)
        continue;
      it->second = 0;
    }
  }
}


template <typename Pose, typename Cam, typename Proj, int ObsDim>
void SlamGraph<Pose,Cam,Proj,ObsDim>
::computeInitialDoubleWin(int root_id,
                          int inner_window_size,
                          int double_window_size)
{
  assert(double_window_.size()==0);
  assert(inner_window_size<double_window_size);

  queue<int> bfs_queue;

  bfs_queue.push(root_id);

  while(double_window_.size()<static_cast<size_t>(double_window_size)
        && bfs_queue.size()!=0)
  {
    int leaf_v = bfs_queue.front();
    bfs_queue.pop();

    //Avoid cycles!
    if (double_window_.find(leaf_v)!=double_window_.end())
      continue;

    if (double_window_.size()<static_cast<size_t>(inner_window_size))
    {
      double_window_.insert(make_pair(leaf_v,INNER));
    }
    else
    {
      double_window_.insert(make_pair(leaf_v,OUTER));
    }

    const Vertex & v  = GET_MAP_ELEM(leaf_v, vertex_table_);
    for (map<int,int>::const_reverse_iterator it
         = v.neighbor_ids_ordered_by_strength.rbegin();
         it!=v.neighbor_ids_ordered_by_strength.rend();
         ++it)
    {
      bfs_queue.push(it->second);
    }
  }
}


template <typename Pose, typename Cam, typename Proj, int ObsDim>
void SlamGraph<Pose,Cam,Proj,ObsDim>
::computeActivePointsAndExtendOuterWindow()
{
  assert(active_point_set_.size()==0);
  WindowTable extend_outer_window;
  for (typename WindowTable::const_iterator
       it_win = double_window_.begin(); it_win!=double_window_.end(); ++it_win)
  {
    int frame_id = it_win->first;
    bool is_in_inner_window = it_win->second==INNER;

    if(is_in_inner_window)
    {
      const Vertex & frame_in_inner_window
          = GET_MAP_ELEM(frame_id, vertex_table_);

      for (typename ImageFeature<ObsDim>::Table::const_iterator
           obs_it = frame_in_inner_window.feature_table.begin();
           obs_it!=frame_in_inner_window.feature_table.end(); ++obs_it)
      {
        int point_id = obs_it->first;

        // If point is not in active set yet
        if (active_point_set_.find(point_id)==active_point_set_.end())
        {
          const Point & point = GET_MAP_ELEM(point_id, point_table_);

          if (double_window_.find(point.anchorframe_id)!=double_window_.end())
          {
            active_point_set_.insert(point_id);
          }
          else
          {
            // TODO: SHALL WE INCLUDE ALL ANCHOR FRAMES??
            //Is there a direct edge between frame_in_inner_window and
            //anchorframe?
            if (edge_table_.orderd_find(frame_in_inner_window.own_id,
                                        point.anchorframe_id)
                !=edge_table_.end())
            {
              // Then add anchorframe to outer window!
              active_point_set_.insert(point_id);
              extend_outer_window.insert(make_pair(point.anchorframe_id,OUTER));
            }
          }
        }
      }
    }
    else
    {
      const Vertex & frame_in_outer_window
          = GET_MAP_ELEM(frame_id, vertex_table_);

      for (typename ImageFeature<ObsDim>::Table::const_iterator
           obs_it = frame_in_outer_window.feature_table.begin();
           obs_it!=frame_in_outer_window.feature_table.end(); ++obs_it)
      {
        int point_id = obs_it->first;
        outer_point_set_.insert(point_id);
      }
    }
  }
  double_window_.insert(extend_outer_window.begin(),extend_outer_window.end());
}

template <typename Pose, typename Cam, typename Proj, int ObsDim>
void SlamGraph<Pose,Cam,Proj,ObsDim>
::reinitializePoses(int root_id,
                    const WindowTable & old_window,
                    int loop_id)
{
  queue<ReinitializeTraversalNode> bfs_queue;
  tr1::unordered_set<int> cycle_check;

  //ssert(GET_MAP_ELEM(root_id, vertex_table_).reinitialize_pose==false);

  bfs_queue.push(ReinitializeTraversalNode(root_id,
                                           -1,
                                           Pose(),
                                           false));
  while(bfs_queue.size()!=0)
  {
    ReinitializeTraversalNode  node = bfs_queue.front();
    bfs_queue.pop();

    //Avoid cycles!
    if (IS_IN_SET(node.own_id,  cycle_check))
      continue;

    //Skip is it is not in double window
    if (IS_IN_SET(node.own_id, double_window_)==false)
      continue;

    cycle_check.insert(node.own_id);

    Vertex & v = GET_MAP_ELEM_REF(node.own_id, &vertex_table_);

    bool reinitialize_me_and_my_childs = false;

    if (node.mark_reinitialize || node.own_id==loop_id)
    {
      reinitialize_me_and_my_childs = true;
    }


    if (node.parent_id>-1 &&
        (reinitialize_me_and_my_childs
         || IS_IN_SET(node.own_id, old_window)==false))
    {
      v.T_me_from_world
          = getRelativePose_1_from_2(node.own_id, node.parent_id)
          *node.T_parent_from_world;
    }

    for (map<int,int>::const_reverse_iterator it
         = v.neighbor_ids_ordered_by_strength.rbegin();
         it!=v.neighbor_ids_ordered_by_strength.rend();
         ++it)
    {
      bfs_queue.push(ReinitializeTraversalNode(it->second,
                                               node.own_id,
                                               v.T_me_from_world,
                                               reinitialize_me_and_my_childs));
    }
  }
}


template <typename Pose, typename Cam, typename Proj, int ObsDim>
void SlamGraph<Pose,Cam,Proj,ObsDim>
::unmargPosesEnteringInnerW()
{
  for (typename WindowTable::const_iterator
       it_win1 = double_window_.begin(); it_win1!=double_window_.end();
       ++it_win1)
  {
    int pose_id_1 = it_win1->first;
    WindowType wtype_1 = it_win1->second;
    if (wtype_1==INNER)
    {
      for (typename WindowTable::const_iterator
           it_win2 = double_window_.begin(); it_win2!=double_window_.end();
           ++it_win2)
      {
        int pose_id_2 = it_win2->first;
        if (pose_id_2==pose_id_1)
          continue;

        WindowType wtype_2 = it_win2->second;
        if (wtype_2==INNER)
        {
          if (edge_table_.orderd_find(pose_id_1,pose_id_2)!=edge_table_.end())
          {
            edge_table_.unMarginalize(pose_id_1,pose_id_2);
          }
        }
      }
    }
  }
}


template <typename Pose, typename Cam, typename Proj, int ObsDim>
Pose SlamGraph<Pose,Cam,Proj,ObsDim>
::computeAbsolutePose(int x_id) const
{
  list<int> path;
  shortestPathToWindow(x_id, &path);
  assert(path.size()>=1);
  list<int>::const_reverse_iterator it = path.rbegin();
  int cur_id = *it;
  Pose T_x_from_w
      = GET_MAP_ELEM(cur_id, vertex_table()).T_me_from_world;
  ++it;
  for (; it!=path.rend(); ++it)
  {
    int new_id = *it;
    T_x_from_w
        = getRelativePose_1_from_2(new_id, cur_id)*T_x_from_w;
    cur_id = new_id;
  }
  return T_x_from_w;
}


template <typename Pose, typename Cam, typename Proj, int ObsDim>
void SlamGraph<Pose,Cam,Proj,ObsDim>
::computeConstraint(const Vertex & v1,
                    const Vertex & v2,
                    Pose * T_1_from_2,
                    Matrix<double,Pose::DoF,Pose::DoF> * Lambda)
{
  typename ALIGNED<Pose>::int_hash_map cache_poses;
  *T_1_from_2 = v1.T_me_from_world*v2.T_me_from_world.inverse();

  multiset<double> depth_set;

  for (typename ImageFeature<ObsDim>::Table::const_iterator it
       = v1.feature_table.begin();
       it!=v1.feature_table.end();++it)
  {
    int point_id = it->first;
    if (v2.feature_table.find(point_id)==v2.feature_table.end())
      continue;

    const Point & p = GET_MAP_ELEM(point_id, point_table_);

    Pose T_anchor_from_w;

    if (IS_IN_SET(p.anchorframe_id, double_window_))
    {
      T_anchor_from_w
          = GET_MAP_ELEM(p.anchorframe_id, vertex_table_).T_me_from_world;
    }
    else if (IS_IN_SET(p.anchorframe_id, cache_poses))
    {
      T_anchor_from_w
          = GET_MAP_ELEM(p.anchorframe_id, cache_poses);
    }
    else
    {
      T_anchor_from_w = computeAbsolutePose(p.anchorframe_id);
      cache_poses.insert(make_pair(p.anchorframe_id, T_anchor_from_w));
    }

    // get point wrt. world origin
    Vector3d xyz_v1
        = v1.T_me_from_world*T_anchor_from_w.inverse()*p.xyz_anchor;

    depth_set.insert(xyz_v1.norm());
  }

  int visibility_strength = depth_set.size();
  if(visibility_strength<covis_thr_)
  {
    cerr << "ATTENTIOn, THIS SHOULD NEVER HAPPEN!" << endl;
  }

  double median_depth = median(depth_set);
  double norm_dist =  T_1_from_2->translation().norm()/median_depth;
  double alpha = 1;

  Lambda->setIdentity();
  *Lambda *= visibility_strength;
  Lambda->topLeftCorner(3,3) *= Po2(350*alpha*norm_dist);
  Lambda->bottomRightCorner(3,3) *= Po2(100*alpha);
}

template <typename Pose, typename Cam, typename Proj, int ObsDim>
void SlamGraph<Pose,Cam,Proj,ObsDim>
::margPosesLeftInnerWindow(const WindowTable & old_window)
{
  for (typename WindowTable::const_iterator
       it_win1 = old_window.begin(); it_win1!=old_window.end(); ++it_win1)
  {
    int pose_id_1 = it_win1->first;
    WindowType wtype_1 = it_win1->second;
    if (wtype_1==INNER)
    {
      for (typename WindowTable::const_iterator
           it_win2 = old_window.begin(); it_win2!=old_window.end(); ++it_win2)
      {
        int pose_id_2 = it_win2->first;
        if (pose_id_2==pose_id_1)
          continue;
        if (edge_table_.orderd_find(pose_id_1,pose_id_2)==edge_table_.end())
          continue;

        WindowType wtype_2 = it_win2->second;
        if (wtype_2==INNER)
        {
          //here: relative pose was in inner window

          typename WindowTable::const_iterator it1
              = double_window_.find(pose_id_1);
          bool pose_1_is_in_inner_window_now
              = it1!=double_window_.end() && it1->second==INNER;

          typename WindowTable::const_iterator it2
              = double_window_.find(pose_id_2);
          bool pose_2_is_in_inner_window_now
              = it2!=double_window_.end() && it2->second==INNER;

          if (pose_1_is_in_inner_window_now==false
              || pose_2_is_in_inner_window_now==false)
          {
            Pose T_1_from_2;
            Matrix<double,Pose::DoF,Pose::DoF> Lambda;


            const Vertex & v1
                = GET_MAP_ELEM(pose_id_1, vertex_table_);
            const Vertex & v2
                = GET_MAP_ELEM(pose_id_2, vertex_table_);
            computeConstraint(v1, v2,
                              &T_1_from_2, &Lambda);

            edge_table_.setConstraint(pose_id_1, pose_id_2,
                                      T_1_from_2, Lambda, Lambda);
          }
        }
      }
    }
  }
}


template <typename Pose, typename Cam, typename Proj, int ObsDim>
void SlamGraph<Pose,Cam,Proj,ObsDim>
::addPointToG2o(const Vector3d & psi_anchor,
                int g2o_point_id,
                g2o::SparseOptimizer * optimizer)
{
  G2oVertexPointXYZ * v_point = new G2oVertexPointXYZ;

  v_point->setId(g2o_point_id);
  v_point->setEstimate(invert_depth(psi_anchor));
  v_point->setMarginalized(true);

  optimizer->addVertex(v_point);
}

template <typename Pose, typename Cam, typename Proj, int ObsDim>
void SlamGraph<Pose,Cam,Proj,ObsDim>
::copyPosesToG2o(g2o::SparseOptimizer * optimizer)
{
  for (typename WindowTable::const_iterator
       it_win = double_window_.begin(); it_win!=double_window_.end(); ++it_win)
  {
    int frame_id = it_win->first;
    const Vertex & v = GET_MAP_ELEM(frame_id, vertex_table_);

    addPoseToG2o(v.T_me_from_world, frame_id, false,
                 optimizer);
  }
}

template <typename Pose, typename Cam, typename Proj, int ObsDim>
void SlamGraph<Pose,Cam,Proj,ObsDim>
::copyContraintsToG2o(g2o::SparseOptimizer * optimizer)
{
  for (typename WindowTable::const_iterator
       it_win1 = double_window_.begin(); it_win1!=double_window_.end();
       ++it_win1)
  {
    int pose_id_1 = it_win1->first;
    WindowType wtype_1 = it_win1->second;

    for (typename WindowTable::const_iterator
         it_win2 = double_window_.begin(); it_win2!=double_window_.end();
         ++it_win2)
    {
      int pose_id_2 = it_win2->first;
      if (pose_id_2==pose_id_1)
        continue;

      if (edge_table_.orderd_find(pose_id_1,pose_id_2)
          == edge_table_.end())
        continue;

      WindowType wtype_2 = it_win2->second;

      if (wtype_1==OUTER || wtype_2==OUTER)
      {
        Pose T_2_from_1;
        Matrix<double,Pose::DoF,Pose::DoF> Lambda_2_from_1;
        bool is_maginalized
            = edge_table_.getConstraint_id1_from_id2(pose_id_2,
                                                     pose_id_1,
                                                     &T_2_from_1,
                                                     &Lambda_2_from_1);

        assert(is_maginalized);
        addConstraintToG2o(T_2_from_1,
                           Lambda_2_from_1,
                           pose_id_1,
                           pose_id_2,
                           optimizer);
      }
    }
  }
}

template <typename Pose, typename Cam, typename Proj, int ObsDim>
void SlamGraph<Pose,Cam,Proj,ObsDim>
::copyDataToG2o(const OptParams & opt_params,
                g2o::SparseOptimizer * optimizer)
{
  copyPosesToG2o(optimizer);

  for (typename tr1::unordered_set<int>::const_iterator
       it = active_point_set_.begin(); it!= active_point_set_.end();
       ++it)
  {
    int point_id = *it;
    const Point & p = GET_MAP_ELEM(point_id, point_table_);

    Vector3d psi_anchor = p.xyz_anchor;

    addPointToG2o(psi_anchor, point_id, optimizer);

    for (tr1::unordered_set<int>::const_iterator it_vset = p.vis_set.begin();
         it_vset!=p.vis_set.end(); ++it_vset)
    {
      int pose_id = *it_vset;
      if (double_window_.find(pose_id)!=double_window_.end())
      {
        const Vertex & v = GET_MAP_ELEM(pose_id, vertex_table_);
        const ImageFeature<ObsDim> & feat
            = GET_MAP_ELEM(point_id, v.feature_table);
        Matrix<double,ObsDim,ObsDim> Lambda;
        Lambda.setIdentity();
        double pyr_level_factor = Po2(pyrFromZero_d(1.,feat.level));
        Lambda(0,0) *= pyr_level_factor;
        Lambda(1,1) *= pyr_level_factor;
        Lambda(2,2) *= Po2(0.333);

        const Point & p = GET_MAP_ELEM(point_id, point_table_);

        addObsToG2o(feat.center,
                    Lambda,
                    point_id,
                    pose_id,
                    p.anchorframe_id,
                    opt_params.use_robust_kernel,
                    opt_params.huber_kernel_width,
                    optimizer);
      }
    }
  }

  copyContraintsToG2o(optimizer);
}


template <typename Pose, typename Cam, typename Proj, int ObsDim>
void SlamGraph<Pose,Cam,Proj,ObsDim>
::restoreDataFromG2o(const g2o::SparseOptimizer & optimizer)
{
  for (typename g2o::HyperGraph::VertexIDMap::const_iterator it
       = optimizer.vertices().begin();
       it!=optimizer.vertices().end(); ++it)
  {
    g2o::HyperGraph::Vertex * v = it->second;

    if (!SlamGraphMethods::restorePoseFromG2o<Pose,Cam,Proj,ObsDim>
        (v,&vertex_table_))
    {
      G2oVertexPointXYZ * g2o_point = dynamic_cast<G2oVertexPointXYZ*>(v);
      assert(g2o_point!=0);
      int point_id =  g2o_point->id();

      Point & p = GET_MAP_ELEM_REF(point_id, &point_table_);

      p.xyz_anchor = invert_depth(g2o_point->estimate());
    }
  }

}


template <typename Pose, typename Cam, typename Proj, int ObsDim>
void SlamGraph<Pose,Cam,Proj,ObsDim>
::setupG2o(G2oCameraParameters * g2o_cam,
           g2o::SparseOptimizer * optimizer)
{
  optimizer->setVerbose(true);
  typename g2o::BlockSolver_6_3::LinearSolverType * linearSolver
      = SlamGraphMethods::allocateLinearSolver<g2o::BlockSolver_6_3>();
  g2o::BlockSolver_6_3 * block_solver
      = new g2o::BlockSolver_6_3(linearSolver);
  g2o::OptimizationAlgorithmLevenberg * lm
      = new g2o::OptimizationAlgorithmLevenberg(block_solver);
  lm->setMaxTrialsAfterFailure(5);
  optimizer->setAlgorithm(lm);

  if (!optimizer->addParameter(g2o_cam))
  {
    assert(false);
  }
}

}
