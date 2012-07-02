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

#ifndef SCAVISLAM_SLAM_GRAPH_H
#define SCAVISLAM_SLAM_GRAPH_H

#include "global.h"
#include "data_structures.h"

#include "g2o_types/anchored_points.h"

namespace g2o
{
class SparseOptimizer;
}

namespace ScaViSLAM
{

using namespace ScaViSLAM;

struct OptParams
{
  OptParams(int num_iters,
            bool use_robust_kernel=false,
            double huber_kernel_width=1)
    : num_iters(num_iters),
      use_robust_kernel(use_robust_kernel),
      huber_kernel_width(huber_kernel_width)
  {
  }

  int num_iters;
  bool use_robust_kernel;
  double huber_kernel_width;
};

// TODO: clean interface by moving stuff to *.cpp files

template <typename Pose,   // frame/pose type such as SE3
          typename Camera, // Camera model (focal length, principle points...)
          typename Proj,   // Projection function (monocular, stereo,...)
          int ObsDim>      // Dimension of observations (2 for monocular...)
class SlamGraph
{
public:

  // Typedefs and Enums
  //typedef typename ALIGNED<ImageFeature<ObsDim> >::int_hash_map FeatureTable;

  struct Vertex
  {
  public:
    Vertex() : fixed(false)
    {
    }

    int covisibilityScore(const Vertex & v) const
    {
      multimap<int,int>::const_iterator it
          = neighbor_ids_ordered_by_strength.begin();
      while (v.own_id!=it->second)
      {
        ++it;
        if (it==neighbor_ids_ordered_by_strength.end())
          return 0;
      }
      return it->first;
    }

    int own_id;

    multimap<int,int>                  // first int: strength (number of
    neighbor_ids_ordered_by_strength;  //   co-visible s between both frames)
    //                                    second int: vertex_id

    bool fixed;                        // Fix vertex during optimization?
    typename ImageFeature<ObsDim>::Table  feature_table;
    Pose T_me_from_world;

  private: DISALLOW_COPY_AND_ASSIGN(Vertex)
    public: EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };

    typedef tr1::shared_ptr<Vertex> VertexPtr;


    struct Point                         // 3D point/landmark
    {
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW

      Point(const Vector3d& xyz_anchor,
            const tr1::unordered_set<int> & vis_set,
            int anchorframe_id,
            const typename VECTOR<ObsDim>::col & anchor_obs_pyr,
            int anchor_level,
            const Vector3d& normal_anchor)
        : xyz_anchor(xyz_anchor),
          vis_set(vis_set),
          anchorframe_id(anchorframe_id),
          anchor_obs_pyr(anchor_obs_pyr),
          anchor_level(anchor_level),
          normal_anchor(normal_anchor)
      {
      }

      Vector3d xyz_anchor;               // actual 3d position

      tr1::unordered_set<int> vis_set;   // set of frame vertex ids from which
      //                                    the point is visible

      int anchorframe_id;                // id of the anchor frame tom which the
      //                                    point is anchored

      typename VECTOR<ObsDim>::col       // initial observation in anchor frame
      anchor_obs_pyr;
      int anchor_level;                  // pyramid level of observation

      Vector3d normal_anchor;            // estimate of surface normal

    private:
      DISALLOW_COPY_AND_ASSIGN(Point)
    };

    typedef tr1::shared_ptr<Point> PointPtr;

    class EdgeTable;

    enum EdgeType { LOCAL, METRIC, APPREARANCE};

    class Edge                           //edge between two pose vertices
    {
      friend class EdgeTable;
      bool is_marginalized_;              /* Are points are used during
                                          optimization (false), or is a relative
                                          pose contraint used instead (true)? */
    public:

      bool is_marginalized() const
      {
        return is_marginalized_;
      }

      Edge(int id1, int id2, int strength, EdgeType et)
        : is_marginalized_(false), strength(strength), error(0), edge_type(et)
      {
        assert(id1!=id2);

        if (id1<id2)                     // Takes care of that id1<id2.
        {
          vertex_id1 = id1;
          vertex_id2 = id2;
        }
        else
        {
          vertex_id1 = id2;
          vertex_id2 = id1;
        }
      }

      int vertex_id1;
      int vertex_id2;

      Pose T_1_from_2;

      int strength;
      double error;

      EdgeType edge_type;

      Matrix<double, Pose::DoF, Pose::DoF> Lambda_2_from_1;
      Matrix<double, Pose::DoF, Pose::DoF> Lambda_1_from_2;

    private:
      DISALLOW_COPY_AND_ASSIGN(Edge)
    };

    typedef tr1::shared_ptr<Edge> EdgePtr;

    typedef tr1::unordered_map<std::pair<int,int>, EdgePtr, IntPairHash>
    StdEdgeTable;

    class EdgeTable : StdEdgeTable     // This is the actuall graph strucuture
        //                                for the pose egdes. It is a hash table
        //                                which maps a pair of vertex indices to
        //                                an edge. Since this is an undirected
        //                                graph, internally it is taken care of
        //                                that the first index is always smaller
        //                                than the second index.
    {
    public:

      // ToDo: This is ugly!
      // Eliminate code doublicates by moving most of this code into Edge and
      // write a unique access method using iterators.

      typename StdEdgeTable::iterator orderd_find(int id1, int id2)
      {
        assert(id1!=id2);

        if (id1<id2)
        {
          return StdEdgeTable::find(make_pair(id1,id2));
        }
        else
          return StdEdgeTable::find(make_pair(id2,id1));
      }

      typename StdEdgeTable::const_iterator orderd_find(int id1, int id2) const
      {
        assert(id1!=id2);

        if (id1<id2)
        {
          return StdEdgeTable::find(make_pair(id1,id2));
        }
        else
          return StdEdgeTable::find(make_pair(id2,id1));
      }

      void unMarginalize(int id1, int id2)
      {
        typename StdEdgeTable::iterator  it = orderd_find(id1, id2);
        assert(it!=end());
        it->second->is_marginalized_ = false;
      }

      bool
      getConstraint_id1_from_id2(int id1, int id2,
                                 Pose * T_1_from_2,
                                 Matrix<double,Pose::DoF,Pose::DoF> *
                                 Lambda_1_from_2) const
      {
        assert(id1!=id2);

        if (id1<id2)
        {
          typename StdEdgeTable::const_iterator it
              = StdEdgeTable::find(make_pair(id1,id2));
          if (it==end())
          {
            assert(false);
          }

          if(it->second->is_marginalized_==false)
          {
            return false;
          }

          *T_1_from_2 = it->second->T_1_from_2;
          *Lambda_1_from_2 = it->second->Lambda_1_from_2;
          return true;
        }

        typename StdEdgeTable::const_iterator it
            = StdEdgeTable::find(make_pair(id2,id1));
        if (it==end())
        {
          assert(false);
        }

        if(it->second->is_marginalized_==false)
        {
          return false;
        }

        *T_1_from_2 = it->second->T_1_from_2.inverse();
        *Lambda_1_from_2 = it->second->Lambda_2_from_1;

        return true;
      }

      bool
      getConstraint_id1_from_id2(int id1, int id2,
                                 Pose * T_1_from_2) const
      {
        Matrix<double,Pose::DoF,Pose::DoF> dummy;
        return getConstraint_id1_from_id2(id1, id2, T_1_from_2,&dummy);
      }

      void setConstraint(int id1,
                         int id2,
                         const Pose & T_1_from_2,
                         const Matrix<double,Pose::DoF,Pose::DoF> &
                         Lambda_1_from_2,
                         const Matrix<double,Pose::DoF,Pose::DoF> &
                         Lambda_2_from_1)
      {
        assert(id1!=id2);

        if (id1<id2)
        {
          typename StdEdgeTable::iterator it
              = StdEdgeTable::find(make_pair(id1,id2));
          if (it==end())
          {
            assert(false);
          }
          it->second->is_marginalized_ = true;
          it->second->T_1_from_2 = T_1_from_2;
          it->second->Lambda_2_from_1 = Lambda_2_from_1;
          it->second->Lambda_1_from_2 = Lambda_1_from_2;
        }
        else
        {
          typename StdEdgeTable::iterator it
              = StdEdgeTable::find(make_pair(id2,id1));
          if (it==end())
          {
            assert(false);
          }
          it->second->is_marginalized_ = true;
          it->second->T_1_from_2 = T_1_from_2.inverse();
          it->second->Lambda_2_from_1 = Lambda_1_from_2;
          it->second->Lambda_1_from_2 = Lambda_2_from_1;
        }
      }

      void insertEdge(int id1, int id2, int strenght, EdgeType edge_type)
      {
        assert(id1!=id2);
        bool inserted;
        if (id1<id2)
        {
          inserted
              = insert(make_pair(make_pair(id1, id2),
                                 EdgePtr(new Edge(id1, id2, strenght, edge_type)))).second;
        }
        else
        {
          inserted
              = insert(make_pair(make_pair(id2, id1),
                                 EdgePtr(new Edge(id2, id1, strenght, edge_type)))).second;
        }
        if (inserted==false)
        {
          cerr << id1 << " " << id2 << endl;
        }
        assert(inserted);
      }

      //Make useful things public of the private base class
      using StdEdgeTable::iterator;
      using StdEdgeTable::const_iterator;
      using StdEdgeTable::begin;
      using StdEdgeTable::end;
      using StdEdgeTable::size;
      using StdEdgeTable::clear;
    };


    struct Statistics
    {
      Statistics()
        : num_frame_edges(0),
          num_point_edges(0),
          num_frames(0),
          num_points(0),
          calc_time(0)
      {

      }

      int num_frame_edges;
      int num_point_edges;
      int num_frames;
      int num_points;
      double calc_time;

    private:
      DISALLOW_COPY_AND_ASSIGN(Statistics)
    };

    enum WindowType {INNER=0, OUTER=1};
    typedef tr1::unordered_map<int,WindowType> WindowTable;

    typedef tr1::unordered_map<int,VertexPtr> VertexTable;

    typedef tr1::unordered_map<int,PointPtr>  PointTable;

    typedef CandidatePoint<ObsDim> MyActivePoint;
    typedef tr1::shared_ptr<MyActivePoint > MyActivePointPtr;

    typedef NewTwoViewPoint<ObsDim> MyNewTwoViewPoint;
    typedef tr1::shared_ptr<MyNewTwoViewPoint> MyNewTwoViewPointPtr;

    typedef TrackPoint<ObsDim> MyTrackPoint;
    typedef tr1::shared_ptr<MyTrackPoint> MyTrackPointPtr;

    SlamGraph(const Camera & cam)
      : cam_(cam),
        proj_(cam),
        inner_window_size_(-1),
        double_window_size_(-1),
        covis_thr_(-1)
    {
    }

    void
    initialize               (int new_inner_window_size,
                              int new_outer_window_size,
                              int new_covis_thr)
    {
      inner_window_size_ = new_inner_window_size;
      double_window_size_ = new_outer_window_size;
      covis_thr_ = new_covis_thr;
    }

    bool
    shortestPathToWindow     (int root_id,
                              list<int> * path) const;
    tr1::unordered_set<int>
    framesInNeighborhood     (int root_id,
                              size_t size) const;
    Pose
    computeAbsolutePose      (int x_id) const;

    void
    addKeyframe              (int oldkey_id,
                              int newkey_id,
                              const Pose & T_newkey_from_oldkey,
                              const list<MyNewTwoViewPointPtr> & newpoint_list,
                              const list<MyTrackPointPtr> & trackpoint_list);
    void
    registerKeyframes        (int root_id,
                              const Pose & T_newroot_from_w,
                              const IntTable & neighborid_to_strength,
                              const list<MyTrackPointPtr> & trackpoint_list);
    void
    addLoopClosure           (int root_id,
                              int loop_id,
                              const Pose & T_root_from_loop,
                              const list<MyTrackPointPtr> & trackpoint_list);
    void
    addFirstKeyframe         (int newkey_id);

    Pose
    getRelativePose_1_from_2 (int id1, int id2) const;

    bool
    prepareForOptimization   (int root_id,
                              int loop_id);
    void
    optimize                 (const OptParams & opt_params);

    void
    optimize                 (const OptParams & opt_params,
                              Statistics * stats);

    // Getter and setter
    const WindowTable & double_window() const
    {
      return double_window_;
    }

    const tr1::unordered_set<int> & active_point_set() const
    {
      return active_point_set_;
    }

    const tr1::unordered_set<int> & outer_point_set() const
    {
      return outer_point_set_;
    }

    const VertexTable & vertex_table() const
    {
      return vertex_table_;
    }

    const PointTable & point_table() const
    {
      return point_table_;
    }

    const EdgeTable & edge_table() const
    {
      return edge_table_;
    }

    int covis_thr() const
    {
      return covis_thr_;
    }

  private:
    struct PathTraversalNode
    {
      PathTraversalNode(){}
      PathTraversalNode(int own_id,
                        const list<int> & path_to_parent)
        : own_id(own_id), path_to_me(path_to_parent)
      {
        path_to_me.push_back(own_id);
      }

      int own_id;
      list<int> path_to_me;
    };

    class ReinitializeTraversalNode
    {
    public:
      ReinitializeTraversalNode(int own_id,
                                int parent_id,
                                const Pose & T_parent_from_world,
                                bool mark_reinitialize)
        : own_id(own_id),
          parent_id(parent_id),
          T_parent_from_world(T_parent_from_world),
          mark_reinitialize(mark_reinitialize)
      {
      }

      int own_id;
      int parent_id;
      Pose T_parent_from_world;
      bool mark_reinitialize;
    };

    void
    addNewPointsToMap        (const list<MyNewTwoViewPointPtr> & newpoint_list,
                              const IntTable & neighborid_to_strength,
                              Vertex * v_newkey);
    void
    addNewObsToOldPoints     (const list<MyTrackPointPtr> & trackpoint_list,
                              Vertex * v_newkey);
    void
    addNewEdges              (const IntTable & neighborid_to_strength,
                              EdgeType edge_type,
                              Vertex * v_newkey);
    void
    computeStrength          (const list<MyNewTwoViewPointPtr> & newpoint_list,
                              const list<MyTrackPointPtr> & trackpoint_list,
                              IntTable *
                              neighborid_to_strength);
    void
    computeInitialDoubleWin  (int root_id,
                              int inner_window_size,
                              int outer_window_size);
    void
    computeActivePointsAndExtendOuterWindow();

    void
    reinitializePoses        (int root_id,
                              const WindowTable & old_window,
                              int loop_id);

    void
    margPosesLeftInnerWindow (const WindowTable & old_window);

    void
    unmargPosesEnteringInnerW();

    void
    computeConstraint        (const Vertex & v1,
                              const Vertex & v2,
                              Pose * T_2_from_1,
                              Matrix<double,Pose::DoF,Pose::DoF> * Lambda);
    void
    setupG2o                 (G2oCameraParameters * g2o_cam,
                              g2o::SparseOptimizer * optimizer);

    void
    copyDataToG2o            (const OptParams & opt_params,
                              g2o::SparseOptimizer * optimizer);
    void
    restoreDataFromG2o       (const g2o::SparseOptimizer & optimizer);

    void
    copyPosesToG2o           (g2o::SparseOptimizer * optimizer);

    void
    copyContraintsToG2o      (g2o::SparseOptimizer * optimizer);

    void
    addPoseToG2o             (const Pose & T_me_from_w,
                              int pose_id,
                              bool fixed,
                              g2o::SparseOptimizer * optimizer);
    void
    addPointToG2o            (const Vector3d & psi_anchor,
                              int point_id,
                              g2o::SparseOptimizer * optimizer);
    void
    addObsToG2o              (const typename VECTOR<ObsDim>::col & obs,
                              const Matrix<double, ObsDim, ObsDim> & Lambda,
                              int point_id,
                              int pose_id,
                              int anchor_id,
                              bool robustify,
                              double huber_kernel_width,
                              g2o::SparseOptimizer * optimizer);
    void
    addConstraintToG2o       (const Pose & T_2_from_1,
                              const Matrix<double, Pose::DoF, Pose::DoF> &
                              Lambda_2_from_1,
                              int pose_id_1,
                              int pose_id_2,
                              g2o::SparseOptimizer * optimizer);

    Camera cam_;
    Proj proj_;
    WindowTable double_window_;
    tr1::unordered_set<int> active_point_set_;
    tr1::unordered_set<int> outer_point_set_;
    VertexTable vertex_table_;

    PointTable  point_table_;          // hash table of 3d points
    //                                    (point_id -> Point)
    EdgeTable  edge_table_;
    int inner_window_size_;
    int double_window_size_;
    int covis_thr_;
  };

}
#endif
