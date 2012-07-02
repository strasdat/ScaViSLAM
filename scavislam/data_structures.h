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

#ifndef SCAVISLAM_DATA_STRUCTURES_H
#define SCAVISLAM_DATA_STRUCTURES_H

#include <list>

#include <sophus/se3.h>

#include "global.h"
#include "keyframes.h"


namespace ScaViSLAM
{

using namespace std;
using namespace Sophus;
using namespace Eigen;


template <int Dim>
class CandidatePoint
{
public:
  CandidatePoint(int point_id,
              const Vector3d & xyz_anchor,

              int anchor_id,
              //const ImageFeature<ObsDim> & keyfeat,
              const typename VECTOR<Dim>::col & anchor_obs_pyr,
              int anchor_level,
              const Vector3d& normal_anchor)
    : point_id(point_id),
      xyz_anchor(xyz_anchor),
      anchor_id(anchor_id),
      anchor_obs_pyr(anchor_obs_pyr),
      anchor_level(anchor_level),
      normal_anchor(normal_anchor)

  {
  }

  int point_id;

  Vector3d xyz_anchor;
  int anchor_id;
  typename VECTOR<Dim>::col anchor_obs_pyr;
  int anchor_level;
  Vector3d normal_anchor;


  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

typedef tr1::shared_ptr<CandidatePoint<2> > ActivePoint2Ptr;
typedef tr1::shared_ptr<CandidatePoint<3> > CandidatePoint3Ptr;


template <int Dim>
struct ImageFeature
{
  ImageFeature(const Matrix<double,Dim,1>& center,
               int level)
    : center(center),
      level(level)
  {
  }

  Matrix<double,Dim,1> center;
  int level;

  typedef typename ALIGNED<ImageFeature<Dim> >::int_hash_map Table;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};



template <int Dim>
class TrackPoint
{
public:
  TrackPoint(int global_id,
             const ImageFeature<Dim> & feat)
    : global_id(global_id), feat(feat)

  {
  }

  int global_id;
  ImageFeature<Dim> feat;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

};

typedef tr1::shared_ptr<TrackPoint<2> > TrackPoint2Ptr;
typedef tr1::shared_ptr<TrackPoint<3> > TrackPoint3Ptr;


template <int Dim>
struct NewTwoViewPoint
{
  NewTwoViewPoint(int point_id,
                  int anchor_id,
                  const Vector3d & xyz_anchor,
                  const typename VECTOR<Dim>::col & anchor_obs_pyr,
                  int anchor_level,
                  const Vector3d & normal_anchor,
                  const ImageFeature<Dim> & feat_newkey)
    : point_id(point_id),
      anchor_id(anchor_id),
      xyz_anchor(xyz_anchor),
      anchor_obs_pyr(anchor_obs_pyr),
      anchor_level(anchor_level),
      normal_anchor(normal_anchor),
      feat_newkey(feat_newkey)

  {}

  int point_id;
  int anchor_id;
  Vector3d xyz_anchor;
  const typename VECTOR<Dim>::col anchor_obs_pyr;
  int anchor_level;
  Vector3d  normal_anchor;
  ImageFeature<Dim> feat_newkey;


  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

typedef tr1::shared_ptr<NewTwoViewPoint<2> > NewTwoViewPoint2Ptr;
typedef tr1::shared_ptr<NewTwoViewPoint<3> > NewTwoViewPoint3Ptr;


struct AddToOptimzer
{
  AddToOptimzer(bool first_frame=false) : first_frame(first_frame)
  {
  }

  SE3 T_newkey_from_oldkey;
  list<NewTwoViewPoint3Ptr> new_point_list;
  list<TrackPoint3Ptr> track_point_list;

  int oldkey_id;
  int newkey_id;

  Frame kf;

  bool first_frame;

};
typedef tr1::shared_ptr<AddToOptimzer> AddToOptimzerPtr;

struct FrontendVertex
{
  SE3 T_me_from_w;
  multimap<int,int> strength_to_neighbors;
  ImageFeature<3>::Table feat_map;
};

struct Neighborhood
{
  list<CandidatePoint3Ptr> point_list;
  ALIGNED<FrontendVertex>::int_hash_map vertex_map;
};

typedef tr1::shared_ptr<Neighborhood> NeighborhoodPtr;

}

#endif
