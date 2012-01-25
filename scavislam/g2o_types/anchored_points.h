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

#ifndef SCAVISLAM_G2O_ANCHORED_POINTS_H
#define SCAVISLAM_G2O_ANCHORED_POINTS_H

#include <g2o/core/base_vertex.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_multi_edge.h>
#include <g2o/core/base_unary_edge.h>

#include <sophus/se3.h>
#ifdef MONO
#include <sophus/sim3.h>
#endif

#include "../global.h"

namespace ScaViSLAM
{

using namespace Eigen;
using namespace ScaViSLAM;
using namespace Sophus;

class G2oCameraParameters : public g2o::Parameter
{
public:
  G2oCameraParameters();

  G2oCameraParameters        (const Vector2d & principle_point,
                              double focal_length,
                              double baseline)
    : principle_point_(principle_point),
      focal_length_(focal_length),
      baseline_(baseline){}

  Vector2d
  cam_map                    (const Vector3d & trans_xyz) const;

  Vector3d
  stereocam_uvu_map          (const Vector3d & trans_xyz) const;

  virtual bool
  read                       (std::istream& is)
  {
    assert(false);
  }

  virtual bool
  write                      (std::ostream& os) const
  {
    assert(false);
  }


  Vector2d principle_point_;
  double focal_length_;
  double baseline_;
};

class G2oVertexSE3 : public g2o::BaseVertex<6, SE3>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  G2oVertexSE3               (){}

  virtual bool
  read                       (std::istream& is);

  virtual bool
  write                      (std::ostream& os) const;

  virtual void
  oplusImpl                  (const double * update);

  virtual void
  setToOriginImpl            () {}
};

class G2oVertexPointXYZ : public g2o::BaseVertex<3, Vector3d>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  G2oVertexPointXYZ          (){}

  virtual bool//SCAVISLAM
  read                       (std::istream& is);
  virtual bool
  write                      (std::ostream& os) const;
  virtual void
  oplusImpl                  (const double * update);

  virtual void
  setToOriginImpl            ()
  {
    _estimate.setZero();
  }
};

class G2oEdgeProjectPSI2UVU : public  g2o::BaseMultiEdge<3, Vector3d>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  G2oEdgeProjectPSI2UVU()
  {
    resizeParameters(1);
    installParameter(g2o_cam, 0);
  }

  virtual bool
  read                       (std::istream& is);
  virtual bool
  write                      (std::ostream& os) const;
  void
  computeError               ();
  virtual
  void linearizeOplus        ();

  G2oCameraParameters * g2o_cam;
};

class G2oEdgeSim3ProjectUVQ : public  g2o::BaseMultiEdge<2, Vector2d>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  G2oEdgeSim3ProjectUVQ      ()
  {
    resizeParameters(1);
    installParameter(g2o_cam, 0);
  }

  virtual bool
  read                       (std::istream& is);
  virtual bool
  write                      (std::ostream& os) const;

  void
  computeError               ();

  G2oCameraParameters * g2o_cam;
};

class G2oEdgeSE3
    : public g2o::BaseBinaryEdge<6, SE3, G2oVertexSE3, G2oVertexSE3>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  G2oEdgeSE3                 () {}

  virtual bool
  read                       (std::istream& is);
  virtual bool
  write                      (std::ostream& os) const;
  void
  computeError               ();
  virtual void
  linearizeOplus             ();
};

#ifdef MONO
class G2oVertexSim3 : public g2o::BaseVertex<6, Sim3>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  G2oVertexSim3              ();

  virtual bool
  read                       (std::istream& is);
  virtual bool
  write                      (std::ostream& os) const;
  virtual void
  oplus                      (double * update);
  Vector2d
  cam_map                    (const Vector2d & v) const;

  virtual void
  setToOrigin                () {}

  Vector2d principle_point;
  double focal_length;
};

class G2oEdgeSim3
    : public g2o::BaseBinaryEdge<6, Sim3, G2oVertexSim3, G2oVertexSim3>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  G2oEdgeSim3                (){}

  virtual bool
  read                       (std::istream& is);
  virtual bool
  write                      (std::ostream& os) const;
  void computeError          ();

};
#endif

}

#endif
