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

#include "ransac_models.h"

// This RANSAC code is based on "http://www.ros.org/wiki/posest"
// written by Kurt Konolige and originally licensed under BSD.
namespace ScaViSLAM
{

namespace AbsoluteOrientation
{
double
belowThreshold (const StereoCamera & cam,
                const Vector3d & trans_train_pt,
                const Vector3d & uvu,
                double maxInlierXDist2)
{
  Vector3d uvu_pred = cam.map_uvu( trans_train_pt);

  double du_left = uvu[0] - uvu_pred.x();
  double dv = uvu[1] - uvu_pred.y();
  double du_right = uvu[2] - uvu_pred.z();

  return du_left*du_left < maxInlierXDist2 &&
      dv*dv < maxInlierXDist2 &&
      du_right*du_right < maxInlierXDist2;
}

Matrix3d
getOrientationAndCentriods(Vector3d & p0a,
                           Vector3d & p0b,
                           Vector3d & p0c,
                           Vector3d & p1a,
                           Vector3d & p1b,
                           Vector3d & p1c,
                           Vector3d & c0,
                           Vector3d & c1)
{
  c0 = (p0a+p0b+p0c)*(1.0/3.0);
  c1 = (p1a+p1b+p1c)*(1.0/3.0);

  // subtract out
  p0a -= c0;
  p0b -= c0;
  p0c -= c0;
  p1a -= c1;
  p1b -= c1;
  p1c -= c1;

  Matrix3d H = p1a*p0a.transpose() + p1b*p0b.transpose() +
               p1c*p0c.transpose();

  JacobiSVD<Matrix3d> svd(H, ComputeFullU | ComputeFullV);
  Matrix3d V = svd.matrixV();

  Matrix3d  R;
  R = V * svd.matrixU().transpose();
  double det = R.determinant();

  if (det < 0.0)
  {
    V.col(2) = V.col(2)*-1.0;
    R = V * svd.matrixU().transpose();
  }
  return R;
}

}//namespace AbsoluteOrientation

#ifdef MONO
void Sim3Model
::calc_motion(const StereoCamera & cam,
              const ALIGNED<Vector3d>::vector & query_obs_vec,
              const ALIGNED<Vector3d>::vector & train_pt_vec,
              Transformation & sim3)
{
  assert(query_obs_vec.size()==num_points);
  assert(train_pt_vec.size()==num_points);

  // get centroids
  Vector3d p0a = cam.unmap_uvu(query_obs_vec[0]);
  Vector3d p0b = cam.unmap_uvu(query_obs_vec[1]);
  Vector3d p0c = cam.unmap_uvu(query_obs_vec[2]);
  Vector3d p1a = train_pt_vec[0];
  Vector3d p1b = train_pt_vec[1];
  Vector3d p1c = train_pt_vec[2];
  Vector3d c0, c1;
  Matrix3d R
      = AbsoluteOrientation::getOrientationAndCentriods(p0a,
                                                        p0b,
                                                        p0c,
                                                        p1a,
                                                        p1b,
                                                        p1c,
                                                        c0,
                                                        c1);
  sim3.set_rotation_matrix(R);
  sim3.scale() =
      sqrt(p0a.squaredNorm()
           + p0b.squaredNorm()
           + p0c.squaredNorm())
      /
      sqrt((R*p1a).squaredNorm()
           + (R*p1b).squaredNorm()
           + (R*p1c).squaredNorm());

  sim3.translation() = c0-sim3.scale()*R*c1;
}

bool Sim3Model
::belowThreshold(const StereoCamera & cam,
                 const Vector3d & trans_train_pt,
                 const Vector3d & uvu,
                 double maxInlierXDist2)
{
  return AbsoluteOrientation::belowThreshold(cam,
                                             trans_train_pt,
                                             uvu,
                                             maxInlierXDist2);
}
#endif

void SE3Model
::calc_motion(const StereoCamera & cam,
              const ALIGNED<Vector3d>::vector & query_obs_vec,
              const ALIGNED<Vector3d>::vector & train_pt_vec,
              Transformation & se3)
{
  assert(query_obs_vec.size()==num_points);
  assert(train_pt_vec.size()==num_points);

  // get centroids
  Vector3d p0a = cam.unmap_uvu(query_obs_vec[0]);
  Vector3d p0b = cam.unmap_uvu(query_obs_vec[1]);
  Vector3d p0c = cam.unmap_uvu(query_obs_vec[2]);
  Vector3d p1a = train_pt_vec[0];
  Vector3d p1b = train_pt_vec[1];
  Vector3d p1c = train_pt_vec[2];

  Vector3d c0, c1;

  Matrix3d R
      = AbsoluteOrientation::getOrientationAndCentriods(p0a,
                                                        p0b,
                                                        p0c,
                                                        p1a,
                                                        p1b,
                                                        p1c,
                                                        c0,
                                                        c1);
  se3.translation() = c0-R*c1;
  se3.setRotationMatrix(R);

}

bool SE3Model::
belowThreshold(const StereoCamera & cam,
               const Vector3d & trans_train_pt,
               const Vector3d & uvu,
               double maxInlierXDist2)
{
  return AbsoluteOrientation::belowThreshold(cam,
                                             trans_train_pt,
                                             uvu,
                                             maxInlierXDist2);
}

}
