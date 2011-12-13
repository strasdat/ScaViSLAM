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

#ifndef SCAVISLAM_TRANSFORMATIONS_H
#define SCAVISLAM_TRANSFORMATIONS_H

#include <list>

#include <sophus/se3.h>
#ifdef MONO
#include <sophus/sim3.h>
#endif

#include <visiontools/linear_camera.h>

#include "maths_utils.h"
#include "stereo_camera.h"

namespace ScaViSLAM
{

using namespace Eigen;
using namespace Sophus;
using namespace VisionTools;

//TODO: clean, hide implementation and remove stuff not needed here

struct AnchoredPoint3d
{
  AnchoredPoint3d(const Vector3d & p_a, int frame_id)
    : p_a(p_a), frame_id(frame_id)
  {
  }
  Vector3d p_a;
  int frame_id;
};

inline Matrix<double,2,3>
d_proj_d_y(const double & f, const Vector3d & xyz)
{
  double z_sq = xyz[2]*xyz[2];
  Matrix<double,2,3> J;
  J << f/xyz[2], 0,           -(f*xyz[0])/z_sq,
      0,           f/xyz[2], -(f*xyz[1])/z_sq;
  return J;
}

inline Matrix3d
d_stereoproj_d_y(const double & f, double b, const Vector3d & xyz)
{
  double z_sq = xyz[2]*xyz[2];
  Matrix3d J;
  J << f/xyz[2], 0,           -(f*xyz[0])/z_sq,
      0,           f/xyz[2], -(f*xyz[1])/z_sq,
      f/xyz[2], 0,           -(f*(xyz[0]-b))/z_sq;
  return J;
}

inline Matrix<double,3,6>
d_expy_d_y(const Vector3d & y)
{
  Matrix<double,3,6> J;
  J.topLeftCorner<3,3>().setIdentity();
  J.bottomRightCorner<3,3>() = -SO3::hat(y);
  return J;
}

inline Matrix3d
d_Tinvpsi_d_psi(const SE3 & T, const Vector3d & psi)
{
  Matrix3d R = T.rotation_matrix();
  Vector3d x = invert_depth(psi);
  Vector3d r1 = R.col(0);
  Vector3d r2 = R.col(1);
  Matrix3d J;
  J.col(0) = r1;
  J.col(1) = r2;
  J.col(2) = -R*x;
  J*=1./psi.z();
  return J;
}

inline void
point_jac_xyz2uv(const Vector3d & xyz,
                 const Matrix3d & R,
                 const double & focal_length,
                 Matrix<double,2,3> & point_jac)
{
  double x = xyz[0];
  double y = xyz[1];
  double z = xyz[2];
  Matrix<double,2,3> tmp;
  tmp(0,0) = focal_length;
  tmp(0,1) = 0;
  tmp(0,2) = -x/z*focal_length;
  tmp(1,0) = 0;
  tmp(1,1) = focal_length;
  tmp(1,2) = -y/z*focal_length;
  point_jac =  -1./z * tmp * R;
}

inline void
frame_jac_xyz2uv(const Vector3d & xyz,
                 const double & focal_length,
                 Matrix<double,2,6> & frame_jac)
{
  double x = xyz[0];
  double y = xyz[1];
  double z = xyz[2];
  double z_2 = z*z;

  frame_jac(0,0) = -1./z *focal_length;
  frame_jac(0,1) = 0;
  frame_jac(0,2) = x/z_2 *focal_length;
  frame_jac(0,3) =  x*y/z_2 * focal_length;
  frame_jac(0,4) = -(1+(x*x/z_2)) *focal_length;
  frame_jac(0,5) = y/z *focal_length;

  frame_jac(1,0) = 0;
  frame_jac(1,1) = -1./z *focal_length;
  frame_jac(1,2) = y/z_2 *focal_length;
  frame_jac(1,3) = (1+y*y/z_2) *focal_length;
  frame_jac(1,4) = -x*y/z_2 *focal_length;
  frame_jac(1,5) = -x/z *focal_length;
}

inline void
frame_jac_xyz2uvu(const Vector3d & xyz,
                  const Vector2d & focal_length,
                  Matrix<double,3,6> & frame_jac)
{
  double x = xyz[0];
  double y = xyz[1];
  double z = xyz[2];
  double z_2 = z*z;

  frame_jac(0,0) = -1./z *focal_length(0);
  frame_jac(0,1) = 0;
  frame_jac(0,2) = x/z_2 *focal_length(0);
  frame_jac(0,3) =  x*y/z_2 * focal_length(0);
  frame_jac(0,4) = -(1+(x*x/z_2)) *focal_length(0);
  frame_jac(0,5) = y/z *focal_length(0);

  frame_jac(1,0) = 0;
  frame_jac(1,1) = -1./z *focal_length(1);
  frame_jac(1,2) = y/z_2 *focal_length(1);
  frame_jac(1,3) = (1+y*y/z_2) *focal_length(1);
  frame_jac(1,4) = -x*y/z_2 *focal_length(1);
  frame_jac(1,5) = -x/z *focal_length(1);
}

//  /**
//   * Abstract prediction class
//   * Frame: How is the frame/pose represented? (e.g. SE3)
//   * FrameDoF: How many DoF has the pose/frame? (e.g. 6 DoF, that is
//   *           3 DoF translation, 3 DoF rotation)
//   * PointParNum: number of parameters to represent a point
//   *              (4 for a 3D homogenious point)
//   * PointDoF: DoF of a point (3 DoF for a 3D homogenious point)
//   * ObsDim: dimensions of observation (2 dim for (u,v) image
//   *         measurement)
//   */
template <typename Frame,
          int FrameDoF,
          typename Point,
          int PointDoF,
          int ObsDim>
class AbstractPrediction
{
public:

  /** Map a world point x into the camera/sensor coordinate frame T
       * and create an observation*/
  virtual Matrix<double,ObsDim,1>
  map                        (const Frame & T,
                              const Point & x) const = 0;

  virtual Matrix<double,ObsDim,1>
  map_n_bothJac              (const Frame & T,
                              const Point & x,
                              Matrix<double,ObsDim,FrameDoF> & frame_jac,
                              Matrix<double,ObsDim,PointDoF> & point_jac) const
  {
    frame_jac = frameJac(T,x);
    point_jac = pointJac(T,x);
    return map(T,x);
  }

  virtual Matrix<double,ObsDim,1>
  map_n_frameJac             (const Frame & T,
                              const Point & x,
                              Matrix<double,ObsDim,FrameDoF> & frame_jac) const
  {
    frame_jac = frameJac(T,x);
    return map(T,x);
  }

  virtual Matrix<double,ObsDim,1>
  map_n_pointJac             (const Frame & T,
                              const Point & x,
                              Matrix<double,ObsDim,PointDoF> & point_jac) const
  {
    point_jac = pointJac(T,x);
    return map(T,x);
  }


  /** Jacobian wrt. frame: use numerical Jacobian as default */
  virtual Matrix<double,ObsDim,FrameDoF>
  frameJac                   (const Frame & T,
                              const Point & x) const
  {
    double h = 0.000000000001;
    Matrix<double,ObsDim,FrameDoF> J_pose
        = Matrix<double,ObsDim,FrameDoF>::Zero();

    Matrix<double,ObsDim,1>  fun = -map(T,x);
    for (unsigned int i=0; i<FrameDoF; ++i)
    {
      Matrix<double,FrameDoF,1> eps
          = Matrix<double,FrameDoF,1>::Zero();
      eps[i] = h;

      J_pose.col(i) = (-map(add(T,eps),x) -fun)/h ;
    }
    return J_pose;
  }

  /** Jacobian wrt. point: use numerical Jacobian as default */
  virtual Matrix<double,ObsDim,PointDoF>
  pointJac                   (const Frame & T,
                              const Point & x) const
  {
    double h = 0.000000000001;
    Matrix<double,ObsDim,PointDoF> J_x
        = Matrix<double,ObsDim,PointDoF>::Zero();
    Matrix<double,ObsDim,1> fun = -map(T,x);
    for (unsigned int i=0; i<PointDoF; ++i)
    {
      Matrix<double,PointDoF,1> eps
          = Matrix<double,PointDoF,1>::Zero();
      eps[i] = h;

      J_x.col(i) = (-map(T,add(x,eps)) -fun)/h ;

    }
    return J_x;
  }

  /** Add an incermental update delta to pose/frame T*/
  virtual Frame
  add                        (const Frame & T,
                              const Matrix<double,FrameDoF,1> & delta) const = 0;

  /** Add an incremental update delta to point x*/
  virtual Point
  add                        (const Point & x,
                              const Matrix<double,PointDoF,1> & delta) const = 0;
};


template <typename Frame,
          int FrameDoF,
          typename Point,
          int PointDoF,
          int ObsDim>
class AbstractAnchoredPrediction
{
public:

  /** Map a world point x into the camera/sensor coordinate frame T
       * and create an observation*/
  virtual Matrix<double,ObsDim,1>
  map                        (const Frame & T_cw,
                              const Frame & A_wa,
                              const Point & x_a) const = 0;

  virtual Matrix<double,ObsDim,1>
  map_n_bothJac              (const Frame & T_cw,
                              const Frame & A_wa,
                              const Point & x_a,
                              Matrix<double,ObsDim,FrameDoF> & frame_jac,
                              Matrix<double,ObsDim,PointDoF> & point_jac) const
  {
    frame_jac = frameJac(T_cw,A_wa,x_a);
    point_jac = pointJac(T_cw,A_wa,x_a);
    return map(T_cw,A_wa,x_a);
  }

  virtual Matrix<double,ObsDim,1>
  map_n_allJac               (const Frame & T_cw,
                              const Frame & A_wa,
                              const Point & x_a,
                              Matrix<double,ObsDim,FrameDoF> & frame_jac,
                              Matrix<double,ObsDim,FrameDoF> & anchor_jac,
                              Matrix<double,ObsDim,PointDoF> & point_jac) const
  {
    frame_jac = frameJac(T_cw,A_wa,x_a);
    anchor_jac = anchorJac(T_cw,A_wa,x_a);
    point_jac = pointJac(T_cw,A_wa,x_a);
    return map(T_cw,A_wa,x_a);
  }


  /** Jacobian wrt. frame: use numerical Jacobian as default */
  virtual Matrix<double,ObsDim,FrameDoF>
  frameJac                   (const Frame & T_cw,
                              const Frame & A_wa,
                              const Point & x_a) const
  {
    double h = 0.000000000001;
    Matrix<double,ObsDim,FrameDoF> J_pose
        = Matrix<double,ObsDim,FrameDoF>::Zero();

    Matrix<double,ObsDim,1>  fun = -map(T_cw,A_wa,x_a);
    for (unsigned int i=0; i<FrameDoF; ++i)
    {
      Matrix<double,FrameDoF,1> eps
          = Matrix<double,FrameDoF,1>::Zero();
      eps[i] = h;

      J_pose.col(i) = (-map(add(T_cw,eps),A_wa,x_a) -fun)/h ;
    }
    return J_pose;
  }

  /** Jacobian wrt. anchor: use numerical Jacobian as default */
  virtual Matrix<double,ObsDim,FrameDoF>
  anchorJac                  (const Frame & T_cw,
                              const Frame & A_wa,
                              const Point & x_a) const
  {
    double h = 0.000000000001;
    Matrix<double,ObsDim,FrameDoF> J_pose
        = Matrix<double,ObsDim,FrameDoF>::Zero();

    Matrix<double,ObsDim,1>  fun = -map(T_cw,A_wa,x_a);
    for (unsigned int i=0; i<FrameDoF; ++i)
    {
      Matrix<double,FrameDoF,1> eps
          = Matrix<double,FrameDoF,1>::Zero();
      eps[i] = h;

      J_pose.col(i) = (-map(T_cw,add(A_wa,eps),x_a) -fun)/h ;
    }
    return J_pose;
  }

  /** Jacobian wrt. point: use numerical Jacobian as default */
  virtual Matrix<double,ObsDim,PointDoF>
  pointJac                   (const Frame & T_cw,
                              const Frame & A_wa,
                              const Point & x_a) const
  {
    double h = 0.000000000001;
    Matrix<double,ObsDim,PointDoF> J_x
        = Matrix<double,ObsDim,PointDoF>::Zero();
    Matrix<double,ObsDim,1> fun = -map(T_cw,A_wa,x_a);
    for (unsigned int i=0; i<PointDoF; ++i)
    {
      Matrix<double,PointDoF,1> eps
          = Matrix<double,PointDoF,1>::Zero();
      eps[i] = h;

      J_x.col(i) = (-map(T_cw,A_wa,add(x_a,eps)) -fun)/h ;

    }
    return J_x;
  }

  /** Add an incermental update delta to pose/frame T*/
  virtual Frame
  add                        (const Frame & T,
                              const Matrix<double,FrameDoF,1> & delta
                              ) const = 0;

  /** Add an incremental update delta to point x*/
  virtual Point
  add                        (const Point & x,
                              const Matrix<double,PointDoF,1> & delta
                              ) const = 0;
};



/** abstract prediction class dependig on
     * 3D rigid body transformations SE3 */
template <int PointParNum, int PointDoF, int ObsDim>
class SE3_AbstractPoint
    : public AbstractPrediction
    <SE3,6,Matrix<double, PointParNum,1>,PointDoF,ObsDim>
{
public:
  SE3 add(const SE3 &T, const Matrix<double,6,1> & delta) const
  {
    return SE3::exp(delta)*T;
  }
};

class SE3XYZ_STEREO: public SE3_AbstractPoint<3, 3, 3>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  SE3XYZ_STEREO              (const StereoCamera & cam)
    : _cam(cam)
  {
  }

  Matrix<double,3,6>
  frameJac(const SE3 & se3,
           const Vector3d & xyz)const
  {
    const Vector3d & xyz_trans = se3*xyz;
    double x = xyz_trans[0];
    double y = xyz_trans[1];
    double z = xyz_trans[2];
    double f = _cam.focal_length();

    double one_b_z = 1./z;
    double one_b_z_sq = 1./(z*z);
    double A = -f*one_b_z;
    double B = -f*one_b_z;
    double C = f*x*one_b_z_sq;
    double D = f*y*one_b_z_sq;
    double E = f*(x-_cam.baseline())*one_b_z_sq;

    Matrix<double, 3, 6> jac;
    jac <<  A, 0, C, y*C,     z*A-x*C, -y*A,
        0, B, D,-z*B+y*D, -x*D,     x*B,
        A, 0, E, y*E,     z*A-x*E, -y*A;
    return jac;
  }

  Vector3d map(const SE3 & T,
               const Vector3d& xyz) const
  {
    return _cam.map_uvu(T*xyz);
  }

  Vector3d add(const Vector3d & x,
               const Vector3d & delta) const
  {
    return x+delta;
  }


private:
  StereoCamera _cam;
};

#ifdef MONO
class Sim3XYZ : public AbstractPrediction<Sim3,6,Vector3d,3,2>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Sim3XYZ(const LinearCamera & cam)
  {
    this->cam = cam;
  }

  inline Vector2d map(const Sim3 & T,
                      const Vector3d& x) const
  {
    return cam.map(project2d(T*x));
  }


  Vector3d add(const Vector3d & x,
               const Vector3d & delta) const
  {
    return x+delta;
  }

  Sim3 add(const Sim3 &T, const Matrix<double,6,1> & delta) const
  {
    Matrix<double,7,1> delta7;
    delta7.head<6>() = delta;
    delta7[6] = 0;
    return Sim3::exp(delta7)*T;
  }

private:
  LinearCamera  cam;

};

class Sim3XYZ_STEREO : public AbstractPrediction<Sim3,7,Vector3d,3,3>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Sim3XYZ_STEREO(const StereoCamera & cam)
  {
    this->cam = cam;
  }

  inline Vector3d map(const Sim3 & T,
                      const Vector3d& x) const
  {
    return cam.map_uvu(T*x);
  }

  Vector3d add(const Vector3d & x,
               const Vector3d & delta) const
  {
    return x+delta;
  }

  Sim3 add(const Sim3 &T, const Matrix<double,7,1> & delta) const
  {
    return Sim3::exp(delta)*T;
  }

private:
  StereoCamera  cam;

};

class AbsoluteOrient : public AbstractPrediction<Sim3,7,Vector3d,3,3>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  AbsoluteOrient()
  {
  }

  inline Vector3d map(const Sim3 & T,
                      const Vector3d& x) const
  {
    return T*x;
  }

  Vector3d add(const Vector3d & x,
               const Vector3d & delta) const
  {
    return x+delta;
  }

  Sim3 add(const Sim3 &T, const Matrix<double,7,1> & delta) const
  {
    return Sim3::exp(delta)*T;
  }
};
#endif



/** 3D Euclidean point class */
class SE3XYZ: public SE3_AbstractPoint<3, 3, 2>{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  SE3XYZ(const LinearCamera & cam)
  {
    this->cam = cam;
  }

  inline Vector2d map(const SE3 & T,
                      const Vector3d& x) const
  {
    return cam.map(project2d(T*x));
  }

  Vector3d add(const Vector3d & x,
               const Vector3d & delta) const
  {
    return x+delta;
  }

private:
  LinearCamera  cam;

};

/** 3D inverse depth point class*/
class SE3UVQ : public SE3_AbstractPoint<3, 3, 2>{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  SE3UVQ ()
  {
  }

  SE3UVQ (const LinearCamera & cam_pars)
  {
    this->cam = cam_pars;
  }

  inline Vector2d map(const SE3 & T,
                      const Vector3d& uvq_w) const
  {
    Vector3d xyz_w = invert_depth(uvq_w);
    return cam.map(project2d(T*xyz_w));
  }

  Vector3d add(const Vector3d & x,
               const Vector3d & delta) const
  {
    return x+delta;
  }

private:
  LinearCamera  cam;
};

/** 3D inverse depth point class*/
class SE3AnchordUVQ : public AbstractAnchoredPrediction<SE3,6,Vector3d,3,2>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  SE3AnchordUVQ ()
  {
  }

  SE3AnchordUVQ (const LinearCamera & cam_pars)
  {
    this->cam = cam_pars;
  }

  inline Vector2d map(const SE3 & T_cw,
                      const SE3 & A_aw,
                      const Vector3d& uvq_a) const
  {
    Vector3d xyz_w = A_aw.inverse()*invert_depth(uvq_a);
    return cam.map(project2d(T_cw*xyz_w));
  }

  Vector3d add(const Vector3d & point,
               const Vector3d & delta) const
  {
    return point+delta;
  }

  Matrix<double,2,3>
  pointJac(const SE3 & T_cw,
           const SE3 & A_aw,
           const Vector3d & psi_a) const
  {
    SE3 T_ca = T_cw*A_aw.inverse();
    Vector3d y = T_ca*invert_depth(psi_a);
    Matrix<double,2,3> J1
        = d_proj_d_y(cam.focal_length(),y);

    Matrix3d J2 = d_Tinvpsi_d_psi(T_ca,  psi_a);
    return -J1*J2;

  }

  Matrix<double,2,6>
  frameJac(const SE3 & T_cw,
           const SE3 & A_aw,
           const Vector3d & psi_a) const
  {
      SE3 T_ca = T_cw*A_aw.inverse();
    Vector3d y = T_ca*invert_depth(psi_a);
    Matrix<double,2,3> J1 = d_proj_d_y(cam.focal_length(),y);
    Matrix<double,3,6> J2 = d_expy_d_y(y);
    return -J1*J2;
  }

  Matrix<double,2,6>
  anchorJac(const SE3 & T_cw,
            const SE3 & A_aw,
            const Vector3d & psi_a) const
  {
    SE3 T_ca = T_cw*A_aw.inverse();
    Vector3d x = invert_depth(psi_a);
    Vector3d y = T_ca*x;
     Matrix<double,2,3> J1
        = d_proj_d_y(cam.focal_length(),y);
    Matrix<double,3,6> d_invexpx_dx
        = -d_expy_d_y(x);
    return -J1*T_ca.rotation_matrix()*d_invexpx_dx;
  }

  SE3 add(const SE3 &T, const Matrix<double,6,1> & delta) const
  {
    return SE3::exp(delta)*T;
  }

private:
  LinearCamera  cam;
};


/** 3D inverse depth point class*/
class SE3NormUVQ : public AbstractPrediction<SE3,5,Vector3d,3,2>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  SE3NormUVQ ()
  {
  }

  SE3NormUVQ (const LinearCamera & cam_pars)
  {
    this->cam = cam_pars;
  }

  inline Vector2d map(const SE3 & T_cw,
                      const Vector3d& uvq_w) const
  {
    Vector3d xyz_w = invert_depth(uvq_w);
    return cam.map(project2d(T_cw*xyz_w));
  }

  Vector3d add(const Vector3d & point,
               const Vector3d & delta) const
  {
    return point+delta;
  }

  SE3 add(const SE3 &T, const Matrix<double,5,1> & delta) const
  {
    Vector6d delta6;
    delta6[0] = delta[0];
    delta6[1] = delta[1];
    delta6[2] = 0;
    delta6.tail<3>() = delta.tail<3>();

    SE3 new_T = SE3::exp(delta6)*T;
    double length = new_T.translation().norm();
    assert(fabs(length)>0.00001);

    new_T.translation() *= 1./length;
    assert(fabs(new_T.translation().norm()-1) < 0.00001);

    return new_T;
  }


private:
  LinearCamera  cam;
};

/** 3D inverse depth point class*/
class SE3AnchordUVQ_STEREO
    : public AbstractAnchoredPrediction<SE3,6,Vector3d,3,3>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  SE3AnchordUVQ_STEREO ()
  {
  }

  SE3AnchordUVQ_STEREO (const StereoCamera & cam_pars)
  {
    this->cam = cam_pars;
  }

  inline Vector3d map(const SE3 & T_cw,
                      const SE3 & A_aw,
                      const Vector3d& uvq_a) const
  {
    Vector3d xyz_w = A_aw.inverse()*invert_depth(uvq_a);
    return cam.map_uvu(T_cw*xyz_w);
  }

  Matrix3d
  pointJac(const SE3 & T_cw,
           const SE3 & A_aw,
           const Vector3d & psi_a) const
  {
    SE3 T_ca = T_cw*A_aw.inverse();
    Vector3d y = T_ca*invert_depth(psi_a);
    Matrix3d J1
        = d_stereoproj_d_y(cam.focal_length(),
                           cam.baseline(),
                           y);
    Matrix3d J2
        = d_Tinvpsi_d_psi(T_ca,
                          psi_a);
    return -J1*J2;
  }

  Matrix<double,3,6>
  frameJac(const SE3 & T_cw,
           const SE3 & A_aw,
           const Vector3d & psi_a) const
  {
    SE3 T_ca = T_cw*A_aw.inverse();
    Vector3d y = T_ca*invert_depth(psi_a);
    Matrix3d J1
        = d_stereoproj_d_y(cam.focal_length(),
                           cam.baseline(),
                           y);
    Matrix<double,3,6> J2
        = d_expy_d_y(y);
    return -J1*J2;
  }


  Matrix<double,3,6>
  anchorJac(const SE3 & T_cw,
            const SE3 & A_aw,
            const Vector3d & psi_a) const
  {
    SE3 T_ca = T_cw*A_aw.inverse();
    Vector3d x = invert_depth(psi_a);
    Vector3d y = T_ca*x;
    Matrix3d J1
        = d_stereoproj_d_y(cam.focal_length(),
                           cam.baseline(),
                           y);
    Matrix<double,3,6> d_invexpx_dx
        = -d_expy_d_y(x);
    return -J1*T_ca.rotation_matrix()*d_invexpx_dx;
  }

  Vector3d add(const Vector3d & point,
               const Vector3d & delta) const
  {
    return point+delta;
  }

  SE3 add(const SE3 &T, const Matrix<double,6,1> & delta) const
  {
    return SE3::exp(delta)*T;
  }

private:
  StereoCamera  cam;
};
/** 3D inverse depth point class*/
class SE3UVU_STEREO : public SE3_AbstractPoint<3, 3, 3>{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  SE3UVU_STEREO ()
  {
  }

  SE3UVU_STEREO (const StereoCamera & cam)
  {
    this->cam = cam;
  }

  inline Vector3d map(const SE3 & T,
                      const Vector3d& uvu) const
  {
    Vector3d x = cam.unmap_uvu(uvu);
    return cam.map_uvu(T*x);
  }


  Vector3d add(const Vector3d & x,
               const Vector3d & delta) const
  {
    return x+delta;
  }

private:
  StereoCamera  cam;
};

/** 3D inverse depth point class*/
class SE3UVQ_STEREO : public SE3_AbstractPoint<3, 3, 3>{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  SE3UVQ_STEREO ()
  {
  }

  SE3UVQ_STEREO (const StereoCamera & cam)
  {
    this->cam = cam;
  }

  inline Vector3d map(const SE3 & T,
                      const Vector3d& uvq) const
  {
    Vector3d x = invert_depth(uvq);
    return cam.map_uvu(T*x);
  }

  Vector3d add(const Vector3d & x,
               const Vector3d & delta) const
  {
    return x+delta;
  }

private:
  StereoCamera  cam;
};


/** observation class */
template <int ObsDim>
class IdObs
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  IdObs(){}
  IdObs(int point_id, int frame_id, const Matrix<double,ObsDim,1> & obs)
    : frame_id(frame_id), point_id(point_id), obs(obs)
  {
  }

  int frame_id;
  int point_id;
  Matrix<double,ObsDim,1> obs;
};


/** observation class with inverse uncertainty*/
template <int ObsDim>
class IdObsLambda : public IdObs<ObsDim>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  IdObsLambda(){}
  IdObsLambda(int point_id,
              int frame_id,
              const Matrix<double,ObsDim,1> & obs,
              const Matrix<double,ObsDim,ObsDim> & lambda)
    : IdObs<ObsDim>(point_id, frame_id,  obs) , lambda(lambda)
  {
  }
  Matrix<double,ObsDim,ObsDim> lambda;
};

}


#endif
