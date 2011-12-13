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

#ifndef SCAVISLAM_STEREO_CAMERA_H
#define SCAVISLAM_STEREO_CAMERA_H

#include <visiontools/linear_camera.h>

#include "maths_utils.h"


namespace ScaViSLAM
{

class StereoCamera : public VisionTools::LinearCamera
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  static const int obs_dim = 3;

  StereoCamera()
    : LinearCamera()
  {
  }

  StereoCamera(const StereoCamera& cam)
    : LinearCamera(cam), baseline_(cam.baseline_)
  {
  }

  StereoCamera(const AbstractCamera& cam,
               double baseline)
    : LinearCamera(cam), baseline_(baseline)
  {
  }

  StereoCamera(const Matrix3d K,
               const cv::Size & size,
               double baseline)
    : LinearCamera(K,size), baseline_(baseline)
  {
  }

  StereoCamera(const double & focal_length,
               const Vector2d & principle_point,
               const cv::Size & size,
               double baseline)
    : LinearCamera(focal_length,
                   principle_point,
                   size),
      baseline_(baseline)
  {
  }

  Matrix4d
  Q                          () const;

  Vector3d
  map_uvu                    (const Vector3d & xyz) const;

  Vector3d
  unmap_uvu                  (const Vector3d& uvu) const;

  float
  depthToDisp                (float depth) const;

  const double& baseline() const
  {
    return baseline_;
  }

private:
  double baseline_;
};
}

#endif

