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

#include "stereo_camera.h"

#include "maths_utils.h"

namespace ScaViSLAM
{
Matrix4d StereoCamera
::Q() const
{
  Matrix4d res;
  res << 1.0, 0.0, 0.0, -principle_point_[0]
      ,  0.0, 1.0, 0.0, -principle_point_[1]
      ,  0.0, 0.0, 0.0, focal_length_
      ,  0.0, 0.0, 1.0/baseline_, 0.0;

  return res;
}

Vector3d StereoCamera
::map_uvu(const Vector3d & xyz) const
{
  Vector2d uv = LinearCamera::map(project2d(xyz));
  double proj_x_right = (xyz[0]-baseline_)/xyz[2];
  double u_right = proj_x_right*focal_length_ + principle_point_[0];
  return Vector3d(uv[0],uv[1],u_right);

}

Vector3d StereoCamera
::unmap_uvu( const Vector3d& uvu) const
{
  double scaled_disparity = (uvu[0]-uvu[2])/baseline_;
  double z = focal_length_/scaled_disparity;
  return unproject2d(LinearCamera::unmap(uvu.head(2)))*z;
}

float StereoCamera
::depthToDisp(float depth) const
{
  float scaled_disparity = focal_length_/depth;
  return scaled_disparity/baseline_;
}

}
