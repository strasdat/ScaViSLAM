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

#ifndef SCAVISLAM_RANSAC_MODELS_H
#define SCAVISLAM_RANSAC_MODELS_H

#include <sophus/se3.h>
#ifdef MONO
#include <sophus/sim3.h>
#endif

#include "global.h"
#include "stereo_camera.h"

// This RANSAC code is based on "http://www.ros.org/wiki/posest"
// written by Kurt Konolige and originally licensed under BSD.

namespace ScaViSLAM
{
using namespace Sophus;

namespace AbsoluteOrientation
{
double
belowThreshold               (const StereoCamera & cam,
                              const Vector3d & trans_train_pt,
                              const Vector3d & uvu,
                              double maxInlierXDist2);
Matrix3d
getOrientationAndCentriods   (Vector3d & p0a,
                              Vector3d & p0b,
                              Vector3d & p0c,
                              Vector3d & p1a,
                              Vector3d & p1b,
                              Vector3d & p1c,
                              Vector3d & c0,
                              Vector3d & c1);
}

struct SE3Model
{
  typedef SE3 Transformation;
  typedef StereoCamera Camera;

  static void
  calc_motion                (const StereoCamera & cam,
                              const ALIGNED<Vector3d>::vector & query_obs_vec,
                              const ALIGNED<Vector3d>::vector & train_pt_vec,
                              Transformation & se3);
  static bool
  belowThreshold             (const StereoCamera & cam,
                              const Vector3d & trans_train_pt,
                              const Vector3d & uvu,
                              double maxInlierXDist2);

  static const int obs_dim = 3;
  static const int point_dim = 3;
  static const uint num_points = 3;
};

#ifdef MONO
struct Sim3Model
{
  typedef Sim3 Transformation;
  typedef StereoCamera Camera;

  static void
  calc_motion                (const StereoCamera & cam,
                              const ALIGNED<Vector3d>::vector & query_obs_vec,
                              const ALIGNED<Vector3d>::vector & train_pt_vec,
                              Transformation & sim3);
  static bool
  belowThreshold             (const StereoCamera & cam,
                              const Vector3d & trans_train_pt,
                              const Vector3d & uvu,
                              double maxInlierXDist2);
  static const int obs_dim = 3;
  static const int point_dim = 3;
  static const uint num_points = 3;
};
#endif

}

#endif
