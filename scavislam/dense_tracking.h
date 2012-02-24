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

#ifndef SCAVISLAM_DENSE_TRACKER_H
#define SCAVISLAM_DENSE_TRACKER_H

#include <sophus/se3.h>

#include "global.h"
#include "quadtree.h"
#include "keyframes.h"
#include "frame_grabber.hpp"
#include "stereo_camera.h"

#ifdef SCAVISLAM_CUDA_SUPPORT
#include <opencv2/gpu/gpu.hpp>
#include "gpu/dense_tracking.cuh"
#endif

// This class implement dense tracking as described in
//
// Richard A. Newcombe, Steven J. Lovegrove and Andrew J. Davison
// "DTAM: Dense Tracking and Mapping in Real-Time"
// Sec. 4.3, ICCV 2011.
//
// 
// This method is an application of the famous Lukas-Kanda tracker:
// S. Baker, I. Matthews: Lucas-Kanade 20 Years On: A Unifying Framework
//
//
// Probably one of the first papers introducing this kind of direct 
// method for dense tracking/3D visual odometry is:
//
// Andrew I. Comport, Enzio Malis, P. Rives.
// "Accurate Quadri-focal Tracking for Robust 3D Visual Odometry"
// ICRA'07.

namespace ScaViSLAM
{
using namespace Sophus;

class DenseTracker
{
public:
  DenseTracker(const FrameData<StereoCamera> & frame_data);

#ifdef SCAVISLAM_CUDA_SUPPORT
  void
  computeDensePointCloudGpu  (const SE3 & T_cur_from_actkey);

  void
  denseTrackingGpu           (SE3 * T_cur_from_actkey);
#else
  void
  computeDensePointCloudCpu  (const SE3 & T_cur_from_actkey);

  void
  denseTrackingCpu           (SE3 * T_cur_from_actkey);
#endif

#ifdef SCAVISLAM_CUDA_SUPPORT
  cv::gpu::GpuMat dev_residual_img[NUM_PYR_LEVELS];
#else
  cv::Mat residual_img[NUM_PYR_LEVELS];
#endif

private:
  static const int EVERY_NTH_PIXEL = 4;
  //todo: different factor for different pyramid levels!

  const FrameData<StereoCamera> & frame_data_;

#ifdef SCAVISLAM_CUDA_SUPPORT
  GpuTracker gpu_tracker_;
  cv::gpu::GpuMat dev_ref_dense_points_[NUM_PYR_LEVELS];
#else
  cv::Mat ref_dense_points_[NUM_PYR_LEVELS];
#endif

  DISALLOW_COPY_AND_ASSIGN(DenseTracker)
};

}

#endif
