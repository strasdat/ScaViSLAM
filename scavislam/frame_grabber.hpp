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

#ifndef SCAVISLAM_FRAME_PIPELINE_H
#define SCAVISLAM_FRAME_PIPELINE_H

#include <boost/filesystem.hpp>
#include <boost/thread.hpp>

#include <opencv2/opencv.hpp>
#include <pangolin/video.h>

#ifdef SCAVISLAM_CUDA_SUPPORT
#include <opencv2/gpu/gpu.hpp>
#endif

#include "filegrabber.h"

#ifdef SCAVISLAM_PCL_SUPPORT
#include "rgbd_grabber.h"
#endif


#include "global.h"

namespace VisionTools
{
class PerformanceMonitor;
}

namespace ScaViSLAM
{

using namespace std;
using namespace Eigen;
using namespace VisionTools;

struct ImageSet
{
public:
  ImageSet(){}
  ImageSet(const cv::Mat & img)
    : uint8(img),
      pyr_uint8(NUM_PYR_LEVELS)
  #ifdef SCAVISLAM_CUDA_SUPPORT
    , gpu_pyr_float32(NUM_PYR_LEVELS)
  #endif
  {
  }

  void
  clone(ImageSet & new_set)
  {
    new_set.uint8 = uint8.clone();
    for (int l = 0; l<NUM_PYR_LEVELS; ++l)
    {
      new_set.pyr_uint8[l] = pyr_uint8[l].clone();
    }

#ifdef SCAVISLAM_CUDA_SUPPORT
    new_set.gpu_uint8 = gpu_uint8.clone();
    for (int l = 0; l<NUM_PYR_LEVELS; ++l)
    {
      new_set.gpu_pyr_float32[l] = gpu_pyr_float32[l].clone();
    }
#endif
  }

  cv::Mat color_uint8;
  cv::Mat uint8;
  vector<cv::Mat> pyr_uint8;
#ifdef SCAVISLAM_CUDA_SUPPORT
  cv::gpu::GpuMat gpu_uint8;
  vector<cv::gpu::GpuMat> gpu_pyr_float32;
#endif
};


template <class Camera>
class FrameData
{
public:
  FrameData()
    :
    #ifdef SCAVISLAM_CUDA_SUPPORT
      gpu_pyr_float32_dx(NUM_PYR_LEVELS),
      gpu_pyr_float32_dy(NUM_PYR_LEVELS),
    #endif
      have_disp_img(false),
      offset(0)
  {}

  ImageSet& cur_left()
  {
    return left[offset];
  }

  ImageSet& prev_left()
  {
    return left[(offset+1)%2];
  }

  const ImageSet& cur_left() const
  {
    return left[offset];
  }

  const ImageSet& prev_left() const
  {
    return left[(offset+1)%2];
  }

  void nextFrame()
  {
    offset = (offset+1)%2;
  }

  Camera cam;
  typename ALIGNED<Camera>::vector cam_vec;
  ImageSet left[2];
  ImageSet right;

  cv::Mat disp;
  cv::Mat color_disp;
#ifdef SCAVISLAM_CUDA_SUPPORT
  cv::gpu::GpuMat gpu_disp_32f;
  cv::gpu::GpuMat gpu_xyzw;
  cv::gpu::GpuMat gpu_disp_16s;
  cv::gpu::GpuMat gpu_color_disp;
  vector<cv::gpu::GpuMat> gpu_pyr_float32_dx;
  vector<cv::gpu::GpuMat> gpu_pyr_float32_dy;
#else
  vector<cv::Mat> pyr_float32;
  vector<cv::Mat> pyr_float32_dx;
  vector<cv::Mat> pyr_float32_dy;
#endif
  int frame_id;
  bool have_disp_img;
private:
  int offset;
};


template<class Camera>
class FrameGrabber
{
public:
  struct Params
  {
    Vector3d rot_left;
    Vector3d rot_right;
    Vector5d dist_coeff_left;
    Vector5d dist_coeff_right;
    bool  livestream;
    std::string  base_str;
    std::string  path_str;
    std::string  format_str;
    int  skip_imgs;
    bool color_img;
    bool right_img;
    bool disp_img;
    bool depth_img;
    bool rectify_frame;
  };

#ifdef SCAVISLAM_PCL_SUPPORT
  RgbdGrabber grabber;
#endif
  FileGrabber file_grabber_;
  FileGrabberMonitor file_grabber_mon_;

  FrameGrabber               (const Camera & cam,
                              const Vector4d & cam_distortion_,
                              PerformanceMonitor * per_mon_);
  void
  initialise                 ();
  void
  processNextFrame           ();

  const Params& params() const
  {
    return params_;
  }

  FrameData<Camera> frame_data;

private:
  void
  loadParams                 ();

  void
  rectifyFrame               ();

  void
  intializeRectifier         ();

  void
  frameFromLiveCamera        ();

  void
  preprocessing              ();

  void
  depthToDisp                (const cv::Mat & depth_img,
                              cv::Mat * disp_img) const;

  PerformanceMonitor * per_mon_;
  cv::Mat rect_map_left_[2];
  cv::Mat rect_map_right_[2];
  Vector4d cam_distortion_;

  Params params_;
  double size_factor_;
  //vector<std::string> file_base_vec_;
  std::string file_extension_;
#ifdef SCAVISLAM_CUDA_SUPPORT
  cv::Ptr<cv::gpu::FilterEngine_GPU> dx_filter_;
  cv::Ptr<cv::gpu::FilterEngine_GPU> dy_filter_;
#endif

private:
  DISALLOW_COPY_AND_ASSIGN(FrameGrabber)
};





}

#endif
