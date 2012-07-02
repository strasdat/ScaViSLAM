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

#include <sophus/so3.h>

#include <visiontools/accessor_macros.h>

#include "frame_grabber.cpp"
#include "stereo_camera.h"

namespace ScaViSLAM
{
using namespace Sophus;
using namespace VisionTools;

template <>
FrameGrabber<StereoCamera>::
FrameGrabber(const StereoCamera & cam,
             const Vector4d & cam_dist,
             PerformanceMonitor * per_mon)
  :
    per_mon_(per_mon),
    cam_distortion_(cam_dist),
    size_factor_(1./cam.image_size().area())
{
  frame_data.cur_left()
      = ImageSet(cv::Mat(cam.image_size(),CV_8UC1));
  frame_data.prev_left()
      = ImageSet(cv::Mat(cam.image_size(),CV_8UC1));
  frame_data.right =
      ImageSet(cv::Mat(cam.image_size(),CV_8UC1));
  frame_data.cam
      = cam;
  frame_data.cam_vec
      = ALIGNED<StereoCamera>::vector(NUM_PYR_LEVELS);
  for (int level = 0; level<NUM_PYR_LEVELS; ++level)
  {

    frame_data.cam_vec.at(level)
        = StereoCamera(pyrFromZero_d(cam.focal_length(),level),
                       pyrFromZero_2d(cam.principal_point(),level),
                       cv::Size(pyrFromZero_d(cam.image_size().width,level),
                                pyrFromZero_d(cam.image_size().height,level)),
                       cam.baseline()*(1<<level));
  }
}

template <>
FrameGrabber<LinearCamera>::
FrameGrabber(const LinearCamera & cam,
             const Vector4d & cam_dist,
             PerformanceMonitor * per_mon)
  :
    per_mon_(per_mon),
    cam_distortion_(cam_dist),
    size_factor_(1./cam.image_size().area())
{
  frame_data.cur_left()
      = ImageSet(cv::Mat(cam.image_size(),CV_8UC1));
  frame_data.prev_left()
      = ImageSet(cv::Mat(cam.image_size(),CV_8UC1));
  frame_data.right =
      ImageSet(cv::Mat(cam.image_size(),CV_8UC1));
  frame_data.cam
      = cam;
  frame_data.cam_vec
      = ALIGNED<LinearCamera>::vector(NUM_PYR_LEVELS);
  for (int level = 0; level<NUM_PYR_LEVELS; ++level)
  {

    frame_data.cam_vec.at(level)
        = LinearCamera(pyrFromZero_d(cam.focal_length(),level),
                       pyrFromZero_2d(cam.principal_point(),level),
                       cv::Size(pyrFromZero_d(cam.image_size().width,level),
                                pyrFromZero_d(cam.image_size().height,level)));
  }
}

template <>
void FrameGrabber<StereoCamera>::
intializeRectifier()
{
  cv::Size image_size = frame_data.cam_vec.at(0).image_size();
  Matrix<double,3,3,RowMajor> Rleft =  SO3::exp(params_.rot_left).matrix();
  Matrix<double,3,3,RowMajor> Rright = SO3::exp(params_.rot_right).matrix();



  Matrix<double,3,3,RowMajor> camera_matrix
      = frame_data.cam_vec.at(0).intrinsics();

 // cerr << camera_matrix << endl;

  Matrix<double,3,4,RowMajor> Pleft;
  Pleft.setZero();
  Pleft.topLeftCorner<3,3>() = camera_matrix;

  Matrix<double,3,4,RowMajor> Pright;
  Pright.setZero();
  Pright.topLeftCorner<3,3>() = camera_matrix;
  Pright(0,3) =  frame_data.cam.focal_length()*frame_data.cam.baseline();

  cv::Mat cv_camera_matrix(3,3,CV_64F, camera_matrix.data());

  cv::Mat cvRleft(3,3,CV_64F, Rleft.data());
  cv::Mat cvRright(3,3,CV_64F, Rright.data());

  cv::Mat cvPleft(3,4,CV_64F, Pleft.data());
  cv::Mat cvPright(3,4,CV_64F, Pright.data());

  cv::Mat cvdist_left(5,1,CV_64F, params_.dist_coeff_left.data());
  cv::Mat cvdist_right(5,1,CV_64F, params_.dist_coeff_right.data());

  cv::initUndistortRectifyMap(cv_camera_matrix, cvdist_left,
                              cvRleft, cvPleft, image_size, CV_16SC2,
                              rect_map_left_[0], rect_map_left_[1]);
  cv::initUndistortRectifyMap(cv_camera_matrix, cvdist_right,
                              cvRright, cvPright, image_size, CV_16SC2,
                              rect_map_right_[0], rect_map_right_[1]);
}

template <>
void FrameGrabber<StereoCamera>::
depthToDisp(const cv::Mat & depth_img,
            cv::Mat * disp_img) const
{
  assert(depth_img.type()==CV_16U);
  *disp_img = cv::Mat(depth_img.size(), CV_32F);
  static const float FACTOR = 1./5000.;
  for (int v=0; v<depth_img.size().height; ++v)
    for (int u=0; u<depth_img.size().width; ++u)
    {
      float depth
          = depth_img.at<uint16_t>(v,u)*FACTOR;
      disp_img->at<float>(v,u) = frame_data.cam.depthToDisp(depth);
    }

}

template class FrameGrabber<StereoCamera>;
template class FrameGrabber<LinearCamera>;
}




