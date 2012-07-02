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

#include "frame_grabber.hpp"

#include <stdint.h>

#include <boost/filesystem.hpp>
#include <boost/static_assert.hpp>
#include <boost/regex.hpp>

#include <visiontools/performance_monitor.h>

namespace ScaViSLAM
{

template <class Camera>
void FrameGrabber<Camera>::
initialise()
{
  loadParams();

  frame_data.disp.create(frame_data.cam_vec[0].image_size(),
                         CV_32F);
#ifdef SCAVISLAM_CUDA_SUPPORT
  frame_data.gpu_disp_32f.create(frame_data.cam_vec[0].image_size(),
                                 CV_32F);
  for (int l=0; l<NUM_PYR_LEVELS; ++l)
  {
    frame_data.cur_left().gpu_pyr_float32[l]
        .create(frame_data.cam_vec[l].image_size(), CV_32FC1);
    frame_data.gpu_pyr_float32_dx[l]
        .create(frame_data.cam_vec[l].image_size(), CV_32FC1);
    frame_data.gpu_pyr_float32_dy[l]
        .create(frame_data.cam_vec[l].image_size(), CV_32FC1);
    frame_data.prev_left().gpu_pyr_float32[l]
        .create(frame_data.cam_vec[l].image_size(), CV_32FC1);
    frame_data.gpu_pyr_float32_dx[l]
        .create(frame_data.cam_vec[l].image_size(), CV_32FC1);
    frame_data.gpu_pyr_float32_dy[l]
        .create(frame_data.cam_vec[l].image_size(), CV_32FC1);
  }
  frame_data.right.gpu_pyr_float32[0]
      .create(frame_data.cam_vec[0].image_size(),  CV_32FC1);
#else
  frame_data.pyr_float32.resize(NUM_PYR_LEVELS);
  frame_data.pyr_float32_dx.resize(NUM_PYR_LEVELS);
  frame_data.pyr_float32_dy.resize(NUM_PYR_LEVELS);
  for (int l=0; l<NUM_PYR_LEVELS; ++l)
  {
    frame_data.pyr_float32[l]
        .create(frame_data.cam_vec[l].image_size(), CV_32FC1);
    frame_data.pyr_float32_dx[l]
        .create(frame_data.cam_vec[l].image_size(), CV_32FC1);
    frame_data.pyr_float32_dy[l]
        .create(frame_data.cam_vec[l].image_size(), CV_32FC1);
  }
#endif

  if (params_.livestream)
  {
#ifdef SCAVISLAM_PCL_SUPPORT
    grabber.initialize();
    boost::thread(boost::ref(grabber));
#else
    assert(false);
#endif
  }
  else
  {
    //preprocessFiles(params_.path_str,true);
    file_grabber_.initialize(params_.path_str,
                             params_.base_str,
                             params_.format_str,
                             frame_data.frame_id,
                             params_.color_img,
                             !params_.color_img,
                             params_.right_img,
                             params_.disp_img,
                             params_.right_img,
                             &file_grabber_mon_);


    if (params_.rectify_frame)
    {
      intializeRectifier();
    }
  }
#ifdef SCAVISLAM_CUDA_SUPPORT
  dx_filter_ =
      cv::gpu::createDerivFilter_GPU(frame_data.cur_left()
                                     .gpu_pyr_float32[0].type(),
                                     frame_data.gpu_pyr_float32_dx[0]
                                     .type(),
                                     1, 0, 1, cv::BORDER_REPLICATE);
  dy_filter_ =
      cv::gpu::createDerivFilter_GPU(frame_data.cur_left()
                                     .gpu_pyr_float32[0].type(),
                                     frame_data.gpu_pyr_float32_dy[0]
                                     .type(),
                                     0, 1, 1, cv::BORDER_REPLICATE);
#endif



  processNextFrame();
}

template <class Camera>
void FrameGrabber<Camera>::
processNextFrame()
{
  frame_data.nextFrame();

  per_mon_->start("grab frame");
  if (!params_.livestream)
  {
    FrameBundle bundle;
    while (file_grabber_mon_.getFrameBundle(frame_data.frame_id,
                                            &bundle)
           ==false)
    {
    }

   
    if (params_.color_img)
    {

      frame_data.cur_left().color_uint8 = bundle.left_color;
      cv::cvtColor(frame_data.cur_left().color_uint8,
                   frame_data.cur_left().uint8,
                   CV_BGR2GRAY);
    }
    else
    {
      frame_data.cur_left().uint8 = bundle.left_gray;

    }
    if (params_.disp_img)
    {

      cv::Mat float_as_4uint = bundle.disp;
      frame_data.disp = cv::Mat(frame_data.cur_left().uint8.size(),
                                CV_32F, float_as_4uint.data).clone();
      cerr << frame_data.disp.size().width << " "
           << frame_data.disp.size().height << endl;
      frame_data.have_disp_img = true;
    }
    else if (params_.depth_img)
    {

      //frame_data.right.uint8 = cv::imread(right_sstr.str(),0);
      cv::Mat depth = bundle.depth;
      depthToDisp(depth, &frame_data.disp);
      frame_data.have_disp_img = true;
    }
    else if (params_.right_img)
    {

      frame_data.right.uint8 = bundle.right;
    }
    if (params_.rectify_frame)
      rectifyFrame();
  }
  else
  {
    frameFromLiveCamera();
  }
  per_mon_->stop("grab frame");
  preprocessing();
  ++frame_data.frame_id;
}

template <class Camera>
void FrameGrabber<Camera>::
depthToDisp(const cv::Mat & depth_img,
            cv::Mat * disp_img) const
{
  assert(false);
}

template <class Camera>
void FrameGrabber<Camera>::
loadParams()
{
  params_.rot_left.x() = pangolin::Var<double>("cam.rotx_left",0.0);
  params_.rot_left.y() = pangolin::Var<double>("cam.roty_left",0.0);
  params_.rot_left.z() = pangolin::Var<double>("cam.rotz_left",0.0);

  params_.rot_right.x() = pangolin::Var<double>("cam.rotx_right",0.0);
  params_.rot_right.y() = pangolin::Var<double>("cam.roty_right",0.0);
  params_.rot_right.z() = pangolin::Var<double>("cam.rotz_right",0.0);

  params_.dist_coeff_left[0] = pangolin::Var<double>("cam.dist_left1",0.0);
  params_.dist_coeff_left[1] = pangolin::Var<double>("cam.dist_left2",0.0);
  params_.dist_coeff_left[2] = pangolin::Var<double>("cam.dist_left3",0.0);
  params_.dist_coeff_left[3] = pangolin::Var<double>("cam.dist_left4",0.0);
  params_.dist_coeff_left[4] = pangolin::Var<double>("cam.dist_left5",0.0);

  params_.dist_coeff_right[0] = pangolin::Var<double>("cam.dist_right1",0.0);
  params_.dist_coeff_right[1] = pangolin::Var<double>("cam.dist_right2",0.0);
  params_.dist_coeff_right[2] = pangolin::Var<double>("cam.dist_right3",0.0);
  params_.dist_coeff_right[3] = pangolin::Var<double>("cam.dist_right4",0.0);
  params_.dist_coeff_right[4] = pangolin::Var<double>("cam.dist_right5",0.0);

  params_.livestream = pangolin::Var<bool>
      ("framepipe.livestream",false);
  params_.base_str = pangolin::Var<string>
      ("framepipe.base_str",string(".*rectified.*"));
  params_.path_str = pangolin::Var<string>
      ("framepipe.path_str",string("/home/strasdat/NC2/"));

  params_.format_str = pangolin::Var<string>
      ("framepipe.format_str", std::string("pnm"));
  params_.skip_imgs = pangolin::Var<size_t>
      ("framepipe.skip_imgs",0);
  params_.color_img = pangolin::Var<bool>
      ("framepipe.color_img",false);
  params_.right_img = pangolin::Var<bool>
      ("framepipe.right_img",false);
  params_.disp_img = pangolin::Var<bool>
      ("framepipe.disp_img",false);
  params_.depth_img = pangolin::Var<bool>
      ("framepipe.depth_img",false);
  params_.rectify_frame = pangolin::Var<bool>
      ("framepipe.rectify_frame",false);

  frame_data.frame_id = params_.skip_imgs;
}

template <class Camera>
void FrameGrabber<Camera>::
rectifyFrame()
{
  cv::Mat tmp = frame_data.cur_left().uint8.clone();
  cv::remap(tmp, frame_data.cur_left().uint8,
            rect_map_left_[0], rect_map_left_[1], CV_INTER_LINEAR);

  tmp = frame_data.right.uint8.clone();
  cv::remap(tmp, frame_data.right.uint8,
            rect_map_right_[0], rect_map_right_[1], CV_INTER_LINEAR);
}

template <class Camera>
void FrameGrabber<Camera>::
intializeRectifier()

{
  //No rectification needed for monocular camera!!
  assert(false);
}

template <class Camera>
void FrameGrabber<Camera>::
frameFromLiveCamera()
{
#ifdef SCAVISLAM_PCL_SUPPORT
  while (grabber.getFrame(&(frame_data.cur_left().color_uint8),
                          &(frame_data.disp)) == false )
  {
  }
  cv::cvtColor(frame_data.cur_left().color_uint8,
               frame_data.cur_left().uint8,
               CV_BGR2GRAY);
  frame_data.have_disp_img = true;
#else
  assert(false);
#endif
}

template <class Camera>
void FrameGrabber<Camera>::
preprocessing()
{
  per_mon_->start("preprocess");
  cv::buildPyramid(frame_data.cur_left().uint8,
                   frame_data.cur_left().pyr_uint8,
                   NUM_PYR_LEVELS-1);
#ifdef SCAVISLAM_CUDA_SUPPORT
  frame_data.cur_left().gpu_uint8.upload(frame_data.cur_left().uint8);
  if (params_.right_img)
    frame_data.right.gpu_uint8.upload(frame_data.right.uint8);
  frame_data.cur_left().gpu_uint8
      .convertTo(frame_data.cur_left().gpu_pyr_float32[0], CV_32F,1./255.);

  dx_filter_->apply(frame_data.cur_left().gpu_pyr_float32[0],
                    frame_data.gpu_pyr_float32_dx[0]);
  dy_filter_->apply(frame_data.cur_left().gpu_pyr_float32[0],
                    frame_data.gpu_pyr_float32_dy[0]);

  for (int l=1; l<NUM_PYR_LEVELS; ++l)
  {
    cv::gpu::pyrDown(frame_data.cur_left().gpu_pyr_float32[l-1],
                     frame_data.cur_left().gpu_pyr_float32[l]);
    dx_filter_->apply(frame_data.cur_left().gpu_pyr_float32[l],
                      frame_data.gpu_pyr_float32_dx[l]);
    dy_filter_->apply(frame_data.cur_left().gpu_pyr_float32[l],
                      frame_data.gpu_pyr_float32_dy[l]);
  }
#else
  frame_data.cur_left().uint8
      .convertTo(frame_data.pyr_float32.at(0), CV_32F,1./255.);
  cv::Sobel(frame_data.pyr_float32.at(0), frame_data.pyr_float32_dx.at(0),
            frame_data.pyr_float32.at(0).depth(),
            1, 0, 1);
  cv::Sobel(frame_data.pyr_float32.at(0), frame_data.pyr_float32_dy.at(0),
            frame_data.pyr_float32.at(0).depth(),
            0, 1, 1);
  for (int l=1; l<NUM_PYR_LEVELS; ++l)
  {
    frame_data.cur_left().pyr_uint8.at(l)
        .convertTo(frame_data.pyr_float32.at(l), CV_32F,1./255.);
    cv::Sobel(frame_data.pyr_float32[l], frame_data.pyr_float32_dx[l],
              frame_data.pyr_float32.at(0).depth(),
              1, 0, 1);
    cv::Sobel(frame_data.pyr_float32[l], frame_data.pyr_float32_dy[l],
              frame_data.pyr_float32.at(0).depth(),
              0, 1, 1);
  }
#endif
  per_mon_->stop("preprocess");
}

}
