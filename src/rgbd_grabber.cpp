#include "rgbd_grabber.h"

#include <boost/thread.hpp>

using namespace std;

namespace ScaViSLAM
{
 boost::mutex global_mutex;
 bool new_frame;

 cv::Mat global_rgb_img(cv::Size(640, 480), CV_8UC3);
 cv::Mat global_disp_img(cv::Size(640, 480), CV_32F);

void
cloud_cb_(const boost::shared_ptr<openni_wrapper::Image>& rgb,
            const boost::shared_ptr<openni_wrapper::DepthImage>& depth,
          float)
{
  boost::mutex::scoped_lock lock(global_mutex);

  rgb->fillRGB(640, 480,
               reinterpret_cast<unsigned char *>(global_rgb_img.data));
  depth->fillDisparityImage(640, 480,
                            reinterpret_cast<float *>(global_disp_img.data));
  new_frame = true;
}

bool RgbdGrabber
::getFrame(cv::Mat * rgb, cv::Mat * disp)
{
  boost::mutex::scoped_lock lock(global_mutex);
  if (new_frame)
  {
    *rgb = global_rgb_img;
    *disp = global_disp_img;
    new_frame = false;
    return true;
  }
  return false;
}

void RgbdGrabber
::initialize()
{
  interface_ = new pcl::OpenNIGrabber();

  boost::function<void (const boost::shared_ptr<openni_wrapper::Image>&,
                        const boost::shared_ptr<openni_wrapper::DepthImage>&,
                        float)>
      f =  boost::bind (cloud_cb_, _1, _2, _3);

  interface_->registerCallback (f);
}

void RgbdGrabber
::operator()()
{
  interface_->start ();

  while (true)
  {
    sleep (1);
  }

  interface_->stop ();
}

}







