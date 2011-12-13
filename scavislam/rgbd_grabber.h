#ifndef RGBD_GRABBER_H
#define RGBD_GRABBER_H

#include <iostream>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/openni_grabber.h>

#include "global.h"

namespace ScaViSLAM
{

class RgbdGrabber
{
public:

  void
  initialize                 ();

  bool
  getFrame                   (cv::Mat * rgb,
                              cv::Mat * depth);

  void
  operator                   ()();

private:

  pcl::Grabber* interface_;

};

}

#endif // RGBD_GRABBER_H
