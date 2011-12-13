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

#include <pangolin/pangolin.h>

#include <visiontools/accessor_macros.h>
#include <visiontools/draw2d.h>
#include <visiontools/draw3d.h>
#include <visiontools/performance_monitor.h>

#include <queue>

#include "global.h"

#include "stereo_camera.h"
#include "frame_grabber.hpp"

using namespace ScaViSLAM;

struct Views
{
  pangolin::OpenGlRenderState pangolin_cam;
  pangolin::View * rgbd;
  pangolin::View * disp;
  pangolin::DataLog logger;
};

struct Modules
{
  Modules()
    : frame_grabber(NULL),
      per_mon(NULL) {}

  ~Modules()
  {
    if(frame_grabber!=NULL)
      delete frame_grabber;
    if(per_mon!=NULL)
      delete per_mon;
  }

  FrameGrabber<StereoCamera> * frame_grabber;
  PerformanceMonitor * per_mon;
};

struct ImagePair
{
  ImagePair() {}
  ImagePair(const cv::Mat & rgb,
            const cv::Mat & disp) : rgb(rgb), disp(disp){}
  cv::Mat rgb;
  cv::Mat disp;
};

class FileWriter
{
public:
  class FileWriterMonitor
  {
  public:
    void
    pushImages               (const ImagePair & images)
    {
      boost::mutex::scoped_lock lock(my_mutex_);
      image_queue_.push(images);
    }

    bool
    getImages                (ImagePair * images)
    {
      boost::mutex::scoped_lock lock(my_mutex_);
      if (image_queue_.size()==0)
        return false;

      cerr << image_queue_.size() << endl;

      *images = image_queue_.front();
      image_queue_.pop();

      return true;
    }

  private:
    queue<ImagePair> image_queue_;
    boost::mutex my_mutex_;
  };

  void
  operator()()
  {
    ImagePair images;
    int counter = 0;
    char str_rgb[80];
    char str_disp[80];
    while(true)
    {
      if (monitor.getImages(&images))
      {
        sprintf(str_rgb, "../data/out/img_%06d_left.png", counter);
        sprintf(str_disp, "../data/out/img_%06d_disp.png", counter);
        ++counter;
        cv::imwrite(string(str_rgb),
                    images.rgb);
        cv::imwrite(string(str_disp),
                    images.disp);
      }
      boost::this_thread::sleep(boost::posix_time::milliseconds(1));
    }
  }

  FileWriterMonitor monitor;
};


Modules
startModules(const StereoCamera & stereo_camera,
             Views * views)
{
  Modules modules;
  modules.per_mon = new PerformanceMonitor;
  modules.frame_grabber
      = new FrameGrabber<StereoCamera>(stereo_camera,
                                       Vector4d(0,0,0,0),
                                       modules.per_mon);
  modules.per_mon->add("drawing");
  modules.per_mon->add("grab frame");
  modules.per_mon->add("preprocess");
  modules.per_mon->setup(&views->logger);

  modules.frame_grabber->initialise();
  modules.frame_grabber->processNextFrame();
  return modules;
}


Views
initializeViews(const StereoCamera & stereo_camera)
{
  pangolin::CreateGlutWindowAndBind("Main",
                                    stereo_camera.width()*2,
                                    stereo_camera.height());
  Views views;
  views.rgbd
      = &(pangolin::Display("rgbd")
          .SetBounds(1.0,0.0,0.0,0.5,4./3.));
  views.disp
      = &(pangolin::Display("depth")
          .SetBounds(1.0,0.0,0.5,1.0,4./3.));

  views.pangolin_cam.Set(stereo_camera.getOpenGlMatrixSpec());
  views.pangolin_cam.Set(pangolin::IdentityMatrix(pangolin::GlModelViewStack));
  return views;
}

//TODO: main method to long...
int main(int argc, const char* argv[])
{
  if (argc<2)
  {
    cout << "please specify configuration file!" << endl;
    exit(0);
  }

  pangolin::ParseVarsFile(argv[1]);
  pangolin::Var<int> cam_width("cam.width",640);
  pangolin::Var<int> cam_height("cam.height",480);
  pangolin::Var<double> cam_f("cam.f",570.342);
  pangolin::Var<double> cam_px("cam.px",320);
  pangolin::Var<double> cam_py("cam.py",240);
  pangolin::Var<double> cam_baseline("cam.baseline",0.075);

  StereoCamera stereo_camera((double)cam_f,
                             Vector2d((double)cam_px,(double)cam_py),
                             cv::Size((int)cam_width,(int)cam_height),
                             cam_baseline);

  Views views = initializeViews(stereo_camera);

  Modules modules = startModules(stereo_camera,
                                 &views);
  pangolin::Var<float> ui_fps("ui.fps",false);

  FileWriter writer;
  boost::thread writer_thread(boost::ref(writer));

  while(true)
  {
    modules.per_mon->new_frame();
    ui_fps = modules.per_mon->fps();
    modules.frame_grabber->processNextFrame();
    cerr << ui_fps << endl;


    cv::Mat float_as_4uint
        (modules.frame_grabber->frame_data.disp.size().height,
         modules.frame_grabber->frame_data.disp.size().width*4,
         CV_8U,
         modules.frame_grabber->frame_data.disp.data);


    writer.monitor.pushImages(
          ImagePair(modules.frame_grabber->frame_data.cur_left().color_uint8,
                    float_as_4uint));

        modules.per_mon->start("drawing");
    views.rgbd->ActivateScissorAndClear();
    Draw2d::activate(modules.frame_grabber
                     ->frame_data.cur_left().uint8.size());
    Draw2d::texture(modules.frame_grabber
                    ->frame_data.cur_left().uint8);
    vector<cv::Mat> hsv_array(3);
    hsv_array[0] = cv::Mat(modules.frame_grabber
                           ->frame_data.disp.size(), CV_8UC1);

    modules.frame_grabber->frame_data.disp
        .convertTo(hsv_array[0],CV_8UC1, 5.,0.);
    hsv_array[1] = cv::Mat(modules.frame_grabber
                           ->frame_data.disp.size(), CV_8UC1, 255);
    hsv_array[2] = cv::Mat(modules.frame_grabber
                           ->frame_data.disp.size(), CV_8UC1, 255);

    cv::Mat hsv(modules.frame_grabber
                ->frame_data.disp.size(), CV_8UC3);
    cv::merge(hsv_array, hsv);
    cv::cvtColor(hsv, modules.frame_grabber
                 ->frame_data.color_disp, CV_HSV2BGR);
    views.disp->ActivateScissorAndClear();
    Draw2d::activate(modules.frame_grabber
                     ->frame_data.cur_left().uint8.size());
    Draw2d::texture(modules.frame_grabber
                    ->frame_data.color_disp);

    modules.per_mon->stop("drawing");
    glutSwapBuffers();
    glutMainLoopEvent();

    if (pangolin::ShouldQuit())
    {
       boost::this_thread::sleep(boost::posix_time::milliseconds(10000));
      exit(0);
    }
    if(pangolin::HasResized())
      pangolin::DisplayBase().ActivateScissorAndClear();


  }
  return 0;
}

