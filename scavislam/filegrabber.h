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


#ifndef FILEGRABBER_H
#define FILEGRABBER_H

#include <algorithm>
#include <string>
#include <sstream>
#include <vector>
#include <queue>

#include <boost/filesystem.hpp>
#include <boost/thread.hpp>

#include <opencv2/opencv.hpp>

namespace ScaViSLAM
{

using namespace std;


struct FrameBundle
{
  cv::Mat left_color;
  cv::Mat left_gray;
  cv::Mat right;
  cv::Mat disp;
  cv::Mat depth;
  int frame_id;
};

class FileGrabberMonitor
{
public:
  bool
  getFrameBundle             (int frame_id,
                              FrameBundle * frame_bundle);

  void
  setFrameBundle             (const FrameBundle & frame_bundle);

  bool
  isBufferFull               ();


private:
  queue<FrameBundle> bundle_buffer_;
  boost::mutex my_mutex_;
};

class FileGrabber
{
public:
  FileGrabber();

  void
  initialize                 (const string & path_str,
                              const string & base_str,
                              const string & format_str,
                              int frame_id,
                              bool get_colorleft,
                              bool get_grayleft,
                              bool get_right,
                              bool get_disp,
                              bool get_depth,
                              FileGrabberMonitor * mon);

   void
   operator()                ();

private:
  cv::Mat
  getLeftColorimage          ();

  cv::Mat
  getLeftGrayimage           ();

  cv::Mat
  getRightImage             ();

  cv::Mat
  getDispImage               ();

  cv::Mat
  getDepthImage               ();

  void
  preprocessFiles            (const boost::filesystem::path & directory,
                              bool recursive=true);

  vector<string> file_base_vec_;
  string path_str_;
  string base_str_;
  string format_str_;
  string basename;
  int internal_frame_id_;

  bool get_colorleft_;
  bool get_grayleft_;
  bool get_right_;
  bool get_disp_;
  bool get_depth_;
  bool initialized_;

  FileGrabberMonitor * mon_;

  boost::thread my_thread_;
};
}

#endif // FILEGRABBER_H
