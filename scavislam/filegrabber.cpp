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

#include "filegrabber.h"

#include <boost/regex.hpp>

namespace ScaViSLAM
{
FileGrabber::FileGrabber()
{
  initialized_ = false;
}

bool FileGrabberMonitor::
getFrameBundle(int frame_id,
               FrameBundle * frame_bundle)
{
  boost::mutex::scoped_lock lock(my_mutex_);

  if (bundle_buffer_.size()<1)
    return false;

  *frame_bundle = bundle_buffer_.front();
  assert(frame_bundle->frame_id == frame_id);
  bundle_buffer_.pop();


  return true;
}

void FileGrabberMonitor::
setFrameBundle(const FrameBundle & frame_bundle)
{
  boost::mutex::scoped_lock lock(my_mutex_);

  bundle_buffer_.push(frame_bundle);
}

bool FileGrabberMonitor::
isBufferFull               ()
{
  boost::mutex::scoped_lock lock(my_mutex_);
  if (bundle_buffer_.size()<50)
    return false;
  return true;
}


void FileGrabber
::initialize(const std::string & path_str,
             const string &base_str,
             const string &format_str,
             int frame_id,
             bool get_colorleft,
             bool get_grayleft,
             bool get_right,
             bool get_disp,
             bool get_depth,
             FileGrabberMonitor * mon)
{
  cout << "Filegrabber initializing...." << endl;
  get_colorleft_ = get_colorleft;
  get_grayleft_ = get_grayleft;
  get_right_ = get_right;
  get_disp_ = get_disp;
  get_depth_ = get_depth;
  path_str_ = path_str;
  base_str_ = base_str;
  format_str_ = format_str;
  internal_frame_id_ = frame_id;
  preprocessFiles(path_str,true);
  sort(file_base_vec_.begin(),file_base_vec_.end());
  mon_ = mon;
  initialized_ = true;

  cout << "Filegrabber initialized." << endl;

  my_thread_ = boost::thread(boost::ref(*this));
}

cv::Mat FileGrabber
::getLeftColorimage()
{
  std::stringstream left_sstr;
  left_sstr <<  basename << "left." << format_str_;
  return cv::imread(left_sstr.str(),1);
}

cv::Mat FileGrabber
::getLeftGrayimage()
{
  std::stringstream left_sstr;
  left_sstr <<  basename << "left." << format_str_;
  return cv::imread(left_sstr.str(),0);
}

cv::Mat FileGrabber
::getRightImage()
{
  std::stringstream right_sstr;
  right_sstr <<  basename << "right." << format_str_;
  return cv::imread(right_sstr.str(),0);
}

cv::Mat FileGrabber
::getDispImage()
{
  std::stringstream sstr;
  sstr <<  basename << "disp." << format_str_;
  return cv::imread(sstr.str(),0);
}

cv::Mat FileGrabber
::getDepthImage()
{
  std::stringstream sstr;
  sstr <<  basename << "depth." << format_str_;
  return cv::imread(sstr.str(),-1);
}
void FileGrabber::
preprocessFiles(const boost::filesystem::path & directory,
                bool recursive)
{
  if(exists(directory))
  {
    boost::filesystem::directory_iterator end;
    for(boost::filesystem::directory_iterator iter(directory);
        iter!=end ; ++iter)
    {
      if (is_directory( *iter) )
      {
        if(recursive)
          preprocessFiles(*iter) ;
      }
      else
      {
        std::string path_name = iter->path().string();
        std::string name = base_str_ + "left." + format_str_;
        boost::regex ex(name);
        if (boost::regex_match(path_name.begin(),path_name.end(),ex))
        {
          file_base_vec_.push_back(path_name.substr(0,path_name.length()-8));
        }
      }
    }
  }
}

void FileGrabber::
operator()                ()
{
  while(initialized_)
  {
    if (mon_->isBufferFull()==false)
    {
      basename = file_base_vec_.at(internal_frame_id_);

      FrameBundle bundle;
      bundle.frame_id = internal_frame_id_;
      if (get_colorleft_)
        bundle.left_color = getLeftColorimage();
      if (get_grayleft_)
        bundle.left_gray = getLeftGrayimage();
      if (get_right_)
        bundle.right = getRightImage();
      if (get_disp_)
        bundle.disp = getDispImage();
      if (get_depth_)
        bundle.depth = getDepthImage();
      ++internal_frame_id_;

      mon_->setFrameBundle(bundle);

    }
    else
    {
      boost::this_thread::sleep(boost::posix_time::milliseconds(1));
    }
  }
}
}
