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

#include "matcher.cpp"

struct StereoMatcher
{
  static const int obs_dim = 3;
};

struct MonoMatcher
{
  static const int obs_dim = 2;
};

namespace ScaViSLAM
{
template <>
bool GuidedMatcher<StereoCamera >
::createObervation(const Vector2f & new_uv_pyr,
                 const cv::Mat & disp,
                 int level,
                 Vector3d * uvu_0)
{
  double d
      = ScaViSLAM::interpolateDisparity(disp,
                                        new_uv_pyr.cast<int>(),
                                        level);
  if (d>0)
  {
    *uvu_0 = zeroFromPyr_3d(Vector3d(new_uv_pyr[0],
                                     new_uv_pyr[1],
                                     new_uv_pyr[0]-d),level);
    return true;
  }
  return false;
}


template <>
bool GuidedMatcher<LinearCamera >
::createObervation(const Vector2f & new_uv_pyr,
                 const cv::Mat & disp,
                 int level,
                 Vector2d * uv_0)
{
  *uv_0 = zeroFromPyr_2d(new_uv_pyr.cast<double>() ,level);
  return true;
}

template class GuidedMatcher<StereoCamera>;
template class GuidedMatcher<LinearCamera>;
}



