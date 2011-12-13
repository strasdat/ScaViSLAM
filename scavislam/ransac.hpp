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

#ifndef SCAVISLAM_RANSAC_HPP
#define SCAVISLAM_RANSAC_HPP

#include <opencv2/opencv.hpp>

#include "global.h"

namespace ScaViSLAM
{


template <class Model>
class RanSaC
{
public:
  static int
  compute                  (int numRansac,
                            const typename Model::Camera & cam,
                            const std::vector<cv::DMatch> &matches,
                            const typename ALIGNED<typename VECTOR<
                            Model::point_dim>::col>::vector & train_pts,
                            const typename ALIGNED<typename VECTOR<
                            Model::obs_dim>::col>::vector & query_obs,
                            vector<cv::DMatch> & inliers,
                            typename Model::Transformation & T_query_from_train,
                            double pixel_thr = 2.5);
};

}
#endif
