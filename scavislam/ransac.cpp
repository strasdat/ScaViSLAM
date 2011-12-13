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

#include <visiontools/sample.h>

#include "ransac.hpp"

namespace ScaViSLAM
{

// This RANSAC code is based on "http://www.ros.org/wiki/posest"
// written by Kurt Konolige and originally licensed under BSD.

template<class Model>
int RanSaC<Model>
::compute(int numRansac,
          const typename Model::Camera & cam,
          const std::vector<cv::DMatch> & matches,
          const typename ALIGNED<typename VECTOR<Model::point_dim>::col>::vector
          & train_pt_vec,
          const typename ALIGNED<typename VECTOR<Model::obs_dim>::col>::vector
          & query_obs_vec,
          vector<cv::DMatch> & inliers,
          typename Model::Transformation & T_query_from_train,
          double pixel_thr)
{
  double maxInlierXDist2 = pixel_thr*pixel_thr;
  typename ALIGNED<typename VECTOR<Model::point_dim>::col>::vector
      train_pt(Model::num_points);
  typename ALIGNED<typename VECTOR<Model::obs_dim>::col>::vector
      query_obs(Model::num_points);
  int rand_num[Model::num_points];
  int q_idx[Model::num_points];
  int t_idx[Model::num_points];
  uint nmatch = matches.size();
  assert(query_obs_vec.size()==nmatch);
  // set up data structures for fast processing
  // indices to good matches
  vector<int> m0, m1;
  for (uint i=0; i<nmatch; i++)
  {
    m0.push_back(matches[i].queryIdx);
    m1.push_back(matches[i].trainIdx);
  }
  nmatch = m0.size();
  if (nmatch < Model::num_points)
    return 0;   // can't do it...
  int bestinl = 0;
  for (int i=0; i<numRansac; i++)
  {
    // find a candidate

    //label: see goto below
create_new_samples:
    for (uint i=0; i<Model::num_points;++i)
    {
      //label: see goto below
calc_new_random_number:
      rand_num[i] = Sample::uniform(0,nmatch-1);

      for (uint j=0; j<i; ++j)
      {
        if (rand_num[j]==rand_num[i])
        {
          goto calc_new_random_number;
        }
      }
    }
    for (uint i=0; i<Model::num_points;++i)
    {
      q_idx[i] = m0[rand_num[i]];
      t_idx[i] = m1[rand_num[i]];

      for (uint j=0; j<i; ++j)
      {
        if (q_idx[i]==q_idx[j]
            || t_idx[i]==t_idx[j])
        {
          goto create_new_samples;
        }
      }
    }
    for (uint i=0; i<Model::num_points; ++i)
    {
      query_obs[i] = query_obs_vec[q_idx[i]];
      train_pt[i] = train_pt_vec[t_idx[i]];
    }
    typename Model::Transformation T;
    Model::calc_motion(cam,
                       query_obs,
                       train_pt,
                       T);

    // find inliers, based on image reprojection
    int inl = 0;
    for (uint i=0; i<nmatch; i++)
    {
      if (Model::belowThreshold(cam,
                                T*(train_pt_vec[m1[i]]),
                                query_obs_vec.at(m0[i]),
                                maxInlierXDist2))
      {
        inl++;
      }
    }
    if (inl > bestinl)
    {
      bestinl = inl;
      T_query_from_train = T;
    }
  }
  for (uint i=0; i<nmatch; i++)
  {
    if (Model::belowThreshold(cam,
                              T_query_from_train*(train_pt_vec[m1[i]]),
                              query_obs_vec.at(m0[i]),
                              maxInlierXDist2))
    {
      inliers.push_back(matches[i]);
    }
  }
  return inliers.size();
}

}
