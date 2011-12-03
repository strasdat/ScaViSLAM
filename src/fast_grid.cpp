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

#include "fast_grid.h"

namespace ScaViSLAM
{

FastGrid::
FastGrid(const cv::Size & img_size,
         int num_features_per_cell,
         int boundary_per_cell,
         int fast_thr,
         const cv::Size & grid_size,
         int fast_min,
         int fast_max)
  : cell_grid2d_(grid_size.height),
    fast_min_thr(fast_min),
    fast_max_thr(fast_max)

{
  min_inner_range_ = num_features_per_cell - boundary_per_cell*0.33;
  min_outer_range_ = num_features_per_cell - boundary_per_cell;
  max_inner_range_ = num_features_per_cell + boundary_per_cell*0.33;
  max_outer_range_ = num_features_per_cell + boundary_per_cell;

  int cell_width = img_size.width/grid_size.width;
  int cell_height = img_size.height/grid_size.height;

  for (int j=0; j<grid_size.height; ++j)
  {
    vector<FastGridCell> cell_grid;

    for (int i=0; i<grid_size.width; ++i)
    {
      int u_offset = i*cell_width;
      int v_offset = j*cell_height;
      cell_grid.push_back(FastGridCell(cv::Range(u_offset,u_offset+cell_width),
                                       cv::Range(v_offset,v_offset+cell_height),
                                       fast_thr));
    }
    cell_grid2d_.at(j) = cell_grid;
  }
}

void FastGrid::
detect(const cv::Mat & img, const CellGrid2d & cell_grid2d, QuadTree<int> * qt)
{
  for (size_t j=0; j<cell_grid2d.size(); ++j)
  {
    const vector<FastGridCell> & cell_grid = cell_grid2d.at(j);

    for (size_t i=0; i<cell_grid.size(); ++i)
    {
      const FastGridCell & c = cell_grid.at(i);
      vector<cv::KeyPoint> cell_corners;

      cv::FastFeatureDetector fast(c.fast_thr,false);
      fast.detect(img(c.vrange,c.urange), cell_corners);

      for (size_t idx=0; idx<cell_corners.size(); ++idx)
      {
        qt->insert(Vector2d( cell_corners[idx].pt.x+c.urange.start,
                             cell_corners[idx].pt.y+c.vrange.start),
                   idx);
      }
    }
  }
}


void FastGrid::
detectAdaptively(const cv::Mat & img, int trials, QuadTree<int> * qt)
{
  for (size_t j=0; j<cell_grid2d_.size(); ++j)
  {
    vector<FastGridCell> & cell_grid = cell_grid2d_.at(j);

    int prev_thr = -1;
    int prev_prev_thr = -2;

    for (size_t i=0; i<cell_grid.size(); ++i)
    {
      FastGridCell & c = cell_grid.at(i);

      vector<cv::KeyPoint>  cell_corners;
      for (int trial=0; trial<trials; ++trial)
      {
        cell_corners.clear();
        cv::FastFeatureDetector fast(c.fast_thr,false);
        fast.detect(img(c.vrange,c.urange), cell_corners);

        int num_detected = cell_corners.size();
        if (prev_prev_thr == c.fast_thr)
        {
          c.fast_thr = (c.fast_thr+prev_prev_thr)/2;
          break;
        }
        prev_prev_thr = prev_thr;
        prev_thr = c.fast_thr;
        if (num_detected<min_inner_range_)
        {
          if (c.fast_thr<= fast_min_thr)
            break;
          --c.fast_thr;
          if (num_detected<min_outer_range_)
          {
            if (c.fast_thr<= fast_min_thr)
              break;
            --c.fast_thr;
            continue;
          }
        }
        else if (num_detected>max_inner_range_)
        {
          if (c.fast_thr >= fast_max_thr)
            break;
          ++c.fast_thr;
          if (num_detected>max_outer_range_)
          {
            if (c.fast_thr >= fast_max_thr)
              break;
            ++c.fast_thr;
            continue;
          }
        }
        break;
      }

      for (size_t idx=0;idx<cell_corners.size(); ++idx)
      {
        qt->insert(Vector2d( cell_corners[idx].pt.x+c.urange.start,
                             cell_corners[idx].pt.y+c.vrange.start),
                   idx);
      }
    }
  }
}
}
