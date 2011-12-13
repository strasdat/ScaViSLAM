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

#ifndef SCAVISLAM_FAST_GRID_H
#define SCAVISLAM_FAST_GRID_H

#include "global.h"
#include "keyframes.h"
#include "quadtree.h"


namespace ScaViSLAM
{

class FastGrid
{
public:
  FastGrid                   (){}
  FastGrid                   (const cv::Size & img_size,
                              int num_features_per_cell,
                              int boundary_per_cell,
                              int fast_thr,
                              const cv::Size & grid_size,
                              int fast_min = 10,
                              int fast_max = 40);
  void
  detectAdaptively           (const cv::Mat & img,
                              int trials,
                              QuadTree<int> * qt);
  static void detect         (const cv::Mat & img,
                              const CellGrid2d & cell_grid2d,
                              QuadTree<int> * qt);

  const vector<vector<FastGridCell> >& cell_grid2d() const
  {
    return cell_grid2d_;
  }

private:
  int num_features_per_cell_;
  int boundary_per_cell_;
  int min_inner_range_;
  int min_outer_range_;
  int max_inner_range_;
  int max_outer_range_;
  CellGrid2d cell_grid2d_;
  int fast_min_thr;
  int fast_max_thr;
};
}

#endif
