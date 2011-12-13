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

#ifndef SCAVISLAM_KEYFRAMES_H
#define SCAVISLAM_KEYFRAMES_H

#include <tr1/memory>
#include <vector>
#include <opencv2/opencv.hpp>

#include "quadtree.h"

namespace ScaViSLAM
{
using namespace std;

struct FastGridCell
{
  FastGridCell(const cv::Range & urange,
               const cv::Range & vrange,
               int fast_thr)
    : urange(urange),
      vrange(vrange),
      fast_thr(fast_thr)
  {
  }
  cv::Range urange;
  cv::Range vrange;
  int fast_thr;
};

typedef vector<vector<FastGridCell> > CellGrid2d;

struct Frame
{
  Frame(){}

  Frame(const vector<cv::Mat> & cur_pyr,
        const cv::Mat & cur_disp)
    : pyr(cur_pyr), disp(cur_disp)
  {
  }

  Frame(const Frame & other)
  {
    disp = other.disp;
    pyr = other.pyr;
    cell_grid2d = other.cell_grid2d;
  }

  void operator=(const Frame & other)
  {
    disp = other.disp;
    pyr = other.pyr;
    cell_grid2d = other.cell_grid2d;
  }

  Frame clone()
  {
    Frame other;
    other.disp = disp.clone();
    other.pyr.resize(pyr.size());
    for (unsigned i=0; i<pyr.size(); ++i)
    {
      other.pyr[i] = pyr[i].clone();
    }
    other.cell_grid2d = cell_grid2d;
    return other;
  }

  vector<cv::Mat> pyr;
  cv::Mat disp;
  vector<CellGrid2d > cell_grid2d;
};


}
#endif
