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

#ifndef SCAVISLAM_DRAW_ITEMS_H
#define SCAVISLAM_DRAW_ITEMS_H


#include <opencv2/core/core.hpp>

#include <visiontools/gl_data.h>

#include "global.h"


namespace ScaViSLAM
{
using namespace VisionTools;

//TODO: clean
class DrawItems
{
public:
  DrawItems() : left(NUM_PYR_LEVELS), right(NUM_PYR_LEVELS)
  {
  }

  struct Line3d
  {
    Line3d(){}
    Line3d(const Vector3d & p1,
           const Vector3d & p2) :
      p1(p1),p2(p2)
    {
    }
    Vector3d p1;
    Vector3d p2;
  };

  struct Line2d
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Line2d(){}
    Line2d(const Vector2d & p1,
           const Vector2d & p2) :
      p1(p1),p2(p2)
    {
    }

    Vector2d p1;
    Vector2d p2;
  };

  struct Patch
  {
    Patch(){}
    Patch(const cv::Mat & patch,
          const Vector2i & top_left)
      : patch(patch),top_left(top_left)
    {
    }

    cv::Mat patch;
    Vector2i top_left;
  };

  struct Ball
  {
    Ball(){}
    Ball(const Vector3d & center,
         double radius) :
      center(center), radius(radius)
    {
    }

    Vector3d center;
    double radius;
  };

  struct Circle
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Circle(){}
    Circle(const Vector2d & center,
           double radius) :
      center(center), radius(radius)
    {
    }

    Vector2d center;
    double radius;
  };

  struct Gauss2d
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Gauss2d(){}
    Gauss2d(const Vector2d & mean,
            const Matrix2d & cov)
      : mean(mean), cov(cov)
    {
    }
    Vector2d mean;
    Matrix2d cov;
  };


  typedef list<Line3d > Line3dList;
  typedef ALIGNED<Line2d >::list Line2dList;
  typedef std::list< Rectangle > BoxList;
  typedef std::list<Patch > PatchList;
  typedef ALIGNED<Circle >::list CircleList;
  typedef list<Ball > BallList;
  typedef std::vector<GlPoint3f> Point3dVec;
  typedef std::vector<GlPoint4f> ColorVec;
  typedef std::vector<GlPoint2f> Point2dVec;
  typedef ALIGNED<Gauss2d >::list Gauss2dList;

  struct _Lines3d
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    _Lines3d(){}
    _Lines3d(const Line3dList & l,
             const Vector4f & color)
      : l(l),color(color)
    {
    }
    Line3dList l;
    Vector4f  color;
  };

  struct _Lines2d
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    _Lines2d(){}
    _Lines2d(const Line2dList & l,
             const Vector4f & color)
      : l(l),color(color)
    {
    }
    Line2dList l;
    Vector4f  color;
  };

  struct _Boxes
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    _Boxes(){}
    _Boxes(const BoxList & l,
           const Vector4f & color)
      : l(l),color(color)
    {
    }
    BoxList l;
    Vector4f  color;
  };

  struct _Patches
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    _Patches(){}
    _Patches(const PatchList & l,
             const Vector4f & color)
      : l(l),color(color)
    {
    }
    PatchList l;
    Vector4f  color;
  };

  struct _Circles
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    _Circles(){}
    _Circles(const CircleList & l,
             const Vector4f & color)
      : l(l),color(color)
    {
    }
    CircleList l;
    Vector4f  color;
  };

  struct _Balls
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    _Balls(){}
    _Balls(const BallList & l,
           const Vector4f & color)
      : l(l),color(color)
    {
    }
    BallList l;
    Vector4f  color;
  };

  struct _Points3d
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    _Points3d(){}
    _Points3d(const Point3dVec & v,
              double pixel_size,
              const Vector4f & color)
      : v(v),pixel_size(pixel_size),color(color)
    {
    }
    Point3dVec v;
    double pixel_size;
    Vector4f  color;
  };

  struct _ColorPoints3d
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    _ColorPoints3d(){}
    _ColorPoints3d(const Point3dVec & v,
                   const ColorVec & c_vec,
                   double pixel_size)
      : v(v),c_vec(c_vec),pixel_size(pixel_size)
    {
      assert(v.size() == c_vec.size());
    }

    Point3dVec v;
    ColorVec c_vec;
    double pixel_size;

  };


  struct _Points2d
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    _Points2d(){}
    _Points2d(const Point2dVec & v,
              double pixel_size,
              const Vector4f & color)
      : v(v),pixel_size(pixel_size),color(color)
    {
    }
    Point2dVec v;
    double pixel_size;
    Vector4f  color;
  };


  struct _Gauss2d
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    _Gauss2d(){}
    _Gauss2d(const Gauss2dList & l,
             const Vector4f & color)
      : l(l),color(color)
    {
    }
    Gauss2dList l;
    Vector4f  color;
  };

  class Data2d
  {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    void clear()
    {
      line2d_list.clear();
      box_list.clear();
      circle_list.clear();
      gauss2d_list.clear();
      patch_list.clear();
      point2d_list.clear();
    }

    ALIGNED<_Lines2d>::list line2d_list;
    ALIGNED<_Boxes >::list box_list;
    ALIGNED<_Points2d>::list point2d_list;
    ALIGNED<_Patches >::list patch_list;
    ALIGNED<_Circles >::list circle_list;
    ALIGNED<_Gauss2d >::list gauss2d_list;
  };

  struct Data3d
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    void clear()
    {
      ball_list.clear();
      line3d_list.clear();
      colorpoint3d_list.clear();
      point3d_list.clear();
    }
    ALIGNED<_Balls>::list ball_list;
    ALIGNED<_Lines3d>::list line3d_list;
    ALIGNED<_ColorPoints3d>::list colorpoint3d_list;
    ALIGNED<_Points3d>::list point3d_list;
  };

  ALIGNED<Data2d>::vector left;
  ALIGNED<Data2d>::vector right;
  Data3d data3d;

  bool show_pose;

  void clear()
  {
    for (int i = 0; i<NUM_PYR_LEVELS;++i)
    {
      left.at(i).clear();
      right.at(i).clear();
    }
    data3d.clear();
  }

private:
  DISALLOW_COPY_AND_ASSIGN(DrawItems)
};
}

#endif
