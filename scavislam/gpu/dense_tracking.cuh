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

#ifndef SCAVISLAM_GPU_DENSE_TRACKING_CUH
#define SCAVISLAM_GPU_DENSE_TRACKING_CUH

#include <vector_types.h>
#include <cassert>

const int BLOCK_WIDTH = 8;
const int BLOCK_SiZE = BLOCK_WIDTH*BLOCK_WIDTH;


struct GpuIntrinsics
{
  void set(double fl,
           double pp_x,
           double pp_y)
  {
    focal_length = fl;
    principal_point.x = pp_x;
    principal_point.y = pp_y;
  }

  float focal_length;
  float2 principal_point;
};

struct GpuVector6
{
  void set(double * vec)
  {
    for (int i = 0; i<NUM_ROWS; ++i)
    {
      data[i] = vec[i];
    }
  }

  void copyTo(double * vec) const
  {
    for (int i = 0; i<NUM_ROWS; ++i)
    {
      vec[i] = data[i];
    }
  }

  __device__ __host__
  void setZero()
  {
#pragma unroll
    for (int i = 0; i<NUM_ROWS; ++i)
    {
      data[i] = 0.f;
    }
  }


  __device__ __host__
  void add(const GpuVector6 & other)
  {
#pragma unroll
    for (int i = 0; i<NUM_ROWS; ++i)
    {
      data[i] += other.data[i];
    }
  }

  __device__ __host__
  void add(volatile GpuVector6 & other) volatile
  {
#pragma unroll
    for (int i = 0; i<NUM_ROWS; ++i)
    {
      data[i] += other.data[i];
    }
  }

  __device__ __host__
  void scaledAdd(const GpuVector6 & other, float scalar)
  {
#pragma unroll
    for (int i = 0; i<NUM_ROWS; ++i)
    {
      data[i] += other.data[i]*scalar;
    }
  }

  __device__ __host__
  void copyFrom(const GpuVector6 & other)
  {
#pragma unroll
    for (int i = 0; i<NUM_ROWS; ++i)
    {
      data[i] = other.data[i];
    }
  }

  static const int NUM_ROWS = 6;
  float data[NUM_ROWS];
};

struct GpuSymMatrix6
{
  void copyTo(double * mat_colmajor) const
  {
    int i=0;
    for (int c = 0; c<NUM_ROWS; ++c)
    {
      for (int r = 0; r<=c; ++r)
      {
        float v = data[i];
        mat_colmajor[r + NUM_ROWS*c] = v;
        mat_colmajor[c + NUM_ROWS*r] = v;
        ++i;
      }
    }

  }

  __device__ __host__
  void setZero()
  {
#pragma unroll
    for (int i = 0; i<NUM_ELEMS; ++i)
    {
      data[i] = 0.f;
    }
  }

  __device__ __host__
  void add(const GpuSymMatrix6 & other)
  {
#pragma unroll
    for (int i = 0; i<NUM_ELEMS; ++i)
    {
      data[i] += other.data[i];
    }
  }

  __device__ __host__
  void add(volatile GpuSymMatrix6 & other) volatile
  {
#pragma unroll
    for (int i = 0; i<NUM_ELEMS; ++i)
    {
      data[i] += other.data[i];
    }
  }

  __device__ __host__
  void addOuter(const GpuVector6 & vec)
  {
    data[0] += vec.data[0]*vec.data[0];
    int i = 1;
#pragma unroll
    for(int c = 0; c<=1; ++c)
    {
      data[i] += vec.data[1]*vec.data[c];
      ++i;
    }
#pragma unroll
    for(int c = 0; c<=2; ++c)
    {
      data[i] += vec.data[2]*vec.data[c];
      ++i;
    }
#pragma unroll
    for(int c = 0; c<=3; ++c)
    {
      data[i] += vec.data[3]*vec.data[c];
      ++i;
    }
#pragma unroll
    for(int c = 0; c<=4; ++c)
    {
      data[i] += vec.data[4]*vec.data[c];
      ++i;
    }
#pragma unroll
    for( int c = 0; c<=5; ++c)
    {
      data[i] += vec.data[5]*vec.data[c];
      ++i;
    }
  }

  __device__ __host__
  void copyFrom(const GpuSymMatrix6 & other)
  {
#pragma unroll
    for (int i = 0; i<NUM_ELEMS; ++i)
    {
      data[i] = other.data[i];
    }
  }

  static const int NUM_ROWS = 6;
  static const int NUM_ELEMS = 21;
  float data[NUM_ELEMS];
};

struct GpuMatrix34
{
  void set(double * mat_colmajor)
  {
    for (int i = 0; i<NUM_ELEMS; ++i)
    {
      data_colmajor[i] = mat_colmajor[i];
    }
  }

  inline __device__ __host__ float operator()(int r, int c) const
  {
    return data_colmajor[c*NUM_ROWS + r];
  }

  static const int NUM_ROWS = 3;
  static const int NUM_COLS = 4;
  static const int NUM_ELEMS = NUM_ROWS*NUM_COLS;
  float data_colmajor[NUM_ELEMS];
};

struct GpuMatrix4
{
  void set(double * mat_colmajor)
  {
    for (int i = 0; i<NUM_ELEMS; ++i)
    {
      data_colmajor[i] = mat_colmajor[i];
    }
  }

  static const int NUM_ROWS = 4;
  static const int NUM_COLS = 4;
  static const int NUM_ELEMS = NUM_ROWS*NUM_COLS;
  float data_colmajor[NUM_ELEMS];
};



struct GpuTrackingData
{
  inline __device__ __host__ void add(const GpuTrackingData& rhs)
  {
    jacobian_times_res.add(rhs.jacobian_times_res);
    hessian.add(rhs.hessian);
  }

  inline __device__ __host__ void add(volatile GpuTrackingData& rhs) volatile
  {
    jacobian_times_res.add(rhs.jacobian_times_res);
    hessian.add(rhs.hessian);
  }

  inline __device__ __host__ void setZero()
  {
    jacobian_times_res.setZero();
    hessian.setZero();
  }

  GpuSymMatrix6 hessian;
  GpuVector6 jacobian_times_res;
};



void
computePointCloud            (const GpuMatrix4 & TQ_actkey_from_cur,
                              const float * disparties,
                              int width,
                              int height,
                              int stride_in,
                              int stride_out,
                              int factor,
                              float4 * point_cloud);

class GpuTracker
{
public:

  GpuTracker(int width, int height);
  ~GpuTracker();

  void
  bindTexture                (const float * img_cur,
                              const float * dx_img_cur,
                              const float * dy_img_cur,
                              int width,
                              int height,
                              int stride_float_img);
  void
  jacobianReduction          (const float * img_prev,
                              const float4* point_cloud_prev,
                              const GpuMatrix34 & T_cur_from_prev,
                              const GpuIntrinsics & intrinsics,
                              int width,
                              int height,
                              int stride_float_img,
                              int stride_float4_img,
                              GpuTrackingData * tracking_result);
  float
  chi2                       (const float * img_prev,
                              const float4* point_cloud_prev,
                              const GpuMatrix34 & T_cur_from_prev,
                              const GpuIntrinsics & intrinsics,
                              int width,
                              int height,
                              int stride_float_img,
                              int stride_float4_img);
  void
  residualImage              (const float * img_prev,
                              const float4* point_cloud_prev,
                              const GpuMatrix34 & T_cur_from_prev,
                              const GpuIntrinsics & intrinsics,
                              int width,
                              int height,
                              int stride_float_img,
                              int stride_float4_img,
                              float4 * res_img);

private:
  GpuTrackingData * gpu_tracking_data_;
  GpuTrackingData * tracking_data_;
  float * gpu_chi2_;
  float * chi2_;
  int mem_size_;
  cudaChannelFormatDesc float_descriptor_;
};

#endif
