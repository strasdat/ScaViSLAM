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

#include "dense_tracking.cuh"
#include <stdio.h>

// Many thanks to Steven Lovegrove for various dicussions/comments and hints
// about efficient programming in CUDA!

inline __device__ __host__ float
dotStride3(const float * m_colmajor, const float4 & v1)
{
  return v1.x*m_colmajor[0] + v1.y*m_colmajor[3]
      + v1.z*m_colmajor[6] + v1.w*m_colmajor[9];
}

inline __device__ __host__ float
dotStride4(const float * m_colmajor, const float4 & v1)
{
  return v1.x*m_colmajor[0] + v1.y*m_colmajor[4]
      + v1.z*m_colmajor[8] + v1.w*m_colmajor[12];
}

inline __device__ __host__ float4
matTimesVec(const GpuMatrix4 & m, const float4 & v)
{
  return make_float4(dotStride4(m.data_colmajor,   v),
                     dotStride4(m.data_colmajor+1, v),
                     dotStride4(m.data_colmajor+2, v),
                     dotStride4(m.data_colmajor+3, v));
}

inline __device__ __host__ float4
matTimesVec(const GpuMatrix34 & m, const float4 & v)
{
  return make_float4(dotStride3(m.data_colmajor,   v),
                     dotStride3(m.data_colmajor+1, v),
                     dotStride3(m.data_colmajor+2, v),
                     1.f);
}

inline __device__ __host__ float2
cameraProject(const GpuIntrinsics & intrinsics, const float4 & p)
{
  return make_float2(intrinsics.focal_length*p.x/p.z
                     + intrinsics.principal_point.x,
                     intrinsics.focal_length*p.y/p.z
                     + intrinsics.principal_point.y);
}

inline __device__ __host__ void
frameJacobian(const float4 & p, float focal_length, float dx, float dy,
              GpuVector6 * jac)
{
  float z_sq = p.z*p.z;

  dx *= focal_length;
  dy *= focal_length;

  jac->data[0] = -dx*(1./p.z);
  jac->data[1] = -dy*1./p.z;
  jac->data[2] = (dx*p.x/z_sq + dy*p.y/z_sq);
  jac->data[3] = (dx*(p.x*p.y)/z_sq + dy*(1.f+p.y*p.y/z_sq));
  jac->data[4] = (-dx*(1.f+(p.x*p.x/z_sq))-dy*(p.x*p.y)/z_sq);
  jac->data[5] = (dx*p.y/p.z-dy*p.x/p.z);
}

__global__ void
pointcloud_kernel(GpuMatrix4 TQ,
                  const float * disparities,
                  int width,
                  int height,
                  int stride_in,
                  int stride_out,
                  int factor,
                  float4 * point_cloud)
{
  int u = (blockIdx.x*blockDim.x + threadIdx.x);
  int v = (blockIdx.y*blockDim.y + threadIdx.y);

  if (u<width && v<height)
  {
    int x = u*factor;
    int idx_in = v*stride_in + x;
    int idx_out = v*stride_out + u;

    float4 point;
    float d = disparities[idx_in]*factor;
    if (d<=0)
    {
      point = make_float4(0.f, 0.f, 0.f, -1.f);
    }
    else
    {
      //      if (d==0)
      //      {
      //        d = 0.00000000001f;
      //      }
      float4 uvd = make_float4(u, v, d, 1.f);
      point = matTimesVec(TQ,uvd);
      point.x /= point.w;
      point.y /= point.w;
      point.z /= point.w;
      point.w = 1.f;
    }
    point_cloud[idx_out] = point;
  }
}

void
computePointCloud(const GpuMatrix4 & TQ_actkey_from_cur,
                  const float * disparties,
                  int width,
                  int height,
                  int stride_in,
                  int stride_out,
                  int factor,
                  float4 * point_cloud)
{
  dim3 block_size(BLOCK_WIDTH, BLOCK_WIDTH, 1);
  dim3 grid_size((width + block_size.x - 1)/ block_size.x,
                 (height + block_size.y - 1) / block_size.y, 1);

  pointcloud_kernel
      <<<grid_size, block_size>>>(TQ_actkey_from_cur,
                                  disparties,
                                  width,
                                  height,
                                  stride_in,
                                  stride_out,
                                  factor,
                                  point_cloud);

}

texture<float, 2, cudaReadModeElementType> tex_image_cur;
texture<float, 2, cudaReadModeElementType> tex_dx_cur;
texture<float, 2, cudaReadModeElementType> tex_dy_cur;

template<int block_size>
__device__ __host__ void
warpReduce(int thread_id, volatile GpuTrackingData * shared_tracking_data)
{
  if (block_size>=64)
    shared_tracking_data[thread_id].add(shared_tracking_data[thread_id+32]);
  if (block_size>=32)
    shared_tracking_data[thread_id].add(shared_tracking_data[thread_id+16]);
  if (block_size>=16)
    shared_tracking_data[thread_id].add(shared_tracking_data[thread_id+8]);
  if (block_size>=8)
    shared_tracking_data[thread_id].add(shared_tracking_data[thread_id+4]);
  if (block_size>=4)
    shared_tracking_data[thread_id].add(shared_tracking_data[thread_id+2]);
  if (block_size>=2)
    shared_tracking_data[thread_id].add(shared_tracking_data[thread_id+1]);
}

template<int block_size>
__global__ void
jacobianReduction_kernel(const float * img_prev,
                          const float4 * point_cloud_prev,
                          GpuMatrix34  T_cur_from_prev,
                          GpuIntrinsics intrinsics,
                          int width,
                          int height,
                          int stride_float_img,
                          int stride_float4_img,
                          GpuTrackingData * global_tracking_data)
{
  int thread_id = threadIdx.y*blockDim.x + threadIdx.x;
  int block_id = blockIdx.y*gridDim.x + blockIdx.x;
  int u_prev = blockIdx.x*blockDim.x + threadIdx.x;
  int v_prev = blockIdx.y*blockDim.y + threadIdx.y;
  int idx_float_img = v_prev*stride_float_img + u_prev;
  int idx_float4_img = v_prev*stride_float4_img + u_prev;

  __shared__ GpuTrackingData shared_tracking_data[BLOCK_SiZE];

  shared_tracking_data[thread_id].setZero();
  float4 xyz_prev = point_cloud_prev[idx_float4_img];

  if (u_prev<width && v_prev<height)
  {
    if( xyz_prev.w > 0 )
    {
      float4 xyz_cur = matTimesVec(T_cur_from_prev,xyz_prev);
      float2 uv_cur = cameraProject(intrinsics, xyz_cur);

      if(uv_cur.x >= 1.f && uv_cur.y >= 1.f
         && uv_cur.x <= (float)(width-2) && uv_cur.y <= (float)(height-2))
      {
        float2 uv_cur_texoffset = make_float2(uv_cur.x+0.5f, uv_cur.y+0.5f);

        float intensity_prev = img_prev[idx_float_img];
        float intensity_cur
            = tex2D(tex_image_cur, uv_cur_texoffset.x, uv_cur_texoffset.y);
        //TODO: variable kernel size of derivatives and different factor
        float dx
            = 0.5f*tex2D(tex_dx_cur, uv_cur_texoffset.x, uv_cur_texoffset.y);
        float dy
            = 0.5f*tex2D(tex_dy_cur, uv_cur_texoffset.x, uv_cur_texoffset.y);

        float res  = (intensity_prev-intensity_cur);
        GpuVector6 jacobian;
        frameJacobian(xyz_cur, intrinsics.focal_length, dx, dy, &jacobian);

        shared_tracking_data[thread_id]
            .jacobian_times_res.scaledAdd(jacobian,res);
        shared_tracking_data[thread_id]
            .hessian.addOuter(jacobian);
      }
    }

    // reduction
    __syncthreads();

    if (block_size>=512)
    {
      if (thread_id < 256)
        shared_tracking_data[thread_id]
            .add(shared_tracking_data[thread_id + 256]);
      __syncthreads();
    }
    if (block_size>=256)
    {
      if (thread_id < 128)
        shared_tracking_data[thread_id]
            .add(shared_tracking_data[thread_id + 128]);
      __syncthreads();
    }
    if (block_size>=128)
    {
      if (thread_id < 64)
        shared_tracking_data[thread_id]
            .add(shared_tracking_data[thread_id + 64]);
      __syncthreads();
    }

    if (thread_id<32)
    {
      warpReduce<block_size>(thread_id, shared_tracking_data);
    }

    if(thread_id == 0)
    {
      global_tracking_data[block_id] = shared_tracking_data[0];
    }
  }
}

GpuTracker::GpuTracker(int width, int height)
{
  mem_size_ = ((width+BLOCK_WIDTH-1)/BLOCK_WIDTH)
      * ((height+BLOCK_WIDTH-1)/BLOCK_WIDTH);
  gpu_tracking_data_ = NULL;
  tracking_data_ = NULL;
  gpu_chi2_ = NULL;
  chi2_ = NULL;

  cudaMalloc(&gpu_tracking_data_, sizeof(GpuTrackingData)*mem_size_);
  tracking_data_ = new GpuTrackingData[mem_size_];

  cudaMalloc(&gpu_chi2_, sizeof(float)*mem_size_);
  chi2_ = new float[mem_size_];

  assert(gpu_tracking_data_!=NULL);
  assert(tracking_data_!=NULL);
  assert(gpu_chi2_!=NULL);
  assert(chi2_!=NULL);

  tex_image_cur.filterMode = cudaFilterModeLinear;
  tex_dx_cur.filterMode = cudaFilterModeLinear;
  tex_dy_cur.filterMode = cudaFilterModeLinear;

  float_descriptor_ =
      cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
}

GpuTracker::~GpuTracker()
{
  cudaFree(gpu_tracking_data_);
  delete[] tracking_data_;
  cudaFree(gpu_chi2_);
  delete[] chi2_;
}

void GpuTracker::
bindTexture(const float * img_cur,
            const float * dx_img_cur,
            const float * dy_img_cur,
            int width,
            int height,
            int stride_float_img)
{
  cudaBindTexture2D(0, tex_image_cur, img_cur, float_descriptor_,
                    width, height, stride_float_img*sizeof(float));
  cudaBindTexture2D(0, tex_dx_cur, dx_img_cur, float_descriptor_,
                    width, height, stride_float_img*sizeof(float));
  cudaBindTexture2D(0, tex_dy_cur, dy_img_cur,float_descriptor_,
                    width, height, stride_float_img*sizeof(float));
}


void GpuTracker::
jacobianReduction(const float * img_prev,
                   const float4* point_cloud_prev,
                   const GpuMatrix34 & T_cur_from_prev,
                   const GpuIntrinsics & intrinsics,
                   int width,
                   int height,
                   int stride_float_img,
                   int stride_float4_img,
                   GpuTrackingData * tracking_result)
{
  const dim3 block_size(BLOCK_WIDTH, BLOCK_WIDTH, 1);
  dim3 grid_size((width + block_size.x-1)/ block_size.x,
                 (height + block_size.y-1) / block_size.y, 1);

  jacobianReduction_kernel<BLOCK_SiZE>
      <<<grid_size, block_size>>>(img_prev,
                                  point_cloud_prev,
                                  T_cur_from_prev,
                                  intrinsics,
                                  width,
                                  height,
                                  stride_float_img,
                                  stride_float4_img,
                                  gpu_tracking_data_);
  cudaThreadSynchronize();

  cudaMemcpy(tracking_data_, gpu_tracking_data_,
             sizeof(GpuTrackingData)*grid_size.x*grid_size.y,
             cudaMemcpyDeviceToHost);

  tracking_result->jacobian_times_res.setZero();
  tracking_result->hessian.setZero();

  int global_mem_size = grid_size.x*grid_size.y;
  assert(mem_size_ >= global_mem_size);
  for (int i=0; i<global_mem_size; ++i)
    tracking_result->add(tracking_data_[i]);
}

template<int block_size>
__device__ __host__ void
warpReduce(int thread_id, volatile float * shared_float)
{
  if (block_size>=64)
    shared_float[thread_id] += shared_float[thread_id+32];
  if (block_size>=32)
    shared_float[thread_id] += shared_float[thread_id+16];
  if (block_size>=16)
    shared_float[thread_id] += shared_float[thread_id+8];
  if (block_size>=8)
    shared_float[thread_id] += shared_float[thread_id+4];
  if (block_size>=4)
    shared_float[thread_id] += shared_float[thread_id+2];
  if (block_size>=2)
    shared_float[thread_id] += shared_float[thread_id+1];
}

template<int block_size>
__global__ void
chi2_kernel(const float * img_prev,
            const float4 * point_cloud_prev,
            GpuMatrix34  T_cur_from_prev,
            GpuIntrinsics intrinsics,
            int width,
            int height,
            int stride_float_img,
            int stride_float4_img,
            float * global_chi2)
{
  int thread_id = threadIdx.y*blockDim.x + threadIdx.x;
  int block_id = blockIdx.y*gridDim.x + blockIdx.x;
  int u_prev = blockIdx.x*blockDim.x + threadIdx.x;
  int v_prev = blockIdx.y*blockDim.y + threadIdx.y;
  int idx_float_img = v_prev*stride_float_img + u_prev;
  int idx_float4_img = v_prev*stride_float4_img + u_prev;

  __shared__ float shared_chi2[BLOCK_WIDTH*BLOCK_WIDTH];

  shared_chi2[thread_id] = 0.f;
  float4 xyz_prev = point_cloud_prev[idx_float4_img];

  if (u_prev<width && v_prev<height)
  {
    if( xyz_prev.w > 0 )
    {
      float4 xyz_cur = matTimesVec(T_cur_from_prev,xyz_prev);
      float2 uv_cur = cameraProject(intrinsics, xyz_cur);

      if(uv_cur.x >= 1.f && uv_cur.y >= 1.f
         && uv_cur.x <= (float)(width-2) && uv_cur.y <= (float)(height-2))
      {
        float2 uv_cur_texoffset = make_float2(uv_cur.x+0.5f, uv_cur.y+0.5f);

        float intensity_prev = img_prev[idx_float_img];
        float intensity_cur
            = tex2D(tex_image_cur, uv_cur_texoffset.x, uv_cur_texoffset.y);
        float res  = (intensity_prev-intensity_cur);
        shared_chi2[thread_id] += res*res;
      }
    }

    // reduction
    __syncthreads();

    if (block_size>=512)
    {
      if (thread_id < 256)
        shared_chi2[thread_id] += shared_chi2[thread_id + 256];
      __syncthreads();
    }
    if (block_size>=256)
    {
      if (thread_id < 128)
        shared_chi2[thread_id] += shared_chi2[thread_id + 128];
      __syncthreads();
    }
    if (block_size>=128)
    {
      if (thread_id < 64)
        shared_chi2[thread_id] += shared_chi2[thread_id + 64];
      __syncthreads();
    }

    // unroll last loop
    if (thread_id<32)
    {
      warpReduce<block_size>(thread_id, shared_chi2);
    }

    if(thread_id == 0)
    {
      global_chi2[block_id] = shared_chi2[0];
    }
  }
}

float GpuTracker::
chi2(const float * img_prev,
     const float4* point_cloud_prev,
     const GpuMatrix34 & T_cur_from_prev,
     const GpuIntrinsics & intrinsics,
     int width,
     int height,
     int stride_float_img,
     int stride_float4_img)
{
  float output_chi2 = 0.f;
  const dim3 block_size(BLOCK_WIDTH, BLOCK_WIDTH, 1);
  dim3 grid_size((width + block_size.x-1)/ block_size.x,
                 (height + block_size.y-1) / block_size.y, 1);

  chi2_kernel<BLOCK_SiZE>
      <<<grid_size, block_size>>>(img_prev,
                                  point_cloud_prev,
                                  T_cur_from_prev,
                                  intrinsics,
                                  width,
                                  height,
                                  stride_float_img,
                                  stride_float4_img,
                                  gpu_chi2_);
  cudaThreadSynchronize();

  cudaMemcpy(chi2_, gpu_chi2_,
             sizeof(float)*grid_size.x*grid_size.y,
             cudaMemcpyDeviceToHost);

  int global_mem_size = grid_size.x*grid_size.y;
  assert(mem_size_ >= global_mem_size);
  for (int i=0; i<global_mem_size; ++i)
    output_chi2 += chi2_[i];
  return output_chi2;
}


__global__ void
residualImage_kernel(const float * img_prev,
                     const float4 * point_cloud_prev,
                     GpuMatrix34  T_cur_from_prev,
                     GpuIntrinsics intrinsics,
                     int width,
                     int height,
                     int stride_float_img,
                     int stride_float4_img,
                     float4 * res_img)
{
  int u_prev = blockIdx.x*blockDim.x + threadIdx.x;
  int v_prev = blockIdx.y*blockDim.y + threadIdx.y;
  int idx_float_img = v_prev*stride_float_img + u_prev;
  int idx_float4_img = v_prev*stride_float4_img + u_prev;

  float4 xyz_prev = point_cloud_prev[idx_float4_img];

  if (u_prev<width && v_prev<height)
  {
    if(xyz_prev.w > 0)
    {
      float4 xyz_cur = matTimesVec(T_cur_from_prev,xyz_prev);
      float2 uv_cur = cameraProject(intrinsics, xyz_cur);

      if(uv_cur.x >= 1.f && uv_cur.y >= 1.f
         && uv_cur.x <= (float)(width-2) && uv_cur.y <= (float)(height-2))
      {
        float2 uv_cur_texoffset = make_float2(uv_cur.x+0.5f, uv_cur.y+0.5f);

        float intensity_prev = img_prev[idx_float_img];
        float intensity_cur
            = tex2D(tex_image_cur, uv_cur_texoffset.x, uv_cur_texoffset.y);
        float res  = (intensity_prev-intensity_cur);
        float v = max(0.f, 1-50.f*res*res);
        res_img[idx_float4_img] = make_float4(v, v, v, 1.f);
      }
      else
      {
        res_img[idx_float4_img] = make_float4(1.f, 0.f, 0.f, 1.f);
      }
    }
    else
    {
      res_img[idx_float4_img] = make_float4(0.f, 1.f, 0.f, 1.f);
    }
  }
}

void GpuTracker::
residualImage(const float * img_prev,
              const float4* point_cloud_prev,
              const GpuMatrix34 & T_cur_from_prev,
              const GpuIntrinsics & intrinsics,
              int width,
              int height,
              int stride_float_img,
              int stride_float4_img,
              float4 * res_img)
{
  const dim3 block_size(BLOCK_WIDTH, BLOCK_WIDTH, 1);
  dim3 grid_size((width + block_size.x-1)/ block_size.x,
                 (height + block_size.y-1) / block_size.y, 1);

  residualImage_kernel
      <<<grid_size, block_size>>>(img_prev,
                                  point_cloud_prev,
                                  T_cur_from_prev,
                                  intrinsics,
                                  width,
                                  height,
                                  stride_float_img,
                                  stride_float4_img,
                                  res_img);
  cudaThreadSynchronize();
}
