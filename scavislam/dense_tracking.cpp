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

#include "dense_tracking.h"

#include "transformations.h"

namespace ScaViSLAM
{

DenseTracker
::DenseTracker(const FrameData<StereoCamera> & frame_data)
  : frame_data_(frame_data)
  #ifdef SCAVISLAM_CUDA_SUPPORT
  , gpu_tracker_(frame_data.cam.width(), frame_data.cam.height())
  #endif
{
  for (int level = 0; level<NUM_PYR_LEVELS; ++level)
  {
    cv::Size mat_size(frame_data.cam_vec[level].width(),
                      frame_data.cam_vec[level].height()
                      );
#ifdef SCAVISLAM_CUDA_SUPPORT
    dev_ref_dense_points_[level].create(mat_size, CV_32FC4);
    dev_residual_img[level].create(mat_size, CV_32FC4);
    dev_residual_img[level].setTo(cv::Scalar(0,0,0,1));
#else
    assert(mat_size.width%EVERY_NTH_PIXEL==0);
    assert(mat_size.height%EVERY_NTH_PIXEL==0);
    // If this asserts fails, we need to verify whether the logic below still
    // works, or we need to adapt the code!

    int width_times_factor = mat_size.width/EVERY_NTH_PIXEL;
    int height_times_factor = mat_size.height/EVERY_NTH_PIXEL;

    ref_dense_points_[level].create(height_times_factor,
                                    width_times_factor, CV_32FC4);
    residual_img[level].create(height_times_factor,
                               width_times_factor, CV_32FC4);
    residual_img[level].setTo(cv::Scalar(0,0,0,1));
#endif
  }
}

#ifdef SCAVISLAM_CUDA_SUPPORT
//TODO: method too long
//Decide: LM or Gauss-Newton
void DenseTracker
::denseTrackingGpu(SE3 * T_cur_from_actkey)
{
  const ALIGNED<StereoCamera>::vector & cam_vec
      = frame_data_.cam_vec;
  GpuTrackingData tracking_data;
  GpuIntrinsics intrinsics;
  GpuMatrix34 gpuT_cur_from_prev;
  GpuMatrix34 gpuT_cur_from_actkey_new;
  //  StopWatch sw;
  //  sw.start();
  for (int l=NUM_PYR_LEVELS-1; l>=0; --l)
  {
    int width = cam_vec[l].width();
    int height = cam_vec[l].height();
    int stride_float_img
        = frame_data_.gpu_pyr_float32_dx[l].step1();
    int stride_float4_img
        = dev_ref_dense_points_[l].step1()/4;
    assert(dev_ref_dense_points_[l].step1() == dev_residual_img[l].step1());
    intrinsics.set(cam_vec[l].focal_length(),
                   cam_vec[l].principal_point()[0],
                   cam_vec[l].principal_point()[1]);
    gpuT_cur_from_prev
        .set(Matrix<double,3,4>(T_cur_from_actkey->matrix()
                                .topLeftCorner<3,4>()).data());
    gpu_tracker_.bindTexture(
          (float *)frame_data_.cur_left().gpu_pyr_float32[l].data,
          (float *)frame_data_.gpu_pyr_float32_dx[l].data,
          (float *)frame_data_.gpu_pyr_float32_dy[l].data,
          width,
          height,
          stride_float_img);
    float chi2 = gpu_tracker_.chi2(
          (float *)frame_data_.prev_left().gpu_pyr_float32[l].data,
          (float4 *)dev_ref_dense_points_[l].data,
          gpuT_cur_from_prev,
          intrinsics,
          width,
          height,
          stride_float_img,
          stride_float4_img);
    double nu = 2;
    bool stop = false;
    double mu = 0.01f;
    int trial = 0;

    for (int i = 0; i<15; ++i)
    {
      double rho = 0;
      do
      {
        gpuT_cur_from_prev
            .set(Matrix<double,3,4>(T_cur_from_actkey->matrix()
                                    .topLeftCorner<3,4>()).data());

        gpu_tracker_.jacobianReduction(
              (float *)frame_data_.prev_left().gpu_pyr_float32[l].data,
              (float4 *)dev_ref_dense_points_[l].data,
              gpuT_cur_from_prev,
              intrinsics,
              width,
              height,
              stride_float_img,
              stride_float4_img,
              &tracking_data);
        //fprintf(stderr, "chi2 %f \n", chi2);
        Matrix6d H;
        tracking_data.hessian.copyTo(H.data());
        H += DiagonalMatrix<double,6>(mu*H.diagonal());
        Vector6d b;
        tracking_data.jacobian_times_res.copyTo(b.data());

        Vector6d x = H.ldlt().solve(-b);
        SE3 T_cur_from_actkey_new =  SE3::exp(x)*(*T_cur_from_actkey);



        gpuT_cur_from_actkey_new
            .set(Matrix<double,3,4>(T_cur_from_actkey_new.matrix()
                                    .topLeftCorner<3,4>()).data());

        float new_chi2
            = gpu_tracker_.chi2(
              (float *)frame_data_.prev_left().gpu_pyr_float32[l].data,
              (float4 *)dev_ref_dense_points_[l].data,
              gpuT_cur_from_actkey_new,
              intrinsics,
              width,
              height,
              stride_float_img,
              stride_float4_img);

        rho = chi2-new_chi2;
        //fprintf(stderr, "mu %f \n", mu);
        if(rho>0)
        {
          //fprintf(stderr, "update  \n");
          *T_cur_from_actkey = T_cur_from_actkey_new;
          chi2 = new_chi2;
          stop = norm_max(b)<=EPS;
          mu *= std::max(1./3.,1-Po3(2*rho-1));
          nu = 2.;
          trial = 0;
        }
        else
        {
          // fprintf(stderr, "%f > %f \n", new_chi2, chi2);
          mu *= nu;
          nu *= 2.;
          ++trial;
          if (trial==2)
            stop = true;
        }
      }while(!(rho>0 ||  stop));
      if (stop)
        break;
    }
    gpu_tracker_.residualImage((float *)frame_data_.prev_left()
                               .gpu_pyr_float32[l].data,
                               (float4 *)dev_ref_dense_points_[l].data,
                               gpuT_cur_from_prev,
                               intrinsics,
                               width,
                               height,
                               stride_float_img,
                               stride_float4_img,
                               (float4 *)dev_residual_img[l].data);
  }
  //  sw.stop();
  //  fprintf(stderr, "time %f \n", sw.get_stopped_time());
}

void DenseTracker
::computeDensePointCloudGpu(const SE3 & T_cur_from_actkey)
// for each pixel create a 3d point
{
  for (int level=0; level<NUM_PYR_LEVELS; ++level)
  {
    const StereoCamera & cam
        = frame_data_.cam_vec[level];
    Matrix4d TQ = T_cur_from_actkey.inverse().matrix()*cam.Q();

    GpuMatrix4 cuTQ_actkey_from_cur;
    cuTQ_actkey_from_cur.set(TQ.data());
    computePointCloud(cuTQ_actkey_from_cur,
                      (float *)frame_data_.gpu_disp_32f.data,
                      dev_ref_dense_points_[level].size().width,
                      dev_ref_dense_points_[level].size().height,
                      frame_data_.gpu_disp_32f.step1(),
                      dev_ref_dense_points_[level].step1()/4,
                      pow(2,level),
                      (float4 *)dev_ref_dense_points_[level].data);
  }
}

#else


// ToDo: method too long (Implement abstract class for LM!)
void DenseTracker
::denseTrackingCpu(SE3 * T_cur_from_actkey)
{
  for (int level = NUM_PYR_LEVELS-1; level>=0; --level)
  {
    const StereoCamera & cam
        = frame_data_.cam_vec[level];
    float chi2 = 0;
    for (int v=0; v<ref_dense_points_[level].size().height; ++v)
    {
      for (int u=0; u<ref_dense_points_[level].size().width; ++u)
      {
        cv::Vec4f cv_float4 = ref_dense_points_[level].at<cv::Vec4f>(v,u);
        if (cv_float4[3]>0)
        {
          Vector3d xyz_prev(cv_float4[0], cv_float4[1], cv_float4[2]);
          Vector3d xyz_cur = (*T_cur_from_actkey)*xyz_prev;
          Vector2f uv_cur = cam.map(project2d(xyz_cur)).cast<float>();
          if (cam.isInFrame(uv_cur.cast<int>(),2))
          {
            float intensity_prev
                = (1./255.)*frame_data_.prev_left()
                .pyr_uint8[level].at<uint8_t>(v*EVERY_NTH_PIXEL,
                                              u*EVERY_NTH_PIXEL);
            float intensity_cur
                = interpolateMat_32f(frame_data_.pyr_float32[level], uv_cur);
            float res = intensity_prev-intensity_cur;
            if (res > 0.1)
            {
              res = 0.1;
            }
            if (res < -0.1)
            {
              res = -0.1;
            }
            chi2 += res*res;
          }
        }
      }
    }

    double nu = 2;
    bool stop = false;
    double mu = 0.01f;
    int trial = 0;
    SE3 T_cur_from_actkey_new;

    for (int i = 0; i<15; ++i)
    {
      double rho = 0;
      do
      {
        Matrix6d H;
        H.setZero();
        Vector6d Jres;
        Jres.setZero();

        for (int v=0; v<ref_dense_points_[level].size().height; ++v)
        {
          for (int u=0; u<ref_dense_points_[level].size().width; ++u)
          {
            cv::Vec4f cv_float4 = ref_dense_points_[level].at<cv::Vec4f>(v,u);
            if (cv_float4[3]>0)
            {
              Vector3d xyz_prev(cv_float4[0], cv_float4[1], cv_float4[2]);
              Vector3d xyz_cur = (*T_cur_from_actkey)*xyz_prev;
              Vector2f uv_cur = cam.map(project2d(xyz_cur)).cast<float>();
              if (cam.isInFrame(uv_cur.cast<int>(),2))
              {
                float intensity_prev
                    = (1./255.)*frame_data_.prev_left()
                    .pyr_uint8[level].at<uint8_t>(v*EVERY_NTH_PIXEL,
                                                  u*EVERY_NTH_PIXEL);
                float intensity_cur
                    = interpolateMat_32f(frame_data_.pyr_float32[level], uv_cur);
                float dx
                    = 0.5*interpolateMat_32f(frame_data_.pyr_float32_dx[level],
                                             uv_cur);
                float dy
                    = 0.5*interpolateMat_32f(frame_data_.pyr_float32_dy[level],
                                             uv_cur);
                float res = intensity_prev-intensity_cur;
                if (res > 0.1)
                {
                  res = 0.1;
                }
                if (res < -0.1)
                {
                  res = -0.1;
                }
                Matrix<double,2,6> frame_jac;
                frame_jac_xyz2uv(xyz_cur, cam.focal_length(), frame_jac);
                Vector6d J = dx*frame_jac.row(0) + dy*frame_jac.row(1);
                H += J*J.transpose();
                Jres += J*res;
                float v = max(0.f, 1-50.f*res*res);
                cv_float4 =  cv::Vec4f(v, v, v, 1.f);
              }
              else
              {
                cv_float4 =  cv::Vec4f(1.f, 0.f, 0.f, 1.f);
              }
            }
            else
            {
              cv_float4 =  cv::Vec4f(0.f, 1.f, 0.f, 1.f);
            }
            residual_img[level].at<cv::Vec4f>(v,u) = cv_float4;
          }
        }
        Vector6d x = H.ldlt().solve(-Jres);
        T_cur_from_actkey_new =  SE3::exp(x)*(*T_cur_from_actkey);

        float new_chi2 = 0;
        for (int v=0; v<ref_dense_points_[level].size().height; ++v)
        {
          for (int u=0; u<ref_dense_points_[level].size().width; ++u)
          {
            cv::Vec4f cv_float4 = ref_dense_points_[level].at<cv::Vec4f>(v,u);
            if (cv_float4[3]>0)
            {
              Vector3d xyz_prev(cv_float4[0], cv_float4[1], cv_float4[2]);
              Vector3d xyz_cur = (T_cur_from_actkey_new)*xyz_prev;
              Vector2f uv_cur = cam.map(project2d(xyz_cur)).cast<float>();
              if (cam.isInFrame(uv_cur.cast<int>(),2))
              {
                float intensity_prev
                    = (1./255.)*frame_data_.prev_left()
                    .pyr_uint8[level].at<uint8_t>(v*EVERY_NTH_PIXEL,
                                                  u*EVERY_NTH_PIXEL);
                float intensity_cur
                    = interpolateMat_32f(frame_data_.pyr_float32[level], uv_cur);
                float res = intensity_prev-intensity_cur;
                if (res > 0.1)
                {
                  res = 0.1;
                }
                if (res < -0.1)
                {
                  res = -0.1;
                }
                new_chi2 += res*res;
              }
            }
          }
        }
        rho = chi2-new_chi2;
        if(rho>0)
        {
          *T_cur_from_actkey = T_cur_from_actkey_new;
          chi2 = new_chi2;
          stop = norm_max(x)<=EPS;
          mu *= std::max(1./3.,1-Po3(2*rho-1));
          nu = 2.;
          trial = 0;
        }
        else
        {
          mu *= nu;
          nu *= 2.;
          ++trial;
          if (trial==2)
            stop = true;
        }
      }while(!(rho>0 ||  stop));
      if (stop)
        break;
    }
  }
}

void DenseTracker
::computeDensePointCloudCpu(const SE3 & T_cur_from_actkey_)
// for each pixel create a 3d point
{
  for (int level=0; level<NUM_PYR_LEVELS; ++level)
  {
    const StereoCamera & cam
        = frame_data_.cam_vec[level];
    Matrix4d TQ = T_cur_from_actkey_.inverse().matrix()*cam.Q();

    for (int v=0; v<ref_dense_points_[level].size().height; ++v)
      for (int u=0; u<ref_dense_points_[level].size().width; ++u)
      {
        float d
            = interpolateDisparity(frame_data_.disp, Vector2i(u*4,v*4), level);

        cv::Vec4f cv_float4;
        if (d<=0)
        {
          cv_float4 = cv::Vec4f(0., 0., 0., -1.);
        }
        else
        {
          Vector4d uvd(u*EVERY_NTH_PIXEL, v*EVERY_NTH_PIXEL, d, 1.f);
          Vector3d xyz = project3d(TQ*uvd);
          cv_float4 = cv::Vec4f(xyz.x(), xyz.y(), xyz.z(), 1.);
        }
        ref_dense_points_[level].at<cv::Vec4f>(v,u) = cv_float4;
      }
  }
}
#endif

}
