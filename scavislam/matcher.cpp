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

#include "matcher.hpp"

#include <stdint.h>

#include <sophus/se3.h>

#include <visiontools/accessor_macros.h>

#include "data_structures.h"
#include "homography.h"
#include "keyframes.h"
#include "transformations.h"

//TODO: improve sub-pixel using LK tracking or ESM
namespace ScaViSLAM
{

template <class Camera>
uint8_t GuidedMatcher<Camera>::KEY_PATCH[BOX_AREA];
template <class Camera>
uint8_t GuidedMatcher<Camera>::CUR_PATCH[BOX_AREA];


//TODO: make even faster by ensure data alignment
template <class Camera>
void GuidedMatcher<Camera>
::matchPatchZeroMeanSSD(int sumA,
                        int sumAA,
                        int *znssd)
{
  uint32_t sumB_uint = 0;
  uint32_t sumBB_uint = 0;
  uint32_t sumAB_uint = 0;

  // Written in a way so set clever compilers (e.g. gcc) can do auto
  // vectorization!
  for(int r = 0; r < BOX_AREA; r++)
  {
    uint8_t cur_pixel = CUR_PATCH[r];
    sumB_uint += cur_pixel;
    sumBB_uint += cur_pixel*cur_pixel;
    sumAB_uint += cur_pixel * KEY_PATCH[r];
  }
  int sumB = sumB_uint;
  int sumBB = sumBB_uint;
  int sumAB = sumAB_uint;

  // zero mean sum of squared differences (SSD)
  // sum[ ((A-mean(A)) - (B-mean(B))^2 ]
  // = sum[ ((A-B) - (mean(A)-mean(B))^2 ]
  // = sum[ (A-B)^2 - 2(A-B)(mean(A)mean(B)) + (mean(A)-mean(B))^2 ]
  // = sum[ (A-B)^2 ] - 2(mean(A)mean(B))(sumA-sumB) + N(mean(A)-mean(B))^2
  // = sum[ (A-B)^2 ] - N * (mean(A)-mean(B))^2
  // = sum[ A^2-2AB-B^2 ] - 1/N * (sumA-sumB)^2
  // = sumAA-2*sumAB-sumBB - 1/N * (sumA^2-2*sumA*sumB-sumB^2)
  *znssd = sumAA-2*sumAB-sumBB - (sumA*sumA - 2*sumA*sumB - sumB*sumB)/BOX_AREA;
}

//TODO: make even faster by ensure data alignment
template <class Camera>
void GuidedMatcher<Camera>
::computePatchScores(int * sumA,
                     int * sumAA)
{
  uint32_t sumA_uint = 0;
  uint32_t sumAA_uint = 0;


  // Written in a way so set clever compilers (e.g. gcc) can do auto
  // vectorization!
  for(int r = 0; r < BOX_AREA; r++)
  {
    uint8_t n = KEY_PATCH[r];
    sumA_uint += n;
    sumAA_uint += n*n;
  }
  *sumA = sumA_uint;
  *sumAA = sumAA_uint ;
}

template <class Camera>
bool GuidedMatcher<Camera>
::computePrediction(const SE3 & T_cur_from_w,
                    const typename ALIGNED<Camera>::vector & cam_vec,
                    const tr1::shared_ptr<CandidatePoint<Camera::obs_dim> >
                    & ap,
                    const ALIGNED<FrontendVertex>::int_hash_map & vertex_map,
                    Vector2d * uv_pyr,
                    SE3 * T_anchorkey_from_w)
{
  ALIGNED<FrontendVertex>::int_hash_map::const_iterator it_T
      = vertex_map.find(ap->anchor_id);

  if(it_T==vertex_map.end())
    return false;

  *T_anchorkey_from_w = it_T->second.T_me_from_w;

  SE3 T_cur_from_anchor = T_cur_from_w*T_anchorkey_from_w->inverse();

  Vector3d xyz_cur = T_cur_from_anchor*ap->xyz_anchor;

  *uv_pyr
      = cam_vec[ap->anchor_level]
      .map(project2d(xyz_cur));

  Vector2d key_uv_pyr
      = ap->anchor_obs_pyr.head(2);

  if (!cam_vec[ap->anchor_level].isInFrame(
        Vector2i(key_uv_pyr[0],key_uv_pyr[1]),HALFBOX_SIZE))
  {
    return false;
  }

  double depth_cur = 1./xyz_cur.z();
  double depth_anchor = 1./ap->xyz_anchor.z();


  if (depth_cur>depth_anchor*3 || depth_anchor>depth_cur*3)
  {
    return false;
  }
  return true;
}

template <class Camera>
void GuidedMatcher<Camera>
::matchCandidates(const ALIGNED<QuadTreeElement<int> >::list & candidates,
                  const Frame & cur_frame,
                  const typename ALIGNED<Camera>::vector & cam_vec,
                  int pixel_sum,
                  int pixel_sum_square,
                  int level,
                  MatchData *match_data)
{
  for (list<QuadTreeElement<int> >::const_iterator it = candidates.begin();
       it!=candidates.end(); ++it)
  {
    Vector2i cur_uvi = it->pos.cast<int>();
    if (!cam_vec[level].isInFrame(cur_uvi,HALFBOX_SIZE + 2))
    {
      continue;
    }

    cv::Mat cur_patch(8,8,CV_8U,&(CUR_PATCH[0]));
    cur_frame.pyr.at(level)
        (cv::Range(cur_uvi[1]-HALFBOX_SIZE,
                   cur_uvi[1]+HALFBOX_SIZE),
         cv::Range(cur_uvi[0]-HALFBOX_SIZE,
                   cur_uvi[0]+HALFBOX_SIZE)).copyTo(cur_patch);
    int znssd = 0;
    matchPatchZeroMeanSSD(pixel_sum, pixel_sum_square,
                          &znssd);

    if (znssd<match_data->min_dist)
    {

      match_data->min_dist = znssd;
      match_data->index = it->content;
      match_data->uv_pyr = it->pos.cast<int>();
    }
  }
}

template <class Camera>
void GuidedMatcher<Camera>
::returnBestMatch(const cv::Mat & key_patch,
                  const Frame & cur_frame,
                  const MatchData & match_data,
                  const Vector3d & xyz_actkey,
                  const tr1::shared_ptr<CandidatePoint<Camera::obs_dim> > & ap,
                  TrackData<Camera::obs_dim> * track_data)
{
  if (match_data.index>-1)
  {
    Matrix<double,Camera::obs_dim,1> obs;

    Vector2f uv_pyr;

    if (subpixelAccuracy(key_patch,
                         cur_frame,
                         match_data.uv_pyr,
                         ap->anchor_level,
                         &uv_pyr))
    {
      if (createObervation(uv_pyr, cur_frame.disp, ap->anchor_level, &obs))
      {

        int point_id = track_data->point_list.size();
        track_data->obs_list.push_back(
              IdObs<Camera::obs_dim>(point_id, 0, obs));

        track_data->point_list.push_back(xyz_actkey);
        track_data->ba2globalptr.push_back(ap);
      }
    }
  }
}

template <class Camera>
cv::Mat GuidedMatcher<Camera>
::warp2d(const cv::Mat & patch_with_border,
         const Vector2f & uv_pyr)
{
  cv::Mat patch_out(cv::Size(patch_with_border.size().width-2,
                             patch_with_border.size().height-2),  CV_32F);

  for(int h = 0; h < patch_out.size().height; ++h)
  {
    float * patch_ptr = patch_out.ptr<float>(h,0);

    for(int w = 0; w < patch_out.size().width; ++w)
    {
      float warped_pixel = interpolateMat_8u(patch_with_border,
                                             uv_pyr + Vector2f(w+1, h+1));
      patch_ptr[w] = warped_pixel;
    }
  }
  return patch_out;
}


template <class Camera>
bool GuidedMatcher<Camera>
::subpixelAccuracy(const cv::Mat & key_patch_with_border_8u,
                   const Frame & cur_frame,
                   const Vector2i & uv_pyr_in,
                   int level,
                   Vector2f * uv_pyr_out)
{
#if 0
  int patch_width = key_patch_with_border_8u.size().width-2;
  assert(patch_width == key_patch_with_border_8u.size().height-2);

  cv::Mat key_patch_with_border_32f;
  key_patch_with_border_8u.convertTo(key_patch_with_border_32f, CV_32F);

  int width_with_border = key_patch_with_border_8u.size().width;
  int halfwidth_with_border = width_with_border/2;

  cv::Mat dx_with_border (width_with_border, width_with_border, CV_32F);
  cv::Mat dy_with_border (width_with_border, width_with_border, CV_32F);
  cv::Sobel(key_patch_with_border_32f, dx_with_border,
            dx_with_border.depth(),
            1., 0, 1, 0.5);
  cv::Sobel(key_patch_with_border_32f, dy_with_border,
            dy_with_border.depth(),
            0., 1., 1, 0.5);

  cv::Rect roi(cv::Point(1, 1),
               cv::Size(patch_width, patch_width));
  cv::Mat key_patch_32f = key_patch_with_border_32f(roi);
  cv::Mat dx = dx_with_border(roi);
  cv::Mat dy = dy_with_border(roi);

  cv::Mat cur_patch_width_border = cur_frame.pyr.at(level)(
        cv::Rect(cv::Point(uv_pyr_in.x()-halfwidth_with_border,
                           uv_pyr_in.y()-halfwidth_with_border),
                 cv::Size(width_with_border, width_with_border)));
  cv::Mat warped_patch = warp2d(cur_patch_width_border, Vector2f(0, 0));

  cv::Mat diff = warped_patch-key_patch_32f;

  Matrix2f H;
  H.setZero();
  Vector2f Jres;
  Jres.setZero();
  float old_chi2 = 0;
  for(int h = 0; h < patch_width; ++h)
  {
    float * diff_ptr = diff.ptr<float>(h,0);
    float * dx_ptr = dx.ptr<float>(h,0);
    float * dy_ptr = dy.ptr<float>(h,0);

    for(int w = 0; w < patch_width; ++w)
    {
      Vector2f J(dx_ptr[w], dy_ptr[w]);
      float d = diff_ptr[w];
      old_chi2 += d*d;
      Jres += J*d;
      H += J*J.transpose();
    }
  }
  Vector2f delta = H.ldlt().solve(-Jres);
  *uv_pyr_out = Vector2f(uv_pyr_in.x() + delta.x(),
                         uv_pyr_in.y() + delta.y());
#endif
  *uv_pyr_out = Vector2f(uv_pyr_in.x(), uv_pyr_in.y());

  return true;
}


template <class Camera>
void GuidedMatcher<Camera>
::match(const tr1::unordered_map<int,Frame> & keyframe_map,
        const SE3 & T_cur_from_actkey,
        const Frame & cur_frame,
        const ALIGNED<QuadTree<int> >::vector & feature_tree,
        const typename ALIGNED<Camera>::vector & cam_vec,
        int actkey_id,
        const ALIGNED<FrontendVertex>::int_hash_map & vertex_map,
        const list< tr1::shared_ptr<CandidatePoint<Camera::obs_dim> > >
        & ap_map,
        int SEARCHRADIUS,
        int thr_mean,
        int thr_std,
        TrackData<Camera::obs_dim> * track_data)
{
  SE3 T_actkey_from_w
      = GET_MAP_ELEM(actkey_id, vertex_map).T_me_from_w;
  SE3 T_w_from_actkey = T_actkey_from_w.inverse();

  SE3 T_cur_from_w = T_cur_from_actkey*T_actkey_from_w;

  for (typename ALIGNED<tr1::shared_ptr<CandidatePoint<Camera::obs_dim> > >
       ::list::const_iterator it = ap_map.begin(); it!=ap_map.end();++it)
  {
    const tr1::shared_ptr<CandidatePoint<Camera::obs_dim> > & ap
        = *it;

    Vector2d uv_pyr;
    SE3 T_anchorkey_from_w;
    bool is_prediction_valid = computePrediction(T_cur_from_w,
                                                 cam_vec,
                                                 ap,
                                                 vertex_map,
                                                 &uv_pyr,
                                                 &T_anchorkey_from_w);
    if (is_prediction_valid==false)
      continue;

    ALIGNED<QuadTreeElement<int> >::list candidates;
    Vector2i uvi_pyr = uv_pyr.cast<int>();
    int DOUBLE_SEARCHRADIUS = SEARCHRADIUS*2+1;
    feature_tree.at(ap->anchor_level).query(
          Rectangle(uvi_pyr[0]-SEARCHRADIUS, uvi_pyr[1]-SEARCHRADIUS,
                    DOUBLE_SEARCHRADIUS, DOUBLE_SEARCHRADIUS),
          &candidates);

    const cv::Mat & kf
        = GET_MAP_ELEM(ap->anchor_id, keyframe_map).pyr.at(ap->anchor_level);

//    Homography homo(T_cur_from_w*T_anchorkey_from_w.inverse());
//    cv::Mat key_patch_with_border
//        = warpPatchProjective(kf, homo, ap->xyz_anchor, ap->normal_anchor,
//                              ap->anchor_obs_pyr.head(2),
//                              cam_vec[ap->anchor_level], HALFBOX_SIZE+1);

    cv::Mat key_patch_with_border
        =   warpAffinve(kf, T_cur_from_w*T_anchorkey_from_w.inverse(),
                        ap->xyz_anchor[2], ap->anchor_obs_pyr.head(2),
                        cam_vec[ap->anchor_level], HALFBOX_SIZE+1);
    cv::Mat key_patch
        = key_patch_with_border(cv::Rect(cv::Point(1,1),
                                         cv::Size(BOX_SIZE, BOX_SIZE)));

    int pixel_sum = 0;
    int pixel_sum_square = 0;

    cv::Mat data_wrap(8,8, CV_8U, &(KEY_PATCH[0]));
    key_patch.copyTo(data_wrap);

    computePatchScores(&pixel_sum, &pixel_sum_square);

    if (pixel_sum*pixel_sum-pixel_sum_square
        <(int)(thr_std*thr_std*BOX_SIZE*BOX_SIZE))
      continue;


    MatchData match_data(thr_mean*thr_mean*BOX_SIZE*BOX_SIZE);
    matchCandidates(candidates, cur_frame, cam_vec,
                    pixel_sum, pixel_sum_square, ap->anchor_level,
                    &match_data);
    SE3 T_anchorkey_from_actkey = T_anchorkey_from_w*T_w_from_actkey;
    Vector3d xyz_actkey = T_anchorkey_from_actkey.inverse()*ap->xyz_anchor;
    returnBestMatch(key_patch_with_border, cur_frame, match_data,
                    xyz_actkey, ap, track_data);
  }
}

//TODO:
// - test affine wraper (e.g. for large in-plane rotations)
// - make it faster by precomputing relevant data
template <class Camera>
cv::Mat GuidedMatcher<Camera>
::warpAffinve                (const cv::Mat & frame,
                              const SE3 & T_c2_from_c1,
                              double depth,
                              const Vector2d & key_uv,
                              const Camera & cam,
                              int halfpatch_size)
{
  Vector2d f = cam.map(project2d(T_c2_from_c1*(depth*unproject2d(cam.unmap(key_uv)))));
  Vector2d f_pu = cam.map(project2d(T_c2_from_c1*(depth*unproject2d(cam.unmap(key_uv+Vector2d(1,0))))));
  Vector2d f_pv = cam.map(project2d(T_c2_from_c1*(depth*unproject2d(cam.unmap(key_uv+Vector2d(0,1))))));
  Matrix2d A;
  A.row(0) = f_pu - f; A.row(1) = f_pv - f;
  Matrix2d inv_A = A.inverse();

  int patch_size = halfpatch_size*2 ;
  cv::Mat ap_patch(patch_size,patch_size,CV_8UC1);

  for (int ix=0; ix<patch_size; ix++)
  {
    for (int iy=0; iy<patch_size; iy++)
    {
      Vector2d idx(ix-halfpatch_size,
                   iy-halfpatch_size);
      Vector2d r = inv_A*idx + key_uv;

      double x = floor(r[0]);
      double y = floor(r[1]);

      uint8_t val;
      if (x<0 || y<0 || x+1>=cam.width() || y+1>=cam.height())
        val = 0;
      else
      {
        double subpix_x = r[0]-x;
        double subpix_y = r[1]-y;
        double wx0 = 1-subpix_x;
        double wx1 =  subpix_x;
        double wy0 = 1-subpix_y;
        double wy1 =  subpix_y;

        double val00 = (frame).at<uint8_t>(y,x);
        double val01 = (frame).at<uint8_t>(y+1,x);
        double val10 = (frame).at<uint8_t>(y,x+1);
        double val11 = (frame).at<uint8_t>(y+1,x+1);
        val = uint8_t(min(255.,(wx0*wy0)*val00
                          + (wx0*wy1)*val01
                          + (wx1*wy0)*val10
                          + (wx1*wy1)*val11));
      }
      ap_patch.at<uint8_t>(iy,ix)= val;
    }
  }
  return ap_patch;
}


template <class Camera>
cv::Mat GuidedMatcher<Camera>
::warpPatchProjective(const cv::Mat & frame,
                      const Homography & homo,
                      const Vector3d & xyz_c1,
                      const Vector3d & normal_c1,
                      const Vector2d & key_uv,
                      const Camera & cam,
                      int halfpatch_size)
{
  Matrix3d H_cur_from_key
      = (cam.intrinsics()
         *homo.calc_c2_from_c1(normal_c1,xyz_c1)
         *cam.intrinsics_inv());

  int patch_size = halfpatch_size*2 ;
  Vector2d center_cur = project2d(H_cur_from_key * unproject2d(key_uv));

  Matrix3d H_key_from_cur = H_cur_from_key.inverse();
  cerr << H_key_from_cur << endl;
  cv::Mat ap_patch(patch_size,patch_size,CV_8UC1);

  for (int ix=0; ix<patch_size; ix++)
  {
    for (int iy=0; iy<patch_size; iy++)
    {
      Vector3d idx(center_cur[0]+ix-halfpatch_size,
                   center_cur[1]+iy-halfpatch_size,1);
      Vector2d r = (project2d(H_key_from_cur*idx));

      if (ix==0 && iy == 0)
      {
      //cerr << ix << " " << iy << endl;
        cerr << r << endl;
      }

      double x = floor(r[0]);
      double y = floor(r[1]);

      uint8_t val;
      if (x<0 || y<0 || x+1>=cam.width() || y+1>=cam.height())
        val = 0;
      else
      {
        double subpix_x = r[0]-x;
        double subpix_y = r[1]-y;
        double wx0 = 1-subpix_x;
        double wx1 =  subpix_x;
        double wy0 = 1-subpix_y;
        double wy1 =  subpix_y;

        double val00 = (frame).at<uint8_t>(y,x);
        double val01 = (frame).at<uint8_t>(y+1,x);
        double val10 = (frame).at<uint8_t>(y,x+1);
        double val11 = (frame).at<uint8_t>(y+1,x+1);
        val = uint8_t(min(255.,(wx0*wy0)*val00
                          + (wx0*wy1)*val01
                          + (wx1*wy0)*val10
                          + (wx1*wy1)*val11));
      }
      ap_patch.at<uint8_t>(iy,ix)= val;
    }
  }
  return ap_patch;
}

}
