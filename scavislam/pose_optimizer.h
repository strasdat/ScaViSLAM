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

#ifndef SCAVISLAM_POSE_OPTIMIZER_H
#define SCAVISLAM_POSE_OPTIMIZER_H

#include <vector>
#include <map>

#include <visiontools/accessor_macros.h>
#include <visiontools/stopwatch.h>

#include "global.h"
#include "maths_utils.h"
#include "transformations.h"

//TODO: separate interface from implementation, rename class, clean

namespace ScaViSLAM
{
/**
     * Parameter class for BundleAdjuster (see below)
     */
class PoseOptimizerParams
{
public:
  PoseOptimizerParams(bool robust_kernel=true,
                       double kernel_param=1,
                       int num_iter=50,
                       double initial_mu=-1):
    robust_kernel(robust_kernel),
    kernel_param(kernel_param),
    num_iter(num_iter),
    initial_mu(initial_mu),
    tau(0.00001)
  {
  }
  bool robust_kernel;
  double kernel_param;
  int num_iter;
  double initial_mu;
  double final_mu_factor;
  double tau;
};

class OptimizerStatistics
{
public:

  OptimizerStatistics(){}

  OptimizerStatistics(const OptimizerStatistics & other) : chi2(other.chi2),
    max_err(other.max_err),
    num_obs(other.num_obs){}

  OptimizerStatistics(double initial_chi2,
                      double chi2,
                      double max_err,
                      int num_obs)
    : initial_chi2(initial_chi2),
      chi2(chi2),
      max_err(max_err),
      num_obs(num_obs)
  {
  }

  double rmse()
  {
    return sqrt(chi2/num_obs);
  }

  double initial_rmse()
  {
    return sqrt(initial_chi2/num_obs);
  }

  double initial_chi2;
  double chi2;
  double max_err;
  int num_obs;
};

/**
   *
   * Frame: How is the frame/pose represented? (e.g. SE3)
   * FrameDoF: How many DoF has the pose/frame? (e.g. 6 DoF, that is
   *           3 DoF translation, 3 DoF rotation)
   * PointParNum: number of parameters to represent a point
   *              (4 for a 3D homogenious point)
   * PointDoF: DoF of a point (3 DoF for a 3D homogenious point)
   * ObsDim: dimensions of observation (2 dim for (u,res) image
   *         measurement)
   */
template <typename Frame,
          int FrameDoF,
          typename Obs,
          int ObsDim>
class PoseOptimizer
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  static const int PointParNum = 3;
  static const int PointDoF = 3;
  typedef AbstractPrediction
  <Frame,FrameDoF,Matrix<double,PointParNum,1>,PointDoF,ObsDim>
  _AbsJac;
  typedef vector<Vector3d> _PointVec;

  typedef typename ALIGNED<Obs>::list  _Track;
  typedef typename ALIGNED<_Track>::int_map  _TrackMap;

  PoseOptimizer(){
    verbose = 0;
  }

  int verbose;

  //TODO: method too long
  OptimizerStatistics
  calcFastMotionOnly(const typename ALIGNED<Obs>::list & obs_list,
                     const _AbsJac & prediction,
                     const PoseOptimizerParams & ba_params,
                     Frame  * frame,
                     _PointVec * point_list)
  {
    assert(obs_list.size()>0);
    double nu = 2;
    bool stop = false;
    double chi2 = 0;
    double max_err=0;
    int num_obs=0;

    Matrix<double,ObsDim,1> mean;
    mean.setZero();

    Frame  new_frame = *frame;
    double norm_max_A = 0.;
    for (typename std::list<Obs>
         ::const_iterator it_cam = obs_list.begin();
         it_cam != obs_list.end();
         ++it_cam)
    {
      int point_id = it_cam->point_id;

      Matrix<double,PointDoF,1> & point
          = GET_VEC_VAL_REF(point_id, point_list);
      const Obs & id_obs = *it_cam;
      Matrix<double,ObsDim,FrameDoF> J_c;
      Matrix<double,ObsDim,1> f =
          id_obs.obs - prediction.map_n_frameJac(*frame, point, J_c);
      mean += f;
      norm_max_A = max(norm_max_A,
                       norm_max(mulW(J_c,J_c,id_obs).diagonal()));
      if(ba_params.robust_kernel)
      {
        double nrm = std::max(EPS,
                              sqrt(sqrW(f,id_obs)));
        double w = sqrt(kernel(nrm,ba_params.kernel_param))/nrm;
        f *= w;
      }
      chi2 += sqrW(f,id_obs);
      ++num_obs;
      max_err = max(norm_max(f),max_err);
    }
    mean/=num_obs;
    double initial_chi2 = chi2;
    double mu = ba_params.initial_mu;
    if (verbose>0)
    {
      std::cout << "init. chi2: "<< chi2 << std::endl;
    }
    if (ba_params.initial_mu==-1)
    {
      mu = ba_params.tau*norm_max_A;
    }
    int trial = 0;

    for (int i_g=0; i_g<ba_params.num_iter; i_g++)
    {
      if (verbose>0)
      {
        std::cout << "iteration: "<< i_g << std::endl;
      }
      double rho = 0; //Assign value, so the compiler is not complaining...
      do{

        if (verbose>0)
        {
          std::cout << "mu: " <<mu<< std::endl;
        }
        typename VECTOR<FrameDoF>::col B = Matrix<double,FrameDoF,1>::Zero();
        Matrix<double,FrameDoF,FrameDoF> A
            = mu*Matrix<double,FrameDoF,FrameDoF>::Identity();
        for (typename std::list<Obs>
             ::const_iterator it_cam = obs_list.begin();
             it_cam != obs_list.end();
             ++it_cam)
        {
          int point_id = it_cam->point_id;
          typename VECTOR<PointDoF>::col & point
              = GET_VEC_VAL_REF(point_id, point_list);
          const Obs & id_obs = *it_cam;
          Matrix<double,ObsDim,FrameDoF> J_c;

          typename VECTOR<ObsDim>::col f = id_obs.obs
              - prediction.map_n_frameJac(*frame,
                                          point,
                                          J_c);
          if(ba_params.robust_kernel)
          {
            double nrm = std::max(EPS, sqrt(sqrW(f,id_obs)));
            double w = sqrt(kernel(nrm,ba_params.kernel_param))/nrm;
            f *= w;
          }
          A += mulW(J_c,J_c,id_obs);
          B.head(FrameDoF) -= mulW(J_c,f,id_obs);
        }
        typename VECTOR<FrameDoF>::col delta
            = Matrix<double,FrameDoF,1>::Zero();
        delta = A.ldlt().solve(B);
        double new_chi2 = 0;
        double new_max_err=0;
        new_frame = prediction.add(*frame,delta);
        mean.setZero();
        new_chi2 = 0;
        new_max_err=0;
        for (typename std::list<Obs>
             ::const_iterator it_cam = obs_list.begin();
             it_cam != obs_list.end();
             ++it_cam)
        {
          int point_id = it_cam->point_id;
          typename VECTOR<PointDoF>::col & point
              = GET_VEC_VAL_REF(point_id, point_list);
          const Obs & id_obs = *it_cam;
          typename VECTOR<ObsDim>::col f
              = id_obs.obs - prediction.map(new_frame, point);
          mean += f;
          if(ba_params.robust_kernel)
          {
            double nrm = std::max(EPS, sqrt(sqrW(f,id_obs)));
            double w = sqrt(kernel(nrm,ba_params.kernel_param))/nrm;
            f *= w;
          }
          new_chi2 += sqrW(f,id_obs);
          new_max_err = max(norm_max(f),new_max_err);
        }
        mean/=num_obs;
        if (isnan(new_chi2))
        {
          throw std::runtime_error("Res is NaN!");
        }
        rho = (chi2-new_chi2);
        if(rho>0)
        {
          *frame = new_frame;
          chi2 = new_chi2;
          max_err = new_max_err;
          stop = norm_max(B)<=EPS;
          mu *= std::max(1./3.,1-Po3(2*rho-1));
          nu = 2.;
          trial = 0;
        }
        else
        {
          if (verbose>0)
            std::cout << "no update: chi vs. new_chi2 "
                      << chi2
                      << " vs. "
                      << new_chi2
                      << std::endl;
          mu *= nu;
          nu *= 2.;
          ++trial;
          if (trial==5)
            stop = true;
        }
      }while(!(rho>0 ||  stop));
      if (stop)
        break;
    }
    return OptimizerStatistics(initial_chi2,chi2,max_err,num_obs);
  }

#if MONO
  /**
     * This function implements an information filter for a single landmark
     * given known frame/pose. It can be used for a landmark initialisation
     * (without any depth prior) within a key-frame BA apporach as described
     * in:
     *
     * > H. Strasdat, J.M.M. Montiel, A.J. Davison:
     *   "Scale Drift-Aware Large Scale Monocular SLAM",
     *   Proc. of Robotics: Science and Systems (RSS),
     *   Zaragoza, Spain, 2010.
     *   http://www.roboticsproceedings.org/rss06/p10.html <
     *
     * frame: pose/frame
     * point: 3D point/landmark (normally in inverse depth coordinates)
     * Lambda: 3D point/landmark inverse covariance
     * prediction: prediction class
     * obs_vec: set of observations
     * ba_params: BA parameters
     */
  //TODO: method too long
  double
  filterSingleFeatureOnly(const Frame & frame,
                          Matrix<double,PointDoF,1>  & point,
                          Matrix<double,PointDoF,PointDoF> & Lambda,
                          _AbsJac& prediction,
                          const Matrix<double,ObsDim,1> & obs,
                          const BundleAdjusterParams & ba_params)
  {
    typename VECTOR<PointDoF>::col point_mean = point;
    typename VECTOR<ObsDim>::col residuals;
    Matrix<double,PointDoF,1> residuals_dist(PointDoF);
    typename VECTOR<PointDoF>::col  new_point;
    Matrix<double,ObsDim,PointDoF> J_point = prediction.pointJac(frame, point);
    typename VECTOR<ObsDim>::col delta_obs = obs - prediction.map(frame, point);
    double res =  delta_obs.dot(delta_obs);
    residuals = delta_obs;
    VectorXd  diff = (point_mean-point);
    VectorXd  LambdaDiff = Lambda*diff;
    residuals_dist = LambdaDiff;
    res += diff.dot(LambdaDiff);
    if (isnan(res))
    {
      throw std::runtime_error("Res is NaN!");
    }
    if(verbose>0)
      std::cout << "res: " << res << std::endl;
    Matrix<double,PointDoF,PointDoF> V = J_point.transpose() * J_point;
    typename VECTOR<PointDoF>::col g = -J_point.transpose() * residuals;
    g += residuals_dist;
    double nu = 2;
    bool stop = false;
    double mu = ba_params.initial_mu;
    if (ba_params.initial_mu==-1)
    {
      double norm_max_A = norm_max(V.diagonal());
      mu = ba_params.tau*norm_max_A;
    }
    double final_mu = ba_params.final_mu_factor*mu;
    for (int i_g=0; i_g<ba_params.num_iter; i_g++)
    {
      if (verbose>0)
      {
        std::cout << "iteration: "<< i_g << std::endl;
      }
      double rho = 0; //Assign value, so the compiler is not complaining...
      do{
        if (verbose>0)
        {
          std::cout << "mu: " <<mu<< std::endl;
        }
        Matrix<double,PointDoF,PointDoF> H = Lambda+ V;
        H +=  Matrix<double,PointDoF,PointDoF>::Identity()*mu ;
        Matrix<double,PointDoF,1>  delta = H.lu().solve(g);
        new_point = prediction.add(point, delta);
        Matrix<double,ObsDim,1> delta_obs
            = obs - prediction.map(frame, new_point);
        double res_new =  delta_obs.dot(delta_obs);
        residuals = delta_obs;
        Matrix<double,PointDoF,1> diff = (point_mean-new_point);
        Matrix<double,PointDoF,1> LambdaDiff = Lambda*diff;
        residuals_dist  = LambdaDiff;
        res_new += diff.dot(LambdaDiff);
        if (isnan(res_new))
        {
          throw std::runtime_error("Res is NaN!");
        }
        rho = (res-res_new);
        if(rho>0)
        {
          if(verbose>0)
            std::cout << "res_new: " << res_new << std::endl;
          point= new_point;
          res = res_new;
          J_point = -prediction.pointJac(frame, point);
          V = J_point.transpose() * J_point;
          g= J_point.transpose() * residuals;
          g += residuals_dist;

          stop = norm_max(g)<=EPS;
          mu *= std::max(1./3.,1-Po3(2*rho-1));
          nu = 2.;
        }
        else
        {
          if (verbose>0)
            std::cout << "no update: res vs.res_new "
                      << res << " vs. " << res_new << std::endl;
          mu *= nu;
          nu *= 2.;
          if (verbose)
            std::cout << "mu" << mu << std::endl;
          stop = (mu>final_mu);
        }

      }while(!(rho>0 ||  stop));
      if (stop)
        break;
    }
    Lambda += V;
    return res;
  }
#endif

protected:

  /** pseudo-huber cost function */
  double
  kernel(double delta, double b)
  {
    double delta_abs = fabs(delta);
    if (delta_abs<b)
      return delta*delta;
    else
      return 2*b*delta_abs - b*b;
  }

  template <int Trans1DoF, int Trans2DoF>
  Matrix<double,Trans1DoF,Trans2DoF>
  mulW(const Matrix<double,ObsDim,Trans1DoF> & J_trans1,
       const Matrix<double,ObsDim,Trans2DoF> &  J_trans2,
       const IdObs<ObsDim> & obs )
  {
    return J_trans1.transpose() * J_trans2;
  }

  template <int Trans1DoF, int Trans2DoF>
  Matrix<double,Trans1DoF,Trans2DoF>
  mulW(const Matrix<double,ObsDim,Trans1DoF> & J_trans1,
       const Matrix<double,ObsDim,Trans2DoF> & J_trans2,
       const IdObsLambda<ObsDim> & obs )
  {
    return J_trans1.transpose() * obs.lambda *J_trans2;
  }

  template <int TransDoF>
  Matrix<double,TransDoF,1>
  mulW(const Matrix<double,ObsDim,TransDoF> & J_trans,
       const Matrix<double,ObsDim,1> & f,
       const IdObs<ObsDim> & obs )
  {
    return J_trans.transpose() * f;
  }

  template <int TransDoF>
  inline Matrix<double,TransDoF,1>
  mulW(const Matrix<double,ObsDim,TransDoF> & J_trans,
       const Matrix<double,ObsDim,1> & f,
       const IdObsLambda<ObsDim> & obs )
  {
    return J_trans.transpose() * obs.lambda * f;
  }

  double
  sqrW(const Matrix<double,ObsDim,1> & f,
       const IdObs<ObsDim> & obs )
  {
    return f.dot(f);
  }

  double
  sqrW(const Matrix<double,ObsDim,1> & f,
       const IdObsLambda<ObsDim> & obs )
  {
    return f * obs.lambda * f;
  }

};

typedef PoseOptimizer<SE3,6,IdObs<2>,2> BA_SE3_XYZ;
typedef PoseOptimizer<SE3,6,IdObs<2>,2> BA_SE3_XYZ_Lambda;
typedef PoseOptimizer<SE3,6,IdObs<3>,3> BA_SE3_XYZ_STEREO;

#ifdef MONO
typedef BundleAdjuster<Sim3,6,IdObs<2>,2> BA_Sim3_XYZ;
typedef BundleAdjuster<Sim3,7,IdObs<3>,3> BA_Sim3_XYZ_STEREO;
#endif

}


#endif
