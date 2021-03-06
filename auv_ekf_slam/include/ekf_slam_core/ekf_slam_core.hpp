#ifndef EKF_SLAM_CORE_HPP
#define EKF_SLAM_CORE_HPP

#include <ros/ros.h>

#include <eigen3/Eigen/Eigen>
#include <eigen3/Eigen/Dense>
#include <Eigen/SparseCore>
#include <tf/transform_datatypes.h>
#include <tf_conversions/tf_eigen.h>
#include <tf/tf.h>

#include "correspondence_class/correspondence_class.hpp"
#include "correspondence_class/correspondence_mbes.hpp"
#include "correspondence_class/correspondence_fls.hpp"

#include "noise_oneD_kf/noise_oneD_kf.hpp"

#include <queue>
#include <math.h>

#include <boost/thread/mutex.hpp>
#include <boost/assign.hpp>
#include <boost/bind.hpp>
#include <boost/scoped_ptr.hpp>

#include <boost/math/distributions/chi_squared.hpp>
#include <boost/math/distributions/inverse_chi_squared.hpp>

#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseArray.h>

#include "utils/ekf_utils.hpp"
#include "munkres/munkres.h"

class EKFCore{

public:

    EKFCore(Eigen::VectorXd& mu, Eigen::MatrixXd& Sigma, Eigen::MatrixXd& R, Eigen::MatrixXd& Q_fls, Eigen::MatrixXd &Q_mbes,
            double& lambda_fls, double& lambda_mbes, tf::StampedTransform& tf_base_sensor, const double mh_dist_fls, const double mh_dist_mbes);
    ~EKFCore();
    std::tuple<Eigen::VectorXd, Eigen::MatrixXd> ekfUpdate();
    void predictMotion(nav_msgs::Odometry odom_reading);
//    void dataAssociation(std::vector<Eigen::Vector3d> z_t, const utils::MeasSensor& sens_type);
    void batchDataAssociation(std::vector<Eigen::Vector3d> z_t, const utils::MeasSensor& sens_type);

private:

    // System state variables
    Eigen::VectorXd mu_;
    Eigen::MatrixXd Sigma_;
    Eigen::VectorXd mu_hat_;
    Eigen::MatrixXd Sigma_hat_;
    Eigen::Vector3d mu_auv_odom_;

    // Noise models
    Eigen::MatrixXd R_;
    Eigen::MatrixXd Q_fls_;
    Eigen::MatrixXd Q_mbes_;

    // Mapping variables
    tf::StampedTransform tf_base_sensor_;
    tf::Transform tf_sensor_base_;
    double lambda_mbes_;
    double lambda_fls_;
    double mh_dist_fls_;
    double mh_dist_mbes_;
    int lm_num_;

//    void predictMeasurement(const Eigen::Vector3d &landmark_j,
//                            const Eigen::Vector3d &z_i,
//                            unsigned int i,
//                            unsigned int j,
//                            const tf::Transform &transf_base_odom,
//                            const Eigen::MatrixXd &temp_sigma,
//                            h_comp h_comps,
//                            const utils::MeasSensor &sens_type,
//                            std::vector<CorrespondenceClass> &ml_i_list);

    void predictBatchMeasurement(const Eigen::Vector3d &landmark_j,
                                const Eigen::Vector3d &z_i,
                                unsigned int i,
                                unsigned int j,
                                const tf::Transform &transf_base_odom,
                                const Eigen::MatrixXd &temp_sigma,
                                h_comp h_comps,
                                const utils::MeasSensor &sens_type,
                                std::vector<CorrespondenceClass> &corresp_i_list,
                                Eigen::MatrixXd &corresp_table);

    void sequentialUpdate(const CorrespondenceClass &c_i_j, Eigen::MatrixXd temp_sigma);

//    void batchUpdate(std::vector<CorrespondenceClass> const& vec_cij, Eigen::MatrixXd temp_sigma);


};

#endif // EKF_SLAM_CORE_HPP
