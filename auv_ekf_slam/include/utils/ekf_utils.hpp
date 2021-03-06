#ifndef EKF_UTILS_HPP
#define EKF_UTILS_HPP

#include <ros/ros.h>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

namespace utils{

    void updateMatrixBlock(const Eigen::MatrixXd& sigma_in, Eigen::MatrixXd& sigma_out, int lm_num);

    void addLMtoFilter(Eigen::VectorXd &mu_hat, Eigen::MatrixXd &Sigma_hat, const Eigen::Vector3d &landmark, const std::tuple<double, double, double> &sigma_new);

    void addLMtoMatrix(Eigen::MatrixXd &Sigma_hat, const std::tuple<double, double, double> &sigma_new);

    void removeLMfromFilter(Eigen::VectorXd &mu_hat, Eigen::MatrixXd &Sigma_hat, int j);

    double angleLimit (double angle);

    enum class MeasSensor { MBES, FLS };

    template <typename T> int sgn(T val) {
        return (T(0) < val) - (val < T(0));
    }
}

#endif // EKF_UTILS_HPP
