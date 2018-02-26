#include "correspondence_class/correspondence_class.hpp"


CorrespondenceClass::CorrespondenceClass(const int &z_id, const double &lm_id){
    i_j_ = std::make_pair(z_id,lm_id);
}

void CorrespondenceClass::computeH(const boost::numeric::ublas::vector<double> &mu_hat,
                          const tf::Vector3 lm_odom, double N_t){

    using namespace std;
    boost::numeric::ublas::matrix<double> h_t = boost::numeric::ublas::identity_matrix<double>(3,9);

    // Store the landmark position
    this->landmark_pos_ = boost::numeric::ublas::vector<int>(3);
    this->landmark_pos_(0) = lm_odom.getX();
    this->landmark_pos_(1) = lm_odom.getY();
    this->landmark_pos_(2) = lm_odom.getZ();

    // Compute low-dimensional jacobian of measurement model
    h_t(0,0) = -cos(mu_hat(4))*cos(mu_hat(5));
    h_t(0,1) = -cos(mu_hat(4))*sin(mu_hat(5));
    h_t(0,2) = sin(mu_hat(4));
    h_t(0,3) = 0;
    h_t(0,4) = mu_hat(2)*cos(mu_hat(4)) - lm_odom.getZ()*cos(mu_hat(4)) - lm_odom.getX()*cos(mu_hat(5))*sin(mu_hat(4)) - lm_odom.getY()*sin(mu_hat(4))
            *sin(mu_hat(5)) + mu_hat(0)*cos(mu_hat(5))*sin(mu_hat(4)) + mu_hat(1)*sin(mu_hat(4))*sin(mu_hat(5));
    h_t(0,5) = cos(mu_hat(4))*(lm_odom.getY()*cos(mu_hat(5)) - lm_odom.getX()*sin(mu_hat(5)) - mu_hat(1)*cos(mu_hat(5)) + mu_hat(0)*sin(mu_hat(5)));
    h_t(0,6) = cos(mu_hat(4))*cos(mu_hat(5));
    h_t(0,7) = cos(mu_hat(4))*sin(mu_hat(5));
    h_t(0,8) = -sin(mu_hat(4));

    h_t(1,0) = cos(mu_hat(3))*sin(mu_hat(5)) - cos(mu_hat(5))*sin(mu_hat(4))*sin(mu_hat(3));
    h_t(1,1) = - cos(mu_hat(3))*cos(mu_hat(5)) - sin(mu_hat(4))*sin(mu_hat(3))*sin(mu_hat(5));
    h_t(1,2) = -cos(mu_hat(4))*sin(mu_hat(3));
    h_t(1,3) = lm_odom.getZ()*cos(mu_hat(4))*cos(mu_hat(3)) - mu_hat(2)*cos(mu_hat(4))*cos(mu_hat(3)) - lm_odom.getY()*cos(mu_hat(5))*sin(mu_hat(3))
            + lm_odom.getX()*sin(mu_hat(3))*sin(mu_hat(5)) + mu_hat(1)*cos(mu_hat(5))*sin(mu_hat(3)) - mu_hat(0)*sin(mu_hat(3))*sin(mu_hat(5))
            + lm_odom.getX()*cos(mu_hat(3))*cos(mu_hat(5))*sin(mu_hat(4)) + lm_odom.getY()*cos(mu_hat(3))*sin(mu_hat(4))*sin(mu_hat(5))
            - mu_hat(0)*cos(mu_hat(3))*cos(mu_hat(5))*sin(mu_hat(4)) - mu_hat(1)*cos(mu_hat(3))*sin(mu_hat(4))*sin(mu_hat(5));
    h_t(1,4) = -sin(mu_hat(3))*(lm_odom.getZ()*sin(mu_hat(4)) - mu_hat(2)*sin(mu_hat(4)) - lm_odom.getX()*cos(mu_hat(4))*cos(mu_hat(5))
              - lm_odom.getY()*cos(mu_hat(4))*sin(mu_hat(5)) + mu_hat(0)*cos(mu_hat(4))*cos(mu_hat(5)) + mu_hat(1)*cos(mu_hat(4))*sin(mu_hat(5)));
    h_t(1,5) = mu_hat(0)*cos(mu_hat(3))*cos(mu_hat(5)) - lm_odom.getY()*cos(mu_hat(3))*sin(mu_hat(5)) - lm_odom.getX()*cos(mu_hat(3))*cos(mu_hat(5))
            + mu_hat(1)*cos(mu_hat(3))*sin(mu_hat(5)) + lm_odom.getY()*cos(mu_hat(5))*sin(mu_hat(4))*sin(mu_hat(3))
            - lm_odom.getX()*sin(mu_hat(4))*sin(mu_hat(3))*sin(mu_hat(5)) - mu_hat(1)*cos(mu_hat(5))*sin(mu_hat(4))*sin(mu_hat(3))
            + mu_hat(0)*sin(mu_hat(4))*sin(mu_hat(3))*sin(mu_hat(5));
    h_t(1,6) = cos(mu_hat(5))*sin(mu_hat(4))*sin(mu_hat(3)) - cos(mu_hat(3))*sin(mu_hat(5));
    h_t(1,7) = cos(mu_hat(3))*cos(mu_hat(5)) + sin(mu_hat(4))*sin(mu_hat(3))*sin(mu_hat(5));
    h_t(1,8) = cos(mu_hat(4))*sin(mu_hat(3));

    h_t(2,0) = - sin(mu_hat(3))*sin(mu_hat(5)) - cos(mu_hat(3))*cos(mu_hat(5))*sin(mu_hat(4));
    h_t(2,1) = cos(mu_hat(5))*sin(mu_hat(3)) - cos(mu_hat(3))*sin(mu_hat(4))*sin(mu_hat(5));
    h_t(2,2) = -cos(mu_hat(4))*cos(mu_hat(3));
    h_t(2,3) = lm_odom.getX()*cos(mu_hat(3))*sin(mu_hat(5)) - lm_odom.getZ()*cos(mu_hat(4))*sin(mu_hat(3)) - lm_odom.getY()*cos(mu_hat(3))*cos(mu_hat(5))
            + mu_hat(1)*cos(mu_hat(3))*cos(mu_hat(5)) + mu_hat(2)*cos(mu_hat(4))*sin(mu_hat(3)) - mu_hat(0)*cos(mu_hat(3))*sin(mu_hat(5))
            - lm_odom.getX()*cos(mu_hat(5))*sin(mu_hat(4))*sin(mu_hat(3)) - lm_odom.getY()*sin(mu_hat(4))*sin(mu_hat(3))*sin(mu_hat(5))
            + mu_hat(0)*cos(mu_hat(5))*sin(mu_hat(4))*sin(mu_hat(3)) + mu_hat(1)*sin(mu_hat(4))*sin(mu_hat(3))*sin(mu_hat(5));
    h_t(2,4) = -cos(mu_hat(3))*(lm_odom.getZ()*sin(mu_hat(4)) - mu_hat(2)*sin(mu_hat(4)) - lm_odom.getX()*cos(mu_hat(4))*cos(mu_hat(5))
              - lm_odom.getY()*cos(mu_hat(4))*sin(mu_hat(5)) + mu_hat(0)*cos(mu_hat(4))*cos(mu_hat(5)) + mu_hat(1)*cos(mu_hat(4))*sin(mu_hat(5)));
    h_t(2,5) = lm_odom.getX()*cos(mu_hat(5))*sin(mu_hat(3)) + lm_odom.getY()*sin(mu_hat(3))*sin(mu_hat(5)) - mu_hat(0)*cos(mu_hat(5))*sin(mu_hat(3))
            - mu_hat(1)*sin(mu_hat(3))*sin(mu_hat(5)) + lm_odom.getY()*cos(mu_hat(3))*cos(mu_hat(5))*sin(mu_hat(4))
            - lm_odom.getX()*cos(mu_hat(3))*sin(mu_hat(4))*sin(mu_hat(5)) - mu_hat(1)*cos(mu_hat(3))*cos(mu_hat(5))*sin(mu_hat(4))
            + mu_hat(0)*cos(mu_hat(3))*sin(mu_hat(4))*sin(mu_hat(5));
    h_t(2,6) = sin(mu_hat(3))*sin(mu_hat(5)) + cos(mu_hat(3))*cos(mu_hat(5))*sin(mu_hat(4));
    h_t(2,7) = cos(mu_hat(3))*sin(mu_hat(4))*sin(mu_hat(5)) - cos(mu_hat(5))*sin(mu_hat(3));
    h_t(2,8) = cos(mu_hat(4))*cos(mu_hat(3));

    ROS_INFO("h_t Done!");

    // Construct F_x projection matrix
    boost::numeric::ublas::matrix<double> F_x_k = boost::numeric::ublas::zero_matrix<double>(9, 6 + 3*N_t);
    boost::numeric::ublas::subrange(F_x_k, 0, 6, 0, 6) = boost::numeric::ublas::identity_matrix<double>(6);
    boost::numeric::ublas::subrange(F_x_k, 6, 9, i_j_.second, i_j_.second+3) = boost::numeric::ublas::identity_matrix<double>(3);
    H_t_ = boost::numeric::ublas::prod(h_t, F_x_k);

}

void CorrespondenceClass::computeS(const boost::numeric::ublas::matrix<double> &sigma,
                          const boost::numeric::ublas::matrix<double> &Q){
    // Intermidiate steps
    boost::numeric::ublas::matrix<double> mat = boost::numeric::ublas::prod(H_t_, sigma);
    boost::numeric::ublas::matrix<double> mat1 = boost::numeric::ublas::trans(H_t_);
    // Computation of S
    S_ = boost::numeric::ublas::prod(mat, mat1);
    S_ += Q;
}

void CorrespondenceClass::computeNu(const boost::numeric::ublas::vector<double> &z_hat_i,
                           const boost::numeric::ublas::vector<double> &z_i){
    nu_ = z_i - z_hat_i;
}

void CorrespondenceClass::computeLikelihood(){
    using namespace boost::numeric::ublas;

    S_inverted_ = matrix<double> (S_.size1(), S_.size2());
    bool inverted = matrices::InvertMatrix(S_, S_inverted_);
    if(!inverted){
        ROS_ERROR("Error inverting S");
        return;
    }

    // Compute Mahalanobis distance (z_i, z_hat_j)
    vector<double> aux = prod(trans(nu_), S_inverted_);
    d_m_ = inner_prod(trans(aux), nu_);

    // Calculate the determinant on the first member of the distribution
    matrix<double> mat_aux = 2 * M_PI_2 * S_;
    double det_mat = matrices::matDeterminant(mat_aux);

    // Likelihood
    psi_ = (1 / (std::sqrt(det_mat))) * std::exp(-0.5 * d_m_);
}

double angleLimit (double angle){ // keep angle within [-pi;pi)
        return std::fmod(angle + M_PI, (M_PI * 2)) - M_PI;
}
