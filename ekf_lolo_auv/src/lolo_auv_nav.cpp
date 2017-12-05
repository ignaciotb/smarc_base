#include "lolo_auv_nav/lolo_auv_nav.hpp"

LoLoEKF::LoLoEKF(std::string node_name, ros::NodeHandle &nh):node_name_(node_name), nh_(&nh){

    std::string imu_topic;
    std::string dvl_topic;
    std::string odom_topic;
    std::string gt_topic;

    nh_->param<std::string>((ros::this_node::getName() + "/imu_topic"), imu_topic, "/imu");
    nh_->param<std::string>((ros::this_node::getName() + "/dvl_topic"), dvl_topic, "/dvl");
    nh_->param<std::string>((ros::this_node::getName() + "/odom_pub_topic"), odom_topic, "/odom_ekf");
    nh_->param<std::string>((ros::this_node::getName() + "/gt_pose_topic"), gt_topic, "/gt_pose");
    nh_->param<std::string>((ros::this_node::getName() + "/odom_frame"), odom_frame_, "/odom");
    nh_->param<std::string>((ros::this_node::getName() + "/world_frame"), world_frame_, "/world");
    nh_->param<std::string>((ros::this_node::getName() + "/base_frame"), base_frame_, "/base_link");
    nh_->param<std::string>((ros::this_node::getName() + "/dvl_frame"), dvl_frame_, "/dvl_link");

    // Node connections
    imu_subs_ = nh_->subscribe(imu_topic, 1, &LoLoEKF::imuCB, this);
    dvl_subs_ = nh_->subscribe(dvl_topic, 1, &LoLoEKF::dvlCB, this);
    tf_gt_subs_ = nh_->subscribe(gt_topic, 1, &LoLoEKF::gtCB, this);
    odom_pub_ = nh_->advertise<nav_msgs::Odometry>(odom_topic, 10);
}


double LoLoEKF::angleLimit (double angle) const{ // keep angle within [-pi;pi)
        return std::fmod(angle + M_PI, (M_PI * 2)) - M_PI;
}

template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

double angleDiff (double angle_new, double angle_old) // return signed difference between new and old angle
{
    double diff = angle_new - angle_old;
    while (diff < -M_PI)
        diff += M_PI * 2;
    while (diff > M_PI)
        diff -= M_PI * 2;
    return diff;
}

// Freq is 50Hz
void LoLoEKF::imuCB(const sensor_msgs::ImuPtr &imu_msg){
    using namespace boost::numeric::ublas;
    boost::mutex::scoped_lock(msg_lock_);
    imu_readings_.push(imu_msg);
}

void LoLoEKF::gtCB(const nav_msgs::OdometryPtr &pose_msg){
    gt_readings_.push(pose_msg);
}

// Freq is 10Hz
void LoLoEKF::dvlCB(const geometry_msgs::TwistWithCovarianceStampedPtr &dvl_msg){
    dvl_readings_.push(dvl_msg);
}

bool LoLoEKF::sendOutput(ros::Time &t, tf::Quaternion &q_auv){

//    try{
        geometry_msgs::Quaternion odom_quat;
        tf::quaternionTFToMsg(q_auv, odom_quat);

        // Broadcast transform over tf
        geometry_msgs::TransformStamped odom_trans;
        odom_trans.header.stamp = t;
        odom_trans.header.frame_id = odom_frame_;
        odom_trans.child_frame_id = base_frame_;
        odom_trans.transform.translation.x = mu_(0);
        odom_trans.transform.translation.y = mu_(1);
        odom_trans.transform.translation.z = mu_(2);
        odom_trans.transform.rotation = odom_quat;
        odom_bc_.sendTransform(odom_trans);

        // Publish odom msg
        nav_msgs::Odometry odom_msg;
        odom_msg.header.stamp = t;
        odom_msg.header.frame_id = odom_frame_;
        odom_msg.child_frame_id = base_frame_;
        odom_msg.pose.pose.position.x = mu_(0);
        odom_msg.pose.pose.position.y = mu_(1);
        odom_msg.pose.pose.position.z = mu_(2);
        odom_msg.pose.pose.orientation = odom_quat;
        odom_pub_.publish(odom_msg);

        return true;
//    }
//    catch(){
//        ROS_ERROR("Odom update could not be sent");
//        return false;
//    }
}

void LoLoEKF::ekfLocalize(){
    ros::Rate rate(10);

    sensor_msgs::ImuPtr imu_msg;
    geometry_msgs::TwistWithCovarianceStampedPtr dvl_msg;
    nav_msgs::OdometryPtr gt_msg;

    double delta_t;
    double t_now;
    double t_prev;
    double dyaw;
    // TODO: full implementation of 6 DOF movement
    mu_ = boost::numeric::ublas::zero_vector<double>(6);

    tf::StampedTransform transf_dvl_base;
    tf::StampedTransform transf_world_odom;
    tf::TransformListener tf_listener;

    bool loop = true;
    bool init_filter = true;

    double theta;
    double theta_prev;
    double prev_imu_yaw;
    tf::Quaternion q_imu;
    tf::Quaternion q_world_odom;


    while(ros::ok()&& loop){
        ros::spinOnce();
        if(!dvl_readings_.empty() && !imu_readings_.empty() && !gt_readings_.empty()){

            // Init filter with initial, true pose (from GPS?)
            if(init_filter){
                ROS_INFO("Starting localization node");
                gt_msg = gt_readings_.back();
                t_prev = gt_msg->header.stamp.toSec();

                // Get fixed transform dvl_link --> base_link frame
                try {
                  tf_listener.lookupTransform(base_frame_, dvl_frame_, ros::Time(0), transf_dvl_base);
                  tf_listener.waitForTransform(base_frame_, dvl_frame_, ros::Time(0), ros::Duration(10.0) );
                  ROS_INFO("Locked transform dvl --> base");
                }
                catch(tf::TransformException &exception) {
                  ROS_ERROR("%s", exception.what());
                  ros::Duration(1.0).sleep();
                }

                // Get fixed transform world --> odom frame
                try {
                  tf_listener.lookupTransform(world_frame_, odom_frame_, ros::Time(0), transf_world_odom);
                  tf_listener.waitForTransform(world_frame_, odom_frame_, ros::Time(0), ros::Duration(10.0) );
                  q_world_odom = transf_world_odom.getRotation();
                  ROS_INFO("Locked transform world --> odom");
                }
                catch(tf::TransformException &exception) {
                  ROS_ERROR("%s", exception.what());
                  ros::Duration(1.0).sleep();
                }

                // Compute initial pose
                tf::Quaternion q_gt;
                tf::quaternionMsgToTF(gt_msg->pose.pose.orientation, q_gt);
                tf::Quaternion q_auv = q_world_odom * q_gt;
                q_auv.normalize();
                theta = tf::getYaw(q_auv);

                // Publish and broadcast
                this->sendOutput(gt_msg->header.stamp, q_auv);

                init_filter = false;
                continue;
            }

            // TODO: interpolate the faster sensors and adapt to slower ones
            // Integrate pose from dvl and imu
            imu_msg = imu_readings_.back();
            dvl_msg = dvl_readings_.back();
            gt_msg = gt_readings_.back();

            // Update time step
            t_now = dvl_msg->header.stamp.toSec();
            delta_t = t_now - t_prev;

            // Transform from dvl input form dvl --> base_link frame
            tf::Vector3 twist_vel(dvl_msg->twist.twist.linear.x,
                                  dvl_msg->twist.twist.linear.y,
                                  dvl_msg->twist.twist.linear.z);

            tf::Vector3 l_vel_base = transf_dvl_base.getBasis() * twist_vel;
            double w_z_base = transf_dvl_base.getOrigin().getX() * dvl_msg->twist.twist.linear.y;

            double disp = std::sqrt(pow((l_vel_base.y() * delta_t),2) +
                                    pow((l_vel_base.x() * delta_t),2));

            double dtheta = std::atan2((l_vel_base.y() * delta_t),
                                       (l_vel_base.x() * delta_t));

            // Transform IMU orientation from world to odom coordinates
            tf::quaternionMsgToTF(imu_msg->orientation, q_imu);
            q_world_odom.normalize();
            tf::Quaternion q_auv = q_world_odom * q_imu;
            q_auv.normalize();  // TODO: implement handling of singularities
            double imu_yaw = tf::getYaw(q_auv);

            theta += dtheta;
            theta = angleLimit(theta);

//            std::cout << "IMU time stamp: " << imu_msg->header.stamp.toSec() << " DVL time: " << dvl_msg->header.stamp.toSec()<< std::endl;

            // Update pose xt = xt-1 + ut
            mu_(0) += std::cos(theta) * disp;
            int sign = (sgn(std::sin(theta) * disp) == 1)? -1: 1;

            mu_(1) += std::sin(theta) * disp + sign * (w_z_base * delta_t) * transf_dvl_base.getOrigin().getX();
            mu_(2) = gt_msg->pose.pose.position.z - transf_world_odom.getOrigin().getZ(); // Imitate depth sensor input
            mu_(3) = imu_yaw;

            // Publish and broadcast
            this->sendOutput(gt_msg->header.stamp, q_auv);

            theta = imu_yaw;
            t_prev = t_now;

            imu_readings_.pop();
            dvl_readings_.pop();
            gt_readings_.pop();
        }
        else{
            ROS_INFO("Still waiting for some good, nice sensor readings...");
        }
        rate.sleep();
    }
}

LoLoEKF::~LoLoEKF(){
    // TODO_NACHO: do some cleaning here
}