//
//  KalmanFilter.cpp
//  GazeDetection
//
//  Created by Chen Yu on 2/20/15.
//  Copyright (c) 2015 Chen Yu. All rights reserved.
//

#include "KalmanFilter.h"

void  KalmanFilter::step(double measurement_vector){
    //---------------------------Prediction step-----------------------------
    double predicted_state_estimate = A * current_state_estimate;
    double predicted_prob_estimate = A * current_prob_estimate * A + Q;
    //--------------------------Observation step-----------------------------
    double innovation = measurement_vector - H * predicted_state_estimate;
    double innovation_covariance = H * predicted_prob_estimate * H + R;
    //-----------------------------Update step-------------------------------
    double kalman_gain = predicted_prob_estimate * H / innovation_covariance;
    current_state_estimate = predicted_state_estimate + kalman_gain * innovation;
    current_prob_estimate = (1 - kalman_gain * H) * predicted_prob_estimate;
}

double KalmanFilter::getcurrentstate(){
    return current_state_estimate;
}
