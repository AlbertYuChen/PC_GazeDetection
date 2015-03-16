//
//  KalmanFilter.h
//  GazeDetection
//
//  Created by Chen Yu on 2/20/15.
//  Copyright (c) 2015 Chen Yu. All rights reserved.
//

#ifndef __GazeDetection__KalmanFilter__
#define __GazeDetection__KalmanFilter__

#include <stdio.h>



class KalmanFilter {
    double const A = 1.0;                   // State transition matrix.
    double const B = 0.0;                   // Control matrix.
    double const H = 1.0;                   // Observation matrix.
    double current_state_estimate = 0;          // Initial state estimate.
    double current_prob_estimate = 1;           // Initial covariance estimate.
    double const Q = 0.1;                   // Estimated error in process.
    double const R = 1;                     // Estimated error in measurements.
    
public:
    void step(double measurement_vector);
    
    double getcurrentstate();
    
};


#endif /* defined(__GazeDetection__KalmanFilter__) */
