//
// Created by 铨 on 2019/8/9.
//

#ifndef QSPROJECT1_CPP_DATASET_H
#define QSPROJECT1_CPP_DATASET_H

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <fstream>
#include <iostream>

class dataset
{
public:
    Eigen::MatrixXf X_train;
    Eigen::VectorXf y_train;
    Eigen::MatrixXf X_test;
    Eigen::VectorXf y_test;

    dataset(std::string filename, int feature_width, int height_train, int height_test);
    void shuffle();
    void normalize();
};

#endif //QSPROJECT1_CPP_DATASET_H
