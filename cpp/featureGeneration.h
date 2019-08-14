//
// Created by 铨 on 2019/7/23.
//

#ifndef QSPROJECT1_CPP_FEATUREGENERATION_H
#define QSPROJECT1_CPP_FEATUREGENERATION_H

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <fstream>
#include <iostream>

namespace feature {
    void generateImg(std::string filename);
    Eigen::VectorXf getHoGFeature(cv::Mat img);
    cv::Mat cropImg(cv::Mat img, int roiTx, int roiTy, int roiBx, int roiBy);
};

#endif //QSPROJECT1_CPP_FEATUREGENERATION_H
