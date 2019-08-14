//
// Created by 铨 on 2019/8/8.
//

#ifndef QSPROJECT1_CPP_MODELS_H
#define QSPROJECT1_CPP_MODELS_H

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <fstream>
#include <ctime>
#include <random>
#include <iostream>

namespace models {
    class LogisticRegression {
    private:
        float lambda;
        std::string optimizer;
        int max_iter;
        float tolerance;
        std::vector<float> loss_history;
        float learning_rate;
        Eigen::VectorXf beta;
    public:
        LogisticRegression(float regularization = 1.0, std::string optim = "sgd", int iter = 500, float toler = 1e-5, float lr = 5e-4);
        void fit(Eigen::MatrixXf X_train, Eigen::VectorXf y_train);
        Eigen::ArrayXXf sigmoid(Eigen::ArrayXXf x);
        void predict(Eigen::MatrixXf X_test, Eigen::VectorXf y_test);
    };

    class FisherModel {
    private:
        Eigen::MatrixXf beta;
        float threshold;
    public:
        FisherModel();
        void fit(Eigen::MatrixXf X_train, Eigen::VectorXf y_train);
        void predict(Eigen::MatrixXf X_test, Eigen::VectorXf y_test);
    };
}

#endif //QSPROJECT1_CPP_MODELS_H
