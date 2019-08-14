//
// Created by 铨 on 2019/8/8.
//
#include "models.h"
#define random(a,b) (rand()%(b-a)+a)

models::FisherModel::FisherModel()
{
    threshold = 0;
}

void models::FisherModel::predict(Eigen::MatrixXf X_test, Eigen::VectorXf y_test)
{
    Eigen::MatrixXf result = X_test * beta;
    for (int i = 0;i < y_test.size();i++)
    {
        if (result(i) >= threshold) result(i) = 1;
        else result(i) = 0;
    }

    std::cout << "Test Accuracy: " << (float)((result.array() == y_test.array()).count()) / (float)(y_test.size()) << std::endl;
}

void models::FisherModel::fit(Eigen::MatrixXf X_train, Eigen::VectorXf y_train)
{
    // Calculating mean value
    Eigen::MatrixXf mu0 = Eigen::MatrixXf::Zero(1, X_train.cols()), mu1 = Eigen::MatrixXf::Zero(1, X_train.cols());
    int cnt0, cnt1;
    for (int i = 0;i < y_train.size();i++)
    {
        if (y_train[i] == 0) {mu0 += X_train.row(i);cnt0++;}
        else {mu1 += X_train.row(i);cnt1++;}
    }

    mu0 = ((mu0.array()) / cnt0).matrix();
    mu1 = ((mu1.array()) / cnt1).matrix();
    std::cout << "mean value generated successfully" << std::endl;

    // Now we calculate the covarience of the matrix
    Eigen::MatrixXf cov = Eigen::MatrixXf::Zero(X_train.cols(), X_train.cols());
    Eigen::MatrixXf X_tmp = X_train;
    for (int i = 0;i < y_train.size();i++)
    {
        //std::cout << "generating cov" << std::endl;
        if (y_train[i] == 0) X_tmp.row(i) -= mu0;
        else X_tmp.row(i) -= mu1;
    }
    cov = X_tmp.transpose() * X_tmp;
    std::cout << "covariance value generated successfully" << std::endl;

    beta = (1e-18 * Eigen::MatrixXf::Identity(X_train.cols(), X_train.cols()) + cov).inverse() * (mu1 - mu0).transpose();
    threshold = (mu0 * beta + mu1 * beta).mean() / 2;

    predict(X_train, y_train);
}

models::LogisticRegression::LogisticRegression(float regularization, std::string optim, int iter, float toler,
                                               float lr)
{
    lambda = regularization;
    optimizer = optim;
    max_iter = iter;
    tolerance = toler;
    learning_rate = lr;
}

Eigen::ArrayXXf models::LogisticRegression::sigmoid(Eigen::ArrayXXf x)
{
    return x.exp() / (1.0 + x.exp());
}


void models::LogisticRegression::fit(Eigen::MatrixXf X_train, Eigen::VectorXf y_train)
{
    // Using Logistic Regression to fit the model
    static std::default_random_engine e(time(0));
    static std::normal_distribution<float> n(0.0, 0.5);

    // Initialization
    beta = Eigen::VectorXf::Zero(X_train.cols()).unaryExpr([](float dummy) {return n(e);});
    int batch_size = 32;
    float loss = 0.0f;
    Eigen::VectorXf grad = Eigen::VectorXf::Zero(X_train.rows());
    std::cout << "Training set size: " << X_train.rows() << 'x' << X_train.cols() << std::endl;

    for (int i = 0;i < max_iter;i++)
    {
        Eigen::ArrayXXf pred(y_train.size(), 1);
        pred = (X_train * beta).array();
        for (int i = 0;i < pred.size();i++)
        {
            if (pred(i) > 0.5) pred(i) = 1.0;
            else pred(i) = 0.0;
        }

        float accuracy = (float)((pred == y_train.array()).count()) / (float)(y_train.size());
        if (i % 10 == 0) std::cout << "Iteration: " << i << ", Accuracy: " << accuracy << std::endl;
        //std::cout << beta << std::endl;

        // Calculating loss here
        for (int j = 0;j < batch_size;j++)
        {
            srand((int)time(0));
            int batch_element = random(0, y_train.size());

            loss -= y_train[batch_element] * pred(batch_element);
            loss += log2f(1.0f + exp2f(X_train.row(batch_element) * beta));
        }
        loss += lambda * beta.squaredNorm();

        // Now update beta according to grad
        grad = X_train.transpose() * (sigmoid((X_train * beta).array()) - y_train.array()).matrix();
        grad += (grad.array() + 2 * lambda * beta.array()).matrix();
        if (optimizer == "sgd") beta = (beta.array() - learning_rate / 20 * grad.array()).matrix();
        grad = Eigen::VectorXf::Zero(X_train.rows());
        loss = 0;

        if (1 - accuracy < tolerance) break;
    }
}