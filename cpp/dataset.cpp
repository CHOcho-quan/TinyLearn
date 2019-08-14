//
// Created by 铨 on 2019/8/9.
//
#include "dataset.h"

dataset::dataset(std::string filename, int feature_width, int height_train, int height_test)
{
    float tmp;
    std::fstream in(filename, std::ios::in | std::ios::binary);
    X_train.resize(height_train, feature_width);
    X_test.resize(height_test, feature_width);
    y_train.resize(height_train);
    y_test.resize(height_test);

    for (int i = 0;i < height_train;i++)
    {
        for (int j = 0;j < feature_width;j++) {
            in >> tmp;
            X_train.row(i)[j] = tmp;
        }
        in >> tmp;
        y_train[i] = tmp;
    }

    for (int i = 0;i < height_test;i++)
    {
        for (int j = 0;j < feature_width;j++) {
            in >> tmp;
            X_test.row(i)[j] = tmp;
        }
        in >> tmp;
        y_test[i] = tmp;
    }

    std::cout << "Successfully generated dataset!" << std::endl;
}

void dataset::shuffle()
{
    // I don't know how to do currently
}

void dataset::normalize()
{
    X_train.normalized();
    X_test.normalized();
}