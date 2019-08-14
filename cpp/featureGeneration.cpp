//
// Created by 铨 on 2019/7/23.
//
#include "featureGeneration.h"

Eigen::VectorXf feature::getHoGFeature(cv::Mat img)
{
    cv::HOGDescriptor hogDescriptor(cv::Size(96, 96), cv::Size(32, 32), cv::Size(16, 16), cv::Size(16, 16), 9);

    // Calculate HoG feature
    std::vector<float> HoGs;
    hogDescriptor.compute(img, HoGs, cv::Size(0, 0));

    Eigen::VectorXf tmp = Eigen::VectorXf::Map(&HoGs[0], HoGs.size());
    return tmp;
}

cv::Mat feature::cropImg(cv::Mat img, int roiTx, int roiTy, int roiBx, int roiBy)
{
    cv::Mat faceCut(img, cv::Rect(roiTx>0?roiTx:0, roiTy>0?roiTy:0,
                              roiBx<img.cols?roiBx-(roiTx>0?roiTx:0):img.cols-(roiTx>0?roiTx:0)-1,
                              roiBy<img.rows?roiBy-(roiTy>0?roiTy:0):img.rows-(roiTy>0?roiTy:0)-1));

    if (roiTx < 0) cv::copyMakeBorder(faceCut, faceCut, 0, 0, -roiTx, 0, cv::BORDER_REPLICATE);
    if (roiTy < 0) cv::copyMakeBorder(faceCut, faceCut, -roiTy, 0, 0, 0, cv::BORDER_REPLICATE);
    if (roiBx > img.cols) cv::copyMakeBorder(faceCut, faceCut, 0, 0, 0, roiBx - img.cols, cv::BORDER_REPLICATE);
    if (roiBy > img.rows) cv::copyMakeBorder(faceCut, faceCut, 0, roiBy - img.rows, 0, 0, cv::BORDER_REPLICATE);

    cv::resize(faceCut, faceCut, cv::Size(96, 96));
    return faceCut;
}

void feature::generateImg(std::string filename) {
    std::string name, prefix = "../originalPics/", suffix = "-ellipseList.txt", line, tmp;
    int cm = 0;
    std::vector<Eigen::VectorXf> X_train, X_test;
    std::vector<float> y_train, y_test;

    // Generating positive samples
    for (int j = 1;j < 11;j++)
    {
        char temp[100];
        sprintf(temp, "FDDB-fold-%d%d", j/10, j%10);
        name = temp;
        name = filename + name + suffix;
        if (j == 9) cm = 0;

        std::cout << "Generating Positive " << name << std::endl;

        // Now we read those files and get the images
        std::fstream file(name);
        std::string picName, picSuffix = ".jpg";
        int faceNum;
        float a, b, angle, x, y, _;
        while (getline(file, line))
        {
            //std::cout << "Reading " << prefix + line + picSuffix << std::endl;
            picName = prefix + line + picSuffix;
            cv::Mat img = cv::imread(prefix + line + picSuffix);
            cm++;

            // Getting Face Num
            getline(file, line);
            const char* num = line.data();
            faceNum = atoi(num);

            // Now getting each of the face and put it into our folder
            for (int i = 0;i < faceNum;i++)
            {
                file >> a >> b >> angle >> x >> y >> _;
                getline(file, line);
                //std::cout << a << ' ' << b << ' ' << x << ' ' << y << ' ' << _ << std::endl;

                int roiTx, roiTy, roiBx, roiBy;
                roiTx = (int)(x-4*b/3);
                roiTy = (int)(y-4*a/3);
                roiBx = (int)(x+4*b/3);
                roiBy = (int)(y+4*a/3);

                cv::Mat faceCut = feature::cropImg(img, roiTx, roiTy, roiBx, roiBy);
                // Getting the HoG feature
                Eigen::VectorXf HoG = feature::getHoGFeature(faceCut);

                if (j < 9) {
                    sprintf(temp, "../refinedPics/train_pos/%d_%d.jpg", cm, i);
                    X_train.push_back(HoG);
                    y_train.push_back(1);
                }
                else {
                    sprintf(temp, "../refinedPics/test_pos/%d_%d.jpg", cm, i);
                    X_test.push_back(HoG);
                    y_test.push_back(1);
                }

                cv::imwrite(temp, faceCut);
            }
        }
    }

    // Generating negative samples
    for (int j = 1;j < 11;j++)
    {
        char temp[100];
        if (j == 5) j = 10;
        if (j == 10) cm = 0;
        sprintf(temp, "FDDB-fold-%d%d", j/10, j%10);
        name = temp;
        name = filename + name + suffix;

        std::cout << "Generating Negative " << name << std::endl;

        // Now we read those files and get the images
        std::fstream file(name);
        std::string picName, picSuffix = ".jpg";
        int faceNum;
        float a, b, angle, x, y, _;
        while (getline(file, line))
        {
            //std::cout << "Reading " << prefix + line + picSuffix << std::endl;
            picName = prefix + line + picSuffix;
            cv::Mat img = cv::imread(prefix + line + picSuffix);
            cm++;

            // Getting Face Num
            getline(file, line);
            const char* num = line.data();
            faceNum = atoi(num);

            cv::Mat faceCut;
            cv::resize(img, faceCut, cv::Size(96, 96));
            // Getting the HoG feature
            Eigen::VectorXf HoG = feature::getHoGFeature(faceCut);
            if (j < 9) {
                sprintf(temp, "../refinedPics/train_neg/%d_%d.jpg", cm, 100);
                X_train.push_back(HoG);
                y_train.push_back(0);
            }
            else {
                sprintf(temp, "../refinedPics/test_neg/%d_%d.jpg", cm, 100);
                X_test.push_back(HoG);
                y_test.push_back(0);
            }

            cv::imwrite(temp, faceCut);

            // Now getting each of the face and put it into our folder
            for (int i = 0;i < faceNum;i++)
            {
                file >> a >> b >> angle >> x >> y >> _;
                getline(file, line);
                //std::cout << a << ' ' << b << ' ' << x << ' ' << y << ' ' << _ << std::endl;

                for (int k1 = -1;k1 < 2;k1++)
                {
                    for (int k2 = -1;k2 < 2;k2++)
                    {
                        int roiTx, roiTy, roiBx, roiBy;
                        roiTx = (int)(x-4*b/3-8*b/9*k1);
                        roiTy = (int)(y-4*a/3-8*a/9*k2);
                        roiBx = (int)(x+4*b/3-8*b/9*k1);
                        roiBy = (int)(y+4*a/3-8*a/9*k2);

                        cv::Mat faceCut = feature::cropImg(img, roiTx, roiTy, roiBx, roiBy);
                        // Getting the HoG feature
                        Eigen::VectorXf HoG = feature::getHoGFeature(faceCut);

                        if (j < 9) {
                            sprintf(temp, "../refinedPics/train_neg/%d_%d_%d%d.jpg", cm, i, k1, k2);
                            X_train.push_back(HoG);
                            y_train.push_back(0);
                        }
                        else {
                            sprintf(temp, "../refinedPics/test_neg/%d_%d_%d%d.jpg", cm, i, k1, k2);
                            X_test.push_back(HoG);
                            y_test.push_back(0);
                        }

                        cv::imwrite(temp, faceCut);
                    }
                }
            }
        }
    }

    std::cout << "X_Train: " << X_train[0].size() << ' ' << X_train.size() << std::endl;
    std::cout << "X_Test: " << X_test[0].size() << ' ' << X_test.size() << std::endl;

    std::fstream out("../dataset.txt", std::ios::out | std::ios::binary);
    for (int i = 0;i < 23899;i++)
    {
        for (int j = 0;j < 900;j++) out << X_train[i][j] << ' ';
        out << y_train[i] << '\n';
    }
    for (int i = 0;i < 6004;i++)
    {
        for (int j = 0;j < 900;j++) out << X_test[i][j] << ' ';
        out << y_test[i] << '\n';
    }
}

