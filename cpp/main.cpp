#include "includes.h"
#define LINE cout<<string(65,'-')<<endl;

using namespace std;
using namespace cv;
using namespace Eigen;

int main() {
    std::cout << "Hello, World!" << std::endl;

    // Generating HoG features & those images
    //feature::generateImg("../FDDB-folds/");

    dataset face("../dataset.txt", 900, 23899, 6004);
    face.shuffle();

    // Linear Discriminant Analysis (Fisher Model)
    models::FisherModel FM;
    FM.fit(face.X_train, face.y_train);

    // Logistic Regression
    /*
    models::LogisticRegression LR;
    LR.fit(face.X_train, face.y_train);
    */

    return 0;
}