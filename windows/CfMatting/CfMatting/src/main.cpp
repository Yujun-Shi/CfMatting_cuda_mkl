#include <iostream>
#include <string>
#include "CfMatting.h"
#include <mkl.h>
#include <sstream>

using namespace std;
using namespace cv;

int main(int argc, char* argv[]) {
	int w_rad = 1;
	double eps = 1e-9;
    double lambda = 100;
    string full_img_path = argv[1];
    string full_tri_path = argv[2];
    string write_path = argv[3];
    double a,b;
    a = second();
    Mat alpha = CfMatting(full_img_path, full_tri_path, eps, w_rad, lambda);
    b = second();
    cout << "total used time: " << b-a << "\n";
	imwrite(write_path, 255*alpha);
    return 0;
}
