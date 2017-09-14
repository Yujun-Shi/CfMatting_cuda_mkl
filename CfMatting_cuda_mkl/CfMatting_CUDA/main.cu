#include <stdlib.h>
#include "CfMatting.h"
#include <mkl.h>

int main(int argc, char *argv[]) {
	CUDA_CONTEXT_INIT;
	if (argc < 2) {
		exit(EXIT_FAILURE);
	}
	
	int w_rad = 1;
	double eps = 1e-7;
	double lambda = 100;

	cv::Mat alpha = CfMatting(argv[1], argv[2], eps, w_rad, lambda);

	cv::namedWindow("demo");
	cv::imshow("demo", alpha);
	cv::waitKey(0);
	return 0;
}
