#include <opencv2/opencv.hpp>

cv::Mat CfMatting(cv::String src_path, cv::String tri_path,
	const double eps, const int w_rad, const double lambda);
