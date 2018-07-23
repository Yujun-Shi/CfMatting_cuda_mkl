#include "CfMatting.h"
#include "utils.h"
using namespace std;

cv::Mat CfMatting(cv::String src_path, cv::String tri_path,
	const double eps, const int w_rad, const double lambda)
{
	// initialize the data
	cv::Mat src = cv::imread(src_path, CV_LOAD_IMAGE_COLOR);
	int imgSize = src.rows*src.cols;
	cv::Mat fore, all, res;
    setTRI(tri_path, src, all, fore);
	int *d_lapRowPtrs, *d_lapColInds, *d_diagInds, lapNnz;
	int *d_lhsRowPtrs_t, *d_lhsColInds_t, *h_lhsRowPtrs, *h_lhsColInds, lhsNnz;
	double *d_lhsVals_t, *h_lhsVals, *d_lapVals;
	double *d_all, *h_res;
	double alpha, beta;
	cusparseHandle_t cusparseHandle = NULL;
	CUDA_CHECK(cusparseCreate(&cusparseHandle));

	// work on lhs
	getCfLaplacianDCoo_gpu(cusparseHandle, eps, w_rad, src, &d_lapRowPtrs, &d_lapColInds, &d_lapVals, &lapNnz);

	createDiagXcsrInds(imgSize, &d_diagInds);

	CUDA_CHECK(cudaMalloc((void**)&d_all, sizeof(double)*imgSize));
	CUDA_CHECK(cudaMemcpy(d_all, (void*)all.data, sizeof(double)*imgSize, cudaMemcpyHostToDevice));

	alpha = 1;
	beta = lambda;
	cusparseAddDSpMat(cusparseHandle, d_lapRowPtrs, d_lapColInds,
		d_lapVals, lapNnz, d_diagInds, d_diagInds, d_all, imgSize,
		imgSize, imgSize, alpha, beta,
		&d_lhsRowPtrs_t, &d_lhsColInds_t, &d_lhsVals_t, &lhsNnz);

	CUDA_CHECK(cudaFree(d_all));
	CUDA_CHECK(cudaFree(d_lapRowPtrs));
	CUDA_CHECK(cudaFree(d_lapColInds));
	CUDA_CHECK(cudaFree(d_lapVals));
	CUDA_CHECK(cudaFree(d_diagInds));

	fore = fore * lambda;
	h_lhsRowPtrs = (int*)malloc(sizeof(int)*(imgSize + 1));
	h_lhsColInds = (int*)malloc(sizeof(int)*lhsNnz);
	h_lhsVals = (double*)malloc(sizeof(double)*lhsNnz);
	h_res = (double*)malloc(sizeof(double)*imgSize);
	CUDA_CHECK(cudaMemcpy(h_lhsRowPtrs, d_lhsRowPtrs_t, sizeof(int)*(imgSize + 1), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(h_lhsColInds, d_lhsColInds_t, sizeof(int)*lhsNnz, cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(h_lhsVals, d_lhsVals_t, sizeof(double)*lhsNnz, cudaMemcpyDeviceToHost));

	CUDA_CHECK(cudaFree(d_lhsVals_t));
	CUDA_CHECK(cudaFree(d_lhsRowPtrs_t));
	CUDA_CHECK(cudaFree(d_lhsColInds_t));

	pardiso_solve_sym(imgSize, lhsNnz, h_lhsRowPtrs, h_lhsColInds, h_lhsVals, \
		(double*)fore.data, h_res);
    
	free(h_lhsColInds);
	free(h_lhsRowPtrs);
	free(h_lhsVals);
	#pragma omp parallel for
	for (int i = 0; i < imgSize; ++i) {
		if (h_res[i] > 1) {
			h_res[i] = 1;
		}
		else if (h_res[i] < 0) {
			h_res[i] = 0;
		}
	}

	CUDA_CHECK(cusparseDestroy(cusparseHandle));
	// it's a shallow copy here
	// so we don't free h_res
	res = cv::Mat(src.rows, src.cols, CV_64FC1, h_res);

	return res;
}

