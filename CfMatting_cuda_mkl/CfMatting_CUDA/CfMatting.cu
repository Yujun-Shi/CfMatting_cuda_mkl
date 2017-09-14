#include "CfMatting.h"

cv::Mat CfMatting(cv::String src_path, cv::String tri_path,
	const double eps, const int w_rad, const double lambda)
{
	// initialize the data
	cv::Mat src = cv::imread(src_path, CV_LOAD_IMAGE_COLOR);
	int imgSize = src.rows*src.cols;
	cv::Mat fore, all, res;
	setTRI(tri_path, src, all, fore);
	int *lapRowPtrs, *lapColInds, *diagInds, lapNnz;
	int *lhsRowPtrs, *lhsColInds, lhsNnz;
	double *lhsVals, *lapVals;
	double *d_all, *h_res;
	double alpha, beta;
	cusparseHandle_t cusparseHandle = NULL;
	CUDA_CHECK(cusparseCreate(&cusparseHandle));

	// work on lhs
	getCfLaplacianDCoo_gpu(cusparseHandle, eps, w_rad, src, &lapRowPtrs, &lapColInds, &lapVals, &lapNnz);

	createDiagXcsrInds(imgSize, &diagInds);

	CUDA_CHECK(cudaMalloc((void**)&d_all, sizeof(double)*imgSize));
	CUDA_CHECK(cudaMemcpy(d_all, (void*)all.data, sizeof(double)*imgSize, cudaMemcpyHostToDevice));

	alpha = 1;
	beta = lambda;
	cusparseAddDSpMat(cusparseHandle, lapRowPtrs, lapColInds,
		lapVals, lapNnz, diagInds, diagInds, d_all, imgSize,
		imgSize, imgSize, alpha, beta,
		&lhsRowPtrs, &lhsColInds, &lhsVals, &lhsNnz);

	CUDA_CHECK(cudaFree(d_all));
	CUDA_CHECK(cudaFree(lapRowPtrs));
	CUDA_CHECK(cudaFree(lapColInds));
	CUDA_CHECK(cudaFree(lapVals));
	CUDA_CHECK(cudaFree(diagInds));

	fore = fore * lambda;
	int *h_lhsRowPtrs = (int*)malloc(sizeof(int)*(imgSize + 1));
	int *h_lhsColInds = (int*)malloc(sizeof(int)*lhsNnz);
	double *h_lhsVals = (double*)malloc(sizeof(double)*lhsNnz);
	h_res = (double*)malloc(sizeof(double)*imgSize);
	CUDA_CHECK(cudaMemcpy(h_lhsRowPtrs, lhsRowPtrs, sizeof(int)*(imgSize + 1), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(h_lhsColInds, lhsColInds, sizeof(int)*lhsNnz, cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(h_lhsVals, lhsVals, sizeof(double)*lhsNnz, cudaMemcpyDeviceToHost));

	CUDA_CHECK(cudaFree(lhsVals));
	CUDA_CHECK(cudaFree(lhsRowPtrs));
	CUDA_CHECK(cudaFree(lhsColInds));

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