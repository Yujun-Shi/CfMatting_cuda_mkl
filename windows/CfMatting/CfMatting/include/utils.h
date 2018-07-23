#include <device_functions.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math_functions.h>

#include <cusparse.h>

#include <opencv2/opencv.hpp>
#include <omp.h>

#include <helper_cuda.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

#include "mkl.h"
// #include "mkl_pardiso.h"
// #include "mkl_types.h"

#include <iostream>

#define  CAFFE_CUDA_NUM_THREADS 512

#define CUDA_CHECK checkCudaErrors

#define ALONG_X 0
#define ALONG_Y 1

inline int CAFFE_GET_BLOCKS(const int n) {
	return (n + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
}

cv::Mat createIndsMat(const int imrows, const int imcols);

template<typename Dtype>
void sumRowCsrSym(const int* rowPtrs, const Dtype* vals, 
				  const int *rowPtrs_trans, const Dtype *vals_trans,
				  const int numRows, Dtype* res);

void createDiagXcsrInds(int size, int **inds);

void cusparseAddDSpMat(cusparseHandle_t handle, const int* rowPtrsA, const int* colIndsA,
	const double* valsA, const int nnzA, const int* rowPtrsB,
	const int* colIndsB, const double* valsB, const int nnzB,
	const int m, const int n, const double alpha, const double beta,
	int** rowPtrsC, int** colIndsC, double** valsC, int* nnzC);

void cusparseDcoodup2coo_compress(cusparseHandle_t cusparseHandle,
	int nnz, int m, int n, const double* vals, const int* rowInds, const int* colInds,
	double** compVals, int** compRowInds, int** compColInds, int* compNnz);

void getCfLaplacianDCoo_gpu(cusparseHandle_t cusparseHandle, const double eps,
	const int w_rad, const cv::Mat src,
	int **rowPtrs, int **colInds, double **vals, int *nnz);

void pardiso_solve_sym(const MKL_INT numRows, const MKL_INT lhsNnz,
	const int* lhsRowPtrs, const int* lhsColInds, const double* lhsVals,
	const double *rhsVals, double *res);

void setTRI(cv::String tri_path, cv::Mat &all, cv::Mat &fore);

void setTRI(cv::String tri_path, const cv::Mat src, cv::Mat &all, cv::Mat &fore);
