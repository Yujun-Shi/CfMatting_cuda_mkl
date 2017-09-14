#include "utils.h"

// sum up rows of a sparse matrix stored in csr format
// and store the value in a vector
template<typename Dtype>
__global__ void sumRowCsrSymKernel(const int* rowPtrs, const Dtype* vals,
	const int *rowPtrs_trans, const Dtype *vals_trans,
	const int numRows, Dtype* res)
{
	unsigned int idx = threadIdx.x + blockDim.x*blockIdx.x;
	if (idx >= 0 && idx < numRows) {
		Dtype temp = 0;
		int startInd = rowPtrs_trans[idx];
		int endInd = rowPtrs_trans[idx + 1];
		for (int i = startInd; i < endInd; ++i) {
			temp += vals_trans[i];
		}

		startInd = rowPtrs[idx];
		endInd = rowPtrs[idx + 1];
		// start at the next spot to avoid adding the diagonal element twice
		for (int i = startInd + 1; i < endInd; ++i) {
			temp += vals[i];
		}
		res[idx] = temp;
	}
}

template<typename Dtype>
void sumRowCsrSym(const int* rowPtrs, const Dtype* vals, const int *rowPtrs_trans, const Dtype *vals_trans,
	const int numRows, Dtype* res)
{
	sumRowCsrSymKernel<Dtype> << <CAFFE_GET_BLOCKS(numRows), CAFFE_CUDA_NUM_THREADS >> >\
		(rowPtrs, vals, rowPtrs_trans, vals_trans, numRows, res);
}

// create the index of a diagonal matrix
__global__ void diagCsrIndsKernel(const int numRows, int *inds) {
	unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;
	if (idx >= 0 && idx <= numRows) {
		inds[idx] = idx;
	}
}

void createDiagXcsrInds(int size, int **inds) {
	CUDA_CHECK(cudaMalloc((void**)&(*inds), sizeof(int)*(size + 1)));
	diagCsrIndsKernel << <CAFFE_GET_BLOCKS(size + 1), CAFFE_CUDA_NUM_THREADS >> >(size, *inds);
}

// wrapper for cusparseDcsrgeam
void cusparseAddDSpMat(cusparseHandle_t handle, const int* rowPtrsA, const int* colIndsA,
	const double* valsA, const int nnzA, const int* rowPtrsB,
	const int* colIndsB, const double* valsB, const int nnzB,
	const int m, const int n, const double alpha, const double beta,
	int** rowPtrsC, int** colIndsC, double** valsC, int* nnzC)
{
	cusparseMatDescr_t descr;
	CUDA_CHECK(cusparseCreateMatDescr(&descr));

	CUDA_CHECK(cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL));
	CUDA_CHECK(cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO));

	CUDA_CHECK(cudaMalloc((void**)&(*rowPtrsC), sizeof(int)*(m + 1)));

	CUDA_CHECK(cusparseXcsrgeamNnz(handle, m, n, descr, nnzA,
		rowPtrsA, colIndsA, descr, nnzB, rowPtrsB, colIndsB, descr,
		*rowPtrsC, nnzC));

	CUDA_CHECK(cudaMalloc((void**)&(*colIndsC), sizeof(int) * (*nnzC)));
	CUDA_CHECK(cudaMalloc((void**)&(*valsC), sizeof(double) * (*nnzC)));

	CUDA_CHECK(cusparseDcsrgeam(handle, m, n, &alpha, descr, nnzA,
		valsA, rowPtrsA, colIndsA, &beta, descr, nnzB, valsB,
		rowPtrsB, colIndsB, descr, *valsC, *rowPtrsC, *colIndsC));

	CUDA_CHECK(cusparseDestroyMatDescr(descr));
}

// cusparseDcoodup2coo_compress and it's helper functions

// this function eliminate the duplicate entries in
// a sparse matrix stored in coo format by sum up
// the value of the same entry

template<typename Dtype>
__global__ void mapVector(const Dtype *vals_src, const int* permut,
	const int num, Dtype *vals_dst) {
	unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;
	if (idx < num) {
		vals_dst[idx] = vals_src[permut[idx]];
	}
}

// based on the assumption that the coo mat is sorted
// look the left element to determine whether it's a duplicate entry
__global__ void maskDuplicate(int* rowInds, int* colInds, int num, int* mask) {
	unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx > 0 && idx < num) {
		if (rowInds[idx] == rowInds[idx - 1] && colInds[idx] == colInds[idx - 1]) {
			mask[idx] = 0; // mark as duplicate
		}
		else {
			mask[idx] = 1;
		}
	}
	else if (idx == 0) {
		mask[idx] = 1;
	}
}

// 1. look left, check if it's the first element
// 2. go right, add all the duplicate element
template<typename Dtype>
__global__ void reduceByMask(int length,
	int* mask,
	int* rowInds,
	int* colInds,
	Dtype* vals,
	int* compRowInds,
	int* compColInds,
	Dtype* compVals)
{
	unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;
	int compInd;
	int offset = 0;
	Dtype temp;

	if (idx >= 0 && idx < length) {
		if (idx == 0 || mask[idx - 1] != mask[idx]) { // thread hit
			temp = vals[idx];
			while (idx + offset + 1 < length &&\
				mask[idx + offset] == mask[idx + offset + 1]) {
				temp += vals[idx + offset + 1];
				++offset;
			}
			// index in compress mode, hard code a -1 for our scenario
			compInd = mask[idx] - 1;
			compRowInds[compInd] = rowInds[idx];
			compColInds[compInd] = colInds[idx];
			compVals[compInd] = temp;
		}
	}
}

void cusparseDcoodup2coo_compress(cusparseHandle_t cusparseHandle,
	int nnz, int m, int n, const double* vals, const int* rowInds, const int* colInds,
	double** compVals, int** compRowInds, int** compColInds, int* compNnz)
{
	size_t pBufferSizeInBytes = 0;
	int* d_p; // permutation
	void* pBuffer;
	int *mask, *rowIndsCpy, *colIndsCpy;
	double *valsCpy, *d_vals_t;

	// step 0: allocation and copy
	CUDA_CHECK(cudaMalloc((void**)&rowIndsCpy, sizeof(int)*nnz));
	CUDA_CHECK(cudaMalloc((void**)&colIndsCpy, sizeof(int)*nnz));
	CUDA_CHECK(cudaMalloc((void**)&valsCpy, sizeof(double)*nnz));
	CUDA_CHECK(cudaMemcpy(rowIndsCpy, rowInds, sizeof(int)*nnz, cudaMemcpyDeviceToDevice));
	CUDA_CHECK(cudaMemcpy(colIndsCpy, colInds, sizeof(int)*nnz, cudaMemcpyDeviceToDevice));
	CUDA_CHECK(cudaMemcpy(valsCpy, vals, sizeof(double)*nnz, cudaMemcpyDeviceToDevice));

	// step 1: allocation and sort
	CUDA_CHECK(cudaMalloc((void**)&d_p, sizeof(int)*nnz));
	CUDA_CHECK(cusparseCreateIdentityPermutation(cusparseHandle, nnz, d_p));
	CUDA_CHECK(cusparseXcoosort_bufferSizeExt(cusparseHandle, m, n, \
		nnz, rowIndsCpy, colIndsCpy, &pBufferSizeInBytes));
	CUDA_CHECK(cudaMalloc((void**)&pBuffer, pBufferSizeInBytes));

	CUDA_CHECK(cusparseXcoosortByRow(cusparseHandle, m, n, \
		nnz, rowIndsCpy, colIndsCpy, d_p, pBuffer));
	CUDA_CHECK(cudaFree(pBuffer));

	CUDA_CHECK(cudaMalloc((void**)&d_vals_t, sizeof(double)*nnz));
	CUDA_CHECK(cudaMemcpy(d_vals_t, valsCpy, sizeof(double)*nnz, cudaMemcpyDeviceToDevice));
	mapVector<double> << <CAFFE_GET_BLOCKS(nnz), CAFFE_CUDA_NUM_THREADS >> >\
		(d_vals_t, d_p, nnz, valsCpy);

	CUDA_CHECK(cudaFree(d_vals_t));
	CUDA_CHECK(cudaFree(d_p));

	// step 2: mask and scan(inclusive)
	CUDA_CHECK(cudaMalloc((void**)&mask, sizeof(int)*nnz));
	maskDuplicate << <CAFFE_GET_BLOCKS(nnz), CAFFE_CUDA_NUM_THREADS >> >\
		(rowIndsCpy, colIndsCpy, nnz, mask);
	CUDA_CHECK(cudaDeviceSynchronize());
	CUDA_CHECK(cudaGetLastError());

	thrust::inclusive_scan(thrust::device, mask, mask + nnz, mask);

	// step 3: allocate and reduce
	CUDA_CHECK(cudaMemcpy(compNnz, &mask[nnz - 1], sizeof(int), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMalloc((void**)&(*compRowInds), sizeof(int) * (*compNnz)));
	CUDA_CHECK(cudaMalloc((void**)&(*compColInds), sizeof(int) * (*compNnz)));
	CUDA_CHECK(cudaMalloc((void**)&(*compVals), sizeof(double) * (*compNnz)));

	reduceByMask<double> << <CAFFE_GET_BLOCKS(nnz), CAFFE_CUDA_NUM_THREADS >> >\
		(nnz, mask, rowIndsCpy, colIndsCpy, \
			valsCpy, *compRowInds, *compColInds, *compVals);

	CUDA_CHECK(cudaDeviceSynchronize());
	CUDA_CHECK(cudaGetLastError());

	CUDA_CHECK(cudaFree(mask));
	CUDA_CHECK(cudaFree(rowIndsCpy));
	CUDA_CHECK(cudaFree(colIndsCpy));
	CUDA_CHECK(cudaFree(valsCpy));
}

//
// laplacian function and it's helper functions
//

// this function extract a small window from the original image
template<typename Dtype>
__device__ void extract_window(const uchar *src, const int idx_x,
	const int idx_y, const int c, const int w_rad,
	const int imcols, Dtype *window)
{
	for (int iy = 0; iy < 2 * w_rad + 1; ++iy) {
		for (int ix = 0; ix < 2 * w_rad + 1; ++ix) {
			for (int iz = 0; iz < c; ++iz) {
				window[c * (iy * (2 * w_rad + 1) + ix) + iz] = \
					(Dtype)src[c*((idx_y - w_rad + iy)*imcols + (idx_x - w_rad + ix)) + iz];
			}
		}
	}
}

// this function calculates the mean of pixels in a small window
template<typename Dtype>
__device__ void calc_mu(const int m, const int n, const Dtype *mat, const int dim,
	Dtype *mu)
{
	if (dim == ALONG_Y) {
		for (int i = 0; i < n; ++i) {
			mu[i] = 0;
			for (int j = 0; j < m; ++j) {
				mu[i] += mat[j*n + i];
			}
			mu[i] /= m;
		}
	}
	else if (dim == ALONG_X) {
		for (int i = 0; i < m; ++i) {
			mu[i] = 0;
			for (int j = 0; j < n; ++j) {
				mu[i] += mat[i*n + j];
			}
			mu[i] /= n;
		}
	}
}

// this function calculates an intermediate value var
// assume n is the remain dimension,  which means the result is n x n
template<typename Dtype>
__device__ void calc_var(const Dtype *data, const Dtype *mu,
	const int m, const int n, const Dtype eps, Dtype *var)
{
	Dtype ele_val = 0;
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			ele_val = 0;
			for (int k = 0; k < m; ++k) {
				ele_val += data[k*n + i] * data[k*n + j];
			}
			ele_val /= m;
			ele_val -= mu[i] * mu[j];
			if (i == j) {
				ele_val += eps / m;
			}
			var[i*n + j] = ele_val;
		}
	}
}

// this function calculates the inverse of the matrix mat
template<typename Dtype>
__device__ void calc_inverse(Dtype *mat, const int dim, Dtype *inv) {
	for (int i = 0; i < dim; ++i) {
		for (int j = 0; j < dim; ++j) {
			if (i == j) {
				inv[i*dim + j] = 1;
			}
			else {
				inv[i*dim + j] = 0;
			}
		}
	}
	int row = dim;
	int col = dim;
	for (int j = 0; j < col; ++j) {
		// find the max index
		int maxInd = j;
		for (int indy = j + 1; indy < row; ++indy) {
			// just see if it works out
			if (fabs((double)mat[indy*col + j]) > fabs((double)mat[maxInd*col + j])) {
				maxInd = indy;
			}
		}
		// swap j, maxInd
		if (maxInd != j) {
			for (int indx = j; indx < col; ++indx) {
				Dtype temp = mat[j*col + indx];
				mat[j*col + indx] = mat[maxInd*col + indx];
				mat[maxInd*col + indx] = temp;
			}
			for (int indx = 0; indx < col; ++indx) {
				Dtype temp = inv[j*col + indx];
				inv[j*col + indx] = inv[maxInd*col + indx];
				inv[maxInd*col + indx] = temp;
			}
		}
		for (int indy = 0; indy < row; ++indy) {
			if (indy == j) {
				continue;
			}
			Dtype coef = mat[indy*col + j] / mat[j*col + j];
			for (int indx = j; indx < col; ++indx) {
				mat[indy*col + indx] -= coef*mat[j*col + indx];
			}
			for (int indx = 0; indx < col; ++indx) {
				inv[indy*col + indx] -= coef * inv[j*col + indx];
			}
		}
	}
	for (int indy = 0; indy < row; ++indy) {
		for (int indx = 0; indx < col; ++indx) {
			inv[indy*col + indx] /= mat[indy*col + indy];
		}
	}
}

// m, n describe the shape of window_data
template<typename Dtype>
__device__ void calc_val(const Dtype *window_data, const Dtype *win_var_inv, const Dtype *win_mu,
	const int m, const int n, Dtype *win_val)
{
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < m; ++j) {
			Dtype ele_res = 0;
			for (int k = 0; k < n; ++k) {
				for (int l = 0; l < n; ++l) {
					ele_res += \
						win_var_inv[k*n + l] * (window_data[i*n + k] - win_mu[k])*(window_data[j*n + l] - win_mu[l]);
				}
			}
			ele_res += 1;
			ele_res /= m;
			win_val[i*m + j] = ele_res;
		}
	}
}

__device__ void fill_inds(const int col, const int idx_x, const int idx_y,
	const int w_rad, int *rowInds, int *colInds)
{
	int width = 2 * w_rad + 1;
	int neb_size = width * width;
	int working = 0;
	for (int i = 0; i < neb_size; ++i) {
		for (int j = i; j < neb_size; ++j) {
			rowInds[working] = (i / width + idx_y - w_rad)*col + i % 3 + idx_x - w_rad; // i-th pixel
			colInds[working] = (j / width + idx_y - w_rad)*col + j % 3 + idx_x - w_rad; // j-th pixel
			++working;
		}
	}
}

template<typename Dtype>
__device__ void fill_vals(const int neb_size, const Dtype *win_vals, Dtype *vals) {
	int working = 0;
	for (int i = 0; i < neb_size; ++i) {
		for (int j = i; j < neb_size; ++j) {
			vals[working] = win_vals[i*neb_size + j];
			++working;
		}
	}
}

// had to hard code for w_rad = 1, channels = 3.
template<typename Dtype>
__global__ void getCfLaplacianKernel(const Dtype eps, const int imrows,
	const int imcols, const uchar* src,
	int *rowInds, int *colInds, Dtype *vals)
{
	int w_rad = 1;
	int neb_size = (2 * w_rad + 1)*(2 * w_rad + 1);
	int c = 3; // hard code the channels
			   // window size is the size of the laplacian of the small neighbor
	int window_size = neb_size*(neb_size + 1) / 2;
	// using y, x coordinate
	unsigned int idx_x = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int idx_y = threadIdx.y + blockIdx.y * blockDim.y;
	Dtype imgW[9 * 3]; // hard coded as neb_size = 9, channels = 3
	Dtype win_mu[3]; // channels = 3
	Dtype win_var[3 * 3]; // channels = 3
	Dtype win_var_inv[3 * 3]; // channels = 3
	Dtype win_val[9 * 9]; // neb_size = 9
	int pBufferOffset = ((idx_y - w_rad)*(imcols - 2 * w_rad) + idx_x - w_rad)*window_size;

	if (idx_y >= w_rad && idx_y < imrows - w_rad) {
		if (idx_x >= w_rad && idx_x < imcols - w_rad) {
			extract_window<Dtype>(src, idx_x, idx_y, c, w_rad, imcols, imgW);

			calc_mu<Dtype>(neb_size, c, imgW, ALONG_Y, win_mu);
			calc_var<Dtype>(imgW, win_mu, neb_size, c, eps, win_var);
			calc_inverse<Dtype>(win_var, c, win_var_inv);
			calc_val<Dtype>(imgW, win_var_inv, win_mu, neb_size, c, win_val);
	
			// fill in val and inds
			fill_inds(imcols, idx_x, idx_y, w_rad, rowInds + pBufferOffset, colInds + pBufferOffset);
			fill_vals<Dtype>(neb_size, win_val, vals + pBufferOffset);
		}
	}
}

void getCfLaplacianDCoo_gpu(cusparseHandle_t cusparseHandle, const double eps, 
	const int w_rad, const cv::Mat src,
	int **rowPtrs, int **colInds, double **vals, int *nnz)
{
	// host set up
	int c = src.channels();
	int imrows = src.rows;
	int imcols = src.cols;
	int neb_size = (2 * w_rad + 1)*(2 * w_rad + 1);
	int imgSize = imrows * imcols;
	// window size is the size of the laplacian of the small neighbor
	int window_size = neb_size * (neb_size + 1) / 2;
	int num_window = (imrows - 2 * w_rad)*(imcols - 2 * w_rad);

	// cuda set up
	int compNnz;
	int *d_rowBuffer, *d_colBuffer, *d_rowInds, *diagInds;
	int *rowPtrs_t, *colInds_t, *rowPtrs_t_trans, *colInds_t_trans;
	uchar *d_src;
	double *vals_t, *vals_t_trans;
	double *d_valBuffer, *sumLD;
	double alpha = 1, beta = -1;
	CUDA_CHECK(cudaMalloc((void**)&d_rowBuffer, sizeof(int)*window_size*num_window));
	CUDA_CHECK(cudaMalloc((void**)&d_colBuffer, sizeof(int)*window_size*num_window));
	CUDA_CHECK(cudaMalloc((void**)&d_valBuffer, sizeof(double)*window_size*num_window));
	CUDA_CHECK(cudaMalloc((void**)&d_src, sizeof(double)*imrows*imcols));
	CUDA_CHECK(cudaMemcpy(d_src, src.data, sizeof(uchar)*c*imrows*imcols, cudaMemcpyDefault));

	getCfLaplacianKernel<double> << <dim3((imcols + 32 - 1) / 32, (imrows + 32 - 1) / 32), dim3(32, 32) >> >\
		(eps, imrows, imcols, d_src, d_rowBuffer, d_colBuffer, d_valBuffer);
	CUDA_CHECK(cudaFree(d_src));

	cusparseDcoodup2coo_compress(\
		cusparseHandle, window_size * num_window, imgSize, imgSize, \
		d_valBuffer, d_rowBuffer, d_colBuffer, \
		&vals_t, &d_rowInds, &colInds_t, &compNnz);

	CUDA_CHECK(cudaFree(d_valBuffer));
	CUDA_CHECK(cudaFree(d_colBuffer));
	CUDA_CHECK(cudaFree(d_rowBuffer));

	// transform into csr format
	CUDA_CHECK(cudaMalloc((void**)&(rowPtrs_t), sizeof(int) * (imgSize + 1)));
	CUDA_CHECK(cusparseXcoo2csr(cusparseHandle, d_rowInds, compNnz, imgSize, rowPtrs_t,
		CUSPARSE_INDEX_BASE_ZERO));
	CUDA_CHECK(cudaFree(d_rowInds));

	// obtain the symmetric matrix to sum up a row
	CUDA_CHECK(cudaMalloc((void**)&rowPtrs_t_trans, sizeof(int)*(imgSize + 1)));
	CUDA_CHECK(cudaMalloc((void**)&colInds_t_trans, sizeof(int)*compNnz));
	CUDA_CHECK(cudaMalloc((void**)&vals_t_trans, sizeof(double)*compNnz));
	CUDA_CHECK(cusparseDcsr2csc(cusparseHandle, imgSize, imgSize, compNnz,
		vals_t, rowPtrs_t, colInds_t, vals_t_trans,
		colInds_t_trans, rowPtrs_t_trans,
		CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO));

	CUDA_CHECK(cudaFree(colInds_t_trans));

	CUDA_CHECK(cudaMalloc((void**)&sumLD, sizeof(double) * imgSize));
	sumRowCsrSym<double>(rowPtrs_t, vals_t, rowPtrs_t_trans, vals_t_trans,
		imgSize, sumLD);
	CUDA_CHECK(cudaFree(vals_t_trans));
	CUDA_CHECK(cudaFree(rowPtrs_t_trans));

	createDiagXcsrInds(imgSize, &diagInds);

	cusparseAddDSpMat(cusparseHandle, diagInds, diagInds,
		sumLD, imgSize, rowPtrs_t, colInds_t, vals_t, compNnz,
		imgSize, imgSize, alpha, beta,
		rowPtrs, colInds, vals, nnz);

	CUDA_CHECK(cudaFree(rowPtrs_t));
	CUDA_CHECK(cudaFree(colInds_t));
	CUDA_CHECK(cudaFree(vals_t));
	CUDA_CHECK(cudaFree(sumLD));
	CUDA_CHECK(cudaFree(diagInds));
}

cv::Mat createIndsMat(const int imrows, const int imcols) {
	cv::Mat inds(imrows, imcols, CV_32S);
	int imagesize = imrows*imcols;
	int *data = (int*)inds.data;

	#pragma omp parallel for
	for (int i = 0; i < imagesize; ++i) {
		data[i] = i;
	}
	return inds;
}

void pardiso_solve_sym(const MKL_INT numRows, const MKL_INT lhsNnz, 
	const int* lhsRowPtrs, const int* lhsColInds, const double* lhsVals,
	const double *rhsVals, double *res) {
	
	// copy and store the data in MKL_INT so the code can be run under mkl_ilp64
	MKL_INT* rowPtrs_t = (MKL_INT*)malloc(sizeof(MKL_INT)*(numRows + 1));
	MKL_INT* colInds_t = (MKL_INT*)malloc(sizeof(MKL_INT)*lhsNnz);
	#pragma omp parallel for
	for (MKL_INT i = 0; i < lhsNnz; ++i) {
		colInds_t[i] = (MKL_INT)lhsColInds[i];
	}
	#pragma omp parallel for
	for (MKL_INT i = 0; i < numRows + 1; ++i) {
		rowPtrs_t[i] = (MKL_INT)lhsRowPtrs[i];
	}

	MKL_INT mtype = -2;       /* Real symmetric matrix */

	MKL_INT nrhs = 1;     /* Number of right hand sides. */
					
	void *pt[64];
	/* Pardiso control parameters. */
	MKL_INT iparm[64];
	MKL_INT maxfct, mnum, phase, error, msglvl;
	/* Auxiliary variables. */
	double ddum;          /* Double dummy */
	MKL_INT idum;         /* Integer dummy. */

	// set up parameter
	for (MKL_INT i = 0; i < 64; i++)
	{
		iparm[i] = 0;
	}
	iparm[0] = 1;         /* No solver default */
	iparm[1] = 2;         /* Fill-in reordering from METIS */
	iparm[3] = 0;         /* No iterative-direct algorithm */
	iparm[4] = 0;         /* No user fill-in reducing permutation */
	iparm[5] = 0;         /* Write solution into x */
	iparm[7] = 2;         /* Max numbers of iterative refinement steps */
	iparm[9] = 13;        /* Perturb the pivot elements with 1E-13 */
	iparm[10] = 1;        /* Use nonsymmetric permutation and scaling MPS */
	iparm[12] = 0;        /* Maximum weighted matching algorithm is switched-off (default for symmetric) */
	iparm[13] = 0;        /* Output: Number of perturbed pivots */
	iparm[17] = -1;       /* Output: Number of nonzeros in the factor LU */
	iparm[18] = -1;       /* Output: Mflops for LU factorization */
	iparm[19] = 0;        /* Output: Numbers of CG Iterations */
	iparm[34] = 1;        /* set to 0-based index */
	maxfct = 1;           /* Maximum number of numerical factorizations. */
	mnum = 1;             /* Which factorization to use. */
	msglvl = 0;           /* Don't print statistical information in file */
	error = 0;            /* Initialize error flag */

	for (MKL_INT i = 0; i < 64; i++){
		pt[i] = 0;
	}
	
	// reorder and allocate memory
	phase = 11;
	PARDISO(pt, &maxfct, &mnum, &mtype, &phase,
		&numRows, lhsVals, rowPtrs_t, colInds_t, &idum, &nrhs, iparm, &msglvl, &ddum, &ddum, &error);
	if (error != 0){
		std::cout << "\nERROR during symbolic factorization: " << error;
		exit(1);
	}

	// factorization
	phase = 22;
	PARDISO(pt, &maxfct, &mnum, &mtype, &phase,
		&numRows, lhsVals, rowPtrs_t, colInds_t, &idum, &nrhs, iparm, &msglvl, &ddum, &ddum, &error);
	if (error != 0){
		std::cout << "\nERROR during numerical factorization: " << error;
		exit(2);
	}

	// Back substitution and iterative refinement.
	phase = 33;
	iparm[7] = 2;         // Max numbers of iterative refinement steps.
	PARDISO(pt, &maxfct, &mnum, &mtype, &phase,
		&numRows, lhsVals, rowPtrs_t, colInds_t, &idum, &nrhs, iparm, &msglvl, (void*)rhsVals, res, &error);
	if (error != 0){
		std::cout << "\nERROR during solution: " << error;
		exit(3);
	}

	// Termination and release of memory.
	phase = -1;           /* Release internal memory. */
	PARDISO(pt, &maxfct, &mnum, &mtype, &phase,
		&numRows, &ddum, rowPtrs_t, colInds_t, &idum, &nrhs,
		iparm, &msglvl, &ddum, &ddum, &error);

	free(rowPtrs_t);
	free(colInds_t);
}


void setTRI(cv::String tri_path, const cv::Mat src, cv::Mat &all, cv::Mat &fore) {
	// read the image and perform some sanity check
	cv::Mat trimap = cv::imread(tri_path, CV_LOAD_IMAGE_COLOR);
	if (src.rows != trimap.rows || src.cols != trimap.cols) {
		std::cout << "Dimension Not Match" << std::endl;
		exit(EXIT_FAILURE);
	}
	cv::Mat channels[3];
	cv::Mat src_tmp;

	src.convertTo(src_tmp, CV_64FC3);
	trimap.convertTo(trimap, CV_64FC3);

	cv::split(src_tmp, channels);
	src_tmp = (channels[0] + channels[1] + channels[2]) / 3.0;
	cv::split(trimap, channels);
	trimap = (channels[0] + channels[1] + channels[2]) / 3.0;

	trimap = trimap - src_tmp;

	fore = trimap > 0.02;
	all = trimap < -0.02 | trimap > 0.02;

	fore.convertTo(fore, CV_64FC1);
	all.convertTo(all, CV_64FC1);
}