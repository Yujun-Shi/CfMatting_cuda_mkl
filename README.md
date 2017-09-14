# CfMatting_cuda_mkl
Implementation of A. Levin D. Lischinski and Y. Weiss. A Closed Form Solution to Natural Image Matting. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), June 2006, New York with the help of nvidia cuda and intel mkl.

I ran the code on my own laptop with:

visual studio 2015

cuda 8.0

mkl 2017.3.210

opencv 3.2.0


note: 
First, apologize for the terrible coding style.

Second, the project is usable with a stuning result.

Third, this code is hard coded for the window size 3 x 3 when getting matting laplacian matrix since kernel/device function can't allocate dynamic memory.

Original authors: Anat Levin (anat.levin@weizmann.ac.il)
