#ifndef KERNEL_H 
#define KERNEL_H 

#include <CL/sycl.hpp>
#define MAX_WINDOW_SIZE 5*5
using  namespace  cl::sycl;

void remove_noise_SYCL(sycl::queue Q, float *im, float *image_out, float * window,
	float thredshold, int window_size,
	int height, int width);
#endif
