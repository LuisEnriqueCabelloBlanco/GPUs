#ifndef ROUTINESGPU_H
#define ROUTINESGPU_H

#include <stdint.h>

__global__ void noise_reduc(uint8_t*im, float*NR ,int height,int width);

__global__ void gradient_img(float *NR,float *G, float *phi, float *Gx, float *Gy,int height,int width);


__global__ void edge();

__global__ void hyteresis_Thresholding();

void canny(uint8_t *im, uint8_t *image_out,
	uint8_t *pedge,
	float level,
	int height, int width);

void lane_assist_GPU(uint8_t *im, int height, int width,
	int *x1, int *y1, int *x2, int *y2, int *nlines);

#endif

