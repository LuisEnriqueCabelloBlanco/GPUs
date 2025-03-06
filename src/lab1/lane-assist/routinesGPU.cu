#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <png_io.h>

#include "routinesGPU.h"

void canny(uint8_t *im, uint8_t *image_out,
	uint8_t *pedge,
	float level,
	int height, int width){

	int i, j;
	int ii, jj;
	float PI = 3.141593;

	float lowthres, hithres;
	
	float *NR_d;
	float * NR;
	
	uint8_t* im_d;
	cudaMalloc((void **)&NR_d,sizeof(float) * width * height);
	cudaMalloc((void **)&im_d,sizeof(uint8_t) * width * height);
	cudaMemcpy(im_d,im,sizeof(uint8_t) * width * height,cudaMemcpyHostToDevice);
	noise_reduc<<<1,1>>>(im_d,NR_d,height,width);
	cudaFree(im_d);

	float *G_d;
	float *Gx;
	float *Gy;
	float *phi_d;
	cudaMalloc((void **)&G_d,sizeof(float) * width * height);
	cudaMalloc((void **)&phi_d,sizeof(float) * width * height);
	cudaMalloc((void **)&Gx,sizeof(float) * width * height);
	cudaMalloc((void **)&Gy,sizeof(float) * width * height);

	gradient_img<<<1,1>>>(NR_d,G_d,phi_d,Gx,Gy,height,width);
	
	float* phi;
	float* G;
	phi = (float *)malloc(sizeof(float) * width * height);
	G = (float *)malloc(sizeof(float) * width * height);
	cudaMemcpy(phi,phi_d,sizeof(float) * width * height,cudaMemcpyDeviceToHost);
	cudaMemcpy(G,G_d,sizeof(float) * width * height,cudaMemcpyDeviceToHost);

	cudaFree(Gx);
	cudaFree(Gy);
	//edge<<<1,1>>>();
	//hyteresis_Thresholding<<<1,1>>>();

	// Edge
	for(i=3; i<height-3; i++)
		for(j=3; j<width-3; j++)
		{
			pedge[i*width+j] = 0;
			if(phi[i*width+j] == 0){
				if(G[i*width+j]>G[i*width+j+1] && G[i*width+j]>G[i*width+j-1]) //edge is in N-S
					pedge[i*width+j] = 1;

			} else if(phi[i*width+j] == 45) {
				if(G[i*width+j]>G[(i+1)*width+j+1] && G[i*width+j]>G[(i-1)*width+j-1]) // edge is in NW-SE
					pedge[i*width+j] = 1;

			} else if(phi[i*width+j] == 90) {
				if(G[i*width+j]>G[(i+1)*width+j] && G[i*width+j]>G[(i-1)*width+j]) //edge is in E-W
					pedge[i*width+j] = 1;

			} else if(phi[i*width+j] == 135) {
				if(G[i*width+j]>G[(i+1)*width+j-1] && G[i*width+j]>G[(i-1)*width+j+1]) // edge is in NE-SW
					pedge[i*width+j] = 1;
			}
		}

	// Hysteresis Thresholding
	lowthres = level/2;
	hithres  = 2*(level);

	for(i=3; i<height-3; i++)
		for(j=3; j<width-3; j++)
		{
			image_out[i*width+j] = 0;
			if(G[i*width+j]>hithres && pedge[i*width+j])
				image_out[i*width+j] = 255;
			else if(pedge[i*width+j] && G[i*width+j]>=lowthres && G[i*width+j]<hithres)
				// check neighbours 3x3
				for (ii=-1;ii<=1; ii++)
					for (jj=-1;jj<=1; jj++)
						if (G[(i+ii)*width+j+jj]>hithres)
							image_out[i*width+j] = 255;
		}	

}


__global__ void noise_reduc(uint8_t*im, float*NR ,int height,int width){

	int i, j;
	for(i=2; i<height-2; i++)
		for(j=2; j<width-2; j++)
		{
			// Noise reduction
			NR[i*width+j] =
				 (2.0*im[(i-2)*width+(j-2)] +  4.0*im[(i-2)*width+(j-1)] +  5.0*im[(i-2)*width+(j)] +  4.0*im[(i-2)*width+(j+1)] + 2.0*im[(i-2)*width+(j+2)]
				+ 4.0*im[(i-1)*width+(j-2)] +  9.0*im[(i-1)*width+(j-1)] + 12.0*im[(i-1)*width+(j)] +  9.0*im[(i-1)*width+(j+1)] + 4.0*im[(i-1)*width+(j+2)]
				+ 5.0*im[(i  )*width+(j-2)] + 12.0*im[(i  )*width+(j-1)] + 15.0*im[(i  )*width+(j)] + 12.0*im[(i  )*width+(j+1)] + 5.0*im[(i  )*width+(j+2)]
				+ 4.0*im[(i+1)*width+(j-2)] +  9.0*im[(i+1)*width+(j-1)] + 12.0*im[(i+1)*width+(j)] +  9.0*im[(i+1)*width+(j+1)] + 4.0*im[(i+1)*width+(j+2)]
				+ 2.0*im[(i+2)*width+(j-2)] +  4.0*im[(i+2)*width+(j-1)] +  5.0*im[(i+2)*width+(j)] +  4.0*im[(i+2)*width+(j+1)] + 2.0*im[(i+2)*width+(j+2)])
				/159.0;
		}
}

__global__ void gradient_img(float *NR,float *G, float *phi, float *Gx, float *Gy,int height,int width){
	int i, j;

	float PI = 3.141593;

	for(i=2; i<height-2; i++)
		for(j=2; j<width-2; j++)
		{
			// Intensity gradient of the image
			Gx[i*width+j] = 
				 (1.0*NR[(i-2)*width+(j-2)] +  2.0*NR[(i-2)*width+(j-1)] +  (-2.0)*NR[(i-2)*width+(j+1)] + (-1.0)*NR[(i-2)*width+(j+2)]
				+ 4.0*NR[(i-1)*width+(j-2)] +  8.0*NR[(i-1)*width+(j-1)] +  (-8.0)*NR[(i-1)*width+(j+1)] + (-4.0)*NR[(i-1)*width+(j+2)]
				+ 6.0*NR[(i  )*width+(j-2)] + 12.0*NR[(i  )*width+(j-1)] + (-12.0)*NR[(i  )*width+(j+1)] + (-6.0)*NR[(i  )*width+(j+2)]
				+ 4.0*NR[(i+1)*width+(j-2)] +  8.0*NR[(i+1)*width+(j-1)] +  (-8.0)*NR[(i+1)*width+(j+1)] + (-4.0)*NR[(i+1)*width+(j+2)]
				+ 1.0*NR[(i+2)*width+(j-2)] +  2.0*NR[(i+2)*width+(j-1)] +  (-2.0)*NR[(i+2)*width+(j+1)] + (-1.0)*NR[(i+2)*width+(j+2)]);


			Gy[i*width+j] = 
				 ((-1.0)*NR[(i-2)*width+(j-2)] + (-4.0)*NR[(i-2)*width+(j-1)] +  (-6.0)*NR[(i-2)*width+(j)] + (-4.0)*NR[(i-2)*width+(j+1)] + (-1.0)*NR[(i-2)*width+(j+2)]
				+ (-2.0)*NR[(i-1)*width+(j-2)] + (-8.0)*NR[(i-1)*width+(j-1)] + (-12.0)*NR[(i-1)*width+(j)] + (-8.0)*NR[(i-1)*width+(j+1)] + (-2.0)*NR[(i-1)*width+(j+2)]
				+    2.0*NR[(i+1)*width+(j-2)] +    8.0*NR[(i+1)*width+(j-1)] +    12.0*NR[(i+1)*width+(j)] +    8.0*NR[(i+1)*width+(j+1)] +    2.0*NR[(i+1)*width+(j+2)]
				+    1.0*NR[(i+2)*width+(j-2)] +    4.0*NR[(i+2)*width+(j-1)] +     6.0*NR[(i+2)*width+(j)] +    4.0*NR[(i+2)*width+(j+1)] +    1.0*NR[(i+2)*width+(j+2)]);

			G[i*width+j]   = sqrtf((Gx[i*width+j]*Gx[i*width+j])+(Gy[i*width+j]*Gy[i*width+j]));	//G = √Gx²+Gy²
			phi[i*width+j] = atan2f(fabs(Gy[i*width+j]),fabs(Gx[i*width+j]));

			if(fabs(phi[i*width+j])<=PI/8 )
				phi[i*width+j] = 0;
			else if (fabs(phi[i*width+j])<= 3*(PI/8))
				phi[i*width+j] = 45;
			else if (fabs(phi[i*width+j]) <= 5*(PI/8))
				phi[i*width+j] = 90;
			else if (fabs(phi[i*width+j]) <= 7*(PI/8))
				phi[i*width+j] = 135;
			else phi[i*width+j] = 0;
	}
}

void lane_assist_GPU(uint8_t *im, int height, int width,
	int *x1, int *y1, int *x2, int *y2, int *nlines)
{
	// Create temporal buffers 
	uint8_t *imEdge = (uint8_t *)malloc(sizeof(uint8_t) * width * height);
	float *NR = (float *)malloc(sizeof(float) * width * height);
	float *G = (float *)malloc(sizeof(float) * width * height);
	float *phi = (float *)malloc(sizeof(float) * width * height);
	float *Gx = (float *)malloc(sizeof(float) * width * height);
	float *Gy = (float *)malloc(sizeof(float) * width * height);
	uint8_t *pedge = (uint8_t *)malloc(sizeof(uint8_t) * width * height);

	/* Canny */
	canny(im, imEdge,
		pedge,
		1000.0f, //level
		height, width);

	write_png_fileBW("out_edges.png",imEdge,width,height);
	//write_png_fileRGB("outRGB_edges.png", imEdge, width, height);
	/* To do */
}
