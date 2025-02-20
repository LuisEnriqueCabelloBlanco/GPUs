#include <stdio.h>
#include "matrix_mul.h"
#include <time.h>
// Thread block size
#define BLOCK_SIZE 16 

#define MULTI_THREAD

// Forward declaration of the device multiplication function
__global__ void Muld(float*, float*, int, int, float*);
__global__ void MuldOp1(float*, float*, int, int, float*);
// Host multiplication function
// Compute C = A * B
// hA is the height of A
// wA is the width of A
// wB is the width of B


void Mul___(float* A, float* B, int hA, int wA, int wB, float* C)
{
	int size;

	// Load A and B to the device
	float* Ad;
	size = hA * wA * sizeof(float);
	cudaMalloc((void**)&Ad, size);
	clock_t tic = clock();
	cudaMemcpy(Ad, A, size, cudaMemcpyHostToDevice);
	clock_t tac = clock();
	double ATime = (double)(tac-tic)/CLOCKS_PER_SEC;

	float* Bd;
	size = wA * wB * sizeof(float);
	cudaMalloc((void**)&Bd, size);
	tic=clock();
	cudaMemcpy(Bd, B, size, cudaMemcpyHostToDevice);
	tac=clock();
	double BTime = (double)(tac-tic)/CLOCKS_PER_SEC;

	// Allocate C on the device
	float* Cd;
	size = hA * wB * sizeof(float);
	cudaMalloc((void**)&Cd, size);

	// Compute the execution configuration assuming
	// the matrix dimensions are multiples of BLOCK_SIZE
#ifdef MULTI_THREAD
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	int sumX = (wB % dimBlock.x)==0?wB/dimBlock.x:wB/dimBlock.x+1;
	int sumY = (hA % dimBlock.y)==0?hA/dimBlock.y:wB/dimBlock.y+1;
	dim3 dimGrid(sumX, sumY);
#endif
	// Launch the device computation
	tic=clock();
#ifndef MULTI_THREAD
	Muld<<1,1>>(Ad, Bd, wA, wB, Cd);
#else
	MuldOp1<<<dimGrid, dimBlock>>>(Ad, Bd, wA, wB, Cd);
#endif	
	//fuerza a que se espere a que termine el kernel ya que el lanzamiento es asincrono
	cudaDeviceSynchronize();
	tac=clock();
	double KerTime = (double)(tac-tic)/CLOCKS_PER_SEC;

	// Read C from the device
	tic = clock();
	cudaMemcpy(C, Cd, size, cudaMemcpyDeviceToHost);
	tac = clock();
	double CTime = (double)(tac-tic)/CLOCKS_PER_SEC;

	double BWA;
	double BWB;
	//2*M*N*K
	double KerPerf;
	double BWC;


//printf("%f; %f; %f; %f; %f; %f; %f; %f;", Ttx1, Ttx2, Tkrnl, Ttx3, BWtx1, BWtx2, Perfkrnl, BWtx3);

	printf("%f; %f; %f; %f; %f; %f; %f; %f;\n",ATime,BTime,KerTime,CTime,BWA,BWB,KerPerf,BWC);
	// Free device memory
	cudaFree(Ad);
	cudaFree(Bd);
	cudaFree(Cd);
}


#ifndef MULTI_THREAD
__global__ void Muld(float* A, float* B, int wA, int wB, float* C)
{
	//To Do
	for (int i = 0; i < wA; i++) {
      for (int j = 0; j < wB; j++) {
        //este se va
		for (int k = 0; k < wA; k++) {
            C[i*wB+j] += A[i*wA+k]*B[k*wB+j];
        }
      }
   }
}
#else
__global__ void MuldOp1(float* A, float* B, int wA, int wB, float* C)
{
	int row = threadIdx.x+blockIdx.x*blockDim.x;
	int col = threadIdx.y+blockIdx.y*blockDim.y;

	if(col<wA&&row<wB){
		float total =0.0;
		for (int k = 0; k < wA; k++) {
			total += A[row*wA+k]*B[k*wB+col];
		}
		C[row*wB+col] = total;
	}
}
#endif

#if 0
// Device multiplication function called by Mul()
// Compute C = A * B
// wA is the width of A
// wB is the width of B
__global__ void Muld(float* A, float* B, int wA, int wB, float* C)
{
	// Block index
	int bx = blockIdx.x;
	int by = blockIdx.y;

	// Thread index
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	// Index of the first sub-matrix of A processed by the block
	int aBegin = BLOCK_SIZE * wA * by;

	// Index of the last sub-matrix of A processed by the block
	int aEnd = BLOCK_SIZE * bx;

	// Step size used to iterate through the sub-matrices of A
	int aStep = BLOCK_SIZE;

	// Index of the first sub-matrix of B processed by the block
	int bBegin = BLOCK_SIZE * bx;

	// Step size used to iterate through the sub-matrices of B
	int bStep = BLOCK_SIZE * wB;

	// The element of the block sub-matrix that is computed
	// by the thread
	float Csub = 0;

	// Loop over all the sub-matrices of A and B required to
	// compute the block sub-matrix
	for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
		// Shared memory for the sub-matrix of A
		__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

		// Shared memory for the sub-matrix of B
		__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

		// Load the matrices from global memory to shared memory;
		// each thread loads one element of each matrixs
		As[ty][tx] = A[a+tx];
		Bs[ty][tx] = B[b+ty];
		// Synchronize to make sure the matrices are loaded
		__syncthreads();

		// Multiply the two matrices together;
		// each thread computes one element
		// of the block sub-matrix
		for (int k = 0; k < BLOCK_SIZE; ++k)
			C[Csub] += As[ty][tx]*Bs[ty][tx];

		// Synchronize to make sure that the preceding
		// computation is done before loading two new
		// sub-matrices of A and B in the next iteration
		__syncthreads();
	}
	
	// Write the block sub-matrix to global memory;
	// each thread writes one element
	...
}
#endif
