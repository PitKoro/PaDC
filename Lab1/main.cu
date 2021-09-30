#include <stdlib.h> 
#include <stdio.h> 

 
void Sin(float*, int, double); 
 
__global__ void SinKernel(float *a, float *b) { 
int idx = threadIdx.x + blockIdx.x * blockDim.x; 
b[idx] = sinf(a[idx]); 
} 
 
void Printer(float *a, int n){ 
 for (int i = 0; i < n; i++){ 
 printf("%f\n", a[i]); 
 } 
} 
 
void Assigner(float *a, int n){ 
 for (int i = 0; i < n; i++){ 
 a[i] = (float)i; 
 } 
} 
 
int main(){ 
 
int n = 1024 * 1024; 
int size = n * sizeof(double);
 
 
float *aDev = NULL, *bDev = NULL; 
float *a = NULL, *b = NULL; 
 
cudaMalloc((void **) &aDev, size); 
cudaMalloc((void **) &bDev, size); 
 
a = (float *) malloc(size); 
b = (float *) malloc(size); 
 
Assigner(a, n); 
 
dim3 threads = dim3(512, 1); 
dim3 blocks = dim3(n / threads.x, 1); 
 
cudaMemcpy(aDev, a, size, cudaMemcpyHostToDevice); 
cudaMemcpy(bDev, b, size, cudaMemcpyHostToDevice); 
 
SinKernel<<<blocks, threads>>> (aDev, bDev); 
 
cudaMemcpy(b, bDev, size, cudaMemcpyDeviceToHost); 
 
Printer(b, n); 
 
cudaFree(aDev); 
cudaFree(bDev); 
 
free(a); 
free(b); 
}

