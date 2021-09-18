#include <stdio.h>
#include <math.h>
#include <time.h>

__global__ void sinX(double *x, double *result) {
    *result = sin(*x);
}

int main() {
    clock_t begin = clock();
    // переменные хоста
    double x, result;
    int size = sizeof(double);

    // копии для устройства
    double *d_x, *d_result;

    // выделяем память на устройстве
    cudaMalloc((void **)&d_x, size);
    cudaMalloc((void **)&d_result, size);

    // инициализируем переменную хоста
    x = 1;

    // копируем данные с хоста на устройство
    cudaMemcpy(d_x, &x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, &result, size, cudaMemcpyHostToDevice);

    // вызов функции на хосте
    // но работать она будет на устройстве
    sinX<<<1,1>>>(d_x, d_result);

    //копируем данные с устройства на хост
    cudaMemcpy(&result, d_result, size, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_x); cudaFree(d_result);
    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

    printf("GPU:\n%f\n", result);
    printf("Time: %f\n", time_spent);

    return 0;
}