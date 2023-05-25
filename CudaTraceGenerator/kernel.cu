
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <windows.h>

#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 60

void printProgress(double percentage) {
    int val = (int)(percentage * 100);
    int lpad = (int)(percentage * PBWIDTH);
    int rpad = PBWIDTH - lpad;
    if (val == 100)
    {
		printf("\r%3d%% [%.*s%*s] done!", val, lpad, PBSTR, rpad, "");
	}
    else
    {
        printf("\r%3d%% [%.*s%*s] recording...", val, lpad, PBSTR, rpad, "");
    }
    fflush(stdout);
}

cudaError_t countWithCuda(int* trace, const unsigned int attackLength, const unsigned int traceLength, const unsigned int P);

__global__ void counterKernel(int* trace, const unsigned int attackLength, const unsigned int P, bool *work, int offset)
{
    int counter = 0;
    while (*work)
    {
        counter += 1;
    }
    trace[offset] = counter;
}

int main()
{
    const unsigned int attackLength = 5;
    const unsigned int traceLength = 1000;
    const unsigned int P = 5;
    int trace[traceLength] = { 0 };


    cudaError_t cudaStatus = countWithCuda(trace, attackLength, traceLength, P);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "countWithCuda failed!");
        return 1;
    }


    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    fprintf(stdout, "Trace: ");
    for (int i = 0; i < traceLength; i++)
    {
        fprintf(stdout, "%d ", trace[i]);
    }

    return 0;
}

cudaError_t countWithCuda(int* trace, const unsigned int attackLength, const unsigned int traceLength, const unsigned int P)
{
    int offset = 0;
    int *devTrace = 0;
    bool *devWork = 0;
    bool *hostWork = (bool*)malloc(sizeof(bool));
    *hostWork = true;
    cudaError_t cudaStatus;
    cudaStream_t execStream, transferStream;

    cudaStatus = cudaStreamCreate(&execStream);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaStreamCreate failed for execStream!");
        goto Error;
    }

    cudaStatus = cudaStreamCreate(&transferStream);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaStreamCreate failed for transferStream!");
        goto Error;
    }

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers
    cudaStatus = cudaMalloc((void**)&devTrace, attackLength * 1000 / P * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&devWork, sizeof(bool));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(devTrace, trace, traceLength * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(devWork, hostWork, sizeof(bool), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "counterKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    LARGE_INTEGER frequency;
    LARGE_INTEGER start, stop;

    QueryPerformanceFrequency(&frequency); // Get timer frequency
    QueryPerformanceCounter(&start); // Start timer

    unsigned int hostTime = (int)1000 * ((float)start.QuadPart / frequency.QuadPart);
    const unsigned int attackTimeout = hostTime;
    double elapsed = 0;

    while (attackLength * 1000 >= hostTime - attackTimeout)
    {
        QueryPerformanceCounter(&stop);
        hostTime = (int)1000 * ((float)stop.QuadPart / frequency.QuadPart);
        const unsigned int traceTimeout = hostTime - attackTimeout;

        elapsed = (float)(stop.QuadPart - start.QuadPart) / frequency.QuadPart;

        printProgress(elapsed / attackLength);

        // Launch a kernel on the GPU.
        counterKernel <<<1, 1, 0, execStream>>> (devTrace, attackLength, P, devWork, offset);

        while (P >= hostTime - traceTimeout) // Spin until timeout
        {
            QueryPerformanceCounter(&stop);
			hostTime = (int)1000 * ((float)stop.QuadPart / frequency.QuadPart);
        }

        *hostWork = false;

        cudaStatus = cudaMemcpyAsync(devWork, hostWork, sizeof(bool), cudaMemcpyHostToDevice, transferStream);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }

        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching counterKernal!\n", cudaStatus);
            goto Error;
        }

        offset++;

        *hostWork = true;

        cudaStatus = cudaMemcpyAsync(devWork, hostWork, sizeof(bool), cudaMemcpyHostToDevice, transferStream);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }

        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching counterKernal!\n", cudaStatus);
            goto Error;
        }
    }

    elapsed = (float)(stop.QuadPart - start.QuadPart) / frequency.QuadPart;

    fprintf(stdout, " Total Time Elapsed: %f\n", elapsed);

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(trace, devTrace, traceLength * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(devTrace);
    cudaFree(devWork);
    cudaStreamDestroy(execStream);
    cudaStreamDestroy(transferStream);
    free(hostWork);

    return cudaStatus;
}