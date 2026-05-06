#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int device;
    cudaDeviceProp prop;

    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    printf("Device ID: %d\n", device);
    printf("Device name: %s\n", prop.name);
    printf("*Number of SMs: %d\n", prop.multiProcessorCount);
    printf("Compute Capability Major: %d\n", prop.major);
    printf("Compute Capability Minor: %d\n", prop.minor);
    printf("*maxThreadsPerBlock: %d\n", prop.maxThreadsPerBlock);
    printf("maxThreadsPerMultiProcessor: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("*totalGlobalMem: %dM\n", (int)(prop.totalGlobalMem / 1024 / 1024));
    printf("sharedMemPerBlock: %dKB\n", (int)(prop.sharedMemPerBlock / 1024));
    printf("*sharedMemPerMultiprocessor: %dKB\n", (int)(prop.sharedMemPerMultiprocessor / 1024));
    printf("totalConstMem: %dKB\n", (int)(prop.totalConstMem / 1024));
    printf("*multiProcessorCount: %d\n", prop.multiProcessorCount);
    printf("*Warp Size: %d\n", prop.warpSize);

    return 0;
}