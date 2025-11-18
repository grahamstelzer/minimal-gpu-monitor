#define _GNU_SOURCE // for RTLD_NEXT ...?

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <dlfcn.h>

#include <pthread.h>

#include <cuda_runtime.h>       // REQUIRED for cudaDeviceProp, dim3, cudaEvent_t, cudaStream_t, etc.
#include <cuda_runtime_api.h>   // ok to include as well

// typedef enum cudaMemcpyKind {
//     cudaMemcpyHostToHost = 0,
//     cudaMemcpyHostToDevice = 1,
//     cudaMemcpyDeviceToHost = 2,
//     cudaMemcpyDeviceToDevice = 3,
//     cudaMemcpyDefault = 4
// } cudaMemcpyKind;


// cuda hooks:
typedef cudaError_t (*cudaMalloc_t)(void **, size_t);
typedef cudaError_t (*cudaFree_t)(void *);
// typedef cudaError_t (*cudaMemcpy_t)(void *, const void *, size_t, int); // cudaMemcpyKind instead of int but not cuda headers (for now)
typedef cudaError_t (*cudaLaunchKernel_t)(const void *, dim3, dim3, void **, size_t, cudaStream_t);

/*
    comments for cudaMalloc ... these roughly apply to all other cudaError_t functions
        prereqs:
        - since we are "hooking" into runtime, we must match the signatures of the functions we
            are tracking so use typedef cudaError_t
        - those functions DO need to run in the end, so we will call and use them with func 
            pointers with the same args (at the end in each return statement)
        args:
        - **ptr : for the ACTUAL cudaMalloc function address
        - size  : for the no. bytes used
        note:
        - mgm.c will be run constantly, so we set real_malloc to static which will cache the location of
            actual cudaMalloc
*/

cudaError_t cudaMalloc(void **ptr, size_t size) {
    static cudaMalloc_t real_malloc = NULL;
    if (!real_malloc) real_malloc = (cudaMalloc_t)dlsym(RTLD_NEXT, "cudaMalloc");

    cudaError_t err = real_malloc(ptr, size);

    // printf("[CUDA] malloc %zu bytes at %p\n", size, *ptr);
    
    static size_t process_total = 0;
    process_total += size;
    double gb = (double)process_total / (1024.0 * 1024.0 * 1024.0);
    printf("[PID %d] total allocated: %.3f GB (%zu bytes)\n", getpid(), gb, process_total);
    fflush(stdout);
    return err;
}

cudaError_t cudaFree(void *ptr) {       
    static cudaFree_t real_free = NULL;
    if (!real_free) real_free = (cudaFree_t)dlsym(RTLD_NEXT, "cudaFree");
    
    fflush(stdout);
    return real_free(ptr);
}

// cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, int kind) {
//     static cudaMemcpy_t real_memcpy = NULL;
//     if (!real_memcpy) real_memcpy = (cudaMemcpy_t)dlsym(RTLD_NEXT, "cudaMemcpy");
//     const char *dir = (kind == cudaMemcpyHostToDevice) ? "H2D" :
//                       (kind == cudaMemcpyDeviceToHost) ? "D2H" :
//                       (kind == cudaMemcpyDeviceToDevice) ? "D2D" : "Other";
//     printf("[CUDA] memcpy %zu bytes %s\n", count, dir);
//     fflush(stdout);
//     return real_memcpy(dst, src, count, kind);
// }

cudaError_t cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim,
                             void **args, size_t sharedMem, cudaStream_t stream) {
    static cudaLaunchKernel_t real_launch = NULL;
    if (!real_launch) real_launch = (cudaLaunchKernel_t)dlsym(RTLD_NEXT, "cudaLaunchKernel");
    printf("[CUDA] Launch kernel %p grid(%d,%d,%d) block(%d,%d,%d)\n",
           func, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z);
    fflush(stdout);
    return real_launch(func, gridDim, blockDim, args, sharedMem, stream);
}

// for gpu /proc polling:
void read_file(const char *path, const char *label) {
    FILE *f = fopen(path, "r");
    if (!f) return;
    char line[256];
    printf("== %s ==\n", label);
    while (fgets(line, sizeof(line), f)) {
        printf("%s", line);  
    }
    fclose(f);
    printf("\n");
}

void print_gpu_processes() {
    FILE *f = fopen("/proc/driver/nvidia/gpus/0/processes", "r");
    if (!f) return;
    char line[256];
    printf("PID     Type   GPU Memory (MiB)\n");
    printf("--------------------------------\n");
    while(fgets(line, sizeof(line), f)) {
        int gpu, pid;
        char type;
        size_t mem;
        if (sscanf(line, "%d %d %c %*s %zu", &gpu, &pid, &type, &mem) == 4) {
            printf("%-7d %-6c %zu\n", pid, type, mem);
        }
    }
    fclose(f);
}


void *gpu_poll_thread(void *arg) {
    int delay_ms = *(int *)arg;
    while(1) {
        // system("clear");
        // printf("\033[2J\033[H"); 
        read_file("/proc/driver/nvidia/gpus/0/information", "GPU Info");
        print_gpu_processes();
        usleep(delay_ms * 1000);
    }
    return NULL;
}




__attribute__((constructor))
void start_monitor() {
    int delay = 1000; // default 1000 ms
    pthread_t poller;
    if (pthread_create(&poller, NULL, gpu_poll_thread, &delay) != 0) {
        perror("pthread_create");
    }
}




__attribute__((constructor))
static void mgm_init() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);

    if (err != cudaSuccess) {
        fprintf(stderr, "[mgm] cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return;
    }

    fprintf(stderr, "[mgm] Detected %d CUDA device(s)\n", device_count);

    for (int d = 0; d < device_count; ++d) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, d);

        size_t freeMem = 0, totalMem = 0;
        cudaSetDevice(d);
        cudaMemGetInfo(&freeMem, &totalMem);

        fprintf(stderr,
            "[mgm] GPU %d: %s\n"
            "        Total memory: %zu bytes\n"
            "        Free memory:  %zu bytes\n",
            d,
            prop.name,
            totalMem,
            freeMem
        );
    }
}







// int main(int argc, char **argv) {
//     int delay = 1000; // default 1000ms
//     if (argc > 1) delay = atoi(argv[1]);

//     pthread_t poller;
//     if (pthread_create(&poller, NULL, gpu_poll_thread, &delay) != 0) {
//         perror("pthread_create");
//         return 1;
//     }

//     // Keep main thread alive for hooks to work
//     while(1) sleep(1);

//     return 0;
// }