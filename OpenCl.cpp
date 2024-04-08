#include <iostream>
#include <fstream>
#include <chrono>
#include <CL/cl.h>

const int SIZE = 300;

void generateRandomMatrix(int matrix[][SIZE]) {
    for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE; ++j) {
            matrix[i][j] = rand() % 100;
        }
    }
}

int main(int argc, char** argv) {
    // Initialize OpenCL
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;

    cl_int err;
    err = clGetPlatformIDs(1, &platform, nullptr);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    queue = clCreateCommandQueue(context, device, 0, &err);

    // Load and build the OpenCL kernel
    std::ifstream kernelFile("matrix_multiplication.cl");
    std::string kernelSource((std::istreambuf_iterator<char>(kernelFile)), std::istreambuf_iterator<char>());
    const char* source = kernelSource.c_str();
    program = clCreateProgramWithSource(context, 1, &source, nullptr, &err);
    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    kernel = clCreateKernel(program, "matrix_multiplication", &err);

    // Allocate memory on the device
    cl_mem matrix1_dev = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * SIZE * SIZE, nullptr, &err);
    cl_mem matrix2_dev = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * SIZE * SIZE, nullptr, &err);
    cl_mem result_dev = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * SIZE * SIZE, nullptr, &err);

    // Generate random matrices
    int matrix1[SIZE][SIZE], matrix2[SIZE][SIZE], result[SIZE][SIZE];
    generateRandomMatrix(matrix1);
    generateRandomMatrix(matrix2);

    // Copy input matrices to the device
    err = clEnqueueWriteBuffer(queue, matrix1_dev, CL_TRUE, 0, sizeof(int) * SIZE * SIZE, matrix1, 0, nullptr, nullptr);
    err = clEnqueueWriteBuffer(queue, matrix2_dev, CL_TRUE, 0, sizeof(int) * SIZE * SIZE, matrix2, 0, nullptr, nullptr);

    // Set kernel arguments
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &matrix1_dev);
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &matrix2_dev);
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &result_dev);
    err = clSetKernelArg(kernel, 3, sizeof(int), &SIZE);

    // Execute the kernel
    size_t global_size[] = {SIZE, SIZE};
    auto start_time = std::chrono::high_resolution_clock::now();
    err = clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, global_size, nullptr, 0, nullptr, nullptr);
    clFinish(queue);
    auto end_time = std::chrono::high_resolution_clock::now();

    // Copy the result back to the host
    err = clEnqueueReadBuffer(queue, result_dev, CL_TRUE, 0, sizeof(int) * SIZE * SIZE, result, 0, nullptr, nullptr);

    // Print the result and execution time
    std::cout << "Multiplication completed in: " << std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() << " microseconds" << std::endl;

    std::ofstream outputFile("Result_matrix.txt");
    for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE; ++j) {
            outputFile << result[i][j] << "\t";
        }
        outputFile << std::endl;
    }
    outputFile << "Execution time: " << std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() << " microseconds";
    outputFile.close();

    // Clean up
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    clReleaseMemObject(matrix1_dev);
    clReleaseMemObject(matrix2_dev);
    clReleaseMemObject(result_dev);

    return 0;
}
