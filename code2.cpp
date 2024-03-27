#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <fstream>
#include <mpi.h>
#include <omp.h>

using namespace std::chrono;
const int SIZE = 300;

void generateRandomMatrix(int matrix[][SIZE])
{

    for (int i = 0; i < SIZE; ++i)
        for (int j = 0; j < SIZE; ++j)
            matrix[i][j] = rand() % 100;
}

void performMatrixMultiplication(const int matrix1[][SIZE], const int matrix2[][SIZE], int resultMatrix[][SIZE], int rank, int size, int numThreads)
{

    int rows = SIZE / size;
    int start = rank * rows;
    int end = (rank + 1) * rows;

#pragma omp parallel for num_threads(numThreads) shared(matrix1, matrix2, resultMatrix) schedule(static)

    for (int i = start; i < end; ++i)
    {
        for (int j = 0; j < SIZE; ++j)
        {
            resultMatrix[i][j] = 0;
            for (int k = 0; k < SIZE; ++k)
            {
                resultMatrix[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }
}

int main(int argc, char **argv)
{

    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int numThreads = omp_get_max_threads();
    srand(time(nullptr) + rank); // Seed the random number generator differently for each process
    int matrix1[SIZE][SIZE];
    int matrix2[SIZE][SIZE];
    int resultMatrix[SIZE][SIZE];
    if (rank == 0)
    {
        generateRandomMatrix(matrix1);
        generateRandomMatrix(matrix2);
    }
    MPI_Bcast(&matrix1, SIZE * SIZE, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&matrix2, SIZE * SIZE, MPI_INT, 0, MPI_COMM_WORLD);
    auto startTime = high_resolution_clock::now();
    performMatrixMultiplication(matrix1, matrix2, resultMatrix, rank, size, numThreads);
    auto stopTime = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stopTime - startTime);
    if (rank == 0)
    {
        std::cout << "Multiplication completed in: " << duration.count() << " microseconds" << std::endl;
        std::ofstream outputFile("Result_matrix.txt");
        for (int i = 0; i < SIZE; ++i)
        {
            for (int j = 0; j < SIZE; ++j)
            {
                outputFile << resultMatrix[i][j] << "\t";
            }
            outputFile << std::endl;
        }
        outputFile << "Execution time: " << duration.count() << " microseconds";
        outputFile.close();
    }
    MPI_Finalize();
    return 0;
}