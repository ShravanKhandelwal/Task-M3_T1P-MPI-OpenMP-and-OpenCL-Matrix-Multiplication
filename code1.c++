#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <fstream>
#include <mpi.h>

using namespace std::chrono;

const int MATRIX_SIZE = 200;

// Function to fill a matrix with random values
void populateMatrix(int matrix[][MATRIX_SIZE])
{
    for (int row = 0; row < MATRIX_SIZE; ++row)
    {
        for (int col = 0; col < MATRIX_SIZE; ++col)
        {
            // Generate random values between 0 and 99
            matrix[row][col] = rand() % 100;
        }
    }
}

// Function to multiply two matrices
void multiplyMatrices(const int firstMatrix[][MATRIX_SIZE], const int secondMatrix[][MATRIX_SIZE], int resultMatrix[][MATRIX_SIZE], int start_row, int end_row)
{
    for (int row = start_row; row < end_row; ++row)
    {
        for (int col = 0; col < MATRIX_SIZE; ++col)
        {
            resultMatrix[row][col] = 0;
            for (int k = 0; k < MATRIX_SIZE; ++k)
            {
                // Perform matrix multiplication
                resultMatrix[row][col] += firstMatrix[row][k] * secondMatrix[k][col];
            }
        }
    }
}

int main(int argc, char* argv[])
{
    srand(time(nullptr)); // Seed the random number generator with current time

    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int firstMatrix[MATRIX_SIZE][MATRIX_SIZE];
    int secondMatrix[MATRIX_SIZE][MATRIX_SIZE];
    int resultMatrix[MATRIX_SIZE][MATRIX_SIZE];

    // Root process fills the input matrices with random values
    if (rank == 0)
    {
        populateMatrix(firstMatrix);
        populateMatrix(secondMatrix);
    }

    // Broadcast the input matrices to all processes
    MPI_Bcast(&firstMatrix, MATRIX_SIZE * MATRIX_SIZE, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&secondMatrix, MATRIX_SIZE * MATRIX_SIZE, MPI_INT, 0, MPI_COMM_WORLD);

    auto startTime = high_resolution_clock::now();

    // Calculate the row range for each process
    int rows_per_process = MATRIX_SIZE / size;
    int start_row = rank * rows_per_process;
    int end_row = (rank == size - 1) ? MATRIX_SIZE : start_row + rows_per_process;

    // Perform local matrix multiplication
    multiplyMatrices(firstMatrix, secondMatrix, resultMatrix, start_row, end_row);

    // Gather the results from all processes
    int* recvbuf = nullptr;
    if (rank == 0)
    {
        recvbuf = new int[MATRIX_SIZE * MATRIX_SIZE];
    }

    MPI_Gatherv(&resultMatrix[0][0], rows_per_process * MATRIX_SIZE, MPI_INT,
                recvbuf, nullptr, nullptr, MPI_INT, 0, MPI_COMM_WORLD);

    auto stopTime = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stopTime - startTime);

    // Root process writes the result to a file
    if (rank == 0)
    {
        std::ofstream outputFile("Parallel_Matrix_Multiplication_Result.txt");
        for (int row = 0; row < MATRIX_SIZE; ++row)
        {
            for (int col = 0; col < MATRIX_SIZE; ++col)
            {
                outputFile << recvbuf[row * MATRIX_SIZE + col] << "\t";
            }
            outputFile << std::endl;
        }
        outputFile << "Execution time: " << duration.count() << " microseconds";
        outputFile.close();
        delete[] recvbuf;
    }

    MPI_Finalize();
    return 0;
}