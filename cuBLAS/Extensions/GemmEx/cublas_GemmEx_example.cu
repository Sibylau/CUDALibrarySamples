/*
 * Copyright 2020 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <string>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <sstream>
#include <stdexcept>
#include "cublas_utils.h"

using data_type = float;

struct MatrixData {
    int rows;
    int cols;
    int nnz;  // Number of non-zero elements (for sparse matrices)
    std::vector<data_type> values;        // For dense format
};

MatrixData read_mtx_file(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    MatrixData matrix;
    std::string line;

    // Skip comments
    do {
        std::getline(file, line);
    } while (line[0] == '%');

    // Read header
    std::istringstream iss(line);
    int rows, cols, nnz;
    iss >> rows >> cols >> nnz;
    
    matrix.rows = rows;
    matrix.cols = cols;
    matrix.nnz = nnz;

    // Read dense matrix
    matrix.values.resize(rows * cols, 0.0f); // Initialize with zeros

    for (int i = 0; i < nnz; i++) {
        std::getline(file, line);
        iss.clear();
        iss.str(line);

        int row, col;
        float value;
        iss >> row >> col >> value;
        
        // MTX format is 1-based, convert to 0-based
        matrix.values[(row - 1) * cols + (col - 1)] = 1;
    }

    file.close();
    return matrix;
}

int main(int argc, char *argv[]) {
    MatrixData matrix;
    matrix = read_mtx_file(argv[1]);
    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;

    // const int m = 2;
    // const int n = 2;
    // const int k = 2;
    // const int lda = 2;
    // const int ldb = 2;
    // const int ldc = 2;
    // /*
    //  *   A = | 1.0 | 2.0 |
    //  *       | 3.0 | 4.0 |
    //  *
    //  *   B = | 5.0 | 6.0 |
    //  *       | 7.0 | 8.0 |
    //  */

    // const std::vector<data_type> A = {1.0, 3.0, 2.0, 4.0};
    // const std::vector<data_type> B = {5.0, 7.0, 6.0, 8.0};

    int m = matrix.rows;
    int n = 1024;
    int k = matrix.cols;
    int lda = m;
    int ldb = k;
    int ldc = m;

    // const std::vector<data_type> A = {1.0, 3.0, 2.0, 4.0};
    std::vector<data_type> A = matrix.values;
    std::vector<data_type> B(k * n);
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n; j++) {
            B[i * n + j] = 1.0;
        }
    }
    std::vector<data_type> C(m * n);
    std::vector<data_type> hC_result(m * n);
    // col-major matmul
    for (int j = 0; j < n; j++) {
        // Middle loop over rows of C and A
        for (int i = 0; i < m; i++) {
            float sum = 0.0f;
            // Inner loop over columns of A and rows of B
            for (int l = 0; l < k; l++) {
                // In column-major:
                // A[i,l] is at A[i + l*m]
                // B[l,j] is at B[l + j*k]
                // C[i,j] is at C[i + j*m]
                sum += A[i + l*m] * B[l + j*k];
            }
            hC_result[i + j*m] = sum;
        }
    }
    const data_type alpha = 1.0;
    const data_type beta = 0.0;

    data_type *d_A = nullptr;
    data_type *d_B = nullptr;
    data_type *d_C = nullptr;

    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;

    // printf("A\n");
    // print_matrix(m, k, A.data(), lda);
    // printf("=====\n");

    // printf("B\n");
    // print_matrix(k, n, B.data(), ldb);
    // printf("=====\n");

    /* step 1: create cublas handle, bind a stream */
    CUBLAS_CHECK(cublasCreate(&cublasH));

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    /* step 2: copy data to device */
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_C), sizeof(data_type) * C.size()));

    CUDA_CHECK(cudaMemcpyAsync(d_A, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice,
                               stream));

    /* step 3: compute */
    cudaEvent_t start, stop;
    cudaDeviceSynchronize();
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < 1000; i++) {
    CUBLAS_CHECK(cublasGemmEx(
        cublasH, transa, transb, m, n, k, &alpha, d_A, traits<data_type>::cuda_data_type, lda, d_B,
        traits<data_type>::cuda_data_type, ldb, &beta, d_C, traits<data_type>::cuda_data_type, ldc,
        CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed_time_ms = 0.0;
    cudaEventElapsedTime(&elapsed_time_ms, start, stop);
    std::cout << "average_time = " << elapsed_time_ms / 1000 << " ms" << std::endl;

    /* step 4: copy data to host */
    CUDA_CHECK(cudaMemcpyAsync(C.data(), d_C, sizeof(data_type) * C.size(), cudaMemcpyDeviceToHost,
                               stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    /*
     *   C = | 19.0 | 22.0 |
     *       | 43.0 | 50.0 |
     */

    // printf("C\n");
    // print_matrix(m, n, C.data(), ldc);
    // printf("=====\n");
    bool error = false;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (abs(C[i * n + j] - hC_result[i * n + j]) > 1e-6) {
                printf("Error at (%d, %d): %f != %f\n", i, j, C[i * n + j], hC_result[i * n + j]);
                error = true;
            }
        }
    }
    if (!error) {
        printf("Test passed\n");
    }


    /* free resources */
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    CUBLAS_CHECK(cublasDestroy(cublasH));

    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}
