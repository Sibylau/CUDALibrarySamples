/*
 * Copyright 1993-2022 NVIDIA Corporation.  All rights reserved.
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
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSpMV
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include <iostream>
#include <mtx_read.h>

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

int main(void) {
    // Host problem definition
    char filename[] = "/work/shared/users/phd/jl3952/workspace/Sparsity_Patten_Detection/format_preprocess/thermal_row_major.mtx";
    parse_COO<float> input(filename);
    int64_t   A_num_rows   = input.num_rows;
    int64_t   A_num_cols   = input.num_cols;
    int64_t   A_nnz        = input.num_nnz;
    std::cout << "A_num_rows: " << A_num_rows << std::endl;
    std::cout << "A_num_cols: " << A_num_cols << std::endl;
    std::cout << "A_nnz: " << A_nnz << std::endl;
    int* hA_rows = input.cooRowInd;
    int* hA_columns = input.cooColInd;
    float* hA_values = input.cooValue;
    
    // -----------------------------------------------------------
    // int   A_num_rows   = 10240;
    // int   A_num_cols   = 10240;
    // int   A_nnz        = 10237*2;
    // int   hA_rows[10237*2]    = { 0};
    // int   hA_columns[10237*2]    = { 0};
    // float   hA_values[10237*2]    = { 1.0f};
    // int ind = 0;
    // for (int i = 0; i < 10240; i++) {
    //     if (i < 10240 - 3) {
    //         hA_rows[ind] = i;
    //         hA_columns[ind] = i+3;
    //         hA_values[ind] = 1.0f;
    //         ind++;
    //     }
    //     if (i >= 3) {
    //         hA_rows[ind] = i;
    //         hA_columns[ind] = i-3;
    //         hA_values[ind] = 1.0f;
    //         ind++;
    //     }
    // }
    // for (int i = 3; i < 10240; i++) {
    //     hA_rows[i-3] = i;
    //     hA_columns[i-3] = i-3;
    //     hA_values[i-3] = 1.0f;
    // } // 10237
    // for (int i = 0; i < 10240-3; i++) {
    //     hA_rows[i+10237] = i;
    //     hA_columns[i+10237] = i+3;
    //     hA_values[i+10237] = 1.0f;
    // } // 10237
    // float hX[10240]         = { 1.0f};
    // float hY[10240]         = { 0.0f};
    // float hY_result[10240]  = { 0.0f};
    int64_t   B_size       = A_num_cols;
    int64_t   C_size       = A_num_rows;
    // -----------------------------------------------------------
    // int   hA_rows[]    = { 0, 0, 0, 1, 2, 2, 2, 3, 3 };
    // int   hA_columns[] = { 0, 2, 3, 1, 0, 2, 3, 1, 3 };
    // float hA_values[]  = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
    //                        6.0f, 7.0f, 8.0f, 9.0f };
    // -----------------------------------------------------------

    float* hX = (float*)malloc(B_size * sizeof(float));
    float* hY = (float*)malloc(C_size * sizeof(float));
    float* hY_result = (float*)malloc(C_size * sizeof(float));
    for (int i = 0; i < B_size; i++) {
        hX[i] = 1.0f;
    }
    for (int i = 0; i < C_size; i++) {
        hY[i] = 0.0f;
        hY_result[i] = 0.0f;
    }
    
    // for (int i = A_nnz - 10; i < A_nnz; i++) {
    //     std::cout << hA_rows[i] << ", " << hA_columns[i] << ", " << hA_values[i] << std::endl;
    // }
    float alpha        = 1.0f;
    float beta         = 0.0f;
    //--------------------------------------------------------------------------
    // Device memory management
    int   *dA_rows, *dA_columns;
    float *dA_values, *dX, *dY;
    CHECK_CUDA( cudaMalloc((void**) &dA_rows,    A_nnz * sizeof(int))        )
    CHECK_CUDA( cudaMalloc((void**) &dA_columns, A_nnz * sizeof(int))        )
    CHECK_CUDA( cudaMalloc((void**) &dA_values,  A_nnz * sizeof(float))      )
    CHECK_CUDA( cudaMalloc((void**) &dX,         B_size * sizeof(float)) )
    CHECK_CUDA( cudaMalloc((void**) &dY,         C_size * sizeof(float)) )

    CHECK_CUDA( cudaMemcpy(dA_rows, hA_rows, A_nnz * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_columns, hA_columns, A_nnz * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_values, hA_values, A_nnz * sizeof(float),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dX, hX, B_size * sizeof(float),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dY, hY, C_size * sizeof(float),
                           cudaMemcpyHostToDevice) )
    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    CHECK_CUSPARSE( cusparseCreate(&handle) )
    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE( cusparseCreateCoo(&matA, A_num_rows, A_num_cols, A_nnz,
                                      dA_rows, dA_columns, dA_values,
                                      CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
    // Create dense vector X
    CHECK_CUSPARSE( cusparseCreateDnVec(&vecX, B_size, dX, CUDA_R_32F) )
    // Create dense vector y
    CHECK_CUSPARSE( cusparseCreateDnVec(&vecY, C_size, dY, CUDA_R_32F) )
    // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseSpMV_bufferSize(
                                 handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
                                 CUSPARSE_SPMV_COO_ALG2, &bufferSize) )
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )

    // execute SpMV
    CHECK_CUSPARSE( cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
                                 CUSPARSE_SPMV_COO_ALG2, dBuffer) )


    cudaDeviceSynchronize();
    //--------------------------------------------------------------------------
    // device result check
    CHECK_CUDA( cudaMemcpy(hY, dY, C_size * sizeof(float),
                           cudaMemcpyDeviceToHost) )
    for (int i = 0; i < A_nnz; i++) {
        hY_result[hA_rows[i]] += hA_values[i] * hX[hA_columns[i]];
    }
    int correct = 1;
    for (int i = 0; i < A_num_rows; i++) {
        if (hY[i] != hY_result[i]) { // direct floating point comparison is not
            correct = 0;             // reliable
            break;
        }
    }
    // for (int i = 3433; i < 3443; i++) {
    //     std::cout << hY[i] << ", " << hY_result[i] << std::endl;
    // }
    if (correct)
        printf("spmv_coo_example test PASSED\n");
    else
        printf("spmv_coo_example test FAILED: wrong result\n");
    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroyDnVec(vecX) )
    CHECK_CUSPARSE( cusparseDestroyDnVec(vecY) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )
    //--------------------------------------------------------------------------
    // device memory deallocation
    CHECK_CUDA( cudaFree(dBuffer) )
    CHECK_CUDA( cudaFree(dA_rows) )
    CHECK_CUDA( cudaFree(dA_columns) )
    CHECK_CUDA( cudaFree(dA_values) )
    CHECK_CUDA( cudaFree(dX) )
    CHECK_CUDA( cudaFree(dY) )
    return EXIT_SUCCESS;
}
