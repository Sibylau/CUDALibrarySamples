/*******************************************************************************
* Read .mtx and .tns file format
*******************************************************************************/

#ifndef _MTX_READ_H_
#define _MTX_READ_H_

#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <set>
#include <math.h> 
#include <cstring>
#include <cinttypes>

#include "mkl_spblas.h"

using namespace std;

static char *toLower(char *token) {
  for (char *c = token; *c; c++)
    *c = tolower(*c);
  return token;
}

static void readMTXHeader(FILE* file, char* fileName, uint64_t* metaData, char* field, char* symmetry) {
    char line[1025];
    char header[64];
    char object[64];
    char format[64];
    
    // Read header line.
    printf("read MTX filename %s\n", fileName);                                                       
    if (fscanf(file, "%63s %63s %63s %63s %63s\n", header, object, format, field,
                symmetry) != 5) {
        fprintf(stderr, "Corrupt header in %s\n", fileName);
        exit(1);
    }
    // Make sure this is a general sparse matrix.
    if (strcmp(toLower(header), "%%matrixmarket") ||
        strcmp(toLower(object), "matrix") ||
        strcmp(toLower(format), "coordinate")) {
        fprintf(stderr,
                "Cannot find a coordinate format sparse matrix in %s\n", fileName);
        exit(1);
    }
    // if (strcmp(toLower(field), "pattern"))
    // strcmp(toLower(symmetry), "general")

    // Skip comments.
    while (1) {
        if (!fgets(line, 1025, file)) {
        fprintf(stderr, "Cannot find data in %s\n", fileName);
        exit(1);
        }
        if (line[0] != '%')
        break;
    }
    // Next line contains M N NNZ.
    metaData[0] = 2; // rank
    if (sscanf(line, "%" PRIu64 "%" PRIu64 "%" PRIu64 "\n", metaData + 2, metaData + 3,
                metaData + 1) != 3) {
        fprintf(stderr, "Cannot find size in %s\n", fileName);
        exit(1);
    }
}

static void readFROSTTHeader(FILE* file, char* fileName, uint64_t* metaData) {

}

template <typename valueTp>
class parse_affine_DIA {
public:
    parse_affine_DIA(char* fileName, int m00, int m01, int m10, int m11) {
        FILE *file = fopen(fileName, "r");   
        printf("filename %s\n", fileName);                                                       
        if (!file) {                                                                                
            fprintf(stderr, "Cannot find %s\n", fileName);                                          
            exit(1);                                                                                
        }                                                                                           
                                                                                                    
        uint64_t metaData[512];  
        char field[64];
        char symmetry[64];                                                                   
        if (strstr(fileName, ".mtx")) {                                                             
            readMTXHeader(file, fileName, metaData, field, symmetry);                                                
        } else if (strstr(fileName, ".tns")) {                                                      
            readFROSTTHeader(file, fileName, metaData);                                             
        } else {                                                                                    
            fprintf(stderr, "Unknown format %s\n", fileName);                                       
            exit(1);                                                                                
        }                                                                                       
                                                                                                    
        num_nnz = metaData[1]; 
        num_rows = metaData[2];
        num_cols = metaData[3];
        if (m10==0)
            size = num_cols;
        else
            size = num_rows;
        

        rowVec = (MKL_INT*)malloc(num_nnz * sizeof(MKL_INT));
        colVec = (MKL_INT*)malloc(num_nnz * sizeof(MKL_INT));
        valVec = (valueTp*)malloc(num_nnz * sizeof(valueTp));

        bool isFieldPattern = strcmp(toLower(field), "pattern");

        if (!strcmp(toLower(field), "complex")) {
            fprintf(stderr, "Complex data type not yet supported.\n");                                       
            exit(1); 
        } 

        if (strcmp(toLower(symmetry), "general") && strcmp(toLower(symmetry), "symmetric")) {
            fprintf(stderr, "Non general matrix structure not yet supported.\n");                                       
            exit(1); 
        } 

        std::set<MKL_INT> diaOffVec;
        for (unsigned i = 0; i < num_nnz; i++) {
            MKL_INT rowInd = -1;                                                                      
            MKL_INT colInd = -1;                                                                      
            if (fscanf(file, "%d", &rowInd) != 1) {                                          
                fprintf(stderr, "Cannot find next row index at line %u\n", i);                    
                exit(1);                                                                        
            }
            if (fscanf(file, "%d", &colInd) != 1) {                                          
                fprintf(stderr, "Cannot find next col index at line %u\n", i);                    
                exit(1);                                                                        
            }
            MKL_INT row = rowInd - 1;
            MKL_INT col = colInd - 1;
            MKL_INT offset = row * m00 + col * m01;
            diaOffVec.insert(offset);

            rowVec[i] = row;
            colVec[i] = col;
            
            if (!isFieldPattern) {
                // Field is Pattern
                valVec[i] = 1;
            } else {
                valueTp value;
                if (fscanf(file, "%f", &value) != 1) {                                          
                    fprintf(stderr, "Cannot find next value at line %u\n", i);                    
                    exit(1);                                                                        
                }
                valVec[i] = value;
            }
        }

        diagSize = diaOffVec.size();
        diaOffset = (MKL_INT*)malloc(diagSize * sizeof(MKL_INT));
        diaValue = (valueTp*)malloc(diagSize * size * sizeof(valueTp));
        // printf("diagSize = %d\ndiagOffset=\n", diagSize);
        for (unsigned i = 0; i < diagSize; i++) {
            diaOffset[i] = *next(diaOffVec.begin(), i);
            // printf("%d ", diaOffset[i]);
            for (unsigned j = 0; j < size; j++) {
                diaValue[i * size + j] = 0;
            }
        }
        for (unsigned i = 0; i < num_nnz; i++) {
            MKL_INT row = rowVec[i];
            MKL_INT col = colVec[i];
            MKL_INT offset = row * m00 + col * m01;
            MKL_INT index = row * m10 + col * m11;
            int ind = getSetIndex(diaOffVec, offset);
            diaValue[ind * size + index] = valVec[i];
        }
    }

    int getSetIndex(std::set<MKL_INT> &s, MKL_INT val) {
        int index = 0;
        for (auto e : s) {
            if (e == val) {
                return index;
            }
            index++;
        }
        return -1;
    }

    ~parse_affine_DIA() {
        free(diaOffset);
        free(diaValue);
        free(rowVec);
        free(colVec);
        free(valVec);
    }

    int num_rows, num_cols, num_nnz, diagSize, size;
    MKL_INT* diaOffset;
    valueTp* diaValue;
    MKL_INT* rowVec;
    MKL_INT* colVec;
    valueTp* valVec;
};


template <typename valueTp>
class parse_CSC {
public:
    parse_CSC(char* fileName) {
        FILE *file = fopen(fileName, "r");   
        printf("filename %s\n", fileName);                                                       
        if (!file) {                                                                                
            fprintf(stderr, "Cannot find %s\n", fileName);                                          
            exit(1);                                                                                
        }                                                                                           
                                                                                                    
        uint64_t metaData[512];  
        char field[64];
        char symmetry[64];                                                                   
        if (strstr(fileName, ".mtx")) {                                                             
            readMTXHeader(file, fileName, metaData, field, symmetry);                                                
        } else if (strstr(fileName, ".tns")) {                                                      
            readFROSTTHeader(file, fileName, metaData);                                             
        } else {                                                                                    
            fprintf(stderr, "Unknown format %s\n", fileName);                                       
            exit(1);                                                                                
        } 

        // printf("in getTensorIndices  :\n");
        // for (unsigned i = 0; i < 4; i++)
        //     printf("metaData[%u] = %lu \n", i, metaData[i]);                                                                                          
                                                                                                    
        num_nnz = metaData[1]; 
        num_rows = metaData[2];
        num_cols = metaData[3];

        cscColPtr = (MKL_INT*)malloc((num_cols + 1) * sizeof(MKL_INT));
        cscRowInd = (MKL_INT*)malloc(num_nnz * sizeof(MKL_INT));
        cscValue = (valueTp*)malloc(num_nnz * sizeof(valueTp));

        bool isFieldPattern = strcmp(toLower(field), "pattern");

        if (!strcmp(toLower(field), "complex")) {
            fprintf(stderr, "Complex data type not yet supported.\n");                                       
            exit(1); 
        } 

        if (strcmp(toLower(symmetry), "general") && strcmp(toLower(symmetry), "symmetric")) {
            fprintf(stderr, "Non general matrix structure not yet supported.\n");                                       
            exit(1); 
        } 

        MKL_INT lastRowInd = 0;
        // cscColPtr[0] = 0;
        for (unsigned i = 0; i < num_nnz; i++) {
            MKL_INT rowInd = -1;                                                                      
            MKL_INT colInd = -1;                                                                      
            if (fscanf(file, "%d", &rowInd) != 1) {                                          
                fprintf(stderr, "Cannot find next row index at line %lu\n", i);                    
                exit(1);                                                                        
            }
            cscRowInd[i] = rowInd - 1;
            if (fscanf(file, "%d", &colInd) != 1) {                                          
                fprintf(stderr, "Cannot find next col index at line %lu\n", i);                    
                exit(1);                                                                        
            }
            while (colInd > lastRowInd) {
                cscColPtr[lastRowInd++] = i;
            }
            if (!isFieldPattern) {
                // Field is Pattern
                cscValue[i] = 1;
            } else {
                valueTp value;
                if (fscanf(file, "%lf", &value) != 1) {                                          
                    fprintf(stderr, "Cannot find next value at line %lu\n", i);                    
                    exit(1);                                                                        
                }
                cscValue[i] = value;
            }
        }
        for(int i = lastRowInd; i <= num_cols; i++) 
            cscColPtr[i] = num_nnz;
    }

    ~parse_CSC() {
        free(cscColPtr);
        free(cscRowInd);
        free(cscValue);
    }

    int num_rows, num_cols, num_nnz;
    MKL_INT* cscColPtr;
    MKL_INT* cscRowInd;
    valueTp* cscValue;
};

template <typename valueTp>
class parse_CSR {
public:
    parse_CSR(char* fileName) {
        FILE *file = fopen(fileName, "r");   
        printf("filename %s\n", fileName);                                                       
        if (!file) {
           fprintf(stderr, "Cannot find %s\n", fileName);
           exit(1);                                                                              
        }
        
        uint64_t metaData[512];  
        char field[64];
        char symmetry[64];                                                                   
        if (strstr(fileName, ".mtx")) {
            readMTXHeader(file, fileName, metaData, field, symmetry);
        } else if (strstr(fileName, ".tns")) {
            readFROSTTHeader(file, fileName, metaData);                                             
        } else {
            fprintf(stderr, "Unknown format %s\n", fileName);                                       
            exit(1);                                                                                
        } 

        // printf("in getTensorIndices  :\n");
        // for (unsigned i = 0; i < 4; i++)
        //     printf("metaData[%u] = %lu \n", i, metaData[i]);                                                                                          
                                                                                                    
        num_nnz = metaData[1]; 
        num_rows = metaData[2];
        num_cols = metaData[3];

        csrRowPtr = (MKL_INT*)malloc((num_rows + 1) * sizeof(MKL_INT));
        csrColInd = (MKL_INT*)malloc(num_nnz * sizeof(MKL_INT));
        csrValue = (valueTp*)malloc(num_nnz * sizeof(valueTp));

        bool isFieldPattern = strcmp(toLower(field), "pattern");

        if (!strcmp(toLower(field), "complex")) {
            fprintf(stderr, "Complex data type not yet supported.\n");                                       
            exit(1); 
        } 

        if (strcmp(toLower(symmetry), "general")) {
            fprintf(stderr, "Non general matrix structure not yet supported.\n");                                       
            exit(1); 
        } 

        MKL_INT lastRowInd = 0;
        // csrRowPtr[0] = 0;
        for (unsigned i = 0; i < num_nnz; i++) {
            MKL_INT rowInd = -1;                                                                      
            MKL_INT colInd = -1;                                                                      
            if (fscanf(file, "%d", &rowInd) != 1) {                                          
                fprintf(stderr, "Cannot find next row index at line %lu\n", i);                    
                exit(1);                                                                        
            }
            while (rowInd > lastRowInd) {
                csrRowPtr[lastRowInd] = i;
                lastRowInd = lastRowInd + 1; 
            }
            if (fscanf(file, "%d", &colInd) != 1) {                                          
                fprintf(stderr, "Cannot find next col index at line %lu\n", i);                    
                exit(1);                                                                        
            }
            csrColInd[i] = colInd - 1;
            
            if (!isFieldPattern) {
                // Field is Pattern
                csrValue[i] = 1;
            } else {
                valueTp value;
                if (fscanf(file, "%lf", &value) != 1) {                                          
                    fprintf(stderr, "Cannot find next value at line %lu\n", i);                    
                    exit(1);                                                                        
                }
                csrValue[i] = value;
            }
        }
        for (unsigned i = lastRowInd; i <= num_rows; i++ )
            csrRowPtr[i] = num_nnz;
    }

    ~parse_CSR() {
        free(csrRowPtr);
        free(csrColInd);
        free(csrValue);
    }

    int num_rows, num_cols, num_nnz;
    MKL_INT* csrRowPtr;
    MKL_INT* csrColInd;
    valueTp* csrValue;
};

template <typename valueTp>
class parse_COO {
public:
    parse_COO(char* fileName) {
        FILE *file = fopen(fileName, "r");   
        printf("filename %s\n", fileName);                                                       
        if (!file) {                                                                                
            fprintf(stderr, "Cannot find %s\n", fileName);                                          
            exit(1);                                                                                
        }                                                                                           
                                                                                                    
        uint64_t metaData[512];  
        char field[64];
        char symmetry[64];                                                                   
        if (strstr(fileName, ".mtx")) {                                                             
            readMTXHeader(file, fileName, metaData, field, symmetry);                                                
        } else if (strstr(fileName, ".tns")) {                                                      
            readFROSTTHeader(file, fileName, metaData);                                             
        } else {                                                                                    
            fprintf(stderr, "Unknown format %s\n", fileName);                                       
            exit(1);                                                                                
        } 

        // printf("in getTensorIndices  :\n");
        // for (unsigned i = 0; i < 4; i++)
        //     printf("metaData[%u] = %lu \n", i, metaData[i]);                                                                                          
                                                                                                    
        num_nnz = metaData[1]; 
        num_rows = metaData[2];
        num_cols = metaData[3];

        cooRowInd = (MKL_INT*)malloc(num_nnz * sizeof(MKL_INT));
        cooColInd = (MKL_INT*)malloc(num_nnz * sizeof(MKL_INT));
        cooValue = (valueTp*)malloc(num_nnz * sizeof(valueTp));

        bool isFieldPattern = strcmp(toLower(field), "pattern");

        if (!strcmp(toLower(field), "complex")) {
            fprintf(stderr, "Complex data type not yet supported.\n");                                       
            exit(1); 
        } 

        if (strcmp(toLower(symmetry), "general")) {
            fprintf(stderr, "Non general matrix structure not yet supported.\n");                                       
            exit(1); 
        } 

        for (unsigned i = 0; i < num_nnz; i++) {
            MKL_INT rowInd = -1;
            MKL_INT colInd = -1;                                                                      
            if (fscanf(file, "%d", &rowInd) != 1) {                                          
                fprintf(stderr, "Cannot find next row index at line %lu\n", i);                    
                exit(1);                                                                        
            }
            cooRowInd[i] = rowInd;
            if (fscanf(file, "%d", &colInd) != 1) {                                          
                fprintf(stderr, "Cannot find next col index at line %lu\n", i);                    
                exit(1);                                                                        
            }
            cooColInd[i] = colInd;
            if (!isFieldPattern) {
                // Field is Pattern
                cooValue[i] = 1;
            } else {
                valueTp value;
                if (fscanf(file, "%lf", &value) != 1) {                                          
                    fprintf(stderr, "Cannot find next value at line %lu\n", i);                    
                    exit(1);                                                                        
                }
                cooValue[i] = val1ue;
            }
        }
    }

    ~parse_COO() {
        free(cooRowInd);
        free(cooColInd);
        free(cooValue);
    }

    int num_rows, num_cols;
    int num_nnz;
    MKL_INT* cooRowInd;
    MKL_INT* cooColInd;
    valueTp* cooValue;
};

#endif
