
#include <stdio.h>
#include <cuda.h>

__global__ void add_matrices(
    float *c,      // out - pointer to result matrix c
    float *a,      // in  - pointer to summand matrix a
    float *b,      // in  - pointer to summand matrix b
    int m,         // in  - matrix length
    int n          // in  - matrix lenght
    )
{
	// To DO: Device a row major indexing
	int rowID = threadIdx.y + blockIdx.y * blockDim.y; 	// Row address
	int colID = threadIdx.x + blockIdx.x * blockDim.x;	// Column Address
	int elemID;											                    // Element address

    // a_ij = a[i][j], where a is in row major order
	if(rowID < m && colID < n){
		elemID = colID + rowID * n; 				
		c[elemID] = a[elemID] + b[elemID];
	}
}

int main( int argc, char* argv[] ){
    // determine matrix length
    int n = 10;      // set default length
    int m = 10;

    if ( argc > 1 ){
        n = atoi( argv[1] );  // override default length
        if ( n <= 0 ){
            fprintf( stderr, "Matrix length must be positive\n" );
            return EXIT_FAILURE;
        }
        if (argc > 2){
            m = atoi( argv[2] );
            if (m <= 0 ){
               fprintf( stderr, "Matrix length must be positive\n" );
               return EXIT_FAILURE;
            }
        }
    }

    // determine matrix size in bytes
    const size_t matrix_size = (n * m) * sizeof( float );

    // declare pointers to matrices in host memory and allocate memory
    float *a, *b, *c;
    a = (float*) malloc( matrix_size );
    b = (float*) malloc( matrix_size );
    c = (float*) malloc( matrix_size );

    // declare pointers to matrices in device memory and allocate memory
    float *a_d, *b_d, *c_d;
    cudaMalloc( (void**) &a_d, matrix_size );
    cudaMalloc( (void**) &b_d, matrix_size );
    cudaMalloc( (void**) &c_d, matrix_size );

    // initialize matrices and copy them to device
    for ( int i = 0; i < n*m; i++ )
    {
        a[i] =   1.0 * i;
        b[i] = 100.0 * i;        
    }
    cudaMemcpy( a_d, a, matrix_size, cudaMemcpyHostToDevice );
    cudaMemcpy( b_d, b, matrix_size, cudaMemcpyHostToDevice );

    // do calculation on device
    dim3 block_size( 16, 16 );
    dim3 num_blocks( ( n - 1 + block_size.x ) / block_size.x, ( m - 1 + block_size.y ) / block_size.y );
                   
    add_matrices<<< num_blocks, block_size >>>( c_d, a_d, b_d, m, n );

    // retrieve result from device and store on host
    cudaMemcpy( c, c_d, matrix_size, cudaMemcpyDeviceToHost );

    // print results for vectors up to length 100
    if ( n <= 100 && m <= 100)
    {
        for ( int i = 0; i < m; i++ )
        {
            for (int j = 0; j < n; j++)
            {
                printf("%4.0f ", a[i*n + j]);
            }
            printf("  ");
            for (int j = 0; j < n; j++)
            {
                printf("%4.0f ", b[i*n + j]);
            }
            printf("  ");
            for (int j = 0; j < n; j++)
            {
                printf("%4.0f ", c[i*n + j]);
            }
            printf("\n");
            
        }
    }

    // cleanup and quit
    cudaFree( a_d );
    cudaFree( b_d );
    cudaFree( c_d );
    free( a );
    free( b );
    free( c );
  
    return 0;
}
