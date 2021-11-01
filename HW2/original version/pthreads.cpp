#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string.h>
#include <vector>
#include <pthread.h>
#include "common.h"
using namespace std;

#define density 0.0005
#define mass    0.01
#define cutoff  0.01
#define min_r   (cutoff/100)
#define dt      0.0005

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

typedef struct{
   vector<particle_t*> *bins;
   particle_t *particles;
   double size;
   int bpr;
   int numbins;
   int n;
   
   int start;
   int finish;
} thread_args;

// calculate particle's bin number
int binNum(particle_t &p, int bpr) 
{
    return ( floor(p.x/cutoff) + bpr*floor(p.y/cutoff) );
}


void* thread_routine(void* arg){
    thread_args *args = (thread_args *) arg;

    for( int step = 0; step < NSTEPS; step++ ){
      // clear bins at each time step
      for (int m = 0; m < args->numbins; m++)
	      args->bins[m].clear();
      
      // place particles in bins
      //for (int i = 0; i < args->n; i++) 
	    //  args->bins[binNum(args->particles[i],args->bpr)].push_back(args->particles + i);
      //
      //  compute forces
      //
      //args->n = args->finish - args->start;

      for( int p = 0; p < args->n; p++ ){
        args->particles[p].ax = args->particles[p].ay = 0;
	      // find current particle's bin, handle boundaries
	      int cbin = binNum( args->particles[p], args->bpr );
	      int lowi = -1, highi = 1, lowj = -1, highj = 1;
	      if (cbin < args->bpr)
	        lowj = 0;
	      if (cbin % args->bpr == 0)
	        lowi = 0;
	      if (cbin % args->bpr == (args->bpr-1))
	        highi = 0;
	      if (cbin >= args->bpr*(args->bpr-1))
	        highj = 0;

	      // apply nearby forces
        
      pthread_mutex_lock(&mutex);

	    for (int i = lowi; i <= highi; i++)
	      for (int j = lowj; j <= highj; j++){
		      int nbin = cbin + i + args->bpr*j;
		      for (int k = 0; k < args->bins[nbin].size(); k++ )
		        apply_force( args->particles[p], *args->bins[nbin][k] );
	      }
      }
      
      pthread_mutex_unlock(&mutex);
        
        //
        //  move particles
        //
        for( int p = 0; p < args->n; p++ ) 
            move( args->particles[p] );
        
        //

    }

    //printf("done\n");
    pthread_exit(0);
    return 0;
}


int main(int argc, char **argv){
    if( find_option( argc, argv, "-h" ) >= 0 ){
        printf( "Options:\n" );
        printf( "-h to see this help\n" );
        printf( "-n <int> to set the number of particles\n" );
        printf( "-t <int> to set the number of threads\n" );
        printf( "-o <filename> to specify the output file name\n" );
        return 0;
    }

    // default number of particles is 1000 ig
    int n = read_int( argc, argv, "-n", 1000 );
    // default number of threads is 10
    int num_of_threads = read_int(argc, argv, "-t", 4);

    char *savename = read_string( argc, argv, "-o", NULL );
    
    FILE *fsave = savename ? fopen( savename, "w" ) : NULL;
    particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );
    set_size( n );
    init_particles( n, particles );
    
      // create spatial bins (of size cutoff by cutoff)
    double size = sqrt( density*n );
    int bpr = ceil(size/cutoff);
    int numbins = bpr*bpr;
    vector<particle_t*> *bins = new vector<particle_t*>[numbins];


    // creating the structs
    thread_args args_struct[num_of_threads];


    double simulation_time = read_timer( );
    //num_of_threads = 8;
    pthread_t tids[num_of_threads];

    for (int i=0; i<num_of_threads; i++){

        args_struct[i].bins = bins;
        args_struct[i].bpr = bpr;
        args_struct[i].numbins = numbins;
        args_struct[i].particles = particles;
        args_struct[i].size = size;

        args_struct[i].start = n/num_of_threads * (i);
        args_struct[i].finish = n/num_of_threads * (i+1);
        args_struct[i].n = args_struct[i].finish - args_struct[i].start;
        
        pthread_attr_t attr;
        pthread_attr_init(&attr);
        // limit is the arg we are sending to the thread_routine function
        pthread_create(&tids[i], &attr, thread_routine, (void *)&args_struct);
    }
        
    for (int i=0; i<num_of_threads; i++){
        pthread_join(tids[i], NULL);
    }

    simulation_time = read_timer( ) - simulation_time;
	
    printf( "n = %d, simulation time = %g seconds\n", n, simulation_time );

    return 0;
}

///root/GPU linux/HW2/original version
