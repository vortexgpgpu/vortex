#ifndef _BACKPROP_H_
#define _BACKPROP_H_

#define BIGRND 0x7fffffff
#define THREADS 256
#define WIDTH 16  // shared memory width  
#define HEIGHT 16 // shared memory height
#define BLOCK_SIZE 16

#define ETA 0.3       //eta value
#define MOMENTUM 0.3  //momentum value
#define NUM_THREAD 4  //OpenMP threads

#include <CL/cl.h>

typedef struct {
  int input_n;                  /* number of input units */
  int hidden_n;                 /* number of hidden units */
  int output_n;                 /* number of output units */

  float *input_units;          /* the input units */
  float *hidden_units;         /* the hidden units */
  float *output_units;         /* the output units */

  float *hidden_delta;         /* storage for hidden unit error */
  float *output_delta;         /* storage for output unit error */

  float *target;               /* storage for target vector */

  float **input_weights;       /* weights from input to hidden layer */
  float **hidden_weights;      /* weights from hidden to output layer */

                                /*** The next two are for momentum ***/
  float **input_prev_weights;  /* previous change on input to hidden wgt */
  float **hidden_prev_weights; /* previous change on hidden to output wgt */
} BPNN;


/*** User-level functions ***/

//void bpnn_initialize();
void bpnn_initialize(int seed);
BPNN *bpnn_create(int n_in, int n_hidden, int n_out);
void bpnn_free(BPNN *net);
//BPNN *bpnn_create();
//void bpnn_free();
void bpnn_train(BPNN *net, float *eo, float *eh);
//void bpnn_train();
//void bpnn_feedforward();
void bpnn_feedforward(BPNN *net);
void bpnn_save(BPNN *net, char *filename);
//void bpnn_save();
//BPNN *bpnn_read();
BPNN *bpnn_read(char *filename);
void load(BPNN *net);
int bpnn_train_kernel(BPNN *net, float *eo, float *eh);
void bpnn_layerforward(float *l1, float *l2, float **conn, int n1, int n2);
void bpnn_output_error(float *delta, float *target, float *output, int nj, float *err);
void bpnn_hidden_error(float *delta_h, int nh, float *delta_o, int no, float **who, float *hidden, float *err); 
void bpnn_adjust_weights(float *delta, int ndelta, float *ly, int nly, float **w, float **oldw);
int setup(int argc, char** argv);
float **alloc_2d_dbl(int m, int n);
float squash(float x);

/*** OpenCL config variables ***/
extern int platform_id_inuse;
extern int device_id_inuse;
extern cl_device_type device_type;

#endif
