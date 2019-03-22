
#ifndef __QUEUE__

#define __QUEUE__



#define SIZE 50
#define WARPS 7


typedef struct Job_t
{
	unsigned wid;
	unsigned n_threads;
	unsigned base_sp;
    unsigned func_ptr;
    void     * args;
    unsigned assigned_warp;

} Job;

typedef struct Queue_t
{
    unsigned start_i;
    unsigned end_i;
    unsigned num_j;
    unsigned total_warps;
    unsigned active_warps;
    struct Job_t jobs[SIZE];

} Queue;

Queue q[8];

void queue_initialize(Queue *);

void queue_enqueue(Queue *, Job *);

void queue_dequeue(Queue *, Job *);

int queue_isFull(Queue *);
int queue_isEmpty(Queue *);
int queue_availableWarps(Queue *);


void func();

#endif