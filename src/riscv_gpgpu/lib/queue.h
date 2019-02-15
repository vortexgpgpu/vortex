
#ifndef __QUEUE__

#define __QUEUE__



#define SIZE 10


typedef struct Job_t
{
    unsigned func_ptr;
    unsigned * x;
    unsigned * y;
    unsigned * z;

} Job;

typedef struct Queue_t
{

    struct Job_t jobs[SIZE];
    unsigned start_i;
    unsigned end_i;
    unsigned num_j;

} Queue;

Queue q;

void initialize_queue(void);

void enqueue(Job);

Job dequeue(void);

int isFull(void);
int isEmpty(void);


void func();

#endif