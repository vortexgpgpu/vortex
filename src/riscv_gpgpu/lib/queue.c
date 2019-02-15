
#include "queue.h"
  
void queue_initialize(void)
{
    q.start_i      = 0;
    q.end_i        = 0;
    q.num_j        = 0;
    q.total_warps  = 7;
    q.active_warps = 0;
}

void queue_enqueue(Job * j)
{
    q.num_j++;

    // q.jobs[q.end_i] = j;
    
     q.jobs[q.end_i].wid       = j->wid;
     q.jobs[q.end_i].n_threads = j->n_threads;
     q.jobs[q.end_i].base_sp   = j->base_sp;
     q.jobs[q.end_i].func_ptr  = j->func_ptr;
     q.jobs[q.end_i].x         = j->x;
     q.jobs[q.end_i].y         = j->y;
     q.jobs[q.end_i].z         = j->z;
    if ((q.end_i + 1) < SIZE)
    {
        q.end_i++;
    }
    else
    {
        q.end_i = 0;
    }

}

void queue_dequeue(Job * r)
{
    q.num_j--;
    Job * j = &(q.jobs[q.start_i]);
    if ((q.start_i + 1) < SIZE)
    {
        q.start_i++;
    }
    else
    {
        q.start_i = 0;
    }

     r->wid       = j->wid;
     r->n_threads = j->n_threads;
     r->base_sp   = j->base_sp;
     r->func_ptr  = j->func_ptr;
     r->x         = j->x;
     r->y         = j->y;
     r->z         = j->z;

}

int queue_isFull(void)
{
    return (q.num_j == SIZE);
}

int queue_isEmpty(void)
{
    return (q.num_j == 0);
}

int queue_availableWarps()
{
    return (q.active_warps < q.total_warps);
}