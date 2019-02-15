
#include "queue.h"

unsigned x[] = {1, 1,  6, 0, 3, 1, 1, 2, 0, 3, 6, 7, 5, 7, 7, 9};
unsigned y[] = {0, 2,  2, 0, 5, 0, 1, 1, 4, 2, 0, 0, 3, 2, 3, 2};
unsigned z[] = {0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

int main() 
{ 

    Job j;
    j.func_ptr = (unsigned) func;
    j.x        = x;
    j.y        = y;
    j.z        = z;

    enqueue(j);
    enqueue(j);
    enqueue(j);
    enqueue(j);
    enqueue(j);
    enqueue(j);
    j = dequeue();
    j = dequeue();
    enqueue(j);
    enqueue(j);

    if (!isFull())
    {
        enqueue(j);
    }

    if (!isFull())
    {
        enqueue(j);
    }

    if (!isFull())
    {
        enqueue(j);
    }


    if (!isFull())
    {
        enqueue(j);
    }

    if (!isFull())
    {
        enqueue(j);
    }

    if (!isFull())
    {
        enqueue(j);
    }



    dequeue();
    dequeue();
    dequeue();
    dequeue();
    dequeue();
    dequeue();
    dequeue();
    dequeue();
    dequeue();
    dequeue();



    return 0; 
} 
  
void initialize_queue(void)
{
    q.start_i = 0;
    q.end_i   = 0;
    q.num_j   = 0;
}

void enqueue(Job j)
{
    q.num_j++;
    q.jobs[q.end_i] = j;
    if ((q.end_i + 1) < SIZE)
    {
        q.end_i++;
    }
    else
    {
        q.end_i = 0;
    }

}

Job dequeue(void)
{
    q.num_j--;
    Job j = q.jobs[q.start_i];
    if ((q.start_i + 1) < SIZE)
    {
        q.start_i++;
    }
    else
    {
        q.start_i = 0;
    }
}

int isFull(void)
{
    return (q.num_j == SIZE);
}

int isEmpty(void)
{
    return (q.num_j == 0);
}

void func()
{

}