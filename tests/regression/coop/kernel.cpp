#include <vx_spawn.h>
#include "common.h"
#include <vx_intrinsics.h> 



void kernel_body(kernel_arg_t* __UNIFORM__ arg) {

	//CASE 0    
    // int i = vx_thread_id();
	// vx_printf("Thread %d launched\n", i);
    // vx_barrier(0,0b10000000);
    // vx_printf("Thread %d synchronized\n", i);
    // vx_barrier(0,0b10000000);
    // vx_printf("Thread %d synchronized again\n", i);

	// //CASE 1
    // int i = vx_thread_id();
	// vx_printf("A (thread %d)\n", i);
    // vx_tile(0b10001000,16);
    // vx_printf("B (thread %d)\n", i);
    // vx_barrier(0,0b10001000); //2 groups
    // vx_printf("C (thread %d)\n", i);
    // vx_barrier(0,0b10001000);
    // vx_printf("D (thread %d)\n", i);

	// vx_tile(0b10000000,32); // default config

	// //CASE 2
	int i = vx_thread_id();
	vx_printf("A (thread %d)\n", i);

	vx_tile(0b10001000,16);

    vx_tile(0b00001000,16);
    {
        vx_printf("B (thread %d)\n", i);
        vx_barrier(0,0b00001000);
        vx_printf("C (thread %d)\n", i);
	}
    vx_tile(0b10000000,16);
    {
        vx_printf("D (thread %d)\n", i);
	}
    

    vx_tile(0b10001000,16);
    if(i%16 == 0)
    {
        vx_printf("E (thread %d)\n", i);
	}

	vx_tile(0b10000000,32); //default config

	// //CASE 3

    // int i = vx_thread_id();
    
    // // Initial thread output
    // vx_printf("A (thread %d)\n", i);

	// vx_tile(0b11111111,4);

    // vx_tile(0b11110000,4);{
    //     vx_printf("B (thread %d)\n", i);
    //     vx_tile(0b11000000,4); {
    //         vx_barrier(0,0b11000000);
    //         vx_printf("C (thread %d)\n", i);
    //         vx_tile(0b10000000,4); {
    //             vx_barrier(0,0b10000000);
    //             vx_printf("D (thread %d)\n", i);
    //         }
    //     }
        // vx_tile(0b00110000,4); {
        //     vx_printf("F (thread %d)\n", i);
        //     vx_barrier(0,0b00110000);
        // }
    // }

	// vx_tile(0b10000000,32); //default config

	// //CASE 4

    // int i = vx_thread_id();
	// int j = i + 1;

    // vx_printf("A (thread %d)\n", i);

    // vx_tile(0b10001000,16);
    
    // if (i % 2 == 1) 
    // {
    //     vx_printf("B (thread %d)\n", i * 2);
    // }
    // else
    // {
    //     vx_printf("B (thread %d)\n", i);
    // }
    // vx_barrier(0,0b10001000);
    
    // vx_printf("C (thread %d)\n", i + j);

	// vx_tile(0b10000000,32); //default config

	// //CASE 5

    // int i = vx_thread_id();
    // int j = i + 1;

    // vx_printf("A (thread %d)\n", i);
    // vx_tile(0b10000000,32);
    // vx_printf("B (thread %d, i = %d, j = %d)\n", vx_thread_id(), i, j);
    // vx_barrier(0,0b10000000);

	// vx_tile(0b10000000,32); //default config

	// //CASE 6

	// vx_printf("A (thread %d)\n", threadIdx.x);
    // vx_tile(0b10001000,16);
    // for (int i = 0; i < 5; i++) {
    //     vx_printf("C (thread %d): iteration %d\n", vx_thread_id(), i);
    // }

	// vx_tile(0b10000000,32); //default config

	// //CASE 7

    // int i = vx_thread_id();

    // vx_printf("A (thread %d)\n", i);
    // vx_tile(0b10001000,16);
    // vx_tile(0b10000000,16); {
    //     vx_printf("B (thread %d)\n", i);
    //     vx_barrier(0,0b10000000);
    //     vx_printf("C (thread %d)\n", i);
    // }
    // vx_tile(0b00001000,16);
    // {
    //     vx_printf("D (thread %d)\n", i);
    // }
    // vx_tile(0b10001000,16);
    // if (vx_thread_id() < 4) {
    //     vx_printf("E (thread %d)\n", i);
    // }

	// vx_tile(0b10000000,32); //default config

	// //CASE 8

	// int tid = vx_thread_id();

    // vx_printf("A (thread %d)\n", tid);

    // vx_tile(0b10101010,8);

    // vx_printf("B (thread %d)\n", vx_thread_id());

    // vx_barrier(0, 0b10101010);

    // vx_printf("C (thread %d)\n", tid);

	// vx_tile(0b10000000,32); //default config

	// //CASE 9
	
	vx_printf("A (thread %d)\n", vx_thread_id());

    vx_tile(0b10101010,8);

    vx_printf("B (thread %d)\n", vx_thread_id());

    vx_barrier(0,0b10101010);

    vx_printf("C (thread %d)\n", vx_thread_id());

	vx_tile(0b10000000,32); //default config

	// //CASE 10

    // int i = vx_thread_id();

	// if (i % 16 == 0)
    // {
    //     vx_printf("A (group %d)\n", i / 16);
    // }

    // vx_barrier(0,0b10000000);

    // vx_tile(0b10001000,16);

    // int group = i / 16;

    // vx_printf("B %d\n", i);
    // vx_barrier(0,0b10001000);

    // vx_barrier(0,0b10001000);
    
    // vx_printf("C %d\n", i);
    // vx_barrier(0,0b10001000);

	// vx_tile(0b10000000,32); //default config

	// //CASE 11

    // int k = vx_thread_id();
	// vx_tile(0b11111111,4);

    // for (int i = 0; i < 3; i++) {
    //     vx_printf("A(threadIdx %d, loopIdx %d)\n", k, i);
    //     vx_barrier(0, 0b11111111);
    //     vx_printf("B(threadIdx %d, loopIdx %d)\n", k, i);
    // }

	// vx_tile(0b10000000,32); //default config

	// //CASE 12

	// int sum = 0;

    // vx_tile(0b10001000,16);

    // for (int i = 0; i < 10; i++) {
    //     int k = i * vx_thread_id();
    //     int j = i * i;
    //     vx_barrier(0,0b10001000);
    //     sum += vx_thread_id() / 16 + k + j;
    // }

    // vx_printf("(group %d, id %d) sum = %d\n", vx_thread_id() / 16, threadIdx.x % 16, sum);

    // vx_tile(0b10000000,32); //default config

    //CASE 13

    // int sum = 0;
    // int j = vx_thread_id();
    // vx_tile(0b10001000,16);

    // for (int i = 0; i < 10; i++) {
    //     int k = vx_thread_id();
    //     sum += k;
    //     vx_barrier(0, 0b10001000);
    // }

    // vx_printf("(group %d, id %d) sum = %d\n", j / 16, j % 16, sum);
    // vx_tile(0b10000000,32); //default config

}                
          
int main() {
	kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
	return vx_spawn_threads(1, &arg->num_tasks, nullptr, (vx_kernel_func_cb)kernel_body, arg);
}
      