


#include "io/io.h" // Printing functions
#include "intrinsics/instrinsics.h" // vx_threadID and vx_WarpID

struct args
{
	void * data;
};


void function(void * arg)
{
	struct args * real_arg = (struct args *) arg;

	unsigned tid = vx_threadID();
	unsigned wid = vx_WarpID();

	__if(something) // Control divergent if
	{

	}
	__else
	{

	}
	__endif
}

int main()
{

	void * data = vx_loadfile("filename.txt"); // The raw char data will be returned by vx_loadfile

	struct args arg;
	arg.data = data;

	vx_spawnWarps(numWarps, numThreads, function, &data);


}