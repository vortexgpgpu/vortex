
void * vx_malloc_shared(unsigned);
void   vx_free(void *);

typedef struct
{
	void   * ptr;
	
} free_t;

void * heap_ptr = (void *) 0xFF000000;

free_t free_array[100];
unsigned free_index = 0;

