// # ifdef __cplusplus
// extern "C" {
// # endif

// #ifndef LIST_H
// # define LIST_H

//===============================================================================================================================================================================================================200
//	DEFINE/INCLUDE
//===============================================================================================================================================================================================================200

//======================================================================================================================================================150
//	INCLUDE (for some reason these are not recognized when defined in main file before this one is included)
//======================================================================================================================================================150

#include <stdint.h>					// (in path known to compiler)			needed by uint32_t
#include <stdbool.h>				// (in path known to compiler)			needed by true/false, bool
#include <stdlib.h>					// (in path known to compiler)			needed by malloc

//======================================================================================================================================================150
//	DEFINE
//======================================================================================================================================================150

#define fp float

#define Version "1.5"

#ifdef WINDOWS
	#define bool char
	#define false 0
	#define true 1
#endif

/* #define DEFAULT_ORDER 256 */

#ifdef RD_WG_SIZE_0_0
        #define  DEFAULT_ORDER RD_WG_SIZE_0_0
#elif defined(RD_WG_SIZE_0)
        #define  DEFAULT_ORDER RD_WG_SIZE_0
#elif defined(RD_WG_SIZE)
        #define  DEFAULT_ORDER RD_WG_SIZE
#else
        #define  DEFAULT_ORDER 256
        //#define  DEFAULT_ORDER 16
#endif

#ifdef RD_WG_SIZE_1_0
        #define  DEFAULT_ORDER_2 RD_WG_SIZE_1_0
#elif defined(RD_WG_SIZE_1)
        #define  DEFAULT_ORDER_2 RD_WG_SIZE_1
#elif defined(RD_WG_SIZE)
        #define  DEFAULT_ORDER_2 RD_WG_SIZE
#else
        #define  DEFAULT_ORDER_2 256
        //#define  DEFAULT_ORDER_2 16
#endif


#define malloc(size) ({                                                   \
  void *_tmp;                                                             \
                                                                          \
  if (!(_tmp = malloc(size))) {                                           \
    fprintf(stderr, "Allocation failed at %s:%d!\n", __FILE__, __LINE__); \
    exit(-1);                                                             \
  }                                                                       \
                                                                          \
  _tmp;                                                                   \
})

//======================================================================================================================================================150
//	STRUCTURES
//======================================================================================================================================================150

// struct list_item;
typedef struct list_item list_item_t;

typedef struct list_t {
  list_item_t *head, *tail;
  uint32_t length;
  int32_t (*compare)(const void *key, const void *with);
  void (*datum_delete)(void *);
} list_t;

typedef list_item_t *list_iterator_t;
typedef list_item_t *list_reverse_iterator_t;

/* Type representing the record
* to which a given key refers.
* In a real B+ tree system, the
* record would hold data (in a database)
* or a file (in an operating system)
* or some other information.
* Users can rewrite this part of the code
* to change the type and content
* of the value field.
*/
typedef struct record {
	int value;
} record;

/* Type representing a node in the B+ tree.
* This type is general enough to serve for both
* the leaf and the internal node.
* The heart of the node is the array
* of keys and the array of corresponding
* pointers.  The relation between keys
* and pointers differs between leaves and
* internal nodes.  In a leaf, the index
* of each key equals the index of its corresponding
* pointer, with a maximum of order - 1 key-pointer
* pairs.  The last pointer points to the
* leaf to the right (or NULL in the case
* of the rightmost leaf).
* In an internal node, the first pointer
* refers to lower nodes with keys less than
* the smallest key in the keys array.  Then,
* with indices i starting at 0, the pointer
* at i + 1 points to the subtree with keys
* greater than or equal to the key in this
* node at index i.
* The num_keys field is used to keep
* track of the number of valid keys.
* In an internal node, the number of valid
* pointers is always num_keys + 1.
* In a leaf, the number of valid pointers
* to data is always num_keys.  The
* last leaf pointer points to the next leaf.
*/
typedef struct node {
	void ** pointers;
	int * keys;
	struct node * parent;
	bool is_leaf;
	int num_keys;
	struct node * next; // Used for queue.
} node;

// 
typedef struct knode {
	int location;
	int indices [DEFAULT_ORDER + 1];
	int  keys [DEFAULT_ORDER + 1];
	bool is_leaf;
	int num_keys;
} knode; 

struct list_item {
  struct list_item *pred, *next;
  void *datum;
};

//===============================================================================================================================================================================================================200
//	PROTOTYPES
//===============================================================================================================================================================================================================200

//======================================================================================================================================================150
// Other
//======================================================================================================================================================150

void 
list_item_init(	list_item_t *li, 
				void *datum);

void 
list_item_delete(	list_item_t *li, 
					void (*datum_delete)(void *datum));

void 
list_insert_item_tail(	list_t *l, 
						list_item_t *i);

void 
list_insert_item_before(list_t *l, 
						list_item_t *next, 
						list_item_t *i);

void 
list_insert_item_after(	list_t *l, 
						list_item_t *pred, 
						list_item_t *i);

void 
list_insert_item_sorted(list_t *l, 
						list_item_t *i);

//======================================================================================================================================================150
// ???
//======================================================================================================================================================150

void 
list_init(	list_t *l,
			int32_t (*compare)(const void *key, const void *with),
			void (*datum_delete)(void *datum));

void 
list_delete(list_t *l);

void 
list_reset(list_t *l);

void 
list_insert_head(	list_t *l, 
					void *v);

void 
list_insert_tail(	list_t *l, 
					void *v);

void 
list_insert_before(list_t *l, 
					list_item_t *next, 
					void *v);

void 
list_insert_after(	list_t *l, 
					list_item_t *pred, 
					void *v);

void 
list_insert_sorted(	list_t *l, 
					void *v);

void 
list_insert_item_head(	list_t *l, 
						list_item_t *i);

void 
list_remove_item(	list_t *l, 
					list_item_t *i);

void 
list_remove_head(list_t *l);

void 
list_remove_tail(list_t *l);

list_item_t *
list_find_item(	list_t *l, 
				void *datum);

list_item_t *
list_get_head_item(list_t *l);

list_item_t *
list_get_tail_item(list_t *l);

void *
list_find(	list_t *l, 
			void *datum);

void *
list_get_head(list_t *l);

void *
list_get_tail(list_t *l);

uint32_t 
list_get_length(list_t *l);

bool 
list_is_empty(list_t *l);

bool 
list_not_empty(list_t *l);

void 
list_visit_items(	list_t *l, 
					void (*visitor)(void *v));

void *
list_item_get_datum(list_item_t *li);

void 
list_iterator_init(	list_t *l, 
					list_iterator_t *li);

void 
list_iterator_delete(list_iterator_t *li);

void 
list_iterator_next(list_iterator_t *li);

void 
list_iterator_prev(list_iterator_t *li);

void *
list_iterator_get_datum(list_iterator_t *li);

bool 
list_iterator_is_valid(list_iterator_t *li);

void 
list_reverse_iterator_init(	list_t *l, 
							list_iterator_t *li);

void 
list_reverse_iterator_delete(list_iterator_t *li);

void 
list_reverse_iterator_next(list_iterator_t *li);

void 
list_reverse_iterator_prev(list_iterator_t *li);

void *
list_reverse_iterator_get_datum(list_iterator_t *li);

bool 
list_reverse_iterator_is_valid(list_reverse_iterator_t *li);

//======================================================================================================================================================150
// Output and utility
//======================================================================================================================================================150

void *
kmalloc(int size);

long 
transform_to_cuda(	node *n, 
					bool verbose); //returns actual mem used in a long

void 
usage_1( void );

void 
usage_2( void );

void 
enqueue( node * new_node );

node * 
dequeue( void );

int 
height( node * root );

int 
path_to_root(	node * root, 
				node * child );

void 
print_leaves( node * root );

void 
print_tree( node * root );

node * 
find_leaf(	node * root, 
			int key, 
			bool verbose );

record * 
find(	node * root, 
		int key, 
		bool verbose );

int 
cut( int length );

//======================================================================================================================================================150
// Insertion
//======================================================================================================================================================150

record * 
make_record(int value);

node * 
make_node( void );

node * 
make_leaf( void );

int 
get_left_index(	node * parent, 
				node * left);

node * 
insert_into_leaf(	node * leaf, 
					int key, record * pointer );

node * 
insert_into_leaf_after_splitting(	node * root, 
									node * leaf, 
									int key, 
									record * pointer);

node * 
insert_into_node(	node * root, 
					node * parent, 
					int left_index, 
					int key, 
					node * right);

node * 
insert_into_node_after_splitting(	node * root, 
									node * parent, 
									int left_index, 
									int key, 
									node * right);

node * 
insert_into_parent(	node * root, 
					node * left, 
					int key, 
					node * right);

node * 
insert_into_new_root(	node * left, 
						int key, 
						node * right);

node * 
start_new_tree(	int key, 
				record * pointer);

node * 
insert(	node * root, 
		int key, 
		int value );

//======================================================================================================================================================150
// Deletion
//======================================================================================================================================================150

int 
get_neighbor_index(node * n );

node * 
adjust_root(node * root);

node * 
coalesce_nodes(	node * root, 
				node * n, 
				node * neighbor, 
				int neighbor_index, 
				int k_prime);

node * 
redistribute_nodes(	node * root, 
					node * n, 
					node * neighbor, 
					int neighbor_index, 
					int k_prime_index, 
					int k_prime);

node * 
delete_entry(	node * root, 
				node * n, 
				int key, 
				void * pointer );

node * 
deleteVal(	node * root, 
			int key );

//===============================================================================================================================================================================================================200
//	HEADER
//===============================================================================================================================================================================================================200

extern int platform_id_inuse;
extern int device_id_inuse;
extern cl_device_type device_type;

// int main(	int argc, 
			// char *argv []);

//===============================================================================================================================================================================================================200
//	END
//===============================================================================================================================================================================================================200

// #endif

// # ifdef __cplusplus
// }
// # endif
