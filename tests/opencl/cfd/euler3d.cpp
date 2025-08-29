/********************************************************************
	euler3d.cpp
	: parallelized code of CFD
	
	- original code from the AIAA-2009-4001 by Andrew Corrigan, acorriga@gmu.edu
	- parallelization with OpenCL API has been applied by
	Jianbin Fang - j.fang@tudelft.nl
	Delft University of Technology
	Faculty of Electrical Engineering, Mathematics and Computer Science
	Department of Software Technology
	Parallel and Distributed Systems Group
	on 24/03/2011
********************************************************************/

#include <iostream>
#include <fstream>
#include <math.h>
#include "CLHelper.h" 
 
/*
 * Options 
 * 
 */ 
#define GAMMA 1.4f
#define iterations 20
#ifndef block_length
	// #define block_length 192
	//#define block_length 16
	#define block_length 256
#endif

#define NDIM 3
#define NNB 4

#define RK 3	// 3rd order RK
#define ff_mach 1.2f
#define deg_angle_of_attack 0.0f

/*
 * not options
 */


#if block_length > 128
#warning "the kernels may fail too launch on some systems if the block length is too large"
#endif


#define VAR_DENSITY 0
#define VAR_MOMENTUM  1
#define VAR_DENSITY_ENERGY (VAR_MOMENTUM+NDIM)
#define NVAR (VAR_DENSITY_ENERGY+1)

//self-defined user type
typedef struct{
	float x;
	float y;
	float z;
} float3;

/*
 * Generic functions
 */
template <typename T>
cl_mem alloc(int N){
	cl_mem mem_d = _clMalloc(sizeof(T)*N);
	return mem_d;
}

template <typename T>
void dealloc(cl_mem array){
	_clFree(array);
}

template <typename T>
void copy(cl_mem dst, cl_mem src, int N){
	//_clMemcpyD2D(dst, src, N*sizeof(T));
	T* temp = new T[N];
	_clMemcpyD2H(temp, src, N*sizeof(T));
	_clMemcpyH2D(dst, temp, N*sizeof(T));
	delete[] temp;
}

template <typename T>
void upload(cl_mem dst, T* src, int N){
	_clMemcpyH2D(dst, src, N*sizeof(T));
}

template <typename T>
void download(T* dst, cl_mem src, int N){
	_clMemcpyD2H(dst, src, N*sizeof(T));
}

void dump(cl_mem variables, int nel, int nelr){
	float* h_variables = new float[nelr*NVAR];
	download(h_variables, variables, nelr*NVAR);

	{
		std::ofstream file("density.txt");
		file << nel << " " << nelr << std::endl;
		for(int i = 0; i < nel; i++) file << h_variables[i + VAR_DENSITY*nelr] << std::endl;
	}


	{
		std::ofstream file("momentum.txt");
		file << nel << " " << nelr << std::endl;
		for(int i = 0; i < nel; i++)
		{
			for(int j = 0; j != NDIM; j++)
				file << h_variables[i + (VAR_MOMENTUM+j)*nelr] << " ";
			file << std::endl;
		}
	}
	
	{
		std::ofstream file("density_energy.txt");
		file << nel << " " << nelr << std::endl;
		for(int i = 0; i < nel; i++) file << h_variables[i + VAR_DENSITY_ENERGY*nelr] << std::endl;
	}
	delete[] h_variables;
}

void initialize_variables(int nelr, cl_mem variables, cl_mem ff_variable){

	int work_items = nelr;
	int work_group_size = BLOCK_SIZE_1;
	int kernel_id = 1;
	int arg_idx = 0;	
	_clSetArgs(kernel_id, arg_idx++, variables);
	_clSetArgs(kernel_id, arg_idx++, ff_variable);
	_clSetArgs(kernel_id, arg_idx++, &nelr, sizeof(int));
	_clInvokeKernel(kernel_id, work_items, work_group_size);
}

void compute_step_factor(int nelr, cl_mem variables, cl_mem areas, cl_mem step_factors){

	int work_items = nelr;
	int work_group_size = BLOCK_SIZE_2;
	int kernel_id = 2;
	int arg_idx = 0;
	_clSetArgs(kernel_id, arg_idx++, variables);
	_clSetArgs(kernel_id, arg_idx++, areas);
	_clSetArgs(kernel_id, arg_idx++, step_factors);
	_clSetArgs(kernel_id, arg_idx++, &nelr, sizeof(int));
	_clInvokeKernel(kernel_id, work_items, work_group_size);
}

void compute_flux(int nelr, cl_mem elements_surrounding_elements, cl_mem normals, cl_mem variables, cl_mem ff_variable, \
			cl_mem fluxes, cl_mem ff_flux_contribution_density_energy,
			cl_mem ff_flux_contribution_momentum_x,
			cl_mem ff_flux_contribution_momentum_y,
			cl_mem ff_flux_contribution_momentum_z){

	int work_items = nelr;
	int work_group_size = BLOCK_SIZE_3;
	int kernel_id = 3;
	int arg_idx = 0;
	_clSetArgs(kernel_id, arg_idx++, elements_surrounding_elements);
	_clSetArgs(kernel_id, arg_idx++, normals);
	_clSetArgs(kernel_id, arg_idx++, variables);
	_clSetArgs(kernel_id, arg_idx++, ff_variable);
	_clSetArgs(kernel_id, arg_idx++, fluxes);
	_clSetArgs(kernel_id, arg_idx++, ff_flux_contribution_density_energy);
	_clSetArgs(kernel_id, arg_idx++, ff_flux_contribution_momentum_x);
	_clSetArgs(kernel_id, arg_idx++, ff_flux_contribution_momentum_y);
	_clSetArgs(kernel_id, arg_idx++, ff_flux_contribution_momentum_z);
	_clSetArgs(kernel_id, arg_idx++, &nelr, sizeof(int));
	_clInvokeKernel(kernel_id, work_items, work_group_size);
}

void time_step(int j, int nelr, cl_mem old_variables, cl_mem variables, cl_mem step_factors, cl_mem fluxes){

	int work_items = nelr;
	int work_group_size = BLOCK_SIZE_4;
	int kernel_id = 4;
	int arg_idx = 0;
	_clSetArgs(kernel_id, arg_idx++, &j, sizeof(int));
	_clSetArgs(kernel_id, arg_idx++, &nelr, sizeof(int));
	_clSetArgs(kernel_id, arg_idx++, old_variables);
	_clSetArgs(kernel_id, arg_idx++, variables);
	_clSetArgs(kernel_id, arg_idx++, step_factors);
	_clSetArgs(kernel_id, arg_idx++, fluxes);
	
	_clInvokeKernel(kernel_id, work_items, work_group_size);
}
inline void compute_flux_contribution(float& density, float3& momentum, float& density_energy, float& pressure, float3& velocity, float3& fc_momentum_x, float3& fc_momentum_y, float3& fc_momentum_z, float3& fc_density_energy)
{
	fc_momentum_x.x = velocity.x*momentum.x + pressure;
	fc_momentum_x.y = velocity.x*momentum.y;
	fc_momentum_x.z = velocity.x*momentum.z;
	
	
	fc_momentum_y.x = fc_momentum_x.y;
	fc_momentum_y.y = velocity.y*momentum.y + pressure;
	fc_momentum_y.z = velocity.y*momentum.z;

	fc_momentum_z.x = fc_momentum_x.z;
	fc_momentum_z.y = fc_momentum_y.z;
	fc_momentum_z.z = velocity.z*momentum.z + pressure;

	float de_p = density_energy+pressure;
	fc_density_energy.x = velocity.x*de_p;
	fc_density_energy.y = velocity.y*de_p;
	fc_density_energy.z = velocity.z*de_p;
}

/*
 * Main function
 */
int main(int argc, char** argv){
  printf("WG size of kernel:initialize = %d, WG size of kernel:compute_step_factor = %d, WG size of kernel:compute_flux = %d, WG size of kernel:time_step = %d\n", BLOCK_SIZE_1, BLOCK_SIZE_2, BLOCK_SIZE_3, BLOCK_SIZE_4);

	if (argc < 2){
		std::cout << "specify data file name and [device type] [device id]" << std::endl;
		return 0;
	}
	const char* data_file_name = argv[1];
	_clCmdParams(argc, argv);
	cl_mem ff_variable, ff_flux_contribution_momentum_x, ff_flux_contribution_momentum_y,ff_flux_contribution_momentum_z,  ff_flux_contribution_density_energy;
	cl_mem areas, elements_surrounding_elements, normals;
	cl_mem variables, old_variables, fluxes, step_factors;
	float h_ff_variable[NVAR];

	try{		
		_clInit(device_type, device_id);		
		// set far field conditions and load them into constant memory on the gpu
		{
			//float h_ff_variable[NVAR];
			const float angle_of_attack = float(3.1415926535897931 / 180.0f) * float(deg_angle_of_attack);
			
			h_ff_variable[VAR_DENSITY] = float(1.4);
			
			float ff_pressure = float(1.0f);
			float ff_speed_of_sound = sqrt(GAMMA*ff_pressure / h_ff_variable[VAR_DENSITY]);
			float ff_speed = float(ff_mach)*ff_speed_of_sound;
			
			float3 ff_velocity;
			ff_velocity.x = ff_speed*float(cos((float)angle_of_attack));
			ff_velocity.y = ff_speed*float(sin((float)angle_of_attack));
			ff_velocity.z = 0.0f;
			
			h_ff_variable[VAR_MOMENTUM+0] = h_ff_variable[VAR_DENSITY] * ff_velocity.x;
			h_ff_variable[VAR_MOMENTUM+1] = h_ff_variable[VAR_DENSITY] * ff_velocity.y;
			h_ff_variable[VAR_MOMENTUM+2] = h_ff_variable[VAR_DENSITY] * ff_velocity.z;
					
			h_ff_variable[VAR_DENSITY_ENERGY] = h_ff_variable[VAR_DENSITY]*(float(0.5f)*(ff_speed*ff_speed)) + (ff_pressure / float(GAMMA-1.0f));

			float3 h_ff_momentum;
			h_ff_momentum.x = *(h_ff_variable+VAR_MOMENTUM+0);
			h_ff_momentum.y = *(h_ff_variable+VAR_MOMENTUM+1);
			h_ff_momentum.z = *(h_ff_variable+VAR_MOMENTUM+2);
			float3 h_ff_flux_contribution_momentum_x;
			float3 h_ff_flux_contribution_momentum_y;
			float3 h_ff_flux_contribution_momentum_z;
			float3 h_ff_flux_contribution_density_energy;
			compute_flux_contribution(h_ff_variable[VAR_DENSITY], h_ff_momentum, h_ff_variable[VAR_DENSITY_ENERGY], ff_pressure, ff_velocity, h_ff_flux_contribution_momentum_x, h_ff_flux_contribution_momentum_y, h_ff_flux_contribution_momentum_z, h_ff_flux_contribution_density_energy);

			// copy far field conditions to the gpu
			//cl_mem ff_variable, ff_flux_contribution_momentum_x, ff_flux_contribution_momentum_y,ff_flux_contribution_momentum_z,  ff_flux_contribution_density_energy;
			ff_variable = _clMalloc(NVAR*sizeof(float));
			ff_flux_contribution_momentum_x = _clMalloc(sizeof(float3));
			ff_flux_contribution_momentum_y = _clMalloc(sizeof(float3));
			ff_flux_contribution_momentum_z = _clMalloc(sizeof(float3));
			ff_flux_contribution_density_energy = _clMalloc(sizeof(float3));
			_clMemcpyH2D(ff_variable,          h_ff_variable,          NVAR*sizeof(float));
			_clMemcpyH2D(ff_flux_contribution_momentum_x, &h_ff_flux_contribution_momentum_x, sizeof(float3));
			_clMemcpyH2D(ff_flux_contribution_momentum_y, &h_ff_flux_contribution_momentum_y, sizeof(float3));
			_clMemcpyH2D(ff_flux_contribution_momentum_z, &h_ff_flux_contribution_momentum_z, sizeof(float3));		
			_clMemcpyH2D(ff_flux_contribution_density_energy, &h_ff_flux_contribution_density_energy, sizeof(float3));
			_clFinish();
		
		}
		int nel;
		int nelr;
		// read in domain geometry
		//float* areas;
		//int* elements_surrounding_elements;
		//float* normals;
		{
			std::ifstream file(data_file_name);
			if(!file){
				throw(string("can not find/open file!")+data_file_name);
			}
			file >> nel;
			nelr = block_length*((nel / block_length )+ std::min(1, nel % block_length));
			std::cout<<"--cambine: nel="<<nel<<", nelr="<<nelr<<std::endl;
			float* h_areas = new float[nelr];
			int* h_elements_surrounding_elements = new int[nelr*NNB];
			float* h_normals = new float[nelr*NDIM*NNB];

					
			// read in data
			for(int i = 0; i < nel; i++)
			{
				file >> h_areas[i];
				for(int j = 0; j < NNB; j++)
				{
					file >> h_elements_surrounding_elements[i + j*nelr];
					if(h_elements_surrounding_elements[i+j*nelr] < 0) h_elements_surrounding_elements[i+j*nelr] = -1;
					h_elements_surrounding_elements[i + j*nelr]--; //it's coming in with Fortran numbering				
					
					for(int k = 0; k < NDIM; k++)
					{
						file >> h_normals[i + (j + k*NNB)*nelr];
						h_normals[i + (j + k*NNB)*nelr] = -h_normals[i + (j + k*NNB)*nelr];
					}
				}
			}
			
			// fill in remaining data
			int last = nel-1;
			for(int i = nel; i < nelr; i++)
			{
				h_areas[i] = h_areas[last];
				for(int j = 0; j < NNB; j++)
				{
					// duplicate the last element
					h_elements_surrounding_elements[i + j*nelr] = h_elements_surrounding_elements[last + j*nelr];	
					for(int k = 0; k < NDIM; k++) h_normals[last + (j + k*NNB)*nelr] = h_normals[last + (j + k*NNB)*nelr];
				}
			}
			
			areas = alloc<float>(nelr);
			upload<float>(areas, h_areas, nelr);

			elements_surrounding_elements = alloc<int>(nelr*NNB);
			upload<int>(elements_surrounding_elements, h_elements_surrounding_elements, nelr*NNB);

			normals = alloc<float>(nelr*NDIM*NNB);
			upload<float>(normals, h_normals, nelr*NDIM*NNB);
					
			delete[] h_areas;
			delete[] h_elements_surrounding_elements;
			delete[] h_normals;
		}
		
		// Create arrays and set initial conditions
		variables = alloc<float>(nelr*NVAR);				
		int tp = 0;
		initialize_variables(nelr, variables, ff_variable);			
		old_variables = alloc<float>(nelr*NVAR);   	
		fluxes = alloc<float>(nelr*NVAR);
		step_factors = alloc<float>(nelr); 
		// make sure all memory is floatly allocated before we start timing
		initialize_variables(nelr, old_variables, ff_variable);	
		initialize_variables(nelr, fluxes, ff_variable);		
		_clMemset(step_factors, 0, sizeof(float)*nelr);
		// make sure CUDA isn't still doing something before we start timing
		_clFinish();
		// these need to be computed the first time in order to compute time step
		std::cout << "Starting..." << std::endl;

		// Begin iterations
		for(int i = 0; i < iterations; i++){
			printf("A\n");
			copy<float>(old_variables, variables, nelr*NVAR);
			printf("B\n");
			// for the first iteration we compute the time step
			compute_step_factor(nelr, variables, areas, step_factors);
			printf("C\n");
			for(int j = 0; j < RK; j++){
				compute_flux(nelr, elements_surrounding_elements, normals, variables, ff_variable, fluxes, ff_flux_contribution_density_energy, \
				ff_flux_contribution_momentum_x, ff_flux_contribution_momentum_y, ff_flux_contribution_momentum_z);
			
			printf("D\n");
				time_step(j, nelr, old_variables, variables, step_factors, fluxes);
			}
		}
		_clFinish();
		std::cout << "Saving solution..." << std::endl;
		dump(variables, nel, nelr);
		std::cout << "Saved solution..." << std::endl;
		_clStatistics();
		std::cout << "Cleaning up..." << std::endl;
		
		//--release resources
		_clFree(ff_variable);
		_clFree(ff_flux_contribution_momentum_x);
		_clFree(ff_flux_contribution_momentum_y);
		_clFree(ff_flux_contribution_momentum_z);
		_clFree(ff_flux_contribution_density_energy);
		_clFree(areas);
		_clFree(elements_surrounding_elements);
		_clFree(normals);
		_clFree(variables);
		_clFree(old_variables);
		_clFree(fluxes);
		_clFree(step_factors);
		_clRelease();
		std::cout << "Done..." << std::endl;
		_clPrintTiming();
	}
	catch(string msg){
		std::cout<<"--cambine:( an exception catched in main body ->"<<msg<<std::endl;		
		_clFree(ff_variable);
		_clFree(ff_flux_contribution_momentum_x);
		_clFree(ff_flux_contribution_momentum_y);
		_clFree(ff_flux_contribution_momentum_z);
		_clFree(ff_flux_contribution_density_energy);
		_clFree(areas);
		_clFree(elements_surrounding_elements);
		_clFree(normals);
		_clFree(variables);
		_clFree(old_variables);
		_clFree(fluxes);
		_clFree(step_factors);
		_clRelease();		
	}
	catch(...){
		std::cout<<"--cambine:( unknow exceptions in main body..."<<std::endl;		
		_clFree(ff_variable);
		_clFree(ff_flux_contribution_momentum_x);
		_clFree(ff_flux_contribution_momentum_y);
		_clFree(ff_flux_contribution_momentum_z);
		_clFree(ff_flux_contribution_density_energy);
		_clFree(areas);
		_clFree(elements_surrounding_elements);
		_clFree(normals);
		_clFree(variables);
		_clFree(old_variables);
		_clFree(fluxes);
		_clFree(step_factors);
		_clRelease();		
	}
		
	return 0;
}
