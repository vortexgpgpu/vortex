#ifndef __COMPUTEQ__
#define __COMPUTEQ__

void computePhiMag_GPU(int numK,cl_mem phiR_d,cl_mem phiI_d,cl_mem phiMag_d,clPrmtr* clPrm);
void computeQ_GPU (int numK,int numX,
		   cl_mem x_d, cl_mem y_d, cl_mem z_d,
		   struct kValues* kVals,
		   cl_mem Qr_d, cl_mem Qi_d,
		   clPrmtr* clPrm);

void createDataStructsCPU(int numK, int numX, float** phiMag,
	 		  float** Qr, float** Qi);

#endif
