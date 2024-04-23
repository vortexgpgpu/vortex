#include "find_ellipse.h"
#include "find_ellipse_opencl.h"
#include <sys/time.h>

// The number of sample points per ellipse
#define NPOINTS 150
// The expected radius (in pixels) of a cell
#define RADIUS 10
// The range of acceptable radii
#define MIN_RAD RADIUS - 2
#define MAX_RAD RADIUS * 2
// The number of different sample ellipses to try
#define NCIRCLES 7


extern MAT * m_inverse(MAT * A, MAT * out);

// Returns the current system time in microseconds
long long get_time() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return (tv.tv_sec * 1000000) + tv.tv_usec;
}


// Returns the specified frame from the specified video file
// If cropped == true, the frame is cropped to pre-determined dimensions
//  (hardcoded to the boundaries of the blood vessel in the test video)
// If scaled == true, all values are scaled to the range [0.0, 1.0]
MAT * get_frame(avi_t *cell_file, int frame_num, int cropped, int scaled) {
	int dummy;
	int width = AVI_video_width(cell_file);
	int height = AVI_video_height(cell_file);
	unsigned char *image_buf = (unsigned char *) malloc(width * height);

	// There are 600 frames in this file (i.e. frame_num = 600 causes an error)
	AVI_set_video_position(cell_file, frame_num);

	//Read in the frame from the AVI
	if(AVI_read_frame(cell_file, (char *)image_buf, &dummy) == -1) {
		AVI_print_error("Error with AVI_read_frame");
		exit(-1);
	}

	// The image is read in upside-down, so we need to flip it
	MAT * image_chopped;
	if (cropped) {
		// Crop and flip image so we deal only with the interior of the vein
		image_chopped = chop_flip_image(image_buf, height, width, TOP, BOTTOM, 0, width - 1, scaled);
	} else {
		// Just flip the image
		image_chopped = chop_flip_image(image_buf, height, width, 0, height - 1, 0, width - 1, scaled);
	}
	
	free(image_buf);
	
	return image_chopped;
}


// Flips the specified image and crops it to the specified dimensions
// If scaled == true, all values are scaled to the range [0.0, 1.0
MAT * chop_flip_image(unsigned char *image, int height, int width, int top, int bottom, int left, int right, int scaled) {
	MAT * result = m_get(bottom - top + 1, right - left + 1);
	int i, j;
	if (scaled) {
		double scale = 1.0 / 255.0;
		for(i = 0; i <= (bottom - top); i++)
			for(j = 0; j <= (right - left); j++)
				//m_set_val(result, i, j, (double) image[((height - (i + top)) * width) + (j + left)] * scale);
				  m_set_val(result, i, j, (double) image[((height - 1 - (i + top)) * width) + (j + left)] * scale);
	} else {
		for(i = 0; i <= (bottom - top); i++)
			for(j = 0; j <= (right - left); j++)
				//m_set_val(result, i, j, (double) image[((height - (i + top)) * width) + (j + left)]);
				  m_set_val(result, i, j, (double) image[((height - 1 - (i + top)) * width) + (j + left)]);
	}

	return result;
}


// Chooses the best GPU on the current machine
void choose_GPU() {
	select_device();
}


// Computes and then transfers to the GPU all of the
//  constant matrices required by the GPU kernels
void compute_constants() {
	// Compute memory sizes
	int strel_m = 12 * 2 + 1;
	int strel_n = 12 * 2 + 1;
	
	int n, k;
	// Compute the sine and cosine of the angle to each point in each sample circle
	//  (which are the same across all sample circles)
	float host_sin_angle[NPOINTS], host_cos_angle[NPOINTS], theta[NPOINTS];
	for(n = 0; n < NPOINTS; n++) {
		theta[n] = (((double) n) * 2.0 * PI) / ((double) NPOINTS);
		host_sin_angle[n] = sin(theta[n]);
		host_cos_angle[n] = cos(theta[n]);
	}

	// Compute the (x,y) pixel offsets of each sample point in each sample circle
	int host_tX[NCIRCLES * NPOINTS], host_tY[NCIRCLES * NPOINTS];
	for (k = 0; k < NCIRCLES; k++) {
		double rad = (double) (MIN_RAD + (2 * k)); 
		for (n = 0; n < NPOINTS; n++) {
			host_tX[(k * NPOINTS) + n] = (int)(cos(theta[n]) * rad);
			host_tY[(k * NPOINTS) + n] = (int)(sin(theta[n]) * rad);
		}
	}
	
	// Compute the structuring element used in dilation
	float *host_strel = structuring_element(12);
	
	// Transfer the computed matrices to the GPU
	transfer_constants(host_sin_angle, host_cos_angle, host_tX, host_tY, strel_m, strel_n, host_strel);
	
	// Free the memory (only need to free strel since the rest are declared statically)
	free(host_strel);
}


// Given x- and y-gradients of a video frame, computes the GICOV
//  score for each sample ellipse at every pixel in the frame
MAT *GICOV(MAT *grad_x, MAT *grad_y) {
	// Determine the dimensions of the frame
	int grad_m = grad_x->m;
	int grad_n = grad_y->n;
	
	// Allocate host memory for grad_x and grad_y
	unsigned int grad_mem_size = sizeof(float) * grad_m * grad_n;
	float *host_grad_x = (float*) malloc(grad_mem_size);
	float *host_grad_y = (float*) malloc(grad_mem_size);
	// initalize float versions of grad_x and grad_y
	int m, n;
	for (m = 0; m < grad_m; m++) {
		for (n = 0; n < grad_n; n++) {
			host_grad_x[(n * grad_m) + m] = (float) m_get_val(grad_x, m, n);
			host_grad_y[(n * grad_m) + m] = (float) m_get_val(grad_y, m, n);
		}
	}

	// Offload the GICOV score computation to the GPU
	float *host_gicov = GICOV_OpenCL(grad_m, grad_n, host_grad_x, host_grad_y);

	// Copy the results into a new host matrix
	MAT *gicov = m_get(grad_m, grad_n);
	for (m = 0; m < grad_m; m++)
		for (n = 0; n < grad_n; n++)
			m_set_val(gicov, m, n, host_gicov[(n * grad_m) + m]);

	// Cleanup memory
	free(host_grad_x);
	free(host_grad_y);
	/* free(host_gicov); */

	return gicov;
}


// Returns a circular structuring element of the specified radius
float * structuring_element(int radius) {
	int m = radius*2+1;
	int n = radius*2+1;
	float *result = (float *) malloc(sizeof(float) * m * n);
	
	int i, j;
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			if (sqrt((float)((i-radius)*(i-radius)+(j-radius)*(j-radius))) <= radius)
				result[(i * m) + j] = 1.0;
			else
				result[(i * m) + j] = 0.0;
		}
	}

	return result;
}


// Performs an image dilation on the specified matrix
MAT *dilate(MAT *img_in) {
	// Determine the dimensions of the frame
	int max_gicov_m = img_in->m;
	int max_gicov_n = img_in->n;

	// Determine the dimensions of the structuring element
	int strel_m = 12 * 2 + 1;
	int strel_n = 12 * 2 + 1;

	// Offload the dilation to the GPU
	float *host_img_dilated = dilate_OpenCL(max_gicov_m, max_gicov_n, strel_m, strel_n);

	// Copy results into a new host matrix
	MAT *dilated = m_get(max_gicov_m, max_gicov_n);
	int m, n;
	for (m = 0; m < max_gicov_m; m++)
		for (n = 0; n < max_gicov_n; n++)
			m_set_val(dilated, m, n, host_img_dilated[(m * max_gicov_n) + n]);

	// Cleanup memory
	free(host_img_dilated);

	return dilated;
}


//M = # of sampling points in each segment
//N = number of segment of curve
//Get special TMatrix
MAT * TMatrix(unsigned int N, unsigned int M)
{
	MAT * B = NULL, * LB = NULL, * B_TEMP = NULL, * B_TEMP_INV = NULL, * B_RET = NULL;
	int * aindex, * bindex, * cindex, * dindex;
	int i, j;

	aindex = (int*) malloc(N*sizeof(int));
	bindex = (int*) malloc(N*sizeof(int));
	cindex = (int*) malloc(N*sizeof(int));
	dindex = (int*) malloc(N*sizeof(int));

	for(i = 1; i < N; i++)
		aindex[i] = i-1;
	aindex[0] = N-1;

	for(i = 0; i < N; i++)
		bindex[i] = i;
	
	for(i = 0; i < N-1; i++)
		cindex[i] = i+1;
	cindex[N-1] = 0;

	for(i = 0; i < N-2; i++)
		dindex[i] = i+2;
	dindex[N-2] = 0;
	dindex[N-1] = 1;


	B = m_get(N*M, N);
	LB = m_get(M, N);

	for(i = 0; i < N; i++)
	{
		m_zero(LB);
		
		for(j = 0; j < M; j++)
		{
			double s = (double)j / (double)M;
			double a, b, c, d;

			a = (-1.0*s*s*s + 3.0*s*s - 3.0*s + 1.0) / 6.0;
			b = (3.0*s*s*s - 6.0*s*s + 4.0) / 6.0;
			c = (-3.0*s*s*s + 3.0*s*s + 3.0*s + 1.0) / 6.0;
			d = s*s*s / 6.0;

			m_set_val(LB, j, aindex[i], a);
			m_set_val(LB, j, bindex[i], b);
			m_set_val(LB, j, cindex[i], c);
			m_set_val(LB, j, dindex[i], d);
		}
		int m, n;

		for(m = i*M; m < (i+1)*M; m++)
			for(n = 0; n < N; n++)
				m_set_val(B, m, n, m_get_val(LB, m%M, n));
	}

	B_TEMP = mtrm_mlt(B, B, B_TEMP);
	B_TEMP_INV = m_inverse(B_TEMP, B_TEMP_INV);
	B_RET = mmtr_mlt(B_TEMP_INV, B, B_RET);
	
	m_free(B);
	m_free(LB);
	m_free(B_TEMP);
	m_free(B_TEMP_INV);

	free(dindex);
	free(cindex);
	free(bindex);
	free(aindex);

	return B_RET;
}

void uniformseg(VEC * cellx_row, VEC * celly_row, MAT * x, MAT * y)
{
	double dx[36], dy[36], dist[36], dsum[36], perm = 0.0, uperm;
	int i, j, index[36];

	for(i = 1; i <= 36; i++)
	{
		dx[i%36] = v_get_val(cellx_row, i%36) - v_get_val(cellx_row, (i-1)%36);
		dy[i%36] = v_get_val(celly_row, i%36) - v_get_val(celly_row, (i-1)%36);
		dist[i%36] = sqrt(dx[i%36]*dx[i%36] + dy[i%36]*dy[i%36]);
		perm+= dist[i%36];
	}
	uperm = perm / 36.0;
	dsum[0] = dist[0];
	for(i = 1; i < 36; i++)
		dsum[i] = dsum[i-1]+dist[i];

	for(i = 0; i < 36; i++)
	{
		double minimum=DBL_MAX, temp;
		int min_index = 0;
		for(j = 0; j < 36; j++)
		{
			temp = fabs(dsum[j]- (double)i*uperm);
			if (temp < minimum)
			{
				minimum = temp;
				min_index = j;
			}
		}
		index[i] = min_index;
	}

	for(i = 0; i < 36; i++)
	{
		m_set_val(x, 0, i, v_get_val(cellx_row, index[i]));
		m_set_val(y, 0, i, v_get_val(celly_row, index[i]));
	}
}

//Get minimum element in a matrix
double m_min(MAT * m)
{
	int i, j;
	double minimum = DBL_MAX, temp;
	for(i = 0; i < m->m; i++)
	{
		for(j = 0; j < m->n; j++)
		{
			temp = m_get_val(m, i, j);
			if(temp < minimum)
				minimum = temp;
		}
	}
	return minimum;
}

//Get maximum element in a matrix
double m_max(MAT * m)
{
	int i, j;
	double maximum = DBL_MIN, temp;
	for(i = 0; i < m->m; i++)
	{
		for(j = 0; j < m->n; j++)
		{
			temp = m_get_val(m, i, j);
			if(temp > maximum)
				maximum = temp;
		}
	}
	return maximum;
}

VEC * getsampling(MAT * m, int ns)
{
	int N = m->n > m->m ? m-> n:m->m, M = ns;
	int * aindex, * bindex, * cindex, * dindex;
	int i, j;
	VEC * retval = v_get(N*M);

	aindex = (int*) malloc(N*sizeof(int));
	bindex = (int*) malloc(N*sizeof(int));
	cindex = (int*) malloc(N*sizeof(int));
	dindex = (int*) malloc(N*sizeof(int));

	for(i = 1; i < N; i++)
		aindex[i] = i-1;
	aindex[0] = N-1;

	for(i = 0; i < N; i++)
		bindex[i] = i;
	
	for(i = 0; i < N-1; i++)
		cindex[i] = i+1;
	cindex[N-1] = 0;

	for(i = 0; i < N-2; i++)
		dindex[i] = i+2;
	dindex[N-2] = 0;
	dindex[N-1] = 1;

	for(i = 0; i < N; i++)
	{
		for(j = 0; j < M; j++)
		{
			double s = (double)j / (double)M;
			double a, b, c, d;

			a = m_get_val(m, 0, aindex[i]) * (-1.0*s*s*s + 3.0*s*s - 3.0*s + 1.0);
			b = m_get_val(m, 0, bindex[i]) * (3.0*s*s*s - 6.0*s*s + 4.0);
			c = m_get_val(m, 0, cindex[i]) * (-3.0*s*s*s + 3.0*s*s + 3.0*s + 1.0);
			d = m_get_val(m, 0, dindex[i]) * s*s*s;
			v_set_val(retval, i*M+j,(a+b+c+d)/6.0);

		}
	}

	free(dindex);
	free(cindex);
	free(bindex);
	free(aindex);

	return retval;
}

VEC * getfdriv(MAT * m, int ns)
{
	int N = m->n > m->m ? m-> n:m->m, M = ns;
	int * aindex, * bindex, * cindex, * dindex;
	int i, j;
	VEC * retval = v_get(N*M);

	aindex = (int*) malloc(N*sizeof(int));
	bindex = (int*) malloc(N*sizeof(int));
	cindex = (int*) malloc(N*sizeof(int));
	dindex = (int*) malloc(N*sizeof(int));

	for(i = 1; i < N; i++)
		aindex[i] = i-1;
	aindex[0] = N-1;

	for(i = 0; i < N; i++)
		bindex[i] = i;
	
	for(i = 0; i < N-1; i++)
		cindex[i] = i+1;
	cindex[N-1] = 0;

	for(i = 0; i < N-2; i++)
		dindex[i] = i+2;
	dindex[N-2] = 0;
	dindex[N-1] = 1;

	for(i = 0; i < N; i++)
	{
		for(j = 0; j < M; j++)
		{
			double s = (double)j / (double)M;
			double a, b, c, d;

			a = m_get_val(m, 0, aindex[i]) * (-3.0*s*s + 6.0*s - 3.0);
			b = m_get_val(m, 0, bindex[i]) * (9.0*s*s - 12.0*s);
			c = m_get_val(m, 0, cindex[i]) * (-9.0*s*s + 6.0*s + 3.0);
			d = m_get_val(m, 0, dindex[i]) * (3.0 *s*s);
			v_set_val(retval, i*M+j, (a+b+c+d)/6.0);

		}
	}

	free(dindex);
	free(cindex);
	free(bindex);
	free(aindex);

	return retval;
}

//Performs bilinear interpolation, getting the values of m specified in the vectors X and Y
MAT * linear_interp2(MAT * m, VEC * X, VEC * Y)
{
	//Kind of assumes X and Y have same len!

	MAT * retval = m_get(1, X->dim);
	double x_coord, y_coord, new_val, a, b;
	int l, k, i;

	for(i = 0; i < X->dim; i++)
	{
		x_coord = v_get_val(X, i);
		y_coord = v_get_val(Y, i);

		l = (int)x_coord;
		k = (int)y_coord;

		a = x_coord - (double)l;
		b = y_coord - (double)k;

		//printf("xc: %f \t yc: %f \t i: %d \t l: %d \t k: %d \t a: %f \t b: %f\n", x_coord, y_coord, i, l, k, a, b);

		new_val = (1.0-a)*(1.0-b)*m_get_val(m, k, l) +
				  a*(1.0-b)*m_get_val(m, k, l+1) +
				  (1.0-a)*b*m_get_val(m, k+1, l) +
				  a*b*m_get_val(m, k+1, l+1);

		m_set_val(retval, 0, i, new_val);
	}

	return retval;
}

void splineenergyform01(MAT * Cx, MAT * Cy, MAT * Ix, MAT * Iy, int ns, double delta, double dt, int typeofcell)
{
	VEC * X, * Y, * Xs, * Ys, * Nx, * Ny, * X1, * Y1, * X2, * Y2, *	XY, * XX, * YY, * dCx, * dCy, * Ix1, * Ix2, *Iy1, *Iy2;
	MAT * Ix1_mat, * Ix2_mat, * Iy1_mat, * Iy2_mat;
	int i,j, N, * aindex, * bindex, * cindex, * dindex;

	X = getsampling(Cx, ns);
	Y = getsampling(Cy, ns);
	Xs = getfdriv(Cx, ns);
	Ys = getfdriv(Cy, ns);

	Nx = v_get(Ys->dim);
	for(i = 0; i < Nx->dim; i++)
		v_set_val(Nx, i, v_get_val(Ys, i) / sqrt(v_get_val(Xs, i)*v_get_val(Xs, i) + v_get_val(Ys, i)*v_get_val(Ys, i)));

	Ny = v_get(Xs->dim);
	for(i = 0; i < Ny->dim; i++)
		v_set_val(Ny, i, -1.0 * v_get_val(Xs, i) / sqrt(v_get_val(Xs, i)*v_get_val(Xs, i) + v_get_val(Ys, i)*v_get_val(Ys, i)));
	
	X1 = v_get(Nx->dim);
	for(i = 0; i < X1->dim; i++)
		v_set_val(X1, i, v_get_val(X, i) + delta*v_get_val(Nx, i));

	Y1 = v_get(Ny->dim);
	for(i = 0; i < Y1->dim; i++)
		v_set_val(Y1, i, v_get_val(Y, i) + delta*v_get_val(Ny, i));

	X2 = v_get(Nx->dim);
	for(i = 0; i < X2->dim; i++)
		v_set_val(X2, i, v_get_val(X, i) - delta*v_get_val(Nx, i));

	Y2 = v_get(Ny->dim);
	for(i = 0; i < Y2->dim; i++)
		v_set_val(Y2, i, v_get_val(Y, i) + delta*v_get_val(Ny, i));

	// seg fault happens at this func call
	Ix1_mat = linear_interp2(Ix, X1, Y1);
	Iy1_mat = linear_interp2(Iy, X1, Y1);
	Ix2_mat = linear_interp2(Ix, X2, Y2);
	Iy2_mat = linear_interp2(Iy, X2, Y2);

	Ix1 = v_get(Ix1_mat->n);
	Iy1 = v_get(Iy1_mat->n);
	Ix2 = v_get(Ix2_mat->n);
	Iy2 = v_get(Iy2_mat->n);

	Ix1 = get_row(Ix1_mat, 0, Ix1);
	Iy1 = get_row(Iy1_mat, 0, Iy1);
	Ix2 = get_row(Ix2_mat, 0, Ix2);
	Iy2 = get_row(Iy2_mat, 0, Iy2);

	N = Cx->m;

	//VEC * retval = v_get(N*ns);

	aindex = (int*) malloc(N*sizeof(int));
	bindex = (int*) malloc(N*sizeof(int));
	cindex = (int*) malloc(N*sizeof(int));
	dindex = (int*) malloc(N*sizeof(int));

	for(i = 1; i < N; i++)
		aindex[i] = i-1;
	aindex[0] = N-1;

	for(i = 0; i < N; i++)
		bindex[i] = i;
	
	for(i = 0; i < N-1; i++)
		cindex[i] = i+1;
	cindex[N-1] = 0;

	for(i = 0; i < N-2; i++)
		dindex[i] = i+2;
	dindex[N-2] = 0;
	dindex[N-1] = 1;

	XY = v_get(Xs->dim);
	for(i = 0; i < Xs->dim; i++)
		v_set_val(XY, i, v_get_val(Xs, i) * v_get_val(Ys, i));

	XX = v_get(Xs->dim);
	for(i = 0; i < Xs->dim; i++)
		v_set_val(XX, i, v_get_val(Xs, i) * v_get_val(Xs, i));

	YY = v_get(Ys->dim);
	for(i = 0; i < Xs->dim; i++)
		v_set_val(YY, i, v_get_val(Ys, i) * v_get_val(Ys, i));

	dCx = v_get(Cx->m);
	dCy = v_get(Cy->m);

	//get control points for splines
	for(i = 0; i < Cx->m; i++)
	{
		for(j = 0; j < ns; j++)
		{
			double s = (double)j / (double)ns;
			double A1, A2, A3, A4, B1, B2, B3, B4, D, D_3, Tx1, Tx2, Tx3, Tx4, Ty1, Ty2, Ty3, Ty4;
			int k;

			A1 = (-1.0*(s-1.0)*(s-1.0)*(s-1.0)) / 6.0;
			A2 = (3.0*s*s*s - 6.0*s*s + 4.0) / 6.0;
			A3 = (-3.0*s*s*s + 3.0*s*s + 3.0*s + 1.0) / 6.0;
			A4 = s*s*s / 6.0;

			B1 = (-3.0*s*s + 6.0*s - 3.0) / 6.0;
			B2 = (9.0*s*s - 12.0*s) / 6.0;
			B3 = (-9.0*s*s + 6.0*s + 3.0) / 6.0;
			B4 = 3.0*s*s / 6.0;

			k = i*ns+j;
			D = sqrt(v_get_val(Xs, k)*v_get_val(Xs, k) + v_get_val(Ys, k)*v_get_val(Ys, k));
			D_3 = D*D*D;
			
			//1st control point
			
			Tx1 = A1 - delta * v_get_val(XY, k) * B1 / D_3;
			Tx2 = -1.0 * delta*(B1/D - v_get_val(XX, k)*B1/D_3);
			Tx3 = A1 + delta * v_get_val(XY, k) * B1 / D_3;
			Tx4 = delta*(B1/D - v_get_val(XX, k)*B1/D_3);

			Ty1 = delta*(B1/D - v_get_val(YY, k)*B1/D_3);
			Ty2 = A1 + delta * v_get_val(XY, k) * B1 / D_3;
			Ty3 = -1.0 * delta*(B1/D - v_get_val(YY, k)*B1/D_3);
			Ty4 = A1 - delta * v_get_val(XY, k) * B1 / D_3;

			v_set_val(dCx, aindex[i], v_get_val(dCx, aindex[i]) + Tx1*v_get_val(Ix1, k) + Tx2*v_get_val(Iy1,k) - Tx3*v_get_val(Ix2, k) - Tx4*v_get_val(Iy2, k));
			v_set_val(dCy, aindex[i], v_get_val(dCy, aindex[i]) + Ty1*v_get_val(Ix1, k) + Ty2*v_get_val(Iy1,k) - Ty3*v_get_val(Ix2, k) - Ty4*v_get_val(Iy2, k));
		
			//2nd control point

			Tx1 = A2 - delta * v_get_val(XY, k) * B2 / D_3;
			Tx2 = -1.0 * delta*(B2/D - v_get_val(XX, k)*B2/D_3);
			Tx3 = A2 + delta * v_get_val(XY, k) * B2 / D_3;
			Tx4 = delta*(B2/D - v_get_val(XX, k)*B2/D_3);

			Ty1 = delta*(B2/D - v_get_val(YY, k)*B2/D_3);
			Ty2 = A2 + delta * v_get_val(XY, k) * B2 / D_3;
			Ty3 = -1.0 * delta*(B2/D - v_get_val(YY, k)*B2/D_3);
			Ty4 = A2 - delta * v_get_val(XY, k) * B2 / D_3;

			v_set_val(dCx, bindex[i], v_get_val(dCx, bindex[i]) + Tx1*v_get_val(Ix1, k) + Tx2*v_get_val(Iy1,k) - Tx3*v_get_val(Ix2, k) - Tx4*v_get_val(Iy2, k));
			v_set_val(dCy, bindex[i], v_get_val(dCy, bindex[i]) + Ty1*v_get_val(Ix1, k) + Ty2*v_get_val(Iy1,k) - Ty3*v_get_val(Ix2, k) - Ty4*v_get_val(Iy2, k));

			//3nd control point

			Tx1 = A3 - delta * v_get_val(XY, k) * B3 / D_3;
			Tx2 = -1.0 * delta*(B3/D - v_get_val(XX, k)*B3/D_3);
			Tx3 = A3 + delta * v_get_val(XY, k) * B3 / D_3;
			Tx4 = delta*(B3/D - v_get_val(XX, k)*B3/D_3);

			Ty1 = delta*(B3/D - v_get_val(YY, k)*B3/D_3);
			Ty2 = A3 + delta * v_get_val(XY, k) * B3 / D_3;
			Ty3 = -1.0 * delta*(B3/D - v_get_val(YY, k)*B3/D_3);
			Ty4 = A3 - delta * v_get_val(XY, k) * B3 / D_3;

			v_set_val(dCx, cindex[i], v_get_val(dCx, cindex[i]) + Tx1*v_get_val(Ix1, k) + Tx2*v_get_val(Iy1,k) - Tx3*v_get_val(Ix2, k) - Tx4*v_get_val(Iy2, k));
			v_set_val(dCy, cindex[i], v_get_val(dCy, cindex[i]) + Ty1*v_get_val(Ix1, k) + Ty2*v_get_val(Iy1,k) - Ty3*v_get_val(Ix2, k) - Ty4*v_get_val(Iy2, k));
	
			//4nd control point

			Tx1 = A4 - delta * v_get_val(XY, k) * B4 / D_3;
			Tx2 = -1.0 * delta*(B4/D - v_get_val(XX, k)*B4/D_3);
			Tx3 = A4 + delta * v_get_val(XY, k) * B4 / D_3;
			Tx4 = delta*(B4/D - v_get_val(XX, k)*B4/D_3);

			Ty1 = delta*(B4/D - v_get_val(YY, k)*B4/D_3);
			Ty2 = A4 + delta * v_get_val(XY, k) * B4 / D_3;
			Ty3 = -1.0 * delta*(B4/D - v_get_val(YY, k)*B4/D_3);
			Ty4 = A4 - delta * v_get_val(XY, k) * B4 / D_3;

			v_set_val(dCx, dindex[i], v_get_val(dCx, dindex[i]) + Tx1*v_get_val(Ix1, k) + Tx2*v_get_val(Iy1,k) - Tx3*v_get_val(Ix2, k) - Tx4*v_get_val(Iy2, k));
			v_set_val(dCy, dindex[i], v_get_val(dCy, dindex[i]) + Ty1*v_get_val(Ix1, k) + Ty2*v_get_val(Iy1,k) - Ty3*v_get_val(Ix2, k) - Ty4*v_get_val(Iy2, k));		
		}
	}

	if(typeofcell==1)
	{
		for(i = 0; i < Cx->n; i++)
			m_set_val(Cx, 0, i, m_get_val(Cx, 1, i) - dt*v_get_val(dCx, i));

		for(i = 0; i < Cy->n; i++)
			m_set_val(Cy, 0, i, m_get_val(Cy, 1, i) - dt*v_get_val(dCy, i));
	}
	else
	{
		for(i = 0; i < Cx->n; i++)
			m_set_val(Cx, 0, i, m_get_val(Cx, 1, i) + dt*v_get_val(dCx, i));

		for(i = 0; i < Cy->n; i++)
			m_set_val(Cy, 0, i, m_get_val(Cy, 1, i) + dt*v_get_val(dCy, i));
	}

	v_free(dCy); v_free(dCx); v_free(YY); v_free(XX); v_free(XY);

	free(dindex); free(cindex); free(bindex); free(aindex); 

	v_free(Iy2); v_free(Ix2); v_free(Iy1); v_free(Ix1); 

	m_free(Iy2_mat); m_free(Ix2_mat); m_free(Iy1_mat); m_free(Ix1_mat); 

	v_free(Y2); v_free(X2); v_free(Y1); v_free(X1); v_free(Ny); v_free(Nx); v_free(Ys); v_free(Xs); v_free(Y); v_free(X); 
}
