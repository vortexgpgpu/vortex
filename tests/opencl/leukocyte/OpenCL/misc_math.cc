#include "misc_math.h"
#include <stdlib.h>
#include <math.h>


#define THRESHOLD_double 0.000001

inline int double_eq(double f1, double f2)
{
	return fabs(f1-f2) < THRESHOLD_double;
}

//Given a matrix, return the matrix containing an approximation of the gradient matrix dM/dx
MAT * gradient_x(MAT * input)
{
	int i, j;
	MAT * result = m_get(input->m, input->n);

	for(i = 0; i < result->m; i++)
	{
		for(j = 0; j < result->n; j++)
		{
			if(j==0)
				m_set_val(result, i, j, m_get_val(input, i, j+1) - m_get_val(input, i, j));
			else if(j==input->n-1)
				m_set_val(result, i, j, m_get_val(input, i, j) - m_get_val(input, i, j-1));
			else
				m_set_val(result, i, j, (m_get_val(input, i, j+1) - m_get_val(input, i, j-1)) / 2.0);
		}
	}

	return result;
}

//Given a matrix, return the matrix containing an approximation of the gradient matrix dM/dy
MAT * gradient_y(MAT * input)
{
	int i, j;
	MAT * result = m_get(input->m, input->n);
	
	for(i = 0; i < result->n; i++)
	{
		for(j = 0; j < result->m; j++)
		{
			if(j==0)
				m_set_val(result, j, i, m_get_val(input, j+1, i) - m_get_val(input, j, i));
			else if(j==input->m-1)
				m_set_val(result, j, i, m_get_val(input, j, i) - m_get_val(input, j-1, i));
			else
				m_set_val(result, j, i, (m_get_val(input, j+1, i) - m_get_val(input, j-1, i)) / 2.0);
		}
	}
	return result;
}

//Return the mean of the values in a vector
double mean(VEC * in)
{
	double sum = 0.0;
	int i;
	for(i = 0; i < in->dim; i++)
		sum+=v_get_val(in, i);

	return sum/(double)in->dim;
}

//Return the standard deviation of the values in a vector
double std_dev(VEC * in)
{
	double m = mean(in), sum =0.0;
	int i;
	for(i = 0; i < in->dim; i++)
	{
		double temp = v_get_val(in, i) - m;
		sum+=temp*temp;
	}
	return sqrt(sum/(double)in->dim);
}
