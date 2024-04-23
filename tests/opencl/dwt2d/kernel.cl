#define THREADS 256
#define BOUNDARY_X 2


int divRndUp(int n, 
             int d)
{
    return (n / d) + ((n % d) ? 1 : 0);
}


/* Store 3 RGB float components */
/*void storeComponents(__global float *d_r, __global float *d_g, __global float *d_b, __global const float r, __global const float g, __global const float b, int pos) 
{
    d_r[pos] = (r/255.0f) - 0.5f;
    d_g[pos] = (g/255.0f) - 0.5f;
    d_b[pos] = (b/255.0f) - 0.5f;
}
*/


// Store 3 RGB intege components 
void storeComponents(__global int *d_r,   
                     __global int *d_g,
                     __global int *d_b,
                     int r,
                     int g,
                     int b,
                     int pos)
{
    d_r[pos] = r - 128;
    d_g[pos] = g - 128;
    d_b[pos] = b - 128;
}   


/* Store float component */
/*__kernel void storeComponent(__global float *d_c, __global const float c, int pos)
{
    d_c[pos] = (c/255.0f) - 0.5f;
}
*/


// Store integer component 
void storeComponent(__global int *d_c,
                    const int c,
                    int pos)
{
    d_c[pos] = c - 128;
}


// Copy img src data into three separated component buffers 
__kernel void c_CopySrcToComponents (__global int *d_r,
                                     __global int *d_g,
                                     __global int *d_b,
                                     __global unsigned char * cl_d_src,
                                     int pixels)
{
	int x = get_local_id(0);
	int gX= get_local_size(0) * get_group_id(0); 
	
	__local unsigned char sData[THREADS*3];
	
    // Copy data to shared mem by 4bytes 
    // other checks are not necessary, since 
    // cl_d_src buffer is aligned to sharedDataSize 
	sData[3 * x + 0] = cl_d_src [gX * 3 + 3 * x + 0];
	sData[3 * x + 1] = cl_d_src [gX * 3 + 3 * x + 1];
	sData[3 * x + 2] = cl_d_src [gX * 3 + 3 * x + 2]; 
	
	barrier(CLK_LOCAL_MEM_FENCE);   
	
	int r, g, b;
	int offset = x*3;
	r = (int)(sData[offset]);
	g = (int)(sData[offset+1]);
	b = (int)(sData[offset+2]);
	
	int globalOutputPosition = gX + x;
	if (globalOutputPosition < pixels)
	{
		storeComponents(d_r, d_g, d_b, r, g, b, globalOutputPosition);
	}   

}


// Copy img src data into three separated component buffers 
__kernel void c_CopySrcToComponent (__global int *d_c, 
									__global unsigned char * cl_d_src,
									int pixels)
{
	int x = get_local_id(0);
	int gX = get_local_size(0) * get_group_id(0);
	
	__local unsigned char sData[THREADS];
	
	sData[ x ] = cl_d_src [gX + x];
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	int c;
	
	c = (int) (sData[x]);
	
	int globalOutputPosition = gX + x;
	if (globalOutputPosition < pixels)
	{
		storeComponent(d_c, c, globalOutputPosition);
	}
	
}


static void mirror( int *d,
                    const int sizeD)
{
	if ((*d )>= sizeD)
	{
		(*d) = 2 * sizeD -2 - (*d);
	} else if((*d) < 0)
	{
		(*d) = -(*d) ;
	}
}


struct VerticalDWTPixelIO
{
	bool CHECKED;
	int end, stride;
};


int initialize_PixelIO(struct VerticalDWTPixelIO *pIO,
                       bool CHECK,
                       const int sizeX,
                       const int sizeY,
                       int firstX,
                       int firstY)
{
	pIO->CHECKED = CHECK;
	pIO->end = pIO->CHECKED ? (sizeY * sizeX + firstX) : 0 ;
	pIO->stride = sizeX;
    return firstX + sizeX * firstY;
}


struct VerticalDWTPixelLoader
{
	bool CHECKED;
	int last;
};


void init_PixelLoader(struct VerticalDWTPixelLoader *loader,
                      const int sizeX,
                      const int sizeY,
                      int firstX,
                      const int firstY,
                      struct VerticalDWTPixelIO *pIO,
                      bool CHECK )
{
	mirror (&firstX, sizeX);
	loader->last = initialize_PixelIO (pIO, CHECK, sizeX, sizeY, firstX, firstY) - sizeX;
}


void clear_PixelLoader(struct VerticalDWTPixelLoader *pLoader,
                       struct VerticalDWTPixelIO *pIO)
{
	pLoader->last = 0;
	pIO->end = 0 ;
	pIO->stride = 0 ;
}


int loadFrom(struct VerticalDWTPixelLoader *pLoader,
             __global const int * const input,
             struct VerticalDWTPixelIO *pIO,
             int CHECK)
{	
	pLoader->last += pIO->stride;   
	if(CHECK && (pLoader->last == pIO->end)) 
	{
        pLoader->last -= 2 * pIO->stride;
        pIO->stride = 0 - pIO->stride;  
    }
	return input[pLoader->last];  
} 


struct VerticalDWTBandIO 
{
	bool CHECKED;
	int end;
	int strideHighToLow;
	int strideLowToHigh;
};


int initialize_BandIO(struct VerticalDWTBandIO *bandIO,
                      const int imageSizeX,
                      const int imageSizeY,
                      int firstX,
                      int firstY) 
{
	int columnOffset = firstX / 2;
    int verticalStride;
	
	if(firstX & 1)
	{
		verticalStride = imageSizeX / 2;
		columnOffset += divRndUp(imageSizeX, 2) * divRndUp(imageSizeY, 2);
        bandIO->strideLowToHigh = (imageSizeX * imageSizeY) / 2;
	}
	else
	{
	verticalStride = imageSizeX / 2 + (imageSizeX & 1);
    bandIO->strideLowToHigh = divRndUp(imageSizeY, 2)  * imageSizeX;		
	} 
	
	bandIO->strideHighToLow = verticalStride - bandIO->strideLowToHigh;
	
	if (bandIO->CHECKED) 
	{
		bandIO->end = columnOffset + (imageSizeY / 2) * verticalStride + (imageSizeY & 1) * bandIO->strideLowToHigh ;
	}
	else
	{
		bandIO->end = 0;
	}
	
	return columnOffset + (firstY / 2) * verticalStride    // right row
              + (firstY & 1) * bandIO->strideLowToHigh;
} 


struct VerticalDWTBandLoader
{
	bool CHECKED;
	int last;
};


struct VerticalDWTBandWriter
{
	bool CHECKED;  
	int next;
};


int saveAndUpdate(struct VerticalDWTBandWriter *writer,
                  bool CHECK,
                  struct VerticalDWTBandIO *bandIO,
                  __global int * const output,
                  __local int *item,
                  int *stride) 
{
	writer->CHECKED = CHECK;   
	if((!writer->CHECKED) || (writer->next != bandIO->end) )  
	{
		output[writer->next] = *item;
        writer->next += *stride;
    } 
	return writer->next; 
}


void clear_BandWriter(struct VerticalDWTBandWriter *writer,
                      struct VerticalDWTBandIO *bandIO)
{
	bandIO->end = 0;
	bandIO->strideHighToLow = 0;
	bandIO->strideLowToHigh = 0;
	writer->next = 0 ;
}


void init_BandWriter(struct VerticalDWTBandWriter *writer,
                     struct VerticalDWTBandIO *bandIO,
                     const int imageSizeX,
                     const int imageSizeY,
                     const int firstX,
                     const int firstY)
{
	if (firstX < imageSizeX)
	{ 
		writer->next = initialize_BandIO (bandIO, imageSizeX, imageSizeY, firstX, firstY);
	}
	else
	{
		clear_BandWriter (writer , bandIO) ;
	}
}


int writeLowInto(struct VerticalDWTBandWriter *writer,
                 struct VerticalDWTBandIO *bandIO,
                 __global int * const output,
                 __local int *primary)
{
	return saveAndUpdate(writer, writer->CHECKED, bandIO, output, primary, &(bandIO->strideLowToHigh));
}

int writeHighInto(struct VerticalDWTBandWriter *writer, struct VerticalDWTBandIO *bandIO,  __global int * const output, __local int *other)
{
	return saveAndUpdate(writer, writer->CHECKED, bandIO, output, other,   &(bandIO->strideHighToLow));
}


//TransformBuffer is contained in cuda_gwt/transform_buffer.h
struct TransformBuffer
{
	int SIZE_X, SIZE_Y;  
	int VERTICAL_STRIDE; 
	int SHM_BANKS, BUFFER_SIZE, PADDING, ODD_OFFSET;
	
	/// buffer for both even and odd columns
    int data[2182];     //data[2 * BUFFER_SIZE + PADDING]
};


void horizontalStep (__local struct TransformBuffer *buffer,
                     const int count,
                     const int prevOffset,
                     const int midOffset,
                     const int nextOffset,
                     int flag)
{
	const int STEPS = count / buffer->SIZE_X;
	const int finalCount = count % buffer->SIZE_X; 
    const int finalOffset = count - finalCount; 
	for(int i = 0; i< STEPS; i++)
	{
		const int previous = buffer->data[prevOffset + i * buffer->SIZE_X + get_local_id(0)] ;
		const int next     = buffer->data[nextOffset + i * buffer->SIZE_X + get_local_id(0)];
		__local int * center = & (buffer->data[midOffset + i *  buffer->SIZE_X + get_local_id(0)]); 
		if (flag == 0)
		{
			*center -= (previous + next) /2; //Forward53Predict()
		} else if (flag == 1)
		{
			*center += (previous + next + 2) /4; //Forward53Update()
		}
	}
	
	if(get_local_id(0) < finalCount) {
        const int previous = buffer->data[prevOffset + finalOffset + get_local_id(0)];
        const int next     = buffer->data[nextOffset + finalOffset + get_local_id(0)];
        __local int * center = & (buffer->data[midOffset + finalOffset + get_local_id(0)]);
		
        if (flag == 0)
		{
			*center -= (previous + next) /2; //Forward53Predict() 
		} else if (flag == 1)
		{
			*center += (previous + next + 2) /4; //Forward53Update()
		}
    }
	
}


void forEachHorizontalOdd(__local struct TransformBuffer *buffer,
                          const int firstLine,
                          const int numLines,
                          int flag) 
{
	const int count = numLines * buffer->VERTICAL_STRIDE - 1 ;
	const int prevOffset = firstLine * buffer->VERTICAL_STRIDE ; 
	const int centerOffset = prevOffset + buffer->ODD_OFFSET ; 
	const int nextOffset = prevOffset + 1;
	
	horizontalStep (buffer, count, prevOffset, centerOffset, nextOffset, flag);

}


void forEachHorizontalEven(__local struct TransformBuffer *buffer,
                           const int firstLine,
                           const int numLines,
                           int flag) 
{
	const int count = numLines * buffer->VERTICAL_STRIDE - 1 ;
	const int centerOffset = firstLine * buffer->VERTICAL_STRIDE + 1; 
	const int prevOffset = firstLine * buffer->VERTICAL_STRIDE + buffer->ODD_OFFSET; 
	const int nextOffset = prevOffset + 1;
	
	horizontalStep (buffer, count, prevOffset, centerOffset, nextOffset, flag);
}


void forEachVerticalOdd (__local struct TransformBuffer *buffer,
                         const int columnOffset,
                         int flag)
{
	int steps = (buffer->SIZE_Y - 1) / 2;
	for (int i = 0; i < steps; i++)
	{
		int row = i * 2 + 1;
		int prev = buffer->data[columnOffset+ (row - 1) * buffer->VERTICAL_STRIDE];
		int next = buffer->data[columnOffset+ (row + 1) * buffer->VERTICAL_STRIDE];

		if (flag == 0)
		{
			buffer->data[columnOffset + row * buffer->VERTICAL_STRIDE] -= (prev + next) /2;	 	
		}
		else if (flag == 1)
		{
			//buffer->data[columnOffset + row * buffer->VERTICAL_STRIDE] += (prev + next + 2) /4;
		}
	}
}


void forEachVerticalEven (__local struct TransformBuffer *buffer,
                          const int columnOffset,
                          int flag)
{
	int i ;
	if(buffer->SIZE_Y > 3)
	{ 
		int steps = (int)( buffer->SIZE_Y / 2) -1 ;
		
		for(i = 0; i < steps; i++) 
		{
			int row = 2 + i * 2;
			int prev = buffer->data[columnOffset+ (row - 1) * buffer->VERTICAL_STRIDE];
			int next = buffer->data[columnOffset + (row + 1) * buffer->VERTICAL_STRIDE];
			
			if (flag == 0)
			{
				//buffer->data[columnOffset + row * buffer->VERTICAL_STRIDE] -= (prev + next) /2; 
			}
			else if (flag == 1)
			{
				buffer->data[columnOffset + row * buffer->VERTICAL_STRIDE] += (prev + next + 2)/4; //real one
			}
			
        }
	}	
}


struct FDWT53Column
{
	bool CHECKED_LOADER;
	// loader for the column
	struct VerticalDWTPixelLoader loader;
	/// offset of the column in shared buffer
    int offset;                   
    // backup of first 3 loaded pixels (not transformed)
    int pixel0, pixel1, pixel2;

};


void clear_FDWT53Column(struct FDWT53Column *st_FDWT53Column,
                        struct VerticalDWTPixelIO *pIO)
{
	st_FDWT53Column->offset = 0;
	st_FDWT53Column->pixel0 = 0;
	st_FDWT53Column->pixel1 = 0;
	st_FDWT53Column->pixel2 = 0;
	clear_PixelLoader(&(st_FDWT53Column->loader), pIO);
}


struct FDWT53 {
	int WIN_SIZE_X, WIN_SIZE_Y;
	struct FDWT53Column column;
	/// Type of shared memory buffer for 5/3 FDWT transforms.
	/// Actual shared buffer used for forward 5/3 DWT.
    struct TransformBuffer buffer;
	
	/// Difference between indices of two vertical neighbors in buffer.
	int STRIDE;
};


//in from transform_buffer.h  
int getColumnOffset(int columnIndex,
                    __local struct TransformBuffer * buffer) 
{
	columnIndex += BOUNDARY_X;  
	return columnIndex / 2        // select right column
          + (columnIndex & 1) * buffer->ODD_OFFSET;  // select odd or even buffer         
}


void initColumn(__local struct FDWT53 * fdwt53,
                struct FDWT53Column *column,
                bool CHECKED,
                __global const int * const input,
                const int sizeX,
                const int sizeY,
                const int colIndex,
                const int firstY,
                struct VerticalDWTPixelIO *pIO)
{	
	column->CHECKED_LOADER = CHECKED;
	column->offset = getColumnOffset(colIndex, &fdwt53->buffer);
	
	const int firstX = get_group_id(0) * fdwt53->WIN_SIZE_X + colIndex;
	if(get_group_id(1) == 0) 
	{
        // topmost block - apply mirroring rules when loading first 3 rows
		init_PixelLoader(&(column->loader), sizeX, sizeY, firstX, firstY, pIO, CHECKED);
		column->pixel2 = loadFrom(&(column->loader),input, pIO, CHECKED);  // loaded pixel #0
        column->pixel1 = loadFrom(&(column->loader),input, pIO, CHECKED);  // loaded pixel #1
        column->pixel0 = loadFrom(&(column->loader),input, pIO, CHECKED);  // loaded pixel #2
		init_PixelLoader(&(column->loader), sizeX, sizeY, firstX, firstY + 1, pIO, CHECKED);
	} 
	else
	{
		init_PixelLoader(&(column->loader), sizeX, sizeY, firstX, firstY - 2, pIO, CHECKED);
		column->pixel0 = loadFrom(&(column->loader),input, pIO, CHECKED);  // loaded pixel #0
        column->pixel1 = loadFrom(&(column->loader),input, pIO, CHECKED);  // loaded pixel #1
        column->pixel2 = loadFrom(&(column->loader),input, pIO, CHECKED);  // loaded pixel #2
	}
	
}


void loadAndVerticallyTransform (__local struct FDWT53 *fdwt53,
                                 struct FDWT53Column *column,
                                 bool CHECKED,
                                 __global const int * const input,
                                 struct VerticalDWTPixelIO *pIO)
{
		fdwt53->buffer.data[column->offset + 0 * fdwt53->STRIDE] = column->pixel0;
		fdwt53->buffer.data[column->offset + 1 * fdwt53->STRIDE] = column->pixel1;
		fdwt53->buffer.data[column->offset + 2 * fdwt53->STRIDE] = column->pixel2;
	

	for (int i = 3; i < (3 + fdwt53->WIN_SIZE_Y); i++) 
	{
		fdwt53->buffer.data[column->offset + i * fdwt53->STRIDE] = loadFrom(&(column->loader),input, pIO, CHECKED);

	} 


	column->pixel0 = fdwt53->buffer.data [column->offset + ( fdwt53->WIN_SIZE_Y + 0 ) * fdwt53->STRIDE] ;
	column->pixel1 = fdwt53->buffer.data [column->offset + ( fdwt53->WIN_SIZE_Y + 1 ) * fdwt53->STRIDE] ;
	column->pixel2 = fdwt53->buffer.data [column->offset + ( fdwt53->WIN_SIZE_Y + 2 ) * fdwt53->STRIDE] ;
	
	
	int flag = 0 ;
	forEachVerticalOdd (&fdwt53->buffer, column->offset, flag);
	flag = 1 ;
	forEachVerticalEven(&fdwt53->buffer, column->offset, flag);

}


void transform(__local struct FDWT53 *fdwt53,
               bool CHECK_LOADS,
               bool CHECK_WRITES,
               __global const int * const in,
               __global int * out,
               const int sizeX,
               const int sizeY,
               const int winSteps)
{ 		
	// info about one main and one boundary columns processed by this thread
	struct FDWT53Column column; column.CHECKED_LOADER = CHECK_LOADS; 
    struct VerticalDWTPixelIO pIO;
    struct FDWT53Column boundaryColumn; boundaryColumn.CHECKED_LOADER = CHECK_LOADS; 
    struct VerticalDWTPixelIO pIO_b;

	// Initialize all column info: initialize loaders, compute offset of 
    // column in shared buffer and initialize loader of column.
	const int firstY = get_group_id(1) * fdwt53->WIN_SIZE_Y * winSteps;
	initColumn(fdwt53, &column, CHECK_LOADS, in, sizeX, sizeY, get_local_id(0), firstY, &pIO); 

	
	// first 3 threads initialize boundary columns, others do not use them
	clear_FDWT53Column(&boundaryColumn, &pIO_b);
	if (get_local_id(0) < 3) {
	// index of boundary column (relative x-axis coordinate of the column)
	const int colId = get_local_id(0) + ((get_local_id(0)== 0) ? fdwt53->WIN_SIZE_X : -3);
		
	// initialize the column		
	initColumn (fdwt53, &boundaryColumn, CHECK_LOADS, in, sizeX, sizeY, colId, firstY, &pIO_b);
	}
	
	// index of column which will be written into output by this thread
	const int outColumnIndex = (get_local_id(0) * 2) - (fdwt53->WIN_SIZE_X - 1) * (get_local_id(0) / ( fdwt53->WIN_SIZE_X / 2));
	
	// offset of column which will be written by this thread into output
    const int outColumnOffset = getColumnOffset(outColumnIndex, &(fdwt53->buffer));
	  
	// initialize output writer for this thread
    const int outputFirstX = get_group_id(0) * fdwt53->WIN_SIZE_X +outColumnIndex;  
	struct VerticalDWTBandWriter writer;  writer.CHECKED = CHECK_WRITES;
	struct VerticalDWTBandIO bandIO; bandIO.CHECKED = CHECK_WRITES;
	
	init_BandWriter(&writer, &bandIO, sizeX, sizeY, outputFirstX, firstY);

	 
	// Sliding window iterations:
    // Each iteration assumes that first 3 pixels of each column are loaded.

	for(int w = 0; w < winSteps; w++)
	{

		loadAndVerticallyTransform(fdwt53, &column, CHECK_LOADS, in, &pIO); 
		if (get_local_id(0) < 3)
		{
			loadAndVerticallyTransform(fdwt53, &boundaryColumn, CHECK_LOADS, in, &pIO_b); 
		}
	
		barrier(CLK_LOCAL_MEM_FENCE); 

		int flag = 0; //flag = 0 execute Forward53Predict, flag = 1 execute Forward53Update

		forEachHorizontalOdd(&(fdwt53->buffer), 2, fdwt53->WIN_SIZE_Y, flag);		
		barrier(CLK_LOCAL_MEM_FENCE);

		flag = 1;
		forEachHorizontalEven(&(fdwt53->buffer), 2, fdwt53->WIN_SIZE_Y, flag);		
 		barrier(CLK_LOCAL_MEM_FENCE);	

		
		for(int r = 2; r < (2+fdwt53->WIN_SIZE_Y); r+= 2)
		{	
			writeLowInto(&writer,  &bandIO, out, &(fdwt53->buffer.data[outColumnOffset + r * fdwt53->buffer.VERTICAL_STRIDE]));
			writeHighInto(&writer, &bandIO, out, &(fdwt53->buffer.data[outColumnOffset + (r+1) * fdwt53->buffer.VERTICAL_STRIDE]));
		}
		
		barrier(CLK_LOCAL_MEM_FENCE);
	}

}


// Forward 5/3 DWT predict operation. 
void Forward53Predict (const int p,
                       __global int * c,
                       const int n) 
{
		*c -= (p + n) /2;
}
// Forward 5/3 DWT update operation.
void Forward53Update (const int p,
                      __global int * c,
                      const int n) 
{
		*c += (p + n + 2) /4;
}


__kernel void cl_fdwt53Kernel(__global const int * const in, 
                              __global int *  out, 
                              const int sx, 
                              const int sy, 
                              const int steps,
                              int WIN_SIZE_X,
                              int WIN_SIZE_Y)
{
	__local struct FDWT53 fdwt53;
    fdwt53.WIN_SIZE_X = WIN_SIZE_X;
    fdwt53.WIN_SIZE_Y = WIN_SIZE_Y;
	
	//initialize
    //Lingjie Zhang modified on 11/02/2015
	//for(int i = 0; i < sizeof(fdwt53.buffer)/sizeof(int); i++){
	for(int i = 0; i < sizeof(fdwt53.buffer.data)/sizeof(int); i++){
		fdwt53.buffer.data[i] = 0;
	}
    //end of Lingjie Zhang modification
	
	fdwt53.buffer.SIZE_X = fdwt53.WIN_SIZE_X;
	fdwt53.buffer.SIZE_Y = fdwt53.WIN_SIZE_Y + 3;
	fdwt53.buffer.VERTICAL_STRIDE = BOUNDARY_X + (fdwt53.buffer.SIZE_X / 2);//BOUNDARY = 2  
	fdwt53.buffer.SHM_BANKS = 32;  // SHM_BANKS = ((__CUDA_ARCH__ >= 200) ? 32 : 16)
	fdwt53.buffer.BUFFER_SIZE = fdwt53.buffer.VERTICAL_STRIDE * fdwt53.buffer.SIZE_Y;
	fdwt53.buffer.PADDING = fdwt53.buffer.SHM_BANKS - ((fdwt53.buffer.BUFFER_SIZE + fdwt53.buffer.SHM_BANKS / 2) % fdwt53.buffer.SHM_BANKS) ;
	fdwt53.buffer.ODD_OFFSET = fdwt53.buffer.BUFFER_SIZE + fdwt53.buffer.PADDING ;
	fdwt53.STRIDE = fdwt53.buffer.VERTICAL_STRIDE ; 

	const int maxX = (get_group_id(0) + 1) * WIN_SIZE_X + 1;
    const int maxY = (get_group_id(1) + 1) * WIN_SIZE_Y * steps + 1;
    const bool atRightBoudary = maxX >= sx;
    const bool atBottomBoudary = maxY >= sy;
	
    // Select specialized version of code according to distance of this
    // threadblock's pixels from image boundary.
	if(atBottomBoudary)
	{
        // near bottom boundary => check both writing and reading
		transform(&fdwt53, true, true, in, out, sx, sy, steps);
	}
	else if(atRightBoudary)
	{
        // near right boundary only => check writing only
		transform(&fdwt53, false, true, in, out, sx, sy, steps);
	}
	else 
	{
        // no nearby boundary => check nothing
		transform(&fdwt53, false, false, in, out, sx, sy, steps);
	}
}
