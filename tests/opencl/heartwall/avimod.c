// #ifdef __cplusplus
// extern "C" {
// #endif

//===============================================================================================================================================================================================================
//	DEFINE / INCLUDE
//===============================================================================================================================================================================================================
#include "avimod.h"

//===============================================================================================================================================================================================================
//	FUNCTIONS
//===============================================================================================================================================================================================================

// Flips the specified image and crops it to the specified dimensions
// If scaled == true, all values are scaled to the range [0.0, 1.0
fp* chop_flip_image(	char *image, 
								int height, 
								int width, 
								int cropped,
								int scaled,
								int converted) {

	// fixed dimensions for cropping or not cropping, square vertices starting from initial point in top left corner going down and right
	int top;
	int bottom;
	int left;
	int right;
	if(cropped==1){
		top = 0;
		bottom = 0;
		left = 0;
		right = 0;
	}
	else{
		top = 0;
		bottom = height - 1;
		left = 0;
		right = width - 1;
	}

	// dimensions of new cropped image
	int height_new = bottom - top + 1;
	int width_new = right - left + 1;

	// counters
	int i, j;

	// allocate memory for cropped/flipped frame
	fp* result = (fp *) malloc(height_new * width_new * sizeof(fp));

	// crop/flip and scale frame
	fp temp;
	if (scaled) {
		fp scale = 1.0 / 255.0;
		for(i = 0; i <height_new; i++){				// rows
			for(j = 0; j <width_new; j++){			// colums
				temp = (fp) image[((height - 1 - (i + top)) * width) + (j + left)] * scale;
				if(temp<0){
					result[i*width_new+j] = temp + 256;
				}
				else{
					result[i*width_new+j] = temp;
				}
			}
		}
	} else {
		for(i = 0; i <height_new; i++){				// rows
			for(j = 0; j <width_new; j++){			// colums
				temp = (fp) image[((height - 1 - (i + top)) * width) + (j + left)] ;
				if(temp<0){
					result[i*width_new+j] = temp + 256;
				}
				else{
					result[i*width_new+j] = temp;
				}
			}
		}
	}

// convert storage method (from row-major to column-major)
	fp* result_converted = (fp *) malloc(height_new * width_new * sizeof(fp));
	if(converted==1){
		for(i = 0; i <width_new; i++){				// rows
			for(j = 0; j <height_new; j++){			// colums
				result_converted[i*height_new+j] = result[j*width_new+i];
			}
		}
	}
	else{
		result_converted = result;
	}
	free(result);

	// return
	return result_converted;
}

// Returns the specified frame from the specified video file
// If cropped == true, the frame is cropped to pre-determined dimensions
//  (hardcoded to the boundaries of the blood vessel in the test video)
// If scaled == true, all values are scaled to the range [0.0, 1.0]
fp* get_frame(	avi_t* cell_file, 
						int frame_num, 
						int cropped, 
						int scaled,
						int converted) {

	// variable
	int dummy;
	int width = AVI_video_width(cell_file);
	int height = AVI_video_height(cell_file);
	int status;

	// There are 600 frames in this file (i.e. frame_num = 600 causes an error)
	AVI_set_video_position(cell_file, frame_num);

	//Read in the frame from the AVI
	char* image_buf = (char*) malloc(width * height * sizeof(char));
	status = AVI_read_frame(cell_file, image_buf, &dummy);
	if(status == -1) {
		AVI_print_error((char*) "Error with AVI_read_frame");
		exit(-1);
	}

	// The image is read in upside-down, so we need to flip it
	fp* image_chopped;
	image_chopped = chop_flip_image(	image_buf, 
														height, 
														width, 
														cropped,
														scaled,
														converted);

	// free image buffer
	free(image_buf);

	// return
	return image_chopped;

} 

// #ifdef __cplusplus
// }
// #endif
