#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <png.h>
#include <sched.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <emmintrin.h>

// argument
int iters, width, height;
double left, right, lower, upper;
// pthread
int num_threads;
pthread_mutex_t mutex;
// mandelbrot set
int row;
int col;
// png
int* image;

void write_png(const char* filename, int iters, int width, int height, const int* buffer) {
    FILE* fp = fopen(filename, "wb");
    assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    for (int y = 0; y < height; ++y) {
        memset(row, 0, row_size);
        for (int x = 0; x < width; ++x) {
            int p = buffer[(height - 1 - y) * width + x];
            png_bytep color = row + x * 3;
            if (p != iters) {
                if (p & 16) {
                    color[0] = 240;
                    color[1] = color[2] = p % 16 * 16;
                } else {
                    color[0] = p % 16 * 16;
                }
            }
        }
        png_write_row(png_ptr, row);
    }
    free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

void* task(void* arg_void_ptr){
    int row_local;
    int row_offset;
    double row_step = ((upper - lower) / height);
    double col_step = ((right - left) / width);
    double digit2 = 2.0;
    __m128d digit2_SSE = _mm_load1_pd(&digit2);
    while(true){    // while all pixels not done
        pthread_mutex_lock(&mutex);
        if(row == height){
            pthread_mutex_unlock(&mutex);
            break;
        }
        else{
            row_local = row;
            row++;
        }
        pthread_mutex_unlock(&mutex);
        row_offset = row_local * width;
        int index0 = 0, index1 = 1, curr_idx = 2;
        int repeats0 = 0, repeats1 = 0;
        __m128d y_init_SSE, x_init_SSE, x_SSE, y_SSE, length_squared_SSE;
        y_init_SSE[0] = row_local*row_step+lower;
        y_init_SSE[1] = row_local*row_step+lower;
        x_init_SSE[0] = index0*col_step+left;
        x_init_SSE[1] = index1*col_step+left;
        x_SSE[0] = 0;
        x_SSE[1] = 0;
        y_SSE[0] = 0;
        y_SSE[1] = 0;
        length_squared_SSE[0] = 0;
        length_squared_SSE[1] = 0;
        __m128d x_square_SSE = _mm_mul_pd(x_SSE, x_SSE);
        __m128d y_square_SSE = _mm_mul_pd(y_SSE, y_SSE);
        while(true){    // while a row not done
            y_SSE = _mm_add_pd(_mm_mul_pd(_mm_mul_pd(digit2_SSE, x_SSE), y_SSE), y_init_SSE);
            x_SSE = _mm_add_pd(_mm_sub_pd(x_square_SSE, y_square_SSE), x_init_SSE);
            x_square_SSE = _mm_mul_pd(x_SSE, x_SSE);
            y_square_SSE = _mm_mul_pd(y_SSE, y_SSE);
            length_squared_SSE = _mm_add_pd(x_square_SSE, y_square_SSE);
            repeats0++;
            repeats1++;
            if(repeats0 == iters || length_squared_SSE[0] >= 4){   // index0 end
                image[row_offset + index0] = repeats0;
                index0 = curr_idx++;
                if(index0 >= width){
                    break;
                }
                repeats0 = 0;
                x_init_SSE[0] = index0*col_step+left;
                x_SSE[0] = 0;
                y_SSE[0] = 0;
                length_squared_SSE[0] = 0;
                if(repeats1 == iters || length_squared_SSE[1] >= 4){   // index1 end
                    image[row_offset + index1] = repeats1;
                    index1 = curr_idx++;
                    if(index1 >= width){
                        break;
                    }
                    repeats1 = 0;
                    x_init_SSE[1] = index1*col_step+left;
                    x_SSE[1] = 0;
                    y_SSE[1] = 0;
                    length_squared_SSE[1] = 0;
                }
                x_square_SSE = _mm_mul_pd(x_SSE, x_SSE);
                y_square_SSE = _mm_mul_pd(y_SSE, y_SSE);
            }
            else if(repeats1 == iters || length_squared_SSE[1] >= 4){   // index1 end
                image[row_offset + index1] = repeats1;
                index1 = curr_idx++;
                if(index1 >= width){
                    break;
                }
                repeats1 = 0;
                x_init_SSE[1] = index1*col_step+left;
                x_SSE[1] = 0;
                y_SSE[1] = 0;
                length_squared_SSE[1] = 0;
                x_square_SSE = _mm_mul_pd(x_SSE, x_SSE);
                y_square_SSE = _mm_mul_pd(y_SSE, y_SSE);
            }
        }
        if(index1 >= width){    // index1 end, finish index0
            if(index0 >= width){   // if index0 finish too -> break
                continue;
            }
            double x = x_SSE[0];
            double y = y_SSE[0];
            double length_squared = length_squared_SSE[0];
            double x_square = x * x;
            double y_square = y * y;
            while(repeats0 < iters && length_squared < 4){
                y = 2 * x * y + y_init_SSE[0];
                x = x_square - y_square + x_init_SSE[0];
                x_square = x * x;
                y_square = y * y;
                length_squared = x_square + y_square;
                ++repeats0;
            }
            image[row_offset + index0] = repeats0;
        }
        else{   // index1 not end, finish index1
            double x = x_SSE[1];
            double y = y_SSE[1];
            double length_squared = length_squared_SSE[1];
            double x_square = x * x;
            double y_square = y * y;
            while(repeats1 < iters && length_squared < 4){
                y = 2 * x * y + y_init_SSE[1];
                x = x_square - y_square + x_init_SSE[1];
                x_square = x * x;
                y_square = y * y;
                length_squared = x_square + y_square;
                ++repeats1;
            }
            image[row_offset + index1] = repeats1;
        }
    }
    pthread_exit(NULL);
}

int main(int argc, char** argv) {
    /* detect how many CPUs are available */
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    num_threads = CPU_COUNT(&cpu_set);

    /* argument parsing */
    assert(argc == 9);
    const char* filename = argv[1];
    iters = strtol(argv[2], 0, 10);
    left = strtod(argv[3], 0);
    right = strtod(argv[4], 0);
    lower = strtod(argv[5], 0);
    upper = strtod(argv[6], 0);
    width = strtol(argv[7], 0, 10);
    height = strtol(argv[8], 0, 10);

    /* allocate memory for image */
    image = (int*)malloc(width * height * sizeof(int));
    assert(image);

    pthread_t threads[num_threads];
	int args[num_threads];

    pthread_mutex_init (&mutex, NULL);
    row = 0;
    col = 0;
	for(int i = 0; i < num_threads; i++){
		args[i] = i;
		pthread_create(&threads[i], NULL, task, (void*)&args[i]);
	}
	for(int i = 0; i < num_threads; i++){
		pthread_join(threads[i], NULL);
	}
    pthread_mutex_destroy(&mutex);

    /* draw and cleanup */
    write_png(filename, iters, width, height, image);
    free(image);
    pthread_exit(NULL);
}
