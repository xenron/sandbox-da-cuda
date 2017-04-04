#include <iostream>
#include <string>
#include <cassert>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

cv::Mat imageInputRGBA;
cv::Mat imageOutputRGBA;

uchar4 *d_inputImageRGBA__;
uchar4 *d_outputImageRGBA__;

float *h_filter__;

size_t numRows() { return imageInputRGBA.rows; }
size_t numCols() { return imageInputRGBA.cols; }

template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
	if (err != cudaSuccess) {
		std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
		std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
		exit(1);
	}
}

void preProcess(uchar4 **h_inputImageRGBA, uchar4 **h_outputImageRGBA,
	uchar4 **d_inputImageRGBA, uchar4 **d_outputImageRGBA,
	const std::string &filename) {
	//make sure the context initializes ok
	checkCudaErrors(cudaFree(0));

	cv::Mat image = cv::imread(filename.c_str(), CV_LOAD_IMAGE_COLOR);
	if (image.empty()) {
		std::cerr << "Couldn't open file: " << filename << std::endl;
		exit(1);
	}

	cv::cvtColor(image, imageInputRGBA, CV_BGR2RGBA);

	//allocate memory for the output
	imageOutputRGBA.create(image.rows, image.cols, CV_8UC4);

	//This shouldn't ever happen given the way the images are created
	//at least based upon my limited understanding of OpenCV, but better to check
	if (!imageInputRGBA.isContinuous() || !imageOutputRGBA.isContinuous()) {
		std::cerr << "Images aren't continuous!! Exiting." << std::endl;
		exit(1);
	}

	*h_inputImageRGBA = (uchar4 *)imageInputRGBA.ptr<unsigned char>(0);
	*h_outputImageRGBA = (uchar4 *)imageOutputRGBA.ptr<unsigned char>(0);

	const size_t numPixels = numRows() * numCols();
	//allocate memory on the device for both input and output
	checkCudaErrors(cudaMalloc(d_inputImageRGBA, sizeof(uchar4) * numPixels));
	checkCudaErrors(cudaMalloc(d_outputImageRGBA, sizeof(uchar4) * numPixels));
	checkCudaErrors(cudaMemset(*d_outputImageRGBA, 0, numPixels * sizeof(uchar4))); //make sure no memory is left laying around

																					//copy input array to the GPU
	checkCudaErrors(cudaMemcpy(*d_inputImageRGBA, *h_inputImageRGBA, sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice));

	d_inputImageRGBA__ = *d_inputImageRGBA;
	d_outputImageRGBA__ = *d_outputImageRGBA;

	//now create the filter that they will use

	checkCudaErrors(cudaFree(0));

}

//This kernel takes in three color channels and recombines them
//into one image.  The alpha channel is set to 255 to represent
//that this image has no transparency.
__global__
void flipMirror(const uchar4* const inputImageRGBA,
	uchar4* const outputImageRGBA,
	int numRows,
	int numCols)
{
	const int2 thread_2D_pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
		blockIdx.y * blockDim.y + threadIdx.y);

	const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;
	const int vertical_thread_1D_pos = (numRows - thread_2D_pos.y) * numCols + thread_2D_pos.x;
	const int horizontal_thread_1D_pos = thread_2D_pos.y * numCols + (numCols - thread_2D_pos.x);

	//make sure we don't try and access memory outside the image
	//by having any threads mapped there return early
	if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
		return;

	outputImageRGBA[vertical_thread_1D_pos].x = inputImageRGBA[thread_1D_pos].x;
	outputImageRGBA[vertical_thread_1D_pos].y = inputImageRGBA[thread_1D_pos].y;
	outputImageRGBA[vertical_thread_1D_pos].z = inputImageRGBA[thread_1D_pos].z;

}

void postProcess(const std::string& output_file, uchar4* data_ptr) {
	cv::Mat output(numRows(), numCols(), CV_8UC4, (void*)data_ptr);
	cv::Mat imageOutputBGR;
	cv::cvtColor(output, imageOutputBGR, CV_RGBA2BGR);
	//output the image
	cv::imwrite(output_file.c_str(), imageOutputBGR);
}

void cleanup() {
	//cleanup
	cudaFree(d_inputImageRGBA__);
	cudaFree(d_outputImageRGBA__);
	delete[] h_filter__;
}

int main(int argc, char* argv[]) {

	//load input file
	std::string input_file = "cinque_terre_small.jpg";
	//define output file
	std::string output_file = "cinque_terre_small_1.jpg";

	uchar4 *h_inputImageRGBA, *d_inputImageRGBA;
	uchar4 *h_outputImageRGBA, *d_outputImageRGBA;

	//load the image and give us our input and output pointers
	preProcess(&h_inputImageRGBA, &h_outputImageRGBA, &d_inputImageRGBA, &d_outputImageRGBA,
		input_file);

	const dim3 blockSize(16, 16);
	const dim3 gridSize(numCols() / blockSize.x + 1, numRows() / blockSize.y + 1);

	cudaDeviceSynchronize(); //checkCudaErrors(cudaGetLastError());

	flipMirror << <gridSize, blockSize >> >(d_inputImageRGBA,
		d_outputImageRGBA,
		numRows(),
		numCols());
	cudaDeviceSynchronize(); //checkCudaErrors(cudaGetLastError());

	size_t numPixels = numRows()*numCols();
	//copy the output back to the host
	checkCudaErrors(cudaMemcpy(h_outputImageRGBA, d_outputImageRGBA__, sizeof(uchar4) * numPixels, cudaMemcpyDeviceToHost));

	postProcess(output_file, h_outputImageRGBA);

	cleanup();

	return 0;
}

