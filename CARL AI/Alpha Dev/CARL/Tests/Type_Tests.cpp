#include "Type_Tests.h"

static void printVector(Vector* v) {
	printf("hit printVector(Vector*)\n");

	for (int i = 0; i < v->size; i++) {
		printf("Vector[%d]: %f\n", i, v->data[i]);
	}
}

void testVector() {
	printf("hit testVector()\n");
	Vector* v = initVector(5);
	printf("initVector(5)\n");
	for (int i = 0; i < v->size; i++) {
		v->data[i] = (float)(i + 1);
	}
	printf("populated Vector\n\n");
	printVector(v);
	printf("\ncleaning up Vector*\n\n");
	delete(v->data);
	delete(v);
}

static void printMatrix(Matrix* m) {
	printf("hit printMatrix(Matrix*)\n");
	for (int i = 0; i < m->rows; i++) {
		printf("Matrix row [%d]: ", i);
		for (int j = 0; j < m->cols; j++) {
			printf("%f, ", m->data[i * m->cols + j]);
		}
		printf("\n");
	}
}


void testMatrix() {
	printf("hit testMatrix()\n");
	Matrix* m = initMatrix(3, 4);
	printf("initMatrix(3,4)\n");
	for (int i = 0; i < m->rows; i++) {
		for (int j = 0; j < m->cols; j++) {
			m->data[i * m->cols + j] = (float)(i * m->cols + j);
		}
	}
	printf("populated Matrix\n\n");
	printMatrix(m);
	printf("\ncleaning up Matrix*\n\n");

	delete(m->data);
	delete(m);
}

static void print3DTensor(Tensor* testTensor) {
	printf("hit print3DTensor(Tensor*)\n\n");

	for (int i = 0; i < testTensor->shape[0]; i++) {
		printf(" // Tensor - Z: %d / %d\n", i, testTensor->shape[0]);
		for (int j = 0; j < testTensor->shape[1]; j++) {
			printf(" - Y : %d / %d ", j, testTensor->shape[1]);
			for (int k = 0; k < testTensor->shape[2]; k++) {
					int index = i * testTensor->shape[1] * testTensor->shape[2] + j * testTensor->shape[2] + k;
					printf("- E%d%d%d: %f -", i, j, k, testTensor->data[index]);

			}
			printf("\n");
		}
		printf("\n");
	}
}

void test3DTensor() {
	printf("hit test3DTensor()\n");
	int dims[3] = { 4, 4, 4 };
	Tensor* testTensor = initTensor(3, dims);
	printf("initTensor(3, { 2, 3, 4 })\n");
	for (int z = 0; z < testTensor->shape[0]; z++) {
		for (int y = 0; y < testTensor->shape[1]; y++) {
			int height = testTensor->shape[1];
			for (int x = 0; x < testTensor->shape[2]; x++) {
				int width = testTensor->shape[2];
				testTensor->data[z * height * width + y * width + x] = (float)(z * height * width + y * width + x + 1);
			}
		}
	}
	printf("populated 3DTensor* \n");
	print3DTensor(testTensor);
	printf("\ncleaning up 3D Tensor*\n\n");

	delete(testTensor->data);
	delete(testTensor);
}

void print4DTensor(Tensor* testTensor) {
	printf("hit print4DTensor(Tensor*)\n");
	for (int i = 0; i < testTensor->shape[0]; i++) {
		printf("\n // -- Tensor -- W: %d / %d\n", i, testTensor->shape[0]);
		for (int j = 0; j < testTensor->shape[1]; j++) {
			printf(" //  - Z: %d / %d \n", j, testTensor->shape[1]);
			for (int k = 0; k < testTensor->shape[2]; k++) {
				printf(" - Y: %d / %d ", k, testTensor->shape[0]);
				for (int l = 0; l < testTensor->shape[3]; l++) {
					int index = i * testTensor->shape[1] * testTensor->shape[2] * testTensor->shape[3]
						+ j * testTensor->shape[2] * testTensor->shape[3]
						+ k * testTensor->shape[3]
						+ l;
					printf("- El%d%d%d%d: %f -", i, j, k, l, testTensor->data[index]);
				}
				printf("\n");
			}
			printf("\n");
		}
	}
}

void test4DTensor() {
	printf("hit test4DTensor()\n");
	int d[4] = { 2, 3, 4, 5 };
	Tensor* testTensor = initTensor(4, d);
	printf("initTensor(4, { 2, 3, 4, 5 })\n");
	for (int i = 0; i < testTensor->shape[0]; i++) {
		for (int j = 0; j < testTensor->shape[1]; j++) {
			for (int k = 0; k < testTensor->shape[2]; k++) {
				for (int l = 0; l < testTensor->shape[3]; l++) {
					int index = i * testTensor->shape[1] * testTensor->shape[2] * testTensor->shape[3]
							  + j * testTensor->shape[2] * testTensor->shape[3]
							  + k * testTensor->shape[3]
							  + l;
					testTensor->data[index] = (float)(index + 1);
				}
			}
		}
	}
	printf("populated 4DTensor()\n");
	print4DTensor(testTensor);
	printf("cleaning up 4D Tensor*\n\n");
	delete(testTensor->data);
	delete(testTensor);
}