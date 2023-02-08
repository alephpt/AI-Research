#include "Data_Tests.h"
#include <stdio.h>

void testLoadImage() {
    Image img = Image("Data/Image/Samples/sample_4.pngc");
    tensorf3d img_data = img.getTensor();

    printf("Printing Red:\n");
    printFMatrix(img_data[0]);
    printf("\nPrinting Green:\n");
    printFMatrix(img_data[1]);
    printf("Printing Blue:\n");
    printFMatrix(img_data[2]);
    printf("Printing Grey:\n");
    printFMatrix(img_data[3]);
}