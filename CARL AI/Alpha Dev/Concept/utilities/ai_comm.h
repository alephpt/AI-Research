/* description:
	This header file defines a set of functions that can be used to communicate between the 
	different AI models. The specific implementation of these functions would depend on the 
	specific application and the goals of the AI system. It is important to note that this 
	is just one possible way to implement communication between the AI models, and other 
	methods could also be used.
*/


#ifndef AI_COMM_H
#define AI_COMM_H

#include <stdio.h>
#include <stdlib.h>

// Function to send data from GAN to CNN
void sendDataFromGANtoCNNviaSMS(int* data, int size);
void sendDataFromGANtoCNNviaFW(int* data, int size);
void sendDataFromGANtoCNNviaNET(int* data, int size);

// Function to send data from CNN to RL AI
void sendDataFromCNNtoRLviaSMS(float* data, int size);
void sendDataFromCNNtoRLviaFW(float* data, int size);
void sendDataFromCNNtoRLviaNET(float* data, int size);

// Function to receive data from RL AI and update GAN parameters
void receiveDataFromRLandUpdateGANviaNET(float* data, int size);
void receiveDataFromRLandUpdateGANviaSMS(float* data, int size);
void receiveDataFromRLandUpdateGANviaFW(float* data, int size);

#endif
