        /////////////////////
        // FROM GAN TO CNN //
        /////////////////////

void sendDataFromGANtoCNNviaFW(int* data, int size) {
    // Open a file for writing
    FILE* fp = fopen("GAN_data.bin", "wb");

    // Write the data to the file
    fwrite(data, sizeof(int), size, fp);

    // Close the file
    fclose(fp);
}

void sendDataFromGANtoCNNviaSMS(int* data, int size) {
    // Create a shared memory segment
    int shmid = shmget(IPC_PRIVATE, size * sizeof(int), IPC_CREAT | 0666);

    // Attach the shared memory segment to the process's address space
    int* shared_data = (int*)shmat(shmid, NULL, 0);

    // Copy the data to the shared memory
    memcpy(shared_data, data, size * sizeof(int));

    // Detach the shared memory segment from the process's address space
    shmdt(shared_data);
}

void sendDataFromGANtoCNNviaNET(int* data, int size) {
    // Create a socket
    int sock = socket(AF_INET, SOCK_STREAM, 0);

    // Set up the server address
    struct sockaddr_in server_address;
    server_address.sin_family = AF_INET;
    server_address.sin_port = htons(12345);
    server_address.sin_addr.s_addr = INADDR_ANY;

    // Connect to the server
    connect(sock, (struct sockaddr*)&server_address, sizeof(server_address));

    // Send the data
    send(sock, data, size * sizeof(int), 0);

    // Close the socket
    close(sock);
}



        ////////////////////
        // FROM GAN TO RL //
        ////////////////////

void sendDataFromCNNtoRLviaFW(float* data, int size) {
    // Open a file for writing
    FILE* fp = fopen("CNN_data.bin", "wb");

    // Write the data to the file
    fwrite(data, sizeof(float), size, fp);

    // Close the file
    fclose(fp);
}


// TODO: implement proper usage of shared_memory_key
void sendDataFromCNNtoRLviaSMS(float* data, int size) {
    // Create a shared memory segment
    int shmid = shmget(IPC_PRIVATE, size * sizeof(float), IPC_CREAT | 0666);

    // Attach the shared memory segment to the process's address space
    float* shared_data = (float*)shmat(shmid, NULL, 0);

    // Copy the data to the shared memory
    memcpy(shared_data, data, size * sizeof(float));

    // Detach the shared memory segment from the process's address space
    shmdt(shared_data);
}

void sendDataFromCNNtoRLviaNET(float* data, int size) {
    // Create a socket
    int sock = socket(AF_INET, SOCK_STREAM, 0);

    // Set up the server address
    struct sockaddr_in server_address;
    server_address.sin_family = AF_INET;
    server_address.sin_port = htons(54321);
    server_address.sin_addr.s_addr = INADDR_ANY;

    // Connect to the server
    connect(sock, (struct sockaddr*)&server_address, sizeof(server_address));

    // Send the data
    send(sock, data, size * sizeof(float), 0);

    // Close the socket
    close(sock);
}


        ////////////////////
        // FROM GAN TO RL //
        ////////////////////

void receiveDataFromRLandUpdateGANviaNET(float* data, int size) {
    // Create a socket
    int sock = socket(AF_INET, SOCK_STREAM, 0);

    // Set up the server address
    struct sockaddr_in server_address;
    server_address.sin_family = AF_INET;
    server_address.sin_port = htons(54321);
    server_address.sin_addr.s_addr = INADDR_ANY;

    // Bind the socket to the address
    bind(sock, (struct sockaddr*)&server_address, sizeof(server_address));

    // Listen for incoming connections
    listen(sock, 1);

    // Accept an incoming connection
    int client_sock = accept(sock, NULL, NULL);

    // Receive the data
    recv(client_sock, data, size * sizeof(float), 0);

    // do we need these?
        // Update the GAN with the received data
        // updateGAN(data, size);

    // Close the socket
    close(client_sock);
}

void receiveDataFromRLandUpdateGANviaSMS(float* data, int size) {
    // Create a shared memory segment
    int shmid = shmget(IPC_PRIVATE, size * sizeof(float), IPC_CREAT | 0666);

    // Attach the shared memory segment to the process
    float* shared_data = shmat(shmid, NULL, 0);

    // Copy the data from the shared memory to the local data array
    memcpy(data, shared_data, size * sizeof(float));

    // Detach the shared memory segment from the process
    shmdt(shared_data);

    // Update the GAN with the received data
    updateGAN(data, size);

    // Remove the shared memory segment
    shmctl(shmid, IPC_RMID, NULL);
}

void receiveDataFromRLandUpdateGANviaFW(float* data, int size) {
    // Open the binary file for reading
    FILE* file = fopen("rl_data.bin", "rb");

    // Read the data from the file
    fread(data, sizeof(float), size, file);

    // Close the file
    fclose(file);

    // Update the GAN with the received data
    updateGAN(data, size);
}