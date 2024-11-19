//Last updated on: 18/11/2024 --> 18th November 2024

// This is DeepCimple, a framework for saving, loading, and running models via a C interface. (I will add more documentation here, but I want
// to do more work with it first. This is essentially a first draft.)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <sys/stat.h>
#include <math.h>
#include "stb_image.h"

// Tensor structure
typedef struct {
    int *shape;
    int dims;
    float *data;
    size_t size;
} Tensor;

// Layer structure
typedef struct {
    Tensor *weights;
    Tensor *biases;
    char *activation_function;
} Layer;

// Neural Network structure
typedef struct {
    int num_layers;
    Layer **layers;
} NN;

// Image dataset structure
typedef struct {
    Tensor **images;
    int *labels;
    int num_samples;
} Img_dataset;


// Tensor functions
Tensor* create_tensor(int *shape, int dims);
void tensor_zeros(Tensor *tensor);
void save_tensor(Tensor *tensor, FILE *file);
Tensor* load_tensor(FILE *file);

// Layer functions
Layer* create_layer(int input_size, int output_size, const char *activation_function);
void initialize_layer(Layer *layer);
void save_layer(Layer *layer, FILE *file);
Layer* load_layer(FILE *file);

// Neural Network functions
NN* create_network(int num_layers, int *layer_sizes, const char **activation_functions);
void initialize_network(NN *network);
void save_network(NN *network, const char *filename);
NN* load_network(const char *filename);
void train_network(NN *network, Tensor *inputs, Tensor *targets, int epochs, float learning_rate);


int load_img_dataset(const char* dataset_path, int num_classes, int image_width, int image_length, int channels, int input_size, int hidden_size, int epochs, float learning_rate);
DIR* load_directory(const char* path, int printcount);


// Create a new tensor with given shape
Tensor* create_tensor(int *shape, int dims) {
    Tensor *tensor = (Tensor*)malloc(sizeof(Tensor));
    tensor->dims = dims;
    tensor->shape = (int*)malloc(dims * sizeof(int));
    tensor->size = 1;
    for (int i = 0; i < dims; i++) {
        tensor->shape[i] = shape[i];
        tensor->size *= shape[i];
    }
    tensor->data = (float*)malloc(tensor->size * sizeof(float));
    return tensor;
}

// Initialize tensor elements to zero
void tensor_zeros(Tensor *tensor) {
    memset(tensor->data, 0, tensor->size * sizeof(float));
}

// Save tensor to binary file
void save_tensor(Tensor *tensor, FILE *file) {
    fwrite(&tensor->dims, sizeof(int), 1, file);
    fwrite(tensor->shape, sizeof(int), tensor->dims, file);
    fwrite(tensor->data, sizeof(float), tensor->size, file);
}

// Load tensor from binary file
Tensor* load_tensor(FILE *file) {
    int dims;
    if (fread(&dims, sizeof(int), 1, file) != 1) {
        perror("Error reading tensor dimensions");
        exit(1);
    }
    int *shape = (int*)malloc(dims * sizeof(int));
    if (fread(shape, sizeof(int), dims, file) != (size_t)dims) {
        perror("Error reading tensor shape");
        exit(1);
    }
    Tensor *tensor = create_tensor(shape, dims);
    if (fread(tensor->data, sizeof(float), tensor->size, file) != tensor->size) {
        perror("Error reading tensor data");
        exit(1);
    }
    free(shape);
    return tensor;
}

// Create a new layer
Layer* create_layer(int input_size, int output_size, const char *activation_function) {
    Layer *layer = (Layer*)malloc(sizeof(Layer));
    int weight_shape[2] = {input_size, output_size};
    layer->weights = create_tensor(weight_shape, 2);

    int bias_shape[1] = {output_size};
    layer->biases = create_tensor(bias_shape, 1);

    layer->activation_function = strdup(activation_function);
    return layer;
}

// Initialize layer weights and biases
void initialize_layer(Layer *layer) {
    // Example: Random initialization between -0.5 and 0.5
    for (size_t i = 0; i < layer->weights->size; i++) {
        layer->weights->data[i] = ((float)rand() / RAND_MAX) - 0.5f;
    }
    for (size_t i = 0; i < layer->biases->size; i++) {
        layer->biases->data[i] = 0.0f;
    }
}

// Save layer to binary file
void save_layer(Layer *layer, FILE *file) {
    // Save activation function length and string
    int activation_length = strlen(layer->activation_function) + 1; // Include null terminator
    fwrite(&activation_length, sizeof(int), 1, file);
    fwrite(layer->activation_function, sizeof(char), activation_length, file);

    // Save weights and biases
    save_tensor(layer->weights, file);
    save_tensor(layer->biases, file);
}

// Load layer from binary file
Layer* load_layer(FILE *file) {
    // Load activation function
    int activation_length;
    if (fread(&activation_length, sizeof(int), 1, file) != 1) {
        perror("Error reading activation function length");
        exit(1);
    }
    char *activation_function = (char*)malloc(activation_length * sizeof(char));
    if (fread(activation_function, sizeof(char), activation_length, file) != (size_t)activation_length) {
        perror("Error reading activation function");
        exit(1);
    }

    // Load weights and biases
    Tensor *weights = load_tensor(file);
    Tensor *biases = load_tensor(file);

    // Create layer
    Layer *layer = (Layer*)malloc(sizeof(Layer));
    layer->weights = weights;
    layer->biases = biases;
    layer->activation_function = activation_function;
    return layer;
}

// Create a new neural network
NN* create_network(int num_layers, int *layer_sizes, const char **activation_functions) {
    NN *network = (NN*)malloc(sizeof(NN));
    network->num_layers = num_layers;
    network->layers = (Layer**)malloc(num_layers * sizeof(Layer*));
    for (int i = 0; i < num_layers; i++) {
        int input_size = layer_sizes[i];
        int output_size = layer_sizes[i + 1];
        network->layers[i] = create_layer(input_size, output_size, activation_functions[i]);
        initialize_layer(network->layers[i]);
    }
    return network;
}

// Initialize network layers
void initialize_network(NN *network) {
    for (int i = 0; i < network->num_layers; i++) {
        initialize_layer(network->layers[i]);
    }
}

// Save neural network to binary file
void save_network(NN *network, const char *filename) {
    FILE *file = fopen(filename, "wb");
    if (file == NULL) {
        perror("Cannot open file for writing");
        exit(1);
    }

    // Save number of layers
    fwrite(&network->num_layers, sizeof(int), 1, file);

    // Save each layer
    for (int i = 0; i < network->num_layers; i++) {
        save_layer(network->layers[i], file);
    }

    fclose(file);
}

// Load neural network from binary file
NN* load_network(const char *filename) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        perror("Cannot open file for reading");
        exit(1);
    }

    // Load number of layers
    int num_layers;
    if (fread(&num_layers, sizeof(int), 1, file) != 1) {
        perror("Error reading number of layers");
        exit(1);
    }

    // Allocate network
    NN *network = (NN*)malloc(sizeof(NN));
    network->num_layers = num_layers;
    network->layers = (Layer**)malloc(num_layers * sizeof(Layer*));

    // Load each layer
    for (int i = 0; i < num_layers; i++) {
        network->layers[i] = load_layer(file);
    }

    fclose(file);
    return network;
}


// Functions for Classifiers:

DIR* load_directory(const char* path, int printcount){
    DIR* directory = opendir(path);
    if(directory == NULL){
        printf("Error Opening Directory.");
        exit(-2);
    }

    int folder_count = 0;
    struct dirent* class;
    while((class = readdir(directory)) != NULL){
        if (strcmp(class->d_name, ".") == 0 || strcmp(class->d_name, "..") == 0) {
            continue;
        }

        if(class->d_type == DT_DIR){
            folder_count++;
        }
    }

    if(folder_count == 0){
        printf("No folders found in your database --> Possible Error with folder name or database management");
        exit(-2); 
    }    

    if(printcount > 0){
        printf("The amount of folders in your database is %i (each folder will be 1 class in the classifier)", folder_count);
    }
    return directory; 
}



// This model saver uses the following non-native Libraries:
/*
- stb_image.h --> Used for Image processing



*/