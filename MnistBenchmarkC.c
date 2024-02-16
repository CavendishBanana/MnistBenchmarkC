/******************************************************************************

                            Online C Compiler.
                Code, Compile, Run and Debug C program online.
Write your code in this editor and press "Run" button to compile and execute it.

*******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "Constants.h"
#define BUFFER_LENGTH 3200
struct Matrix {
    int rows_count, cols_count;
    float* matrix;
};

void set_matrix_cell(struct Matrix* mat, int row, int col, float value)
{
    mat->matrix[(mat->cols_count * row + col)] = value;
};

float get_matrix_cell(struct Matrix* mat, int row, int col)
{
    return mat->matrix[(mat->cols_count * row + col)];
}

struct Matrix* matrix_constructor_with_default_value(int rows_count, int cols_count, float start_value)
{
    struct Matrix* mat = (struct Matrix*)(malloc(sizeof(struct Matrix)));
    mat->rows_count = rows_count;
    mat->cols_count = cols_count;
    mat->matrix = (float*)malloc(rows_count * cols_count * sizeof(float));
    for (int i = 0; i < rows_count; i++)
    {
        for (int j = 0; j < cols_count; j++)
        {
            mat->matrix[(i * cols_count + j)] = start_value;
        }
    }
    return mat;
}

struct Matrix* matrix_constructor(int rows_count, int cols_count)
{
    struct Matrix* mat = (struct Matrix*)malloc(sizeof(struct Matrix));
    mat->rows_count = rows_count;
    mat->cols_count = cols_count;
    mat->matrix = (float*)malloc(rows_count * cols_count * sizeof(float));
    return mat;
}

struct Matrix* matrix_constructor_from_float_array(int rows_count, int cols_count, float* arr)
{
    struct Matrix* mat = matrix_constructor(rows_count, cols_count);
    int cells_count = rows_count * cols_count;
    int k = 0;
    for (int i = 0; i < cells_count; i++)
    {
        for (int j = 0; j < cols_count; j++)
        {
            mat->matrix[(i * cols_count + j)] = arr[k];
            k++;
        }
    }
    return mat;
}

struct Matrix* matrix_constructor_empty_object()
{
    struct Matrix* mat = (struct Matrix*)malloc(sizeof(struct Matrix));
    mat->rows_count = 0;
    mat->cols_count = 0;
    mat->matrix = NULL;
    return mat;
}

void set_float_array_as_matrix(struct Matrix* mat, int rows_count, int cols_count, float* arr)
{
    mat->rows_count = rows_count;
    mat->cols_count = cols_count;
    mat->matrix = arr;
}

float* set_matrix_as_empty_object(struct Matrix* mat)
{
    mat->rows_count = 0;
    mat->cols_count = 0;
    float* arr = mat->matrix;
    mat->matrix = NULL;
    return arr;
}

struct Matrix* constructor_copy(struct Matrix* original)
{
    struct Matrix* copy = (struct Matrix*)malloc(sizeof(struct Matrix));
    int rows_count = original->rows_count;
    int cols_count = original->cols_count;
    copy->rows_count = rows_count;
    copy->cols_count = cols_count;
    copy->matrix = (float*)malloc(rows_count * cols_count * sizeof(float));
    for (int i = 0; i < rows_count; i++)
    {
        for (int j = 0; j < cols_count; j++)
        {
            copy->matrix[(cols_count * i + j)] = original->matrix[(cols_count * i + j)];
        }
    }
    return copy;
}
/*
float* detach_matrix_array(struct Matrix* mat)
{
    float* arr = mat->matrix;
    mat->matrix = NULL;
    mat->rows_count = 0;
    mat->cols_count = 0;
    return arr;
}
*/

void matrix_destructor(struct Matrix* mat)
{
    if (mat->matrix != NULL)
    {
        free(mat->matrix);
    }
    free(mat);
}

void matrix_add(struct Matrix* mat_left, struct Matrix* mat_right, struct Matrix* mat_result)
{
    for (int i = 0; i < mat_left->rows_count; i++)
    {
        for (int j = 0; j < mat_left->cols_count; j++)
        {
            mat_result->matrix[(mat_left->cols_count * i + j)] = mat_left->matrix[(mat_left->cols_count * i + j)] + mat_right->matrix[(mat_left->cols_count * i + j)];
        }
    }

}

void matrix_sub(struct Matrix* mat_left, struct Matrix* mat_right, struct Matrix* mat_result)
{
    for (int i = 0; i < mat_left->rows_count; i++)
    {
        for (int j = 0; j < mat_left->cols_count; j++)
        {
            mat_result->matrix[(mat_left->cols_count * i + j)] = mat_left->matrix[(mat_left->cols_count * i + j)] - mat_right->matrix[(mat_left->cols_count * i + j)];
        }
    }

}

void matrix_mul_by_number(struct Matrix* mat, struct Matrix* mat_result, float num)
{
    int rows_count = mat->rows_count, cols_count = mat->cols_count;
    for (int i = 0; i < rows_count; i++)
    {
        for (int j = 0; j < cols_count; j++)
        {
            mat_result->matrix[(cols_count * i + j)] = ((mat->matrix[(cols_count * i + j)]) * num);
        }
    }
}

void matrix_mul(struct Matrix* mat_left, struct Matrix* mat_right, struct Matrix* mat_result)
{
    int count_in_dot_product = mat_left->cols_count;
    for (int i = 0; i < mat_left->rows_count; i++)
    {
        for (int j = 0; j < mat_right->cols_count; j++)
        {
            float sum = 0.0;
            for (int k = 0; k < count_in_dot_product; k++)
            {
                /*
                if (i * mat_left->rows_count + k >= mat_left->rows_count * mat_left->cols_count)
                {
                    printf("Error mul, mat_left, rows: %d, cols: %d, i: %d, j: %d, k: %d, rows*cols: %d, cell no expr: %d\n", mat_left->rows_count, mat_left->cols_count, i, j, k, mat_left->rows_count * mat_left->cols_count, i * mat_left->rows_count + k);
                }
                if ((k * mat_right->rows_count + j) >= mat_right->rows_count * mat_right->cols_count )
                {
                    printf("mat left dims: rows: %d, cols %d, arr[783]: %f\n", mat_left->rows_count, mat_left->cols_count, mat_left->matrix[783]);
                    printf("Error mul, mat_right, rows: %d, cols: %d, i: %d, j: %d, k: %d, rows*cols: %d, cell no expr: %d\n", mat_right->rows_count, mat_right->cols_count, i, j, k, mat_right->rows_count * mat_right->cols_count, k * mat_right->rows_count + j);
                }*/
                sum += (mat_left->matrix[(i * mat_left->cols_count + k)] * mat_right->matrix[(k * mat_right->cols_count + j)]);
            }
            mat_result->matrix[(i * mat_result->cols_count + j)] = sum;
        }
    }
}

struct Matrix* prepare_result_matrix_for_multiplication(struct Matrix* mat_left, struct Matrix* mat_right)
{
    return matrix_constructor(mat_left->rows_count, mat_right->cols_count);
}



void print_matrix(struct Matrix* mat)
{
    printf("[");
    for (int i = 0; i < mat->rows_count; i++)
    {
        printf("[");
        for (int j = 0; j < mat->cols_count; j++)
        {
            printf("%f", mat->matrix[(i * mat->cols_count + j)]);
            if (j < mat->cols_count - 1)
            {
                printf(" ");
            }
        }
        printf("]");
        if (i < mat->rows_count - 1)
        {
            printf("\n");
        }
    }
    printf("]");
}

// softmax works on vectors represented by matrix of 1 row and n columns
void softmax(struct Matrix* input, struct Matrix* output)
{
    double sum = 0.0;
    int cols_count = input->cols_count;
    for (int i = 0; i < cols_count; i++)
    {
        input->matrix[i] = (float)exp(input->matrix[i]);
        sum += input->matrix[i];
    }
    float sum_2 = (float)(1.0 / sum);
    for (int i = 0; i < cols_count; i++)
    {
        output->matrix[i] = input->matrix[i] * sum_2;
    }
}

float relu(float x)
{
    relu_table[1] = x;
    return relu_table[(x > 0.0)];
}

void relu_vector(struct Matrix* input, struct Matrix* output)
{
    for (int i = 0; i < input->cols_count; i++)
    {
        output->matrix[i] = relu(input->matrix[i]);
        //set_matrix_cell(output, 0, i, relu(get_matrix_cell(input, 0, i)));
    }
}

struct Model
{
    struct Matrix* flattened_image;
    struct Matrix* layer_1_weights;
    struct Matrix* layer_1_biases;
    struct Matrix* layer_1_processing_result;
    struct Matrix* layer_2_weights;
    struct Matrix* layer_2_biases;
    struct Matrix* layer_2_processing_result;
    int input_image_rows_count;
    int input_image_cols_count;

};

struct Model* prepare_model()
{
    

    float* l_1_weights_memory_block = (float*)layer_1_weights;
    float* l_1_biases_memory_block = (float*)layer_1_biases;
    float* l_2_weights_memory_block = (float*)layer_2_weights;
    float* l2_biases_memory_block = (float*)layer_2_weights;

    struct Matrix* flattened_image_matrix = matrix_constructor_empty_object();
    
    struct Matrix* l_1_weights = matrix_constructor_empty_object();
    set_float_array_as_matrix(l_1_weights, l_1_weights_rows_count, l_1_weights_cols_count, l_1_weights_memory_block);
    struct Matrix* l_1_biases = matrix_constructor_empty_object();
    set_float_array_as_matrix(l_1_biases, l_1_biases_rows_count, l_1_biases_cols_count, l_1_biases_memory_block);
    struct Matrix* l_1_processing_result = matrix_constructor(l_1_biases_rows_count, l_1_biases_cols_count);

    struct Matrix* l_2_weights = matrix_constructor_empty_object();
    set_float_array_as_matrix(l_2_weights, l_2_weights_rows_count, l_2_weights_cols_count, l_2_weights_memory_block);
    struct Matrix* l_2_biases = matrix_constructor_empty_object();
    set_float_array_as_matrix(l_2_biases, l_2_biases_rows_count, l_2_biases_cols_count, l_1_biases_memory_block);
    struct Matrix* l_2_processing_result = matrix_constructor(l_2_biases_rows_count, l_2_biases_cols_count);


    struct Model* model = (struct Model*)malloc(sizeof(struct Model));
    model->flattened_image = flattened_image_matrix;
    model->layer_1_weights = l_1_weights;
    model->layer_1_biases = l_1_biases;
    model->layer_1_processing_result = l_1_processing_result;
    model->layer_2_weights = l_2_weights;
    model->layer_2_biases = l_2_biases;
    model->layer_2_processing_result = l_2_processing_result;
    model->input_image_rows_count = flattened_image_rows_count;
    model->input_image_cols_count = flattened_image_cols_count;
    return model;
}

float* model_predict_raw(struct Model* model, float* flattened_image, float* result)
{
    set_float_array_as_matrix(model->flattened_image, model->input_image_rows_count, model->input_image_cols_count, flattened_image);
    matrix_mul(model->flattened_image, model->layer_1_weights, model->layer_1_processing_result);
    set_matrix_as_empty_object(model->flattened_image);
    matrix_add(model->layer_1_processing_result, model->layer_1_biases, model->layer_1_processing_result);
    relu_vector(model->layer_1_processing_result, model->layer_1_processing_result);
    matrix_mul(model->layer_1_processing_result, model->layer_2_weights, model->layer_2_processing_result);
    matrix_add(model->layer_2_processing_result, model->layer_2_biases, model->layer_2_processing_result);
    softmax(model->layer_2_processing_result, model->layer_2_processing_result);
    for (int i = 0; i < model->layer_2_processing_result->cols_count; i++)
    {
        result[i] = model->layer_2_processing_result->matrix[i];
    }
    return result;
}
int model_predict(struct Model* model, float* flattened_image)
{
    set_float_array_as_matrix(model->flattened_image, model->input_image_rows_count, model->input_image_cols_count, flattened_image);
    matrix_mul(model->flattened_image, model->layer_1_weights, model->layer_1_processing_result);
    set_matrix_as_empty_object(model->flattened_image);
    matrix_add(model->layer_1_processing_result, model->layer_1_biases, model->layer_1_processing_result);
    relu_vector(model->layer_1_processing_result, model->layer_1_processing_result);
    matrix_mul(model->layer_1_processing_result, model->layer_2_weights, model->layer_2_processing_result);
    matrix_add(model->layer_2_processing_result, model->layer_2_biases, model->layer_2_processing_result);
    softmax(model->layer_2_processing_result, model->layer_2_processing_result);
    float highest = 0.0;
    int predicted_number = 0;
    int result_vector_len = model->layer_2_processing_result->cols_count;
    for (int i = 0; i < result_vector_len; i++)
    {
        float ithNeuronVal = get_matrix_cell(model->layer_2_processing_result, 0, i);

        if (ithNeuronVal > highest)
        {
            highest = ithNeuronVal;
            predicted_number = i;
        }
    }
    return predicted_number;
}


void model_destructor(struct Model* model)
{
    set_matrix_as_empty_object(model->layer_1_weights);
    set_matrix_as_empty_object(model->layer_1_biases);
    set_matrix_as_empty_object(model->layer_2_weights);
    set_matrix_as_empty_object(model->layer_2_biases);

    matrix_destructor(model->flattened_image);
    matrix_destructor(model->layer_1_weights);
    matrix_destructor(model->layer_1_biases);
    matrix_destructor(model->layer_1_processing_result);
    matrix_destructor(model->layer_2_weights);
    matrix_destructor(model->layer_2_biases);
    matrix_destructor(model->layer_2_processing_result);
    free(model);
}

void prepare_buffer(char* buff, int buff_len)
{
    for (int i = 0; i < buff_len; i++)
    {
        buff[i] = '\0';
    }
}

float** loadNumbersFromFile(char* filePath, int linesCount, int numbersPerLine)
{
    int buffer_len = BUFFER_LENGTH;
    char buffer[BUFFER_LENGTH];
    prepare_buffer(buffer, buffer_len);
    float** lines = (float**)malloc(linesCount * sizeof(float*));
    for (int i = 0; i < linesCount; i++)
    {
        lines[i] = (float*)malloc(numbersPerLine * sizeof(float));
    }

    FILE* fptr;
    // Open a file in read mode
    errno_t error_code = fopen_s(&fptr, filePath, "r");
    // Store the content of the file

    // Read the content and print it
    int i = 0;
    int j = 0;
    float one_by_255 = (float)(1.0 / 255.0);
    while (fgets(buffer, buffer_len, fptr) && i < linesCount ) {
        int buffIdx = 0;
        int doLineReading = 1;
        int sum = 0;
        j = 0;
        while(doLineReading)
        {
            /*
            if (!(buffer[buffIdx] == '\0' || buffer[buffIdx] == '\n' || buffer[buffIdx] == ' ' || buffer[buffIdx] == ',' || (buffer[buffIdx] >= 48 && buffer[buffIdx] <= 57)))
            {
                int undesiredChar = (int)buffer[buffIdx];
                printf("undesired character: %d\n", undesiredChar);
            }
            */
//            if (buffer[buffIdx] == '\0' || buffer[buffIdx] == '\n' || buffer[buffIdx] == ' ' || buffer[buffIdx] == (char)13 || buffer[buffIdx] == (char)27)
            if (buffer[buffIdx] != ',' && (buffer[buffIdx] < 48  || buffer[buffIdx] > 57))
            {
                lines[i][j] = sum * one_by_255 > 1.0f ? 1.0f : sum * one_by_255;
                //printf("read number - j: %d\n", j);
                prepare_buffer(buffer, buffer_len);
                doLineReading = 0;
            }
            if (doLineReading)
            {
                if (buffer[buffIdx] == ',')
                {
                    lines[i][j] = sum * one_by_255 > 1.0f ? 1.0f : sum * one_by_255;
                    sum = 0;
                    j++;
                }
                else
                {
                    sum *= 10;
                    sum += ( (int)buffer[buffIdx] - 48);
                }
                buffIdx++;
            }
        }
        i++;
    }

    // Close the file
    fclose(fptr);
    return lines;
}

void freeNumberLines(float** numberLines, int linesCount)
{
    for (int i = 0; i < linesCount; i++)
    {
        free(numberLines[i]);
    }
    free(numberLines);
}

int main()
{
    int numbersPerLine = 785;
    int loadedNumbersCount = 10000;
    //char inputFilePath[] = "C:\\Users\\krzys\\source\\repos\\MnistBenchmarkC\\assets\\mnist_train.csv\0";
    char inputFilePath[] = "mnist_data\\mnist_train.csv\0";

    float** lines = loadNumbersFromFile(inputFilePath, loadedNumbersCount, numbersPerLine);
    printf("numbers loaded\n");
    struct Model* model = prepare_model();
    printf("model prepared\n");
    
    int predictions[2];
    predictions[0] = 0;
    predictions[1] = 0;
    printf("Begin making predictions\n");
    //float* data = malloc((numbersPerLine - 1) * sizeof(float));

    clock_t start, end;
    double cpu_time_used;

    start = clock();

    for (int i = 0; i < loadedNumbersCount; i++)
    {
        float* dataBegin = lines[i];
        int label = (int)round(dataBegin[0] * 255.0);
        //for (int i = 1; i < numbersPerLine; i++)
        //{
        //    data[i - 1] = dataBegin[i];
        //}
        //int predictedNumber = model_predict(model, data);
        int predictedNumber = model_predict(model, dataBegin + 1);
        (predictions[(predictedNumber == label)])++;
        /*
        if (predictedNumber == label)
        {
            printf("Prediction correct - label: %d, model: %d\n", label, predictedNumber);
        }
        else
        {
            printf("Prediction incorrect - label: %d, model: %d\n", label, predictedNumber);
        }
        */
    }
    //free(data);
    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Predictions making done\n");
    printf("Make predictions for %d samples, computation time: %f\n", loadedNumbersCount, cpu_time_used);
    printf("Predictions correct: %d, percentage: %f, predictions incorrect: %d, percentage: %f\n", predictions[1], (predictions[1]*1.0/loadedNumbersCount), predictions[0], (predictions[0] * 1.0 / loadedNumbersCount));
    model_destructor(model);
    printf("model destroyed\n");
    freeNumberLines(lines, loadedNumbersCount);
    printf("memory for numbers deallocated");
    return 0;
}
