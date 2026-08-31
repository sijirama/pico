#include <cjson/cJSON.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pico.h"
#include "safetensor/st.h"
#include "tensor.h"
#include "utest.h"

UTEST(safetensor, save_writes_header_offsets_and_raw_tensor_data) {
    const char* path = "/tmp/pico_safetensor_save_test.safetensors";
    remove(path);

    struct PicoContext* ctx = pico_init_verbose(false);

    int64_t weight_shape[] = {2, 2};
    struct PicoTensor* weight = pico_param_named(ctx, "l1.weight", weight_shape, 2);
    weight->data[0] = 1.0f;
    weight->data[1] = 2.0f;
    weight->data[2] = 3.0f;
    weight->data[3] = 4.0f;

    int64_t bias_shape[] = {2};
    struct PicoTensor* bias = pico_param_named(ctx, "l1.bias", bias_shape, 1);
    bias->data[0] = 5.0f;
    bias->data[1] = 6.0f;

    save_tensor(ctx, (char*)path);

    FILE* file = fopen(path, "rb");
    ASSERT_TRUE(file != NULL);

    uint64_t header_size = 0;
    ASSERT_EQ(fread(&header_size, sizeof(uint64_t), 1, file), (size_t)1);
    ASSERT_TRUE(header_size > 0);
    ASSERT_EQ(header_size % 8, (uint64_t)0);

    char* header_bytes = malloc(header_size + 1);
    ASSERT_TRUE(header_bytes != NULL);
    ASSERT_EQ(fread(header_bytes, 1, header_size, file), header_size);
    header_bytes[header_size] = '\0';

    cJSON* header = cJSON_Parse(header_bytes);
    ASSERT_TRUE(header != NULL);

    cJSON* weight_obj = cJSON_GetObjectItemCaseSensitive(header, "l1.weight");
    ASSERT_TRUE(weight_obj != NULL);
    ASSERT_STREQ(cJSON_GetObjectItemCaseSensitive(weight_obj, "dtype")->valuestring, "F32");

    cJSON* weight_shape_json = cJSON_GetObjectItemCaseSensitive(weight_obj, "shape");
    ASSERT_EQ(cJSON_GetArraySize(weight_shape_json), 2);
    ASSERT_EQ((int)cJSON_GetArrayItem(weight_shape_json, 0)->valuedouble, 2);
    ASSERT_EQ((int)cJSON_GetArrayItem(weight_shape_json, 1)->valuedouble, 2);

    cJSON* weight_offsets = cJSON_GetObjectItemCaseSensitive(weight_obj, "data_offsets");
    ASSERT_EQ(cJSON_GetArraySize(weight_offsets), 2);
    ASSERT_EQ((int)cJSON_GetArrayItem(weight_offsets, 0)->valuedouble, 0);
    ASSERT_EQ((int)cJSON_GetArrayItem(weight_offsets, 1)->valuedouble, 16);

    cJSON* bias_obj = cJSON_GetObjectItemCaseSensitive(header, "l1.bias");
    ASSERT_TRUE(bias_obj != NULL);

    cJSON* bias_offsets = cJSON_GetObjectItemCaseSensitive(bias_obj, "data_offsets");
    ASSERT_EQ(cJSON_GetArraySize(bias_offsets), 2);
    ASSERT_EQ((int)cJSON_GetArrayItem(bias_offsets, 0)->valuedouble, 16);
    ASSERT_EQ((int)cJSON_GetArrayItem(bias_offsets, 1)->valuedouble, 24);

    float values[6] = {0};
    ASSERT_EQ(fread(values, sizeof(float), 6, file), (size_t)6);
    ASSERT_NEAR(values[0], 1.0f, 1e-6f);
    ASSERT_NEAR(values[1], 2.0f, 1e-6f);
    ASSERT_NEAR(values[2], 3.0f, 1e-6f);
    ASSERT_NEAR(values[3], 4.0f, 1e-6f);
    ASSERT_NEAR(values[4], 5.0f, 1e-6f);
    ASSERT_NEAR(values[5], 6.0f, 1e-6f);

    ASSERT_EQ(fgetc(file), EOF);

    cJSON_Delete(header);
    free(header_bytes);
    fclose(file);
    remove(path);
    pico_shutdown(ctx);
}

UTEST(safetensor, save_handles_many_context_params) {
    const char* path = "/tmp/pico_safetensor_many_params_test.safetensors";
    remove(path);

    struct PicoContext* ctx = pico_init_verbose(false);
    const int param_count = 40;
    struct PicoTensor* params[param_count];
    char names[param_count][32];
    uint64_t expected_start[param_count];
    uint64_t expected_end[param_count];
    uint64_t data_offset = 0;
    int total_values = 0;

    for(int i = 0; i < param_count; i++) {
        snprintf(names[i], sizeof(names[i]), "layer%d.weight", i);

        int64_t shape[] = {(i % 7) + 1};
        params[i] = pico_param_named(ctx, names[i], shape, 1);
        ASSERT_TRUE(params[i] != NULL);

        expected_start[i] = data_offset;
        data_offset += params[i]->numel * sizeof(float);
        expected_end[i] = data_offset;
        total_values += params[i]->numel;

        for(int j = 0; j < params[i]->numel; j++) {
            params[i]->data[j] = (float)(i * 100 + j);
        }
    }

    ASSERT_EQ(ctx->params.size, (size_t)param_count);
    save_tensor(ctx, (char*)path);

    FILE* file = fopen(path, "rb");
    ASSERT_TRUE(file != NULL);

    uint64_t header_size = 0;
    ASSERT_EQ(fread(&header_size, sizeof(uint64_t), 1, file), (size_t)1);
    ASSERT_TRUE(header_size > 0);
    ASSERT_EQ(header_size % 8, (uint64_t)0);

    char* header_bytes = malloc(header_size + 1);
    ASSERT_TRUE(header_bytes != NULL);
    ASSERT_EQ(fread(header_bytes, 1, header_size, file), header_size);
    header_bytes[header_size] = '\0';

    cJSON* header = cJSON_Parse(header_bytes);
    ASSERT_TRUE(header != NULL);

    for(int i = 0; i < param_count; i++) {
        cJSON* tensor_obj = cJSON_GetObjectItemCaseSensitive(header, names[i]);
        ASSERT_TRUE(tensor_obj != NULL);
        ASSERT_STREQ(cJSON_GetObjectItemCaseSensitive(tensor_obj, "dtype")->valuestring, "F32");

        cJSON* shape = cJSON_GetObjectItemCaseSensitive(tensor_obj, "shape");
        ASSERT_EQ(cJSON_GetArraySize(shape), 1);
        ASSERT_EQ((int)cJSON_GetArrayItem(shape, 0)->valuedouble, params[i]->numel);

        cJSON* offsets = cJSON_GetObjectItemCaseSensitive(tensor_obj, "data_offsets");
        ASSERT_EQ(cJSON_GetArraySize(offsets), 2);
        ASSERT_EQ((uint64_t)cJSON_GetArrayItem(offsets, 0)->valuedouble, expected_start[i]);
        ASSERT_EQ((uint64_t)cJSON_GetArrayItem(offsets, 1)->valuedouble, expected_end[i]);
    }

    float* values = malloc(total_values * sizeof(float));
    ASSERT_TRUE(values != NULL);
    ASSERT_EQ(fread(values, sizeof(float), total_values, file), (size_t)total_values);

    int value_index = 0;
    for(int i = 0; i < param_count; i++) {
        for(int j = 0; j < params[i]->numel; j++) {
            ASSERT_NEAR(values[value_index], (float)(i * 100 + j), 1e-6f);
            value_index++;
        }
    }

    ASSERT_EQ(fgetc(file), EOF);

    free(values);
    cJSON_Delete(header);
    free(header_bytes);
    fclose(file);
    remove(path);
    pico_shutdown(ctx);
}
