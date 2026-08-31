/*
    custom safetensor serializer and deserializer.
    custom loading safetetnsor in c - https://leetarxiv.substack.com/p/parsing-safetensors-file-format
    format spec brief - https://huggingface.co/docs/safetensors/en/index#format

    The reference to 8-byte alignment applies strictly to the JSON Header to ensure the start of the data buffer lands
   smoothly on a memory-aligned boundary.Because the total file position where the data buffer starts is determined by
   8+ N (where 8 is the size indicator and N is the length of the JSON string), the size of your JSON header must
   satisfy a simple condition:

   (8+N)(mod 8) == 0

   Since 8 is already a multiple of 8, this simply means the JSON header string length (\(N\)) itself must be a perfect
   multiple of 8 bytes.
*/

#pragma once
#include <stddef.h>

#include "arena.h"
#define _POSIX_C_SOURCE 200809L
#include <assert.h>
#include <cjson/cJSON.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#include "ctx.h"
#include "tensor.h"

void save_tensor(struct PicoContext* ctx, char* file_name) {
    if(ctx == NULL || file_name == NULL) {
        fprintf(stderr, "PicoSaveSafeTensor: missing context or file name\n");
        return;
    }

    FILE* file;
    file = fopen(file_name, "wb");
    if(file == NULL) {
        fprintf(stderr, "PicoSaveSafeTensor: Could not open file \n");
        return;
    }

    char* header_string = NULL;
    char* padded_header = NULL;

    // construct header
    cJSON* header = cJSON_CreateObject();
    if(header == NULL) {
        goto end;
    }

    // INFO: build out header section.

    struct PicoTensor* tensor = NULL;
    uint64_t data_offset = 0;
    uint64_t tensor_bytes;
    uint64_t start;
    uint64_t end;

    for(int i = 0; i < ctx->params.size; i++) {
        tensor = ctx->params.data[i];
        if(tensor->name == NULL) {
            fprintf(stderr, "PicoSaveSafeTensor: cannot save unnamed tensor\n");
            goto end;
        }

        cJSON* tensorObj = cJSON_CreateObject();

        if(tensorObj == NULL) {
            goto end;
        }

        // add dtype to object
        if(cJSON_AddStringToObject(tensorObj, "dtype", "F32") == NULL) {
            goto end;
        }

        // add array of shape to object
        cJSON* shape = cJSON_AddArrayToObject(tensorObj, "shape");
        if(shape == NULL)
            goto end;

        for(int i = 0; i < tensor->ndim; i++) {
            cJSON* num = cJSON_CreateNumber(tensor->shape[i]);
            cJSON_AddItemToArray(shape, num);
        }

        // add offsets - add it later
        tensor_bytes = tensor->numel * sizeof(float);
        start = data_offset;
        end = start + tensor_bytes;
        data_offset = end;

        cJSON* offsets = cJSON_AddArrayToObject(tensorObj, "data_offsets");
        if(offsets == NULL)
            goto end;
        cJSON_AddItemToArray(offsets, cJSON_CreateNumber(start));
        cJSON_AddItemToArray(offsets, cJSON_CreateNumber(end));

        cJSON_AddItemToObject(header, tensor->name, tensorObj);
    }

    header_string = cJSON_PrintUnformatted(header);
    if(header_string == NULL) {
        goto end;
    }

    uint64_t raw_header_size = strlen(header_string);
    uint64_t header_size = raw_header_size;
    uint64_t padding = header_size % 8;
    if(padding != 0) {
        header_size += 8 - padding;
    }

    padded_header = malloc(header_size + 1);
    if(padded_header == NULL) {
        goto end;
    }
    memcpy(padded_header, header_string, raw_header_size);
    memset(padded_header + raw_header_size, ' ', header_size - raw_header_size);
    padded_header[header_size] = '\0';

    fwrite(&header_size, sizeof(uint64_t), 1, file);
    fwrite(padded_header, 1, header_size, file);

    for(int i = 0; i < ctx->params.size; i++) {
        tensor = ctx->params.data[i];
        fwrite(tensor->data, sizeof(float), tensor->numel, file);
    }

    goto end;
end:
    cJSON_Delete(header);
    fclose(file);
    free(header_string);
    free(padded_header);
    return;
}

size_t GetFileSize(char* fileName) {
    FILE* fp = fopen(fileName, "rb");
    assert(fp != NULL);
    fseek(fp, 0L, SEEK_END);
    size_t currentFileSize = ftell(fp);
    rewind(fp);
    fclose(fp);
    return currentFileSize;
}

unsigned char* LoadSafeTensorData(char* fileName, size_t* fileSizeHolder) {
    size_t fileSize = GetFileSize(fileName);
    FILE* fp = fopen(fileName, "rb");
    assert(fp != NULL);

    int fd = fileno(fp);
    unsigned char* fileData = mmap(NULL, fileSize, PROT_READ, MAP_PRIVATE, fd, 0);
    assert(fileData != NULL);
    assert(fileData != MAP_FAILED);

    fclose(fp);
    *fileSizeHolder = fileSize;
    return fileData;
}

size_t* parseSafeTensorHeaderSizeData(struct PicoContext* ctx, unsigned char* mmapd) {
    size_t* headerLength = arena_alloc(ctx->arena, sizeof(size_t));
    for(int i = 7; i >= 0; i--) {
        *headerLength <<= 8;
        headerLength += mmapd[i];
    }
    return headerLength;
}

char* parseSafeTensorHeader(struct PicoContext* ctx, const unsigned char* mmapd, const size_t* headerLength) {
    //
    // make sure that the 8 byte is the beginning of the header string
    assert(mmapd[8] == '{');

    cJSON* tensorData = cJSON_ParseWithLength(mmapd + 8, *headerLength);
    assert(tensorData != NULL);

    char* formatted_json = cJSON_Print(tensorData);
    assert(formatted_json != NULL);

    char* header = arena_alloc(ctx->arena, *headerLength);
    strcpy(header, formatted_json);

    free(formatted_json);
    cJSON_Delete(tensorData);
    return header;
}

void load_tensor(struct PicoContext* ctx, char* file_name) {
    size_t fileSize = 0;
    unsigned char* safeTensorData = LoadSafeTensorData(file_name, &fileSize);
    assert(safeTensorData != NULL);

    size_t* headerLength = parseSafeTensorHeaderSizeData(ctx, safeTensorData);
    char * header = parseSafeTensorHeader(ctx, safeTensorData, headerLength);
}
