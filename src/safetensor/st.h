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
#include <cjson/cJSON.h>

#include "stdlib.h"
#include "ctx.h"

void save_tensor(struct PicoContext* ctx, char* file_name) {
    char* header_length_in_bytes = malloc(8 * sizeof(char));
    cJSON* header = cJSON_CreateObject();
    if(header == NULL) {
        goto end;
    }

    //INFO: build out header section.

end:
    cJSON_Delete(header);
    free(header_length_in_bytes);
    return;
}

struct PicoContext* load_tensor() {
    return NULL;
}
