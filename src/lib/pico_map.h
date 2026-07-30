#pragma once

// NOTE: this hashmap was written by codex while helping with the tokenizer/data work.
// it is intentionally small: string keys, void* values, copied keys, linear probing.

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define PICO_HASHMAP_INITIAL_CAPACITY 16
#define PICO_HASHMAP_MAX_LOAD_NUM 7
#define PICO_HASHMAP_MAX_LOAD_DEN 10

struct PicoHashEntry {
    char* key;
    void* value;
    bool occupied;
};

struct PicoHashMap {
    struct PicoHashEntry* entries;
    size_t size;
    size_t capacity;
};

static inline char* pico_hashmap_key_copy(const char* key) {
    size_t len = strlen(key);
    char* copy = (char*)malloc(len + 1);
    if(copy == NULL) {
        return NULL;
    }

    memcpy(copy, key, len + 1);
    return copy;
}

// INFO: djb2 gives us a stable integer for a string. the hashmap turns that
// integer into a bucket with `% capacity`, then probes forward if the bucket is busy.
static inline size_t pico_hash_string(const char* key) {
    size_t hash = 5381;
    int c = 0;

    while((c = *key++) != '\0') {
        hash = ((hash << 5) + hash) + (size_t)c;
    }

    return hash;
}

static inline bool pico_hashmap_should_grow(struct PicoHashMap* map) {
    return (map->size + 1) * PICO_HASHMAP_MAX_LOAD_DEN >
           map->capacity * PICO_HASHMAP_MAX_LOAD_NUM;
}

static inline struct PicoHashMap* pico_hashmap_init_with_capacity(size_t capacity) {
    struct PicoHashMap* map = (struct PicoHashMap*)calloc(1, sizeof(struct PicoHashMap));
    if(map == NULL) {
        return NULL;
    }

    map->entries = (struct PicoHashEntry*)calloc(capacity, sizeof(struct PicoHashEntry));
    if(map->entries == NULL) {
        free(map);
        return NULL;
    }

    map->capacity = capacity;
    map->size = 0;
    return map;
}

static inline struct PicoHashMap* pico_hashmap_init(void) {
    return pico_hashmap_init_with_capacity(PICO_HASHMAP_INITIAL_CAPACITY);
}

static inline bool pico_hashmap_place_entry(struct PicoHashMap* map, char* key, void* value) {
    size_t index = pico_hash_string(key) % map->capacity;

    for(size_t probe = 0; probe < map->capacity; probe++) {
        struct PicoHashEntry* entry = &map->entries[index];
        if(!entry->occupied) {
            entry->key = key;
            entry->value = value;
            entry->occupied = true;
            map->size += 1;
            return true;
        }

        if(strcmp(entry->key, key) == 0) {
            entry->value = value;
            return true;
        }

        index = (index + 1) % map->capacity;
    }

    return false;
}

static inline bool pico_hashmap_grow(struct PicoHashMap* map) {
    size_t old_capacity = map->capacity;
    struct PicoHashEntry* old_entries = map->entries;

    map->capacity *= 2;
    map->entries = (struct PicoHashEntry*)calloc(map->capacity, sizeof(struct PicoHashEntry));
    if(map->entries == NULL) {
        map->entries = old_entries;
        map->capacity = old_capacity;
        return false;
    }

    map->size = 0;
    for(size_t i = 0; i < old_capacity; i++) {
        if(old_entries[i].occupied) {
            pico_hashmap_place_entry(map, old_entries[i].key, old_entries[i].value);
        }
    }

    free(old_entries);
    return true;
}

static inline bool pico_hashmap_insert(struct PicoHashMap* map, const char* key, void* value) {
    if(map == NULL || key == NULL) {
        return false;
    }

    if(pico_hashmap_should_grow(map) && !pico_hashmap_grow(map)) {
        return false;
    }

    size_t index = pico_hash_string(key) % map->capacity;
    for(size_t probe = 0; probe < map->capacity; probe++) {
        struct PicoHashEntry* entry = &map->entries[index];
        if(entry->occupied && strcmp(entry->key, key) == 0) {
            entry->value = value;
            return true;
        }

        if(!entry->occupied) {
            char* owned_key = pico_hashmap_key_copy(key);
            if(owned_key == NULL) {
                return false;
            }

            entry->key = owned_key;
            entry->value = value;
            entry->occupied = true;
            map->size += 1;
            return true;
        }

        index = (index + 1) % map->capacity;
    }

    return false;
}

static inline struct PicoHashEntry* pico_hashmap_find_entry(struct PicoHashMap* map, const char* key) {
    if(map == NULL || key == NULL || map->capacity == 0) {
        return NULL;
    }

    size_t index = pico_hash_string(key) % map->capacity;
    for(size_t probe = 0; probe < map->capacity; probe++) {
        struct PicoHashEntry* entry = &map->entries[index];
        if(!entry->occupied) {
            return NULL;
        }

        if(strcmp(entry->key, key) == 0) {
            return entry;
        }

        index = (index + 1) % map->capacity;
    }

    return NULL;
}

static inline void* pico_hashmap_get(struct PicoHashMap* map, const char* key) {
    struct PicoHashEntry* entry = pico_hashmap_find_entry(map, key);
    return entry == NULL ? NULL : entry->value;
}

static inline bool pico_hashmap_contains(struct PicoHashMap* map, const char* key) {
    return pico_hashmap_find_entry(map, key) != NULL;
}

static inline void pico_hashmap_free(struct PicoHashMap* map) {
    if(map == NULL) {
        return;
    }

    for(size_t i = 0; i < map->capacity; i++) {
        if(map->entries[i].occupied) {
            free(map->entries[i].key);
        }
    }

    free(map->entries);
    free(map);
}
