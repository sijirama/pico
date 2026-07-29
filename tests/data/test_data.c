/*
 * Tests for the generic Dataset/DataLoader path.
 * NOTE: no UTEST_MAIN here, test_basic.c owns main + UTEST_STATE.
 */

#include <stddef.h>

#include "pico.h"
#include "utest.h"

struct TestDatasetData {
    struct PicoTensor** xs;
    struct PicoTensor** ys;
    size_t len;
};

static size_t test_dataset_len(const struct Dataset* dataset) {
    struct TestDatasetData* data = (struct TestDatasetData*)dataset->data;
    return data->len;
}

static struct DatasetItem test_dataset_get(const struct Dataset* dataset, size_t idx) {
    struct TestDatasetData* data = (struct TestDatasetData*)dataset->data;
    struct DatasetItem item = {
        .x = data->xs[idx],
        .y = data->ys[idx],
    };
    return item;
}

static void test_dataset_free(struct Dataset* dataset) {
    (void)dataset;
}

static const struct DatasetVTable TEST_DATASET_FUNCS = {
    .len = test_dataset_len,
    .get = test_dataset_get,
    .free = test_dataset_free,
};

static struct Dataset make_test_dataset(struct TestDatasetData* data) {
    struct Dataset dataset = {
        .funcs = &TEST_DATASET_FUNCS,
        .data = data,
    };
    return dataset;
}

static void fill_test_tensors(struct PicoContext* ctx, struct PicoTensor** xs, struct PicoTensor** ys,
                              size_t len) {
    int64_t shape[] = {1};

    for(size_t i = 0; i < len; i++) {
        float x_value[] = {(float)i};
        float y_value[] = {(float)(i * 10)};
        xs[i] = pico_tensor_from_data(ctx, shape, 1, x_value);
        ys[i] = pico_tensor_from_data(ctx, shape, 1, y_value);
    }
}

UTEST(dataloader, init_rejects_invalid_inputs) {
    struct PicoContext ctx = pico_context_init();

    struct TestDatasetData data = {.xs = NULL, .ys = NULL, .len = 0};
    struct Dataset dataset = make_test_dataset(&data);

    ASSERT_TRUE(pico_dataloader_init(NULL, &dataset, 2, false) == NULL);
    ASSERT_TRUE(pico_dataloader_init(&ctx, NULL, 2, false) == NULL);
    ASSERT_TRUE(pico_dataloader_init(&ctx, &dataset, 0, false) == NULL);

    pico_context_destroy(&ctx);
}

UTEST(dataloader, init_builds_sequential_indices) {
    struct PicoContext ctx = pico_context_init();

    struct PicoTensor* xs[4];
    struct PicoTensor* ys[4];
    fill_test_tensors(&ctx, xs, ys, 4);

    struct TestDatasetData data = {.xs = xs, .ys = ys, .len = 4};
    struct Dataset dataset = make_test_dataset(&data);

    struct DataLoader* loader = pico_dataloader_init(&ctx, &dataset, 2, false);
    ASSERT_TRUE(loader != NULL);
    ASSERT_TRUE(loader->dataset == &dataset);
    ASSERT_EQ(loader->batch_size, (size_t)2);
    ASSERT_EQ(loader->cursor, (size_t)0);

    for(size_t i = 0; i < 4; i++) {
        ASSERT_EQ(loader->indices[i], i);
    }

    pico_context_destroy(&ctx);
}

UTEST(dataloader, next_returns_full_batches_and_tail) {
    struct PicoContext ctx = pico_context_init();

    struct PicoTensor* xs[5];
    struct PicoTensor* ys[5];
    fill_test_tensors(&ctx, xs, ys, 5);

    struct TestDatasetData data = {.xs = xs, .ys = ys, .len = 5};
    struct Dataset dataset = make_test_dataset(&data);
    struct DataLoader* loader = pico_dataloader_init(&ctx, &dataset, 2, false);

    struct DataBatch* b1 = pico_dataloader_next(loader);
    ASSERT_TRUE(b1 != NULL);
    ASSERT_EQ(b1->size, (size_t)2);
    ASSERT_TRUE(b1->items[0].x->data[0] == 0.0f);
    ASSERT_TRUE(b1->items[1].x->data[0] == 1.0f);
    ASSERT_TRUE(b1->items[1].y->data[0] == 10.0f);

    struct DataBatch* b2 = pico_dataloader_next(loader);
    ASSERT_TRUE(b2 != NULL);
    ASSERT_EQ(b2->size, (size_t)2);
    ASSERT_TRUE(b2->items[0].x->data[0] == 2.0f);
    ASSERT_TRUE(b2->items[1].x->data[0] == 3.0f);

    struct DataBatch* b3 = pico_dataloader_next(loader);
    ASSERT_TRUE(b3 != NULL);
    ASSERT_EQ(b3->size, (size_t)1);
    ASSERT_TRUE(b3->items[0].x->data[0] == 4.0f);

    ASSERT_TRUE(pico_dataloader_next(loader) == NULL);

    pico_context_destroy(&ctx);
}

UTEST(dataloader, reset_rewinds_iteration) {
    struct PicoContext ctx = pico_context_init();

    struct PicoTensor* xs[3];
    struct PicoTensor* ys[3];
    fill_test_tensors(&ctx, xs, ys, 3);

    struct TestDatasetData data = {.xs = xs, .ys = ys, .len = 3};
    struct Dataset dataset = make_test_dataset(&data);
    struct DataLoader* loader = pico_dataloader_init(&ctx, &dataset, 2, false);

    ASSERT_TRUE(pico_dataloader_next(loader) != NULL);
    ASSERT_TRUE(pico_dataloader_next(loader) != NULL);
    ASSERT_TRUE(pico_dataloader_next(loader) == NULL);

    pico_dataloader_reset(loader);

    struct DataBatch* batch = pico_dataloader_next(loader);
    ASSERT_TRUE(batch != NULL);
    ASSERT_EQ(batch->size, (size_t)2);
    ASSERT_TRUE(batch->items[0].x->data[0] == 0.0f);

    pico_context_destroy(&ctx);
}

UTEST(dataloader, shuffle_keeps_index_permutation) {
    struct PicoContext ctx = pico_context_init();

    struct PicoTensor* xs[6];
    struct PicoTensor* ys[6];
    fill_test_tensors(&ctx, xs, ys, 6);

    struct TestDatasetData data = {.xs = xs, .ys = ys, .len = 6};
    struct Dataset dataset = make_test_dataset(&data);
    struct DataLoader* loader = pico_dataloader_init(&ctx, &dataset, 3, true);

    int seen[6] = {0};
    for(size_t i = 0; i < 6; i++) {
        ASSERT_TRUE(loader->indices[i] < 6);
        seen[loader->indices[i]] += 1;
    }

    for(size_t i = 0; i < 6; i++) {
        ASSERT_EQ(seen[i], 1);
    }

    pico_context_destroy(&ctx);
}

UTEST(dataloader, reset_null_guard_returns) {
    pico_dataloader_reset(NULL);
}
