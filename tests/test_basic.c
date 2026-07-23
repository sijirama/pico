#include "utest.h"
#include "pico.h"

UTEST(Basic, Version) {
    ASSERT_STREQ("0.01", PICO_VERSION);
}

UTEST_STATE();

int main(int argc, const char* const argv[]) {
    int result = utest_main(argc, argv);
    pico_shutdown();
    return result;
}
