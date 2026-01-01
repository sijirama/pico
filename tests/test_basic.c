#include "utest.h"
#include "pico.h"

UTEST(Basic, Version) {
    ASSERT_STREQ("0.01", PICO_VERSION);
}

UTEST_MAIN()
