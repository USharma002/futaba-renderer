#pragma once

// Shared OptiX error-checking helpers. Every host .cpp/.cu file that calls
// into the OptiX API should use these instead of rolling its own copy.

#include <optix.h>
#include <cstdio>
#include <cstdlib>

// Logs and continues (used for calls where a soft failure is acceptable,
// e.g. context/module setup performed once at startup).
#define OPTIX_CHECK(call)                                                      \
    do {                                                                       \
        const OptixResult _res = (call);                                      \
        if (_res != OPTIX_SUCCESS) {                                          \
            fprintf(stderr, "OptiX error at %s:%d  code=%d\n",                \
                    __FILE__, __LINE__, (int)_res);                           \
        }                                                                      \
    } while (0)

// Same as above, but also prints the compiler/linker log buffer and aborts.
// Used for calls that produce a log (module/program-group/pipeline creation),
// where continuing after a failure would just crash later with less context.
#define OPTIX_CHECK_LOG(call, logBuffer)                                       \
    do {                                                                       \
        const OptixResult _res = (call);                                      \
        if (_res != OPTIX_SUCCESS) {                                          \
            fprintf(stderr, "OptiX call (" #call ") failed with error code: %d\n", (int)_res); \
            fprintf(stderr, "OptiX Log:\n%s\n", (logBuffer));                 \
            exit(1);                                                          \
        }                                                                      \
    } while (0)
