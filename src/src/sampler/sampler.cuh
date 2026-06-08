#pragma once

#include <cstdint>
#include "types.cuh"

namespace futaba {

struct pcg32 {
    uint64_t state = 0x853c49e6748fea9bull;
    uint64_t inc = 0xda3e39cb94b95bdbull;

    HD pcg32() : state(0x853c49e6748fea9bull), inc(0xda3e39cb94b95bdbull) {}

    HD explicit pcg32(uint64_t seed) {
        seed_rng(seed);
    }

    HD void seed_rng(uint64_t seed) {
        state = 0u;
        inc = (seed << 1u) | 1u;
        nextUInt32();
        state += seed;
        nextUInt32();
    }

    HD uint32_t nextUInt32() {
        uint64_t oldstate = state;
        state = oldstate * 6364136223846793005ull + inc;
        uint32_t xorshifted = static_cast<uint32_t>(((oldstate >> 18u) ^ oldstate) >> 27u);
        uint32_t rot = static_cast<uint32_t>(oldstate >> 59u);
        return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
    }

    HD float nextFloat() {
        return static_cast<float>(nextUInt32() >> 8) * (1.0f / 16777216.0f);
    }
};

struct Sampler {
    pcg32 m_random;

    HD Sampler() : m_random(1u) {}
    HD explicit Sampler(unsigned int seed) : m_random(static_cast<uint64_t>(seed ? seed : 1u)) {}

    HD float next1D() {
        return m_random.nextFloat();
    }

    HD Point2f next2D() {
        return Point2f(
            m_random.nextFloat(),
            m_random.nextFloat()
        );
    }
};

} // namespace futaba
