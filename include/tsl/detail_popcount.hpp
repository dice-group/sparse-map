#ifndef TSL_SPARSE_MAP_TESTS_DETAIL_POPCOUNT_HPP
#define TSL_SPARSE_MAP_TESTS_DETAIL_POPCOUNT_HPP

#include <cstdint>

#ifdef __INTEL_COMPILER
#include <immintrin.h>  // For _popcnt32 and _popcnt64
#endif

#ifdef _MSC_VER
#include <intrin.h>  // For __cpuid, __popcnt and __popcnt64
#endif

namespace tsl::detail_popcount {
/**
 * Define the popcount(ll) methods and pick-up the best depending on the
 * compiler.
 */

// From Wikipedia: https://en.wikipedia.org/wiki/Hamming_weight
    inline int fallback_popcountll(unsigned long long int x) {
        static_assert(
                sizeof(unsigned long long int) == sizeof(std::uint64_t),
                "sizeof(unsigned long long int) must be equal to sizeof(std::uint64_t). "
                "Open a feature request if you need support for a platform where it "
                "isn't the case.");

        const std::uint64_t m1 = 0x5555555555555555ull;
        const std::uint64_t m2 = 0x3333333333333333ull;
        const std::uint64_t m4 = 0x0f0f0f0f0f0f0f0full;
        const std::uint64_t h01 = 0x0101010101010101ull;

        x -= (x >> 1ull) & m1;
        x = (x & m2) + ((x >> 2ull) & m2);
        x = (x + (x >> 4ull)) & m4;
        return static_cast<int>((x * h01) >> (64ull - 8ull));
    }

    inline int fallback_popcount(unsigned int x) {
        static_assert(sizeof(unsigned int) == sizeof(std::uint32_t) ||
                      sizeof(unsigned int) == sizeof(std::uint64_t),
                      "sizeof(unsigned int) must be equal to sizeof(std::uint32_t) "
                      "or sizeof(std::uint64_t). "
                      "Open a feature request if you need support for a platform "
                      "where it isn't the case.");

        if (sizeof(unsigned int) == sizeof(std::uint32_t)) {
            const std::uint32_t m1 = 0x55555555;
            const std::uint32_t m2 = 0x33333333;
            const std::uint32_t m4 = 0x0f0f0f0f;
            const std::uint32_t h01 = 0x01010101;

            x -= (x >> 1) & m1;
            x = (x & m2) + ((x >> 2) & m2);
            x = (x + (x >> 4)) & m4;
            return static_cast<int>((x * h01) >> (32 - 8));
        } else {
            return fallback_popcountll(x);
        }
    }

#if defined(__clang__) || defined(__GNUC__)
    inline int popcountll(unsigned long long int value) {
        return __builtin_popcountll(value);
    }

    inline int popcount(unsigned int value) { return __builtin_popcount(value); }

#elif defined(_MSC_VER)
    /**
 * We need to check for popcount support at runtime on Windows with __cpuid
 * See https://msdn.microsoft.com/en-us/library/bb385231.aspx
 */
inline bool has_popcount_support() {
  int cpu_infos[4];
  __cpuid(cpu_infos, 1);
  return (cpu_infos[2] & (1 << 23)) != 0;
}

inline int popcountll(unsigned long long int value) {
#ifdef _WIN64
  static_assert(
      sizeof(unsigned long long int) == sizeof(std::int64_t),
      "sizeof(unsigned long long int) must be equal to sizeof(std::int64_t). ");

  static const bool has_popcount = has_popcount_support();
  return has_popcount
             ? static_cast<int>(__popcnt64(static_cast<std::int64_t>(value)))
             : fallback_popcountll(value);
#else
  return fallback_popcountll(value);
#endif
}

inline int popcount(unsigned int value) {
  static_assert(sizeof(unsigned int) == sizeof(std::int32_t),
                "sizeof(unsigned int) must be equal to sizeof(std::int32_t). ");

  static const bool has_popcount = has_popcount_support();
  return has_popcount
             ? static_cast<int>(__popcnt(static_cast<std::int32_t>(value)))
             : fallback_popcount(value);
}

#elif defined(__INTEL_COMPILER)
inline int popcountll(unsigned long long int value) {
  static_assert(sizeof(unsigned long long int) == sizeof(__int64), "");
  return _popcnt64(static_cast<__int64>(value));
}

inline int popcount(unsigned int value) {
  return _popcnt32(static_cast<int>(value));
}

#else
inline int popcountll(unsigned long long int x) {
  return fallback_popcountll(x);
}

inline int popcount(unsigned int x) { return fallback_popcount(x); }

#endif
}  // namespace detail_popcount
#endif //TSL_SPARSE_MAP_TESTS_DETAIL_POPCOUNT_HPP
