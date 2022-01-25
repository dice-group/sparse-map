#ifndef TSL_SPARSE_MAP_TESTS_UTILITY_FUNCTIONS_HPP
#define TSL_SPARSE_MAP_TESTS_UTILITY_FUNCTIONS_HPP


#ifdef TSL_DEBUG
#define tsl_sh_assert(expr) assert(expr)
#else
#define tsl_sh_assert(expr) (static_cast<void>(0))
#endif

#include <memory>

#include "sparse_growth_policy.h"

namespace tsl::detail_sparse_hash {

    /* to_address can convert any raw or fancy pointer into a raw pointer.
     * It is needed for the allocator construct and destroy calls.
     * This specific implementation is based on boost 1.71.0.
     */
#if __cplusplus >= 201400L  // with 14-features

    template<typename T>
    T *to_address(T *v) noexcept { return v; }

    namespace fancy_ptr_detail {
        template<typename T>
        inline T *ptr_address(T *v, int) noexcept { return v; }

        template<typename T>
        inline auto ptr_address(const T &v, int) noexcept
        -> decltype(std::pointer_traits<T>::to_address(v)) {
            return std::pointer_traits<T>::to_address(v);
        }

        template<typename T>
        inline auto ptr_address(const T &v, long) noexcept {
            return fancy_ptr_detail::ptr_address(v.operator->(), 0);
        }
    } // namespace detail

    template<typename T>
    inline auto to_address(const T &v) noexcept {
        return fancy_ptr_detail::ptr_address(v, 0);
    }

#else // without 14-features
    template <typename T>
    inline T *to_address(T *v) noexcept { return v; }

    template <typename T>
    inline typename std::pointer_traits<T>::element_type * to_address(const T &v) noexcept {
        return detail_sparse_hash::to_address(v.operator->());
    }
#endif


    template<typename T>
    struct make_void {
        using type = void;
    };

    template<typename T, typename = void>
    struct has_is_transparent : std::false_type {
    };

    template<typename T>
    struct has_is_transparent<T,
            typename make_void<typename T::is_transparent>::type>
            : std::true_type {
    };

    template<typename U>
    struct is_power_of_two_policy : std::false_type {
    };

    template<std::size_t GrowthFactor>
    struct is_power_of_two_policy<tsl::sh::power_of_two_growth_policy<GrowthFactor>>
            : std::true_type {
    };

    inline constexpr bool is_power_of_two(std::size_t value) {
        return value != 0 && (value & (value - 1)) == 0;
    }

    inline std::size_t round_up_to_power_of_two(std::size_t value) {
        if (is_power_of_two(value)) {
            return value;
        }

        if (value == 0) {
            return 1;
        }

        --value;
        for (std::size_t i = 1; i < sizeof(std::size_t) * CHAR_BIT; i *= 2) {
            value |= value >> i;
        }

        return value + 1;
    }

    template<typename T, typename U>
    static T numeric_cast(U value,
                          const char *error_message = "numeric_cast() failed.") {
        T ret = static_cast<T>(value);
        if (static_cast<U>(ret) != value) {
            throw std::runtime_error(error_message);
        }

        const bool is_same_signedness =
                (std::is_unsigned<T>::value && std::is_unsigned<U>::value) ||
                (std::is_signed<T>::value && std::is_signed<U>::value);
        if (!is_same_signedness && (ret < T{}) != (value < U{})) {
            throw std::runtime_error(error_message);
        }

        return ret;
    }

/**
 * Fixed size type used to represent size_type values on serialization. Need to
 * be big enough to represent a std::size_t on 32 and 64 bits platforms, and
 * must be the same size on both platforms.
 */
    using slz_size_type = std::uint64_t;
    static_assert(std::numeric_limits<slz_size_type>::max() >=
                  std::numeric_limits<std::size_t>::max(),
                  "slz_size_type must be >= std::size_t");

    template<class T, class Deserializer>
    static T deserialize_value(Deserializer &deserializer) {
        // MSVC < 2017 is not conformant, circumvent the problem by removing the
        // template keyword
#if defined(_MSC_VER) && _MSC_VER < 1910
        return deserializer.Deserializer::operator()<T>();
#else
        return deserializer.Deserializer::template operator()<T>();
#endif
    }
}

#endif //TSL_SPARSE_MAP_TESTS_UTILITY_FUNCTIONS_HPP
