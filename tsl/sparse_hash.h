/**
 * MIT License
 * 
 * Copyright (c) 2017 Tessil
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#ifndef TSL_SPARSE_HASH_H
#define TSL_SPARSE_HASH_H


#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <iterator>
#include <limits>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>
#include "sparse_growth_policy.h"



#ifndef tsl_assert
    #ifdef TSL_DEBUG
    #define tsl_assert(expr) assert(expr)
    #else
    #define tsl_assert(expr) (static_cast<void>(0))
    #endif
#endif


#ifdef __INTEL_COMPILER
#include <immintrin.h> // For _popcnt32 and _popcnt64
#endif

#ifdef _MSC_VER
#include <intrin.h> // For __cpuid, __popcnt and __popcnt64
#endif

namespace tsl {
    
namespace sh {    
    enum class probing {
        linear,
        quadratic
    };    
    
    enum class exception_safety {
        basic,
        strong
    };    
}


namespace detail_popcount {
/**
 * Define the popcount(ll) methods and pick-up the best depending on the compiler.
 */
// From Wikipedia: https://en.wikipedia.org/wiki/Hamming_weight
inline int default_popcountll(unsigned long long int x) {
    static_assert(sizeof(unsigned long long int) == sizeof(std::uint64_t),
                  "sizeof(unsigned long long int) must be equal to sizeof(std::uint64_t). "
                  "Open a feature request if you need support for a platform where it isn't the case.");

    const std::uint64_t m1 = 0x5555555555555555ull;
    const std::uint64_t m2 = 0x3333333333333333ull;
    const std::uint64_t m4 = 0x0f0f0f0f0f0f0f0full;
    const std::uint64_t h01 = 0x0101010101010101ull;

    x -= (x >> 1ull) & m1;
    x = (x & m2) + ((x >> 2ull) & m2);
    x = (x + (x >> 4ull)) & m4;
    return (x * h01) >> (64ull - 8ull);
}

inline int default_popcount(unsigned int x) {
    static_assert(sizeof(int) == sizeof(std::uint32_t) || sizeof(int) == sizeof(std::uint64_t),
                  "sizeof(unsigned long long int) must be equal to sizeof(std::uint32_t) or sizeof(std::uint64_t). "
                  "Open a feature request if you need support for a platform where it isn't the case.");

    if (sizeof(int) == sizeof(std::uint32_t)) {
        const std::uint32_t m1 = 0x55555555;
        const std::uint32_t m2 = 0x33333333;
        const std::uint32_t m4 = 0x0f0f0f0f;
        const std::uint32_t h01 = 0x01010101;

        x -= (x >> 1) & m1;
        x = (x & m2) + ((x >> 2) & m2);
        x = (x + (x >> 4)) & m4;
        return (x * h01) >> (32 - 8);
    }
    else {
        return default_popcountll(x);
    }
}

#if defined(__clang__) || defined(__GNUC__) 
inline int popcountll(unsigned long long int value) {
    return __builtin_popcountll(value);
}

inline int popcount(unsigned int value) {
    return __builtin_popcount(value);
}

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
#ifdef  _WIN64
    static_assert(sizeof(unsigned long long int) == sizeof(std::int64_t),
        "sizeof(unsigned long long int) must be equal to sizeof(std::int64_t). ");

    static const bool has_popcount = has_popcount_support();
    return has_popcount?static_cast<int>(__popcnt64(static_cast<std::int64_t>(value))):default_popcountll(value);
#else
    return default_popcountll(value);
#endif
}

inline int popcount(unsigned int value) {
    static_assert(sizeof(unsigned int) == sizeof(std::int32_t),
        "sizeof(unsigned int) must be equal to sizeof(std::int32_t). ");

    static const bool has_popcount = has_popcount_support();
    return has_popcount?static_cast<int>(__popcnt(static_cast<std::int32_t>(value))):default_popcount(value);
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
    return default_popcountll(x);
}

inline int popcount(unsigned int x) {
    return default_popcount(x);
}

#endif
}


namespace detail_sparse_hash {
    
template<typename T>
struct make_void {
    using type = void;
};

template<typename T, typename = void>
struct has_is_transparent: std::false_type {
};

template<typename T>
struct has_is_transparent<T, typename make_void<typename T::is_transparent>::type>: std::true_type {
};

template<typename U>
struct is_power_of_two_policy: std::false_type {
};

template<std::size_t GrowthFactor>
struct is_power_of_two_policy<tsl::sh::power_of_two_growth_policy<GrowthFactor>>: std::true_type {
};


inline constexpr bool is_power_of_two(std::size_t value) {
    return value != 0 && (value & (value - 1)) == 0;
}

inline std::size_t round_up_to_power_of_two(std::size_t value) {
    if(is_power_of_two(value)) {
        return value;
    }
    
    if(value == 0) {
        return 1;
    }
        
    --value;
    for(std::size_t i = 1; i < sizeof(std::size_t) * CHAR_BIT; i *= 2) {
        value |= value >> i;
    }
    
    return value + 1;
}
    
    
/**
 * WARNING: the sparse_array class doesn't free the ressources allocated through the allocator passed in parameter
 * in each method. You have to manually call `clear(Allocator&)` when you don't need the class anymore.
 * 
 * The reason is that the sparse_array doesn't store the allocator. It only allocate/deallocate objects with
 * the allocator that is passed in parameter.
 * 
 * WARNING: Strong exception guarantee only holds if std::is_nothrow_move_constructible<T>::value is true. 
 * Otherwise, only the basic guarantee is there (all ressources will be released but the array may be in
 * an inconsitent state if an exception is thrown).
 * 
 * Index designs a value between [0, BITMAP_NB_BITS), it is an index similar to std::vector.
 * Offset designs the real position in `m_values` corresponding to an index.
 * 
 * TODO Check to use std::realloc and std::memmove when possible
 */
template<typename T, typename Allocator>
class sparse_array {
public:
    using value_type = T;
    using size_type = std::uint_least8_t;
    using allocator_type = Allocator;
    using iterator = value_type*;
    using const_iterator = const value_type*;
    
    /**
     * Bitmap size configuration.
     * Use 32 bits on 32-bits or less environnements as popcount on 64 bits is slow on these environnements,
     * 64 bits otherwise.
     */
#if SIZE_MAX <= UINT32_MAX
    using bitmap_type = std::uint_least32_t;
    static const std::size_t BITMAP_NB_BITS = 32;
    static const std::size_t SHIFT = 5;
    static const std::size_t MASK = BITMAP_NB_BITS - 1;
    static const size_type CAPACITY_GROWTH_STEP = 4;
#else
    using bitmap_type = std::uint_least64_t;
    static const std::size_t BITMAP_NB_BITS = 64;
    static const std::size_t SHIFT = 6;
    static const std::size_t MASK = BITMAP_NB_BITS - 1;
    static const size_type CAPACITY_GROWTH_STEP = 8;
#endif    
    
    static_assert(is_power_of_two(BITMAP_NB_BITS), "BITMAP_NB_BITS must be a power of two.");
    static_assert(std::numeric_limits<bitmap_type>::digits >= BITMAP_NB_BITS, 
                        "bitmap_type must be able to hold at least BITMAP_NB_BITS.");
    static_assert((std::size_t(1) << SHIFT) == BITMAP_NB_BITS, "(1 << SHIFT) must be equal to BITMAP_NB_BITS.");
    
public:
    sparse_array() noexcept: m_values(nullptr), m_bitmap_vals(0), m_bitmap_deleted_vals(0), 
                             m_nb_elements(0), m_capacity(0), m_last_array(false)
    {
    }
    
    sparse_array(const sparse_array& other, Allocator& alloc): 
                             m_values(nullptr), m_bitmap_vals(other.m_bitmap_vals), 
                             m_bitmap_deleted_vals(other.m_bitmap_deleted_vals), 
                             m_nb_elements(0), m_capacity(other.m_capacity), 
                             m_last_array(other.m_last_array)
    {
        if(m_capacity == 0) {
            return;
        }
        
        m_values = alloc.allocate(m_capacity);
        tsl_assert(m_values != nullptr);  // allocate should throw if there is a failure
        try {
            for(size_type i = 0; i < other.m_nb_elements; i++) {
                construct_value(alloc, m_values + i, other.m_values[i]);
                m_nb_elements++;
            }
        }
        catch(...) {
            clear(alloc);
            throw;
        }
    }
    
    sparse_array(sparse_array&& other) noexcept: m_values(other.m_values), m_bitmap_vals(other.m_bitmap_vals), 
                                                 m_bitmap_deleted_vals(other.m_bitmap_deleted_vals), 
                                                 m_nb_elements(other.m_nb_elements), m_capacity(other.m_capacity), 
                                                 m_last_array(other.m_last_array)
    {
        other.m_values = nullptr;
        other.m_bitmap_vals = 0;
        other.m_bitmap_deleted_vals = 0;
        other.m_nb_elements = 0;
        other.m_capacity = 0;
    }
    
    sparse_array(sparse_array&& other, Allocator& alloc): 
                                                 m_values(nullptr), m_bitmap_vals(other.m_bitmap_vals), 
                                                 m_bitmap_deleted_vals(other.m_bitmap_deleted_vals), 
                                                 m_nb_elements(0), m_capacity(other.m_capacity), 
                                                 m_last_array(other.m_last_array)
    {
        if(m_capacity == 0) {
            return;
        }
        
        m_values = alloc.allocate(m_capacity);
        tsl_assert(m_values != nullptr);  // allocate should throw if there is a failure
        try {
            for(size_type i = 0; i < other.m_nb_elements; i++) {
                construct_value(alloc, m_values + i, std::move(other.m_values[i]));
                m_nb_elements++;
            }
        }
        catch(...) {
            clear(alloc);
            throw;
        }
    }
    
    sparse_array& operator=(const sparse_array& ) = delete;
    sparse_array& operator=(sparse_array&& ) = delete;
    
    iterator begin() noexcept { return m_values; }
    iterator end() noexcept { return m_values + m_nb_elements; }
    const_iterator begin() const noexcept { return cbegin(); }
    const_iterator end() const noexcept { return cend(); }
    const_iterator cbegin() const noexcept { return m_values; }
    const_iterator cend() const noexcept { return m_values + m_nb_elements; }

    bool empty() const noexcept {
        return m_nb_elements == 0;
    }
    
    size_type size() const noexcept {
        return m_nb_elements;
    }
    
    void clear(allocator_type& alloc) noexcept {
        destroy_and_deallocate_values(alloc, m_values, m_nb_elements, m_capacity);
        
        m_values = nullptr;
        m_bitmap_vals = 0;
        m_bitmap_deleted_vals = 0;
        m_nb_elements = 0;
        m_capacity = 0;
    }
    
    bool last() const noexcept {
        return m_last_array;
    }
    
    void set_as_last() noexcept {
        m_last_array = true;
    }
    
    bool has_value_at_index(size_type index) const noexcept {
        tsl_assert(index < BITMAP_NB_BITS);
        return (m_bitmap_vals & (bitmap_type(1) << index)) != 0;
    }
    
    bool has_deleted_value_at_index(size_type index) const noexcept {
        tsl_assert(index < BITMAP_NB_BITS);
        return (m_bitmap_deleted_vals & (bitmap_type(1) << index)) != 0;
    }
    
    iterator value_at_index(size_type index) noexcept {
        tsl_assert(has_value_at_index(index));
        return m_values + index_to_offset(index);
    }
    
    const_iterator value_at_index(size_type index) const noexcept {
        tsl_assert(has_value_at_index(index));
        return m_values + index_to_offset(index);
    }
    
    /**
     * Return iterator to set value.
     */
    template<typename... Args>
    iterator set_value_at_index(allocator_type& alloc, size_type index, Args&&... value_args) {
        tsl_assert(!has_value_at_index(index));
        
        const size_type offset = index_to_offset(index);
        if(m_nb_elements < m_capacity) {
            insert_at_offset_no_alloc(alloc, offset, std::forward<Args>(value_args)...);
        }
        else { 
            insert_at_offset_alloc(alloc, offset, std::forward<Args>(value_args)...);
        }
        
        m_bitmap_vals = (m_bitmap_vals | (bitmap_type(1) << index));
        m_bitmap_deleted_vals = (m_bitmap_deleted_vals & ~(bitmap_type(1) << index));
        
        tsl_assert(has_value_at_index(index));
        tsl_assert(!has_deleted_value_at_index(index));
        return m_values + offset;
    }
    
    iterator erase_value_at_position(allocator_type& alloc, iterator position) {
        const size_type offset = static_cast<size_type>(std::distance(begin(), position));
        return erase_value_at_position(alloc, position, offset_to_index(offset));
    }
    
    // Return the next value or end if no next value
    iterator erase_value_at_position(allocator_type& alloc, iterator position, size_type index) {
        tsl_assert(has_value_at_index(index));
        tsl_assert(!has_deleted_value_at_index(index));
        
        const size_type offset = static_cast<size_type>(std::distance(begin(), position));
        for(std::size_t i = offset + 1; i < m_nb_elements; i++) {
            m_values[i - 1] = std::move(m_values[i]);
        }
        destroy_value(alloc, m_values + m_nb_elements - 1);
        
        m_bitmap_vals = (m_bitmap_vals ^ (bitmap_type(1) << index));
        m_bitmap_deleted_vals = (m_bitmap_deleted_vals | (bitmap_type(1) << index)); 
        
        tsl_assert(!has_value_at_index(index));
        tsl_assert(has_deleted_value_at_index(index));
        
        m_nb_elements--;
        
        return position;
    }
    
    void swap(sparse_array& other) {
        using std::swap;
        
        swap(m_values, other.m_values);
        swap(m_bitmap_vals, other.m_bitmap_vals);
        swap(m_bitmap_deleted_vals, other.m_bitmap_deleted_vals);
        swap(m_nb_elements, other.m_nb_elements);
        swap(m_capacity, other.m_capacity);
        swap(m_last_array, other.m_last_array);
    }
    
    static iterator mutable_iterator(const_iterator pos) {
        return const_cast<iterator>(pos);
    }
    
private:
    template<typename... Args>
    static void construct_value(allocator_type& alloc, value_type* value, Args&&... value_args) {
        std::allocator_traits<allocator_type>::construct(alloc, value, std::forward<Args>(value_args)...);
    }
    
    static void destroy_value(allocator_type& alloc, value_type* value) noexcept {
        std::allocator_traits<allocator_type>::destroy(alloc, value);
    }
    
    static void destroy_and_deallocate_values(allocator_type& alloc, value_type* values, 
                                              size_type nb_values, size_type capacity_value) noexcept 
    {
        for(size_type i = 0; i < nb_values; i++) {
            destroy_value(alloc, values + i);
        }
        
        alloc.deallocate(values, capacity_value);
    }
    
    static size_type popcount(bitmap_type val) noexcept {
        if(sizeof(bitmap_type) <= sizeof(unsigned int)) {
            return static_cast<size_type>(tsl::detail_popcount::popcount(static_cast<unsigned int>(val)));
        }
        else {
            return static_cast<size_type>(tsl::detail_popcount::popcountll(val));
        }
    }
    
    size_type index_to_offset(size_type index) const noexcept {
        tsl_assert(index < BITMAP_NB_BITS);
        return popcount(m_bitmap_vals & ((bitmap_type(1) << index) - bitmap_type(1)));
    }
    
    //TODO optimize
    size_type offset_to_index(size_type offset) const noexcept {
        tsl_assert(offset < m_nb_elements);
        
        bitmap_type bitmap_vals = m_bitmap_vals;
        size_type index = 0;
        size_type nb_ones = 0;
        
        while(bitmap_vals != 0) {
            if((bitmap_vals & 0x1) == 1) {
                if(nb_ones == offset) {
                    break;
                }
                
                nb_ones++;
            }
            
            index++;
            bitmap_vals = bitmap_vals >> 1;
        }
        
        return index;
    }
    
    size_type next_capacity() const noexcept {
        return static_cast<size_type>(m_capacity + CAPACITY_GROWTH_STEP);
    }
    

    
    template<typename... Args, typename U = value_type, 
             typename std::enable_if<!std::is_nothrow_move_constructible<U>::value>::type* = nullptr>
    void insert_at_offset_no_alloc(allocator_type& alloc, size_type offset, Args&&... value_args) {
        tsl_assert(offset <= m_nb_elements);
        
        size_type i = m_nb_elements;
        try {
            while(i > offset) {
                construct_value(alloc, m_values + i, std::move(m_values[i - 1]));
                destroy_value(alloc, m_values + i - 1);
                i--;
            }
            
            construct_value(alloc, m_values + offset, std::forward<Args>(value_args)...);
        }
        catch(...) {
            for(size_type j = m_nb_elements;  j != i; j--) {
                destroy_value(alloc, m_values + j);
            }
            
            throw;
        }
        
        m_nb_elements++;
    }
    
    template<typename... Args, typename U = value_type, 
             typename std::enable_if<std::is_nothrow_move_constructible<U>::value>::type* = nullptr>
    void insert_at_offset_no_alloc(allocator_type& alloc, size_type offset, Args&&... value_args) {
        tsl_assert(offset <= m_nb_elements);
        
        for(size_type i = m_nb_elements; i > offset; i--) {
            construct_value(alloc, m_values + i, std::move(m_values[i - 1]));
            destroy_value(alloc, m_values + i - 1);
        }
        
        try {
            construct_value(alloc, m_values + offset, std::forward<Args>(value_args)...);
        }
        catch(...) {
            for(size_type i = offset; i < m_nb_elements; i++) {
                construct_value(alloc, m_values + i, std::move(m_values[i + 1]));
                destroy_value(alloc, m_values + i + 1);
            }
            throw;
        }
        
        m_nb_elements++;
    }
   
   
   
    template<typename... Args, typename U = value_type, 
             typename std::enable_if<!std::is_nothrow_move_constructible<U>::value>::type* = nullptr>
    void insert_at_offset_alloc(allocator_type& alloc, size_type offset, Args&&... value_args) {
        value_type* new_values = alloc.allocate(next_capacity());
        tsl_assert(new_values != nullptr); // allocate should throw if there is a failure
        
        size_type nb_new_elements = 0;
        try {
            for(size_type i = 0; i < offset; i++) {
                construct_value(alloc, new_values + i, m_values[i]);
                nb_new_elements++;
            }
            
            construct_value(alloc, new_values + offset, std::forward<Args>(value_args)...);
            nb_new_elements++;
            
            for(size_type i = offset; i < m_nb_elements; i++) {
                construct_value(alloc, new_values + i + 1, m_values[i]);
                nb_new_elements++;
            }
        }
        catch(...) {
            destroy_and_deallocate_values(alloc, new_values, nb_new_elements, next_capacity());
            throw;
        }
        
        destroy_and_deallocate_values(alloc, m_values, m_nb_elements, m_capacity);
        m_values = new_values;
        
        m_capacity = next_capacity();
        m_nb_elements++;
    }
   
    template<typename... Args, typename U = value_type, 
             typename std::enable_if<std::is_nothrow_move_constructible<U>::value>::type* = nullptr>
    void insert_at_offset_alloc(allocator_type& alloc, size_type offset, Args&&... value_args) {
        value_type* new_values = alloc.allocate(next_capacity());
        tsl_assert(new_values != nullptr); // allocate should throw if there is a failure
        
        try {
            construct_value(alloc, new_values + offset, std::forward<Args>(value_args)...);
        }
        catch(...) {
            alloc.deallocate(new_values, next_capacity());
            throw;
        }
        
        // Should not throw from here
        for(size_type i = 0; i < offset; i++) {
            construct_value(alloc, new_values + i, std::move(m_values[i]));
        }
        
        for(size_type i = offset; i < m_nb_elements; i++) {
            construct_value(alloc, new_values + i + 1, std::move(m_values[i]));
        }
        
        
        destroy_and_deallocate_values(alloc, m_values, m_nb_elements, m_capacity);
        m_values = new_values;
        
        m_capacity = next_capacity();
        m_nb_elements++;
    }
    
private:
    value_type* m_values;
    
    bitmap_type m_bitmap_vals;
    bitmap_type m_bitmap_deleted_vals;
    
    size_type m_nb_elements;
    size_type m_capacity;
    bool m_last_array;
};


template<class ValueType,
         class KeySelect,
         class ValueSelect,
         class Hash,
         class KeyEqual,
         class Allocator,
         class GrowthPolicy,
         tsl::sh::exception_safety ExceptionSafety,
         tsl::sh::probing Probing>
class sparse_hash: private Allocator, private Hash, private KeyEqual, private GrowthPolicy {
private:    
    template<typename U>
    using has_mapped_type = typename std::integral_constant<bool, !std::is_same<U, void>::value>;
    
    static_assert(ExceptionSafety != tsl::sh::exception_safety::strong ||
                  (std::is_nothrow_move_constructible<ValueType>::value &&
                   std::is_copy_constructible<ValueType>::value), 
                  "If ExceptionSafety is set to tsl::sh::exception_safety::strong, the value_type must be "
                  "copy constructible and nothrow move constructible.");
    
public:   
    template<bool IsConst>
    class sparse_iterator;
    
    using key_type = typename KeySelect::key_type;
    using value_type = ValueType;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using hasher = Hash;
    using key_equal = KeyEqual;
    using allocator_type = Allocator;
    using reference = value_type&;
    using const_reference = const value_type&;
    using pointer = value_type*;
    using const_pointer = const value_type*;
    using iterator = sparse_iterator<false>;
    using const_iterator = sparse_iterator<true>;
        
    
private:
    using sparse_array = tsl::detail_sparse_hash::sparse_array<ValueType, Allocator>;
    
    using sparse_buckets_allocator = 
                            typename std::allocator_traits<allocator_type>::template rebind_alloc<sparse_array>;
    using sparse_buckets_container = std::vector<sparse_array, sparse_buckets_allocator>;
    
public:
    template<bool IsConst>
    class sparse_iterator {
        friend class sparse_hash;
        
    private:
        using sparse_bucket_iterator = typename std::conditional<IsConst, 
                                                                 typename sparse_buckets_container::const_iterator, 
                                                                 typename sparse_buckets_container::iterator>::type;
                                                                 
        using sparse_array_iterator = typename std::conditional<IsConst, 
                                                                typename sparse_array::const_iterator, 
                                                                typename sparse_array::iterator>::type;
                                                          
        /**
         * sparse_array_it should be nullptr if sparse_bucket_it == m_sparse_buckets.end(). (TODO better way?)
         */
        sparse_iterator(sparse_bucket_iterator sparse_bucket_it,
                        sparse_array_iterator sparse_array_it): m_sparse_buckets_it(sparse_bucket_it), 
                                                                m_sparse_array_it(sparse_array_it) 
        {
        }
        
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = const typename sparse_hash::value_type;
        using difference_type = std::ptrdiff_t;
        using reference = value_type&;
        using pointer = value_type*;
        
        
        sparse_iterator() noexcept {
        }
        
        sparse_iterator(const sparse_iterator<false>& other) noexcept: m_sparse_buckets_it(other.m_sparse_buckets_it),
                                                                       m_sparse_array_it(other.m_sparse_array_it)
        {
        }
        
        const typename sparse_hash::key_type& key() const {
            return KeySelect()(*m_sparse_array_it);
        }

        template<class U = ValueSelect, typename std::enable_if<has_mapped_type<U>::value && IsConst>::type* = nullptr>
        const typename U::value_type& value() const {
            return U()(*m_sparse_array_it);
        }

        template<class U = ValueSelect, typename std::enable_if<has_mapped_type<U>::value && !IsConst>::type* = nullptr>
        typename U::value_type& value() {
            return U()(*m_sparse_array_it);
        }
        
        reference operator*() const {
            return *m_sparse_array_it;
        }
        
        pointer operator->() const {
            return std::addressof(*m_sparse_array_it);
        }
        
        sparse_iterator& operator++() {
            ++m_sparse_array_it;
            
            if(m_sparse_array_it == m_sparse_buckets_it->end()) {
                do {
                    if(m_sparse_buckets_it->last()) {
                        ++m_sparse_buckets_it;
                        m_sparse_array_it = nullptr;
                        return *this;
                    }
                    
                    ++m_sparse_buckets_it;
                } while(m_sparse_buckets_it->empty());
                
                m_sparse_array_it = m_sparse_buckets_it->begin();              
            }
            
            return *this;
        }
        
        sparse_iterator operator++(int) {
            sparse_iterator tmp(*this);
            ++*this;
            
            return tmp;
        }
        
        friend bool operator==(const sparse_iterator& lhs, const sparse_iterator& rhs) {
            return lhs.m_sparse_buckets_it == rhs.m_sparse_buckets_it && 
                   lhs.m_sparse_array_it == rhs.m_sparse_array_it;
        }
        
        friend bool operator!=(const sparse_iterator& lhs, const sparse_iterator& rhs) { 
            return !(lhs == rhs); 
        }
        
    private:
        sparse_bucket_iterator m_sparse_buckets_it;
        sparse_array_iterator m_sparse_array_it;
    };
    
    
public:
    sparse_hash(size_type bucket_count, 
                const Hash& hash,
                const KeyEqual& equal,
                const Allocator& alloc,
                float max_load_factor): Allocator(alloc), Hash(hash), KeyEqual(equal),
                                        // We need a non-zero bucket_count
                                        GrowthPolicy(bucket_count == 0?++bucket_count:bucket_count),
                                        m_sparse_buckets(alloc), 
                                        m_bucket_count(bucket_count),
                                        m_nb_elements(0)
    {
        if(m_bucket_count > max_bucket_count()) {
            throw std::length_error("The map exceeds its maxmimum size.");
        }
        
        /*
         * We can't use the `vector(size_type count, const Allocator& alloc)` constructor
         * as it's only available in C++14 and we need to support C++11. We thus must resize after using
         * the `vector(const Allocator& alloc)` constructor.
         * 
         * We can't use `vector(size_type count, const T& value, const Allocator& alloc)` as it requires the
         * value T to be copyable.
         */
        const size_type nb_sparse_buckets = std::max(size_type(1), 
                                                     tsl::detail_sparse_hash::round_up_to_power_of_two(bucket_count) >> sparse_array::SHIFT);
        m_sparse_buckets.resize(nb_sparse_buckets);
        m_sparse_buckets.back().set_as_last();
        
        
        this->max_load_factor(max_load_factor);
    }
    
    ~sparse_hash() {
        clear();
    }
    
    sparse_hash(const sparse_hash& other): 
                    Allocator(std::allocator_traits<Allocator>::select_on_container_copy_construction(other)),
                    Hash(other),
                    KeyEqual(other),
                    GrowthPolicy(other),
                    m_sparse_buckets(std::allocator_traits<Allocator>::select_on_container_copy_construction(other)),
                    m_bucket_count(other.m_bucket_count),
                    m_nb_elements(other.m_nb_elements),
                    m_load_threshold(other.m_load_threshold),
                    m_max_load_factor(other.m_max_load_factor)
    {
        m_sparse_buckets.reserve(other.m_sparse_buckets.size());
        for(const auto& bucket: other.m_sparse_buckets) {
            m_sparse_buckets.emplace_back(bucket, static_cast<Allocator&>(*this));
        }
    }
    
    sparse_hash(sparse_hash&& other) noexcept(std::is_nothrow_move_constructible<Allocator>::value &&
                                              std::is_nothrow_move_constructible<Hash>::value &&
                                              std::is_nothrow_move_constructible<KeyEqual>::value &&
                                              std::is_nothrow_move_constructible<GrowthPolicy>::value &&
                                              std::is_nothrow_move_constructible<sparse_buckets_container>::value)
                                          : Allocator(std::move(other)),
                                            Hash(std::move(other)),
                                            KeyEqual(std::move(other)),
                                            GrowthPolicy(std::move(other)),
                                            m_sparse_buckets(std::move(other.m_sparse_buckets)),
                                            m_bucket_count(other.m_bucket_count),
                                            m_nb_elements(other.m_nb_elements),
                                            m_load_threshold(other.m_load_threshold),
                                            m_max_load_factor(other.m_max_load_factor)
    {
        other.clear();
    }
    
    sparse_hash& operator=(const sparse_hash& other) {
        if(this != &other) {
            clear();
            
            if(std::allocator_traits<Allocator>::propagate_on_container_copy_assignment::value) {
                Allocator::operator=(other);
            }
            Hash::operator=(other);
            KeyEqual::operator=(other);
            GrowthPolicy::operator=(other);
            
            if(std::allocator_traits<Allocator>::propagate_on_container_copy_assignment::value) {
                m_sparse_buckets = sparse_buckets_container(other);
            }
            else {
                m_sparse_buckets.clear();
            }
            
            m_sparse_buckets.reserve(other.m_sparse_buckets.size());
            for(const auto& bucket: other.m_sparse_buckets) {
                m_sparse_buckets.emplace_back(bucket, static_cast<Allocator&>(*this));
            }
            
            
            m_bucket_count = other.m_bucket_count;
            m_nb_elements = other.m_nb_elements;
            m_load_threshold = other.m_load_threshold;
            m_max_load_factor = other.m_max_load_factor;
        }
        
        return *this;
    }
    
    sparse_hash& operator=(sparse_hash&& other) {
        clear();
        if(std::allocator_traits<Allocator>::propagate_on_container_move_assignment::value) {
            static_cast<Allocator&>(*this) = std::move(static_cast<Allocator&>(other));
            m_sparse_buckets = std::move(other.m_sparse_buckets);
        }
        else if(static_cast<Allocator&>(*this) != static_cast<Allocator&>(other)) {
            m_sparse_buckets.reserve(other.m_sparse_buckets.size());
            for(auto&& bucket: other.m_sparse_buckets) {
                m_sparse_buckets.emplace_back(std::move(bucket), static_cast<Allocator&>(*this));
            }
        }
        else {
            static_cast<Allocator&>(*this) = std::move(static_cast<Allocator&>(other));
            m_sparse_buckets = std::move(other.m_sparse_buckets);
        }

        static_cast<Hash&>(*this) = std::move(static_cast<Hash&>(other));
        static_cast<KeyEqual&>(*this) = std::move(static_cast<KeyEqual&>(other));
        static_cast<GrowthPolicy&>(*this) = std::move(static_cast<GrowthPolicy&>(other));
        m_bucket_count = other.m_bucket_count;
        m_nb_elements = other.m_nb_elements;
        m_load_threshold = other.m_load_threshold;
        m_max_load_factor = other.m_max_load_factor;
        
        other.clear();
        
        return *this;
    }
    
    allocator_type get_allocator() const {
        return static_cast<Allocator&>(*this);
    }
    
    
    /*
     * Iterators
     */    
    iterator begin() noexcept {
        auto begin = m_sparse_buckets.begin();
        while(begin != m_sparse_buckets.end() && begin->empty()) {
            ++begin;
        }
        
        return iterator(begin, (begin != m_sparse_buckets.end())?begin->begin():nullptr);
    }
    
    const_iterator begin() const noexcept {
        return cbegin();
    }
    
    const_iterator cbegin() const noexcept {
        auto begin = m_sparse_buckets.cbegin();
        while(begin != m_sparse_buckets.cend() && begin->empty()) {
            ++begin;
        }
        
        return const_iterator(begin, (begin != m_sparse_buckets.cend())?begin->cbegin():nullptr);
    }
    
    iterator end() noexcept {
        return iterator(m_sparse_buckets.end(), nullptr);
    }
    
    const_iterator end() const noexcept {
        return cend();
    }
    
    const_iterator cend() const noexcept {
        return const_iterator(m_sparse_buckets.cend(), nullptr);
    }
    
    
    /*
     * Capacity
     */
    bool empty() const noexcept {
        return m_nb_elements == 0;
    }
    
    size_type size() const noexcept {
        return m_nb_elements;
    }
    
    size_type max_size() const noexcept {
        return std::min(std::allocator_traits<Allocator>::max_size(), m_sparse_buckets.max_size());
    }
    
    /*
     * Modifiers
     */
    void clear() noexcept {
        for(auto& bucket: m_sparse_buckets) {
            bucket.clear(*this);
        }
        
        m_nb_elements = 0;
    }
    
    
    
    template<typename P>
    std::pair<iterator, bool> insert(P&& value) {
        return insert_impl(KeySelect()(value), std::forward<P>(value));
    }
    
    template<typename P>
    iterator insert(const_iterator hint, P&& value) { 
        if(hint != cend() && compare_keys(KeySelect()(*hint), KeySelect()(value))) { 
            return mutable_iterator(hint); 
        }
        
        return insert(std::forward<P>(value)).first; 
    }
    
    template<class InputIt>
    void insert(InputIt first, InputIt last) {
        if(std::is_base_of<std::forward_iterator_tag, 
                           typename std::iterator_traits<InputIt>::iterator_category>::value) 
        {
            const auto nb_elements_insert = std::distance(first, last);
            const size_type nb_free_buckets = m_load_threshold - size();
            tsl_assert(m_load_threshold >= size());
            
            if(nb_elements_insert > 0 && nb_free_buckets < size_type(nb_elements_insert)) {
                reserve(size() + size_type(nb_elements_insert));
            }
        }
        
        for(; first != last; ++first) {
            insert(*first);
        }
    }
    
    
    
    template<class K, class M>
    std::pair<iterator, bool> insert_or_assign(K&& key, M&& obj) { 
        auto it = try_emplace(std::forward<K>(key), std::forward<M>(obj));
        if(!it.second) {
            it.first.value() = std::forward<M>(obj);
        }
        
        return it;
    }
    
    template<class K, class M>
    iterator insert_or_assign(const_iterator hint, K&& key, M&& obj) {
        if(hint != cend() && compare_keys(KeySelect()(*hint), key)) { 
            auto it = mutable_iterator(hint); 
            it.value() = std::forward<M>(obj);
            
            return it;
        }
        
        return insert_or_assign(std::forward<K>(key), std::forward<M>(obj)).first;
    }

    
    template<class... Args>
    std::pair<iterator, bool> emplace(Args&&... args) {
        return insert(value_type(std::forward<Args>(args)...));
    }
    
    template<class... Args>
    iterator emplace_hint(const_iterator hint, Args&&... args) {
        return insert(hint, value_type(std::forward<Args>(args)...));        
    }
    
    
    
    template<class K, class... Args>
    std::pair<iterator, bool> try_emplace(K&& key, Args&&... args) {
        return insert_impl(key, std::piecewise_construct, 
                                std::forward_as_tuple(std::forward<K>(key)), 
                                std::forward_as_tuple(std::forward<Args>(args)...));
    }
    
    template<class K, class... Args>
    iterator try_emplace(const_iterator hint, K&& key, Args&&... args) { 
        if(hint != cend() && compare_keys(KeySelect()(*hint), key)) { 
            return mutable_iterator(hint); 
        }
        
        return try_emplace(std::forward<K>(key), std::forward<Args>(args)...).first;
    }
    
    /**
     * Here to avoid `template<class K> size_type erase(const K& key)` being used when
     * we use a iterator instead of a const_iterator.
     */
    iterator erase(iterator pos) {
        auto it_sparse_array_next = pos.m_sparse_buckets_it->erase_value_at_position(*this, pos.m_sparse_array_it);
        m_nb_elements--;

        if(it_sparse_array_next == pos.m_sparse_buckets_it->end()) {
            auto it_sparse_buckets_next = pos.m_sparse_buckets_it;
            do {
                ++it_sparse_buckets_next;
            } while(it_sparse_buckets_next != m_sparse_buckets.end() && it_sparse_buckets_next->empty());
            
            if(it_sparse_buckets_next == m_sparse_buckets.end()) {
                return end();
            }
            else {
                return iterator(it_sparse_buckets_next, it_sparse_buckets_next->begin());
            }
        }
        else {
            return iterator(pos.m_sparse_buckets_it, it_sparse_array_next);
        }
    }
    
    iterator erase(const_iterator pos) {
        return erase(mutable_iterator(pos));
    }
    
    iterator erase(const_iterator first, const_iterator last) {
        if(first == last) {
            return mutable_iterator(first);
        }
        
        const std::size_t nb_elements_to_erase = std::distance(first, last);
        auto to_delete = mutable_iterator(first);
        for(std::size_t i = 0; i < nb_elements_to_erase; i++) {
            to_delete = erase(to_delete);
        }
        
        return to_delete;
    }
    
    
    template<class K>
    size_type erase(const K& key) {
        return erase(key, hash_key(key));
    }
    
    template<class K>
    size_type erase(const K& key, std::size_t hash) {
        return erase_impl(key, hash);
    }
    
    
    
    void swap(sparse_hash& other) {
        using std::swap;
        
        if(std::allocator_traits<Allocator>::propagate_on_container_swap::value) {
            swap(static_cast<Allocator&>(*this), static_cast<Allocator&>(other));
        }
        else {
            tsl_assert(static_cast<Allocator&>(*this) == static_cast<Allocator&>(other));
        }
        
        swap(static_cast<Hash&>(*this), static_cast<Hash&>(other));
        swap(static_cast<KeyEqual&>(*this), static_cast<KeyEqual&>(other));
        swap(static_cast<GrowthPolicy&>(*this), static_cast<GrowthPolicy&>(other));
        swap(m_sparse_buckets, other.m_sparse_buckets);
        swap(m_bucket_count, other.m_bucket_count);
        swap(m_nb_elements, other.m_nb_elements);
        swap(m_load_threshold, other.m_load_threshold);
        swap(m_max_load_factor, other.m_max_load_factor);
    }
    
    
    /*
     * Lookup
     */
    template<class K, class U = ValueSelect, typename std::enable_if<has_mapped_type<U>::value>::type* = nullptr>
    typename U::value_type& at(const K& key) {
        return at(key, hash_key(key));
    }
    
    template<class K, class U = ValueSelect, typename std::enable_if<has_mapped_type<U>::value>::type* = nullptr>
    typename U::value_type& at(const K& key, std::size_t hash) {
        return const_cast<typename U::value_type&>(static_cast<const sparse_hash*>(this)->at(key, hash));
    }
    
    
    template<class K, class U = ValueSelect, typename std::enable_if<has_mapped_type<U>::value>::type* = nullptr>
    const typename U::value_type& at(const K& key) const {
        return at(key, hash_key(key));
    }
    
    template<class K, class U = ValueSelect, typename std::enable_if<has_mapped_type<U>::value>::type* = nullptr>
    const typename U::value_type& at(const K& key, std::size_t hash) const {
        auto it = find(key, hash);
        if(it != cend()) {
            return it.value();
        }
        else {
            throw std::out_of_range("Couldn't find key.");
        }
    }
    
    template<class K, class U = ValueSelect, typename std::enable_if<has_mapped_type<U>::value>::type* = nullptr>
    typename U::value_type& operator[](K&& key) {
        return try_emplace(std::forward<K>(key)).first.value();
    }
    
    
    template<class K>
    size_type count(const K& key) const {
        return count(key, hash_key(key));
    }
    
    template<class K>
    size_type count(const K& key, std::size_t hash) const {
        if(find(key, hash) != cend()) {
            return 1;
        }
        else {
            return 0;
        }
    }
    
    
    template<class K>
    iterator find(const K& key) {
        return find_impl(key, hash_key(key));
    }
    
    template<class K>
    iterator find(const K& key, std::size_t hash) {
        return find_impl(key, hash);
    }
    
    
    template<class K>
    const_iterator find(const K& key) const {
        return find_impl(key, hash_key(key));
    }
    
    template<class K>
    const_iterator find(const K& key, std::size_t hash) const {
        return find_impl(key, hash);
    }
    
    
    template<class K>
    std::pair<iterator, iterator> equal_range(const K& key) {
        return equal_range(key, hash_key(key));
    }
    
    template<class K>
    std::pair<iterator, iterator> equal_range(const K& key, std::size_t hash) {
        iterator it = find(key, hash);
        return std::make_pair(it, (it == end())?it:std::next(it));
    }
    
    
    template<class K>
    std::pair<const_iterator, const_iterator> equal_range(const K& key) const {
        return equal_range(key, hash_key(key));
    }
    
    template<class K>
    std::pair<const_iterator, const_iterator> equal_range(const K& key, std::size_t hash) const {
        const_iterator it = find(key, hash);
        return std::make_pair(it, (it == cend())?it:std::next(it));
    }
    
    /*
     * Bucket interface 
     */
    size_type bucket_count() const {
        return m_bucket_count;
    }
    
    size_type max_bucket_count() const {
        return m_sparse_buckets.max_size();
    }
    
    /*
     * Hash policy 
     */
    float load_factor() const {
        return float(m_nb_elements)/float(bucket_count());
    }
    
    float max_load_factor() const {
        return m_max_load_factor;
    }
    
    void max_load_factor(float ml) {
        m_max_load_factor = std::max(0.1f, std::min(ml, 0.95f));
        m_load_threshold = size_type(float(bucket_count())*m_max_load_factor);
    }
    
    void rehash(size_type count) {
        count = std::max(count, size_type(std::ceil(float(size())/max_load_factor())));
        rehash_impl(count);
    }
    
    void reserve(size_type count) {
        rehash(size_type(std::ceil(float(count)/max_load_factor())));
    }  
    
    /*
     * Observers
     */
    hasher hash_function() const {
        return static_cast<const Hash&>(*this);
    }
    
    key_equal key_eq() const {
        return static_cast<const KeyEqual&>(*this);
    }
    
    
    /*
     * Other
     */    
    iterator mutable_iterator(const_iterator pos) {
        auto it_sparse_buckets = m_sparse_buckets.begin() + 
                                 std::distance(m_sparse_buckets.cbegin(), pos.m_sparse_buckets_it);
                                    
        return iterator(it_sparse_buckets, sparse_array::mutable_iterator(pos.m_sparse_array_it));
    }
    
private:
    template<class K>
    std::size_t hash_key(const K& key) const {
        return Hash::operator()(key);
    }
    
    template<class K1, class K2>
    bool compare_keys(const K1& key1, const K2& key2) const {
        return KeyEqual::operator()(key1, key2);
    }
    
    size_type bucket_for_hash(std::size_t hash) const {
        return GrowthPolicy::bucket_for_hash(hash);
    }
    
    template<class U = GrowthPolicy, typename std::enable_if<is_power_of_two_policy<U>::value>::type* = nullptr>
    size_type next_bucket(size_type ibucket, size_type iprobe) const {
        (void) iprobe;
        if(Probing == tsl::sh::probing::linear) {
            return (ibucket + 1) & this->m_mask;
        }
        else {
            tsl_assert(Probing == tsl::sh::probing::quadratic);
            return (ibucket + iprobe) & this->m_mask;
        }
    }
    
    template<class U = GrowthPolicy, typename std::enable_if<!is_power_of_two_policy<U>::value>::type* = nullptr>
    size_type next_bucket(size_type ibucket, size_type iprobe) const {
        (void) iprobe;
        if(Probing == tsl::sh::probing::linear) {
            ibucket++;
            return (ibucket != bucket_count())?ibucket:0;
        }
        else {
            tsl_assert(Probing == tsl::sh::probing::quadratic);
            ibucket += iprobe;
            return (ibucket < bucket_count())?ibucket:ibucket % bucket_count();
        }
    }
    
    
    
    template<class K, class... Args>
    std::pair<iterator, bool> insert_impl(const K& key, Args&&... value_type_args) {
        if(size() >= m_load_threshold) {
            rehash_impl(GrowthPolicy::next_bucket_count());
        }
        
        const std::size_t hash = hash_key(key);
        std::size_t ibucket = bucket_for_hash(hash);
        
        std::size_t probe = 0;
        while(true) {
            const std::size_t sparse_ibucket = ibucket >> sparse_array::SHIFT;
            const auto index_in_sparse_bucket = 
                        static_cast<typename sparse_array::size_type>(ibucket & sparse_array::MASK);
        
            if(m_sparse_buckets[sparse_ibucket].has_value_at_index(index_in_sparse_bucket)) {
                auto value_it = m_sparse_buckets[sparse_ibucket].value_at_index(index_in_sparse_bucket);
                if(compare_keys(key, KeySelect()(*value_it))) {
                    return std::make_pair(iterator(m_sparse_buckets.begin() + sparse_ibucket, value_it), false);
                }
            }
            else {
                auto value_it = m_sparse_buckets[sparse_ibucket].set_value_at_index(*this, index_in_sparse_bucket, 
                                                                                std::forward<Args>(value_type_args)...);
                m_nb_elements++;
                
                return std::make_pair(iterator(m_sparse_buckets.begin() + sparse_ibucket, value_it), true);
            }
            
            probe++;
            ibucket = next_bucket(ibucket, probe);
        }
    }
    
    
    
    template<class K>
    size_type erase_impl(const K& key, std::size_t hash) {
        std::size_t ibucket = bucket_for_hash(hash);
        
        std::size_t probe = 0;
        while(true) {
            const std::size_t sparse_ibucket = ibucket >> sparse_array::SHIFT;
            const auto index_in_sparse_bucket = 
                        static_cast<typename sparse_array::size_type>(ibucket & sparse_array::MASK);
        
            if(m_sparse_buckets[sparse_ibucket].has_value_at_index(index_in_sparse_bucket)) {
                auto value_it = m_sparse_buckets[sparse_ibucket].value_at_index(index_in_sparse_bucket);
                if(compare_keys(key, KeySelect()(*value_it))) {
                    m_sparse_buckets[sparse_ibucket].erase_value_at_position(*this, value_it, index_in_sparse_bucket);
                    m_nb_elements--;
                    return 1;
                }
            }
            else if(!m_sparse_buckets[sparse_ibucket].has_deleted_value_at_index(index_in_sparse_bucket)) {
                return 0;
            }
            
            probe++;
            ibucket = next_bucket(ibucket, probe);
        }
    }
    
    
    
    template<class K>
    iterator find_impl(const K& key, std::size_t hash) {
        return mutable_iterator(static_cast<const sparse_hash*>(this)->find(key, hash));        
    }
    
    template<class K>
    const_iterator find_impl(const K& key, std::size_t hash) const {
        std::size_t ibucket = bucket_for_hash(hash);
        
        std::size_t probe = 0;
        while(true) {
            const std::size_t sparse_ibucket = ibucket >> sparse_array::SHIFT;
            const auto index_in_sparse_bucket = 
                        static_cast<typename sparse_array::size_type>(ibucket & sparse_array::MASK);
        
            if(m_sparse_buckets[sparse_ibucket].has_value_at_index(index_in_sparse_bucket)) {
                auto value_it = m_sparse_buckets[sparse_ibucket].value_at_index(index_in_sparse_bucket);
                if(compare_keys(key, KeySelect()(*value_it))) {
                    return const_iterator(m_sparse_buckets.cbegin() + sparse_ibucket, value_it);
                }
            }
            else if(!m_sparse_buckets[sparse_ibucket].has_deleted_value_at_index(index_in_sparse_bucket)) {
                return cend();
            }
            
            probe++;
            ibucket = next_bucket(ibucket, probe);
        }
    }
    

    template<tsl::sh::exception_safety U = ExceptionSafety, 
             typename std::enable_if<U == tsl::sh::exception_safety::basic>::type* = nullptr>
    void rehash_impl(size_type count) {
        sparse_hash new_table(count, static_cast<Hash&>(*this), static_cast<KeyEqual&>(*this), 
                              static_cast<Allocator&>(*this), m_max_load_factor);
            
        for(auto& bucket: m_sparse_buckets) {
            for(auto& val: bucket) {
                new_table.insert(std::move(val));
            }
            
            bucket.clear(*this);
        }
        
        new_table.swap(*this);
    }

    /**
     * TODO: For now we copy each element into the new map. We could move
     * them if they are nothrow_move_constructible without triggering
     * any exception if we reserve enough space in the sparse arrays beforehand.
     */
    template<tsl::sh::exception_safety U = ExceptionSafety, 
             typename std::enable_if<U == tsl::sh::exception_safety::strong>::type* = nullptr>
    void rehash_impl(size_type count) {
        sparse_hash new_table(count, static_cast<Hash&>(*this), static_cast<KeyEqual&>(*this), 
                              static_cast<Allocator&>(*this), m_max_load_factor);
            
        for(auto& bucket: m_sparse_buckets) {
            for(auto& val: bucket) {
                new_table.insert(val);
            }
        }
        
        new_table.swap(*this);
    }
    
private:
    static const size_type MIN_BUCKETS_SIZE = sparse_array::BITMAP_NB_BITS;
    
public:    
    static const size_type DEFAULT_INIT_BUCKETS_SIZE = MIN_BUCKETS_SIZE;
    static constexpr float DEFAULT_MAX_LOAD_FACTOR = 0.5f;
    
private:
    sparse_buckets_container m_sparse_buckets;
    size_type m_bucket_count;
    size_type m_nb_elements;
    
    size_type m_load_threshold;
    float m_max_load_factor;
};

}
}

#endif