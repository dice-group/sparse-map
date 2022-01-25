#ifndef TSL_SPARSE_MAP_TESTS_SPARSE_HASH2_HPP
#define TSL_SPARSE_MAP_TESTS_SPARSE_HASH2_HPP

#include <algorithm>
#include <cassert>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <boost/container/vector.hpp>
#include <boost/interprocess/offset_ptr.hpp>

#include "sparse_growth_policy.h"
#include "template_parameters.hpp"
#include "detail_popcount.hpp"
#include "const_cast.hpp"
#include "utility_functions.hpp"
#include "sparse_array2.hpp"


namespace tsl::detail_sparse_hash {

/**
 * Internal common class used by `sparse_map` and `sparse_set`.
 *
 * `ValueType` is what will be stored by `sparse_hash` (usually `std::pair<Key,
 * T>` for map and `Key` for set).
 *
 * `KeySelect` should be a `FunctionObject` which takes a `ValueType` in
 * parameter and returns a reference to the key.
 *
 * `ValueSelect` should be a `FunctionObject` which takes a `ValueType` in
 * parameter and returns a reference to the value. `ValueSelect` should be void
 * if there is no value (in a set for example).
 *
 * The strong exception guarantee only holds if `ExceptionSafety` is set to
 * `tsl::sh::exception_safety::strong`.
 *
 * `ValueType` must be nothrow move constructible and/or copy constructible.
 * Behaviour is undefined if the destructor of `ValueType` throws.
 *
 *
 * The class holds its buckets in a 2-dimensional fashion. Instead of having a
 * linear `std::vector<bucket>` for [0, bucket_count) where each bucket stores
 * one value, we have a `std::vector<sparse_array>` (m_sparse_buckets_data)
 * where each `sparse_array` stores multiple values (up to
 * `sparse_array::BITMAP_NB_BITS`). To convert a one dimensional `ibucket`
 * position to a position in `std::vector<sparse_array>` and a position in
 * `sparse_array`, use respectively the methods
 * `sparse_array::sparse_ibucket(ibucket)` and
 * `sparse_array::index_in_sparse_bucket(ibucket)`.
 */
    template<class ValueType,
            class KeySelect,
            class ValueSelect,
            class Hash,
            class KeyEqual,
            class Allocator,
            class GrowthPolicy,
            tsl::sh::exception_safety ExceptionSafety,
            tsl::sh::sparsity Sparsity,
            tsl::sh::probing Probing>
    class sparse_hash2 {
        Allocator allocator;
        static_assert(sizeof(Hash) == 0);
        static_assert(sizeof(KeyEqual) == 0);
        static_assert(sizeof(GrowthPolicy) == 0);

        static constexpr bool has_mapped_type = std::is_same_v<ValueType, void>;


        static_assert(
                noexcept(std::declval<GrowthPolicy>().bucket_for_hash(std::size_t(0))),
                "GrowthPolicy::bucket_for_hash must be noexcept.");
        static_assert(noexcept(std::declval<GrowthPolicy>().clear()),
                      "GrowthPolicy::clear must be noexcept.");

    public:
        template<bool IsConst>
        class sparse_iterator;

        using key_type = typename KeySelect::key_type;
        using value_type = ValueType;
        using hasher = Hash;
        using key_equal = KeyEqual;
        using allocator_type = Allocator;
        using reference = value_type &;
        using const_reference = const value_type &;
        using size_type = typename std::allocator_traits<allocator_type>::size_type;
        using pointer = typename std::allocator_traits<allocator_type>::pointer;
        using const_pointer = typename std::allocator_traits<allocator_type>::const_pointer;
        using difference_type = typename std::allocator_traits<allocator_type>::difference_type;
        using iterator = sparse_iterator<false>;
        using const_iterator = sparse_iterator<true>;

    private:
        using sparse_array =
        tsl::detail_sparse_hash::sparse_array<ValueType, Allocator, Sparsity>;

        using sparse_buckets_allocator = typename std::allocator_traits<
                allocator_type>::template rebind_alloc<sparse_array>;
        using sparse_buckets_container =
        boost::container::vector<sparse_array, sparse_buckets_allocator>;

    public:
        static constexpr size_type DEFAULT_INIT_BUCKET_COUNT = 0;
        static constexpr float DEFAULT_MAX_LOAD_FACTOR = 0.5f;

        /**
         * Return an always valid pointer to an static empty bucket_entry with
         * last_bucket() == true.
         */
        // TODO:: that might be problematic as well
        sparse_array *static_empty_sparse_bucket_ptr() {
            static sparse_array empty_sparse_bucket(true);
            return &empty_sparse_bucket;
        }

    private:
        sparse_buckets_container m_sparse_buckets_data;

        /**
         * Points to m_sparse_buckets_data.data() if !m_sparse_buckets_data.empty()
         * otherwise points to static_empty_sparse_bucket_ptr. This variable is useful
         * to avoid the cost of checking if m_sparse_buckets_data is empty when trying
         * to find an element.
         *
         * TODO Remove m_sparse_buckets_data and only use a pointer instead of a
         * pointer+vector to save some space in the sparse_hash object.
         */
        boost::interprocess::offset_ptr<sparse_array> m_sparse_buckets;

        size_type m_bucket_count;
        size_type m_nb_elements;
        size_type m_nb_deleted_buckets;

        /**
         * Maximum that m_nb_elements can reach before a rehash occurs automatically
         * to grow the hash table.
         */
        size_type m_load_threshold_rehash;

        /**
         * Maximum that m_nb_elements + m_nb_deleted_buckets can reach before cleaning
         * up the buckets marked as deleted.
         */
        size_type m_load_threshold_clear_deleted;
        float m_max_load_factor;

    };

}
#endif //TSL_SPARSE_MAP_TESTS_SPARSE_HASH2_HPP
