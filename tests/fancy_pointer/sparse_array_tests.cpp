//
// Created by lukas on 22.04.21.
//

#include <boost/test/unit_test.hpp>
#include <tsl/sparse_hash.h>
#include "CustomAllocator.h"

// Globals
constexpr auto MAX_INDEX = 32; //BITMAP_NB_BITS

/* templated tests
 * T is a struct with T::Array and T::Allocator.
*/
template <typename T>
void compilation() {
    typename T::Array test;
}

template <typename T>
void construction() {
    typename T::Allocator a;
    typename T::Array test(MAX_INDEX, a);
    test.clear(a); //needed because destructor asserts
}

namespace details {
    template<typename T>
    auto generate_test_array(typename T::Allocator &a) {
        typename T::Array arr(MAX_INDEX, a);
        for (std::size_t i = 0; i < MAX_INDEX; ++i) {
            arr.set(a, i, i);
        }
        return arr;
    }

    template<typename T>
    auto generate_check_for_test_array() {
        std::vector<typename T::Allocator::value_type> check(MAX_INDEX);
        for (std::size_t i = 0; i < MAX_INDEX; ++i) {
            check[i] = i;
        }
        return check;
    }
}

template <typename T>
void set() {
    typename T::Allocator a;
    auto test = details::generate_test_array<T>(a);
    auto check = details::generate_check_for_test_array<T>();
    BOOST_TEST_REQUIRE(std::equal(test.begin(), test.end(), check.begin()),
                       "'set' did not create the correct order of items");
    test.clear(a); //needed because destructor asserts
}

template <typename T>
void copy_construction() {
    typename T::Allocator a;
    //needs to be its own line, otherwise the move-construction would take place
    auto test = details::generate_test_array<T>(a);
    typename T::Array copy(test, a);
    auto check = details::generate_check_for_test_array<T>();
    BOOST_TEST_REQUIRE(std::equal(copy.begin(), copy.end(), check.begin()),
                       "'copy' changed the order of the items");
    test.clear(a);
    copy.clear(a);
}

template <typename T>
void move_construction() {
    typename T::Allocator a;
    auto moved_to(details::generate_test_array<T>(a));
    auto check = details::generate_check_for_test_array<T>();
    BOOST_TEST_REQUIRE(std::equal(moved_to.begin(), moved_to.end(), check.begin()),
                       "'move' changed the order of the items");
    check.clear(a);
}



// Template test structs
template <typename T, tsl::sh::sparsity Sparsity = tsl::sh::sparsity::medium>
struct STD {
    using Allocator = std::allocator<T>;
    using Array = tsl::detail_sparse_hash::sparse_array<T, std::allocator<T>, Sparsity>;
};

template<typename T, tsl::sh::sparsity Sparsity = tsl::sh::sparsity::medium>
struct CUSTOM {
    using Allocator = OffsetAllocator<T>;
    using Array = tsl::detail_sparse_hash::sparse_array<T, OffsetAllocator<T>, Sparsity>;
};


/* Tests.
 * I don't use the boost template test cases because with this I can set the title of every test case.
 */
BOOST_AUTO_TEST_SUITE(sparse_array_tests)

BOOST_AUTO_TEST_CASE(std_alloc_compile) {compilation<STD<int>>();}
BOOST_AUTO_TEST_CASE(std_alloc_construction) {construction<STD<int>>();}
BOOST_AUTO_TEST_CASE(std_alloc_set) {set<STD<int>>();}
BOOST_AUTO_TEST_CASE(std_alloc_copy_construction) {copy_construction<STD<int>>();}
BOOST_AUTO_TEST_CASE(std_alloc_move_construction) {copy_construction<STD<int>>();}

BOOST_AUTO_TEST_CASE(custom_alloc_compile) {compilation<CUSTOM<int>>();}
BOOST_AUTO_TEST_CASE(custom_alloc_construction) {construction<CUSTOM<int>>();}
BOOST_AUTO_TEST_CASE(custom_alloc_set) {set<CUSTOM<int>>();}
BOOST_AUTO_TEST_CASE(custom_alloc_copy_construction) {copy_construction<CUSTOM<int>>();}
BOOST_AUTO_TEST_CASE(custom_alloc_move_construction) {copy_construction<CUSTOM<int>>();}

BOOST_AUTO_TEST_SUITE_END()