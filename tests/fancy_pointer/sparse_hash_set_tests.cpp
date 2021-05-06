//
// Created by lukas on 22.04.21.
//
#include <boost/test/unit_test.hpp>
#include <tsl/sparse_hash.h>
#include "CustomAllocator.h"

namespace details {
    template<typename Key>
    struct KeySelect {
        using key_type = Key;
        const key_type &operator()(Key const &key) const noexcept { return key; }
        key_type &operator()(Key &key) noexcept { return key; }
    };

    template<typename T, typename Alloc>
    using sparse_set = tsl::detail_sparse_hash::sparse_hash<
            T, KeySelect<T>, void, std::hash<T>, std::equal_to<T>, Alloc,
            tsl::sh::power_of_two_growth_policy<2>,
            tsl::sh::exception_safety::basic,
            tsl::sh::sparsity::medium,
            tsl::sh::probing::quadratic>;

    template<typename T>
    auto default_construct_set() {
        using Type = typename T::value_type;
        return typename T::Set(T::Set::DEFAULT_INIT_BUCKET_COUNT, std::hash<Type>(), std::equal_to<Type>(),
                               typename T::Allocator(), T::Set::DEFAULT_MAX_LOAD_FACTOR);
    }

    /**
     *  checks if all values of the set are in the initializer_list and than if the lengths are equal.
     *  So basically Set \subset l and |Set| == |l|.
     *  Needs 'set.contains(.)' to work correctly.
     */
    template <typename Set>
    bool is_equal(Set const& set, std::initializer_list<typename Set::value_type> l) {
        return std::all_of(l.begin(), l.end(), [&set](auto i){return set.contains(i);})
            and set.size() == l.size();
    }
}

template<typename T>
void construction() {
    auto set = details::default_construct_set<T>();
}

template <typename T>
void insert(std::initializer_list<typename T::value_type> l) {
    auto set = details::default_construct_set<T>();
    for (auto const& i: l)  set.insert(i);
    BOOST_TEST_REQUIRE(details::is_equal(set, l), "'insert' did not create exactly the values needed");
}

template <typename T>
void iterator_insert(std::initializer_list<typename T::value_type> l) {
   auto set = details::default_construct_set<T>();
   set.insert(l.begin(), l.end());
   BOOST_TEST_REQUIRE(details::is_equal(set, l), "'insert' with iterators did not create exactly the values needed");
}

template <typename T>
void iterator_access(typename T::value_type single_value) {
    auto set = details::default_construct_set<T>();
    set.insert(single_value);
    BOOST_TEST_REQUIRE(*(set.begin()) == single_value, "iterator cannot access single value");
}

template <typename T>
void iterator_access_multi(std::initializer_list<typename T::value_type> l) {
    auto set = details::default_construct_set<T>();
    set.insert(l.begin(), l.end());
    std::vector<typename T::value_type> l_sorted = l;
    std::vector<typename T::value_type> set_sorted(set.begin(), set.end());
    std::sort(l_sorted.begin(), l_sorted.end());
    std::sort(set_sorted.begin(), set_sorted.end());
    BOOST_TEST_REQUIRE(std::equal(l_sorted.begin(), l_sorted.end(),
                                  set_sorted.begin(), set_sorted.end()),
                       "iterating over the set didn't work");
}


template<typename T>
struct STD {
    using value_type = T;
    using Allocator = std::allocator<value_type>;
    using Set = details::sparse_set<value_type, Allocator>;
};

template<typename T>
struct CUSTOM {
    using value_type = T;
    using Allocator = OffsetAllocator<value_type>;
    using Set = details::sparse_set<value_type, Allocator>;
};


BOOST_AUTO_TEST_SUITE(sparse_hash_set_tests)

BOOST_AUTO_TEST_CASE(std_alloc_compiles) {construction<STD<int>>();}
BOOST_AUTO_TEST_CASE(std_alloc_insert) {insert<STD<int>>({1,2,3,4});}
BOOST_AUTO_TEST_CASE(std_alloc_iterator_insert) {iterator_insert<STD<int>>({1,2,3,4});}
BOOST_AUTO_TEST_CASE(std_alloc_iterator_access) {iterator_access<STD<int>>(42);}
BOOST_AUTO_TEST_CASE(std_alloc_iterator_access_multi) {iterator_access_multi<STD<int>>({1,2,3,4});}

BOOST_AUTO_TEST_CASE(custom_alloc_compiles) {construction<CUSTOM<int>>();}
BOOST_AUTO_TEST_CASE(custom_alloc_insert) {insert<CUSTOM<int>>({1,2,3,4});}
BOOST_AUTO_TEST_CASE(custom_alloc_iterator_insert) {iterator_insert<CUSTOM<int>>({1,2,3,4});}
BOOST_AUTO_TEST_CASE(custom_alloc_iterator_access) {iterator_access<CUSTOM<int>>(42);}
BOOST_AUTO_TEST_CASE(custom_alloc_iterator_access_multi) {iterator_access_multi<CUSTOM<int>>({1,2,3,4});}

BOOST_AUTO_TEST_SUITE_END()

#include <tsl/sparse_set.h>

BOOST_AUTO_TEST_SUITE(test_full_set)

BOOST_AUTO_TEST_CASE(full_set) {
   tsl::sparse_set<int, std::hash<int>, std::equal_to<int>, OffsetAllocator<int>> set;
   std::vector<int> data = {1,2,3,4,5,6,7,8,9};
   set.insert(data.begin(), data.end());
   std::cout << std::boolalpha;
   for (auto d : data) {
       std::cout << d << " in set: " << set.contains(d) << std::endl;
   }

}
BOOST_AUTO_TEST_SUITE_END()