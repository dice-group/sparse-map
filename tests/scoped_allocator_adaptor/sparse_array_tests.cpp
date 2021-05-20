#include <boost/test/unit_test.hpp>
#include <iostream>
#include <memory>
#include <scoped_allocator>
#include <tsl/sparse_set.h>
#include <unordered_set>
#include <type_traits>

// Globals
constexpr auto MAX_INDEX = 32; // BITMAP_NB_BITS

template <typename T> void compilation() { typename T::Array test; }

template <typename T> void construction() {
  typename T::Allocator a;
  typename T::Array test(MAX_INDEX, a);
  test.clear(a);
}

template <typename T>
void set(std::initializer_list<typename T::value_type> l) {
  typename T::Allocator a;
  typename T::Array array(MAX_INDEX, a);
  std::vector<typename T::value_type> check;
  check.reserve(l.size());
  std::size_t counter = 0;
  for (auto const &value : l) {
    array.set(a, counter++, value);
    check.emplace_back(value);
  }
  BOOST_TEST_REQUIRE(std::equal(array.begin(), array.end(), check.begin()),
                     "'set' did not create the correct order of items");
  array.clear(a);
}

template <typename T> void uses_allocator() {
  BOOST_TEST_REQUIRE((std::uses_allocator<typename T::Array, typename T::Allocator>::value),
                     "uses_allocator returns false");
}

template <typename T, typename ...Args> void trailing_allocator_convention(Args ...) {
  using Alloc = typename T::Allocator;
  BOOST_TEST_REQUIRE((std::is_constructible<typename T::Array, Args..., const Alloc&>::value),
                     "trailing_allocator thinks construction is not possible");
}

template <typename T> void is_constructable_test() {
  auto alloc = typename std::allocator_traits<typename T::Allocator>::rebind_alloc<
      typename T::Array>();
  auto const& alloc_ref = alloc;
  std::vector<typename T::Array, typename T::Allocator> vector(alloc_ref);
}

template <typename T, tsl::sh::sparsity Sparsity = tsl::sh::sparsity::medium>
struct NORMAL {
  using value_type = T;
  using Allocator = std::allocator<T>;
  using Array = tsl::detail_sparse_hash::sparse_array<T, Allocator, Sparsity>;
};

template <typename T, tsl::sh::sparsity Sparsity = tsl::sh::sparsity::medium>
struct SCOPED {
  using value_type = T;
  using Allocator = std::scoped_allocator_adaptor<std::allocator<T>>;
  using Array = tsl::detail_sparse_hash::sparse_array<T, Allocator, Sparsity>;
};

BOOST_AUTO_TEST_SUITE(scoped_allocators)
BOOST_AUTO_TEST_SUITE(sparse_array_tests)

BOOST_AUTO_TEST_CASE(normal_compilation) { compilation<NORMAL<int>>(); }
BOOST_AUTO_TEST_CASE(normal_construction) { construction<NORMAL<int>>(); }
BOOST_AUTO_TEST_CASE(normal_set) { set<NORMAL<int>>({0, 1, 2, 3, 4}); }
BOOST_AUTO_TEST_CASE(normal_uses_allocator) { uses_allocator<NORMAL<int>>(); }
BOOST_AUTO_TEST_CASE(normal_leading_allocator_convention) {trailing_allocator_convention<NORMAL<int>>(0); }
BOOST_AUTO_TEST_CASE(normal_is_constructable_test) {is_constructable_test<NORMAL<int>>(); }

BOOST_AUTO_TEST_CASE(scoped_compilation) { compilation<SCOPED<int>>(); }
BOOST_AUTO_TEST_CASE(scoped_construction) { construction<SCOPED<int>>(); }
BOOST_AUTO_TEST_CASE(scoped_set) { set<SCOPED<int>>({0, 1, 2, 3, 4}); }
BOOST_AUTO_TEST_CASE(scoped_uses_allocator) { uses_allocator<SCOPED<int>>(); }
BOOST_AUTO_TEST_CASE(scoped_leading_allocator_convention) {trailing_allocator_convention<SCOPED<int>>(0); }
BOOST_AUTO_TEST_CASE(scoped_is_constructable_test) {is_constructable_test<SCOPED<int>>(); }

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
