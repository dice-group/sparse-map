#ifndef TSL_SPARSE_MAP_TESTS_SPARSE_ARRAY_HPP
#define TSL_SPARSE_MAP_TESTS_SPARSE_ARRAY_HPP

#include <cinttypes>
#include <memory>
#include <numeric>

#include "utility_functions.hpp"
#include "sparse_growth_policy.h"

namespace tsl::detail_sparse_hash {
    template<typename T, typename Allocator, tsl::sh::sparsity Sparsity>
    class sparse_array {
    public:
        using value_type = T;
        using size_type = std::uint_least8_t;
        using allocator_type = Allocator;
        using allocator_traits = std::allocator_traits<allocator_type>;
        using pointer = typename allocator_traits::pointer;
        using const_pointer = typename allocator_traits::const_pointer;
        using iterator = pointer;
        using const_iterator = const_pointer;

    private:
        static constexpr size_type CAPACITY_GROWTH_STEP =
                (Sparsity == tsl::sh::sparsity::high)
                ? 2
                : (Sparsity == tsl::sh::sparsity::medium)
                  ? 4
                  : 8;  // (Sparsity == tsl::sh::sparsity::low)

        using bitmap_type = std::uint_least64_t;
        static constexpr std::size_t BITMAP_NB_BITS = 64;
        static constexpr std::size_t BUCKET_SHIFT = 6;

        static constexpr std::size_t BUCKET_MASK = BITMAP_NB_BITS - 1;

        static_assert(is_power_of_two(BITMAP_NB_BITS),
                      "BITMAP_NB_BITS must be a power of two.");
        static_assert(std::numeric_limits<bitmap_type>::digits >= BITMAP_NB_BITS,
                      "bitmap_type must be able to hold at least BITMAP_NB_BITS.");
        static_assert((std::size_t(1) << BUCKET_SHIFT) == BITMAP_NB_BITS,
                      "(1 << BUCKET_SHIFT) must be equal to BITMAP_NB_BITS.");
        static_assert(std::numeric_limits<size_type>::max() >= BITMAP_NB_BITS,
                      "size_type must be big enough to hold BITMAP_NB_BITS.");
        static_assert(std::is_unsigned<bitmap_type>::value,
                      "bitmap_type must be unsigned.");
        static_assert((std::numeric_limits<bitmap_type>::max() & BUCKET_MASK) ==
                      BITMAP_NB_BITS - 1);

    public:
        /**
         * Map an ibucket [0, bucket_count) in the hash table to a sparse_ibucket
         * (a sparse_array holds multiple buckets, so there is less sparse_array than
         * bucket_count).
         *
         * The bucket ibucket is in
         * m_sparse_buckets[sparse_ibucket(ibucket)][index_in_sparse_bucket(ibucket)]
         * instead of something like m_buckets[ibucket] in a classical hash table.
         */
        static constexpr std::size_t sparse_ibucket(std::size_t ibucket) {
            return ibucket >> BUCKET_SHIFT;
        }

        /**
         * Map an ibucket [0, bucket_count) in the hash table to an index in the
         * sparse_array which corresponds to the bucket.
         *
         * The bucket ibucket is in
         * m_sparse_buckets[sparse_ibucket(ibucket)][index_in_sparse_bucket(ibucket)]
         * instead of something like m_buckets[ibucket] in a classical hash table.
         */
        static constexpr typename sparse_array::size_type index_in_sparse_bucket(
                std::size_t ibucket) {
            return static_cast<typename sparse_array::size_type>(
                    ibucket & sparse_array::BUCKET_MASK);
        }

        static constexpr std::size_t nb_sparse_buckets(std::size_t bucket_count) noexcept {
            if (bucket_count == 0) {
                return 0;
            }

            return std::max<std::size_t>(
                    1, sparse_ibucket(tsl::detail_sparse_hash::round_up_to_power_of_two(
                            bucket_count)));
        }

    private:
        /**
         * Pointer to the beginning of an array values. Length: capacity_
         */
        pointer values_ptr_;

        /**
         * One-hot bitmap that maps indices [0 ... (size_ - 1)] to offsets in values_ptr_.
         */
        bitmap_type bitmap_vals_{};
        /**
         * One-hot bitmap that marks entries that were deleted.
         * // TODO: what do we need that for?
         */
        bitmap_type bitmap_deleted_vals_{};
        /**
         * Number of values currently stored.
         */
        size_type size_{};
        /**
         * Maximum size_ before reallocation becomes necessary
         */
        size_type capacity_: 7{};
        bool is_last_array_: 1{}; // TODO: make it tight

    public:
        /**
         * Constructors and assignment/move operators
         */
        sparse_array() = default;

        //needed for "is_constructible" with no parameters
        sparse_array(std::allocator_arg_t, Allocator const &) noexcept: sparse_array{} {}

        //const Allocator needed for MoveInsertable requirement
        sparse_array(size_type capacity, Allocator const &const_alloc) noexcept:
                capacity_(capacity) {
            if (capacity_ > 0) {
                auto alloc = const_alloc;
                values_ptr_ = alloc.allocate(capacity_);
                tsl_sh_assert(values_ptr_ != nullptr);  // allocate should throw if there is a failure
            }
        }

        //const Allocator needed for MoveInsertable requirement
        sparse_array(sparse_array const &other, Allocator const &const_alloc) noexcept
                : bitmap_vals_(other.bitmap_vals_),
                  bitmap_deleted_vals_(other.bitmap_deleted_vals_),
                  capacity_(other.capacity_),
                  is_last_array_(other.is_last_array_) {
            tsl_sh_assert(capacity_ >= size_);
            if (capacity_ == 0) return;

            auto alloc = const_alloc;
            values_ptr_ = alloc.allocate(capacity_);
            tsl_sh_assert(values_ptr_ != nullptr);  // allocate should throw if there is a failure
            size_type i;
            // TODO: should we really guarantee exception-safety?
            try {
                while (size_ < other.size_)
                    construct_value(alloc, values_ptr_ + (size_++), other.values_ptr_[i]);
            } catch (...) {
                clear(alloc);
                throw;
            }
        }

        sparse_array(sparse_array &&other) noexcept
                : values_ptr_(other.values_ptr_),
                  bitmap_vals_(other.bitmap_vals_),
                  bitmap_deleted_vals_(other.bitmap_deleted_vals_),
                  size_(other.size_),
                  capacity_(other.capacity_),
                  is_last_array_(other.is_last_array_) {
            other.values_ptr_ = nullptr;
            other.bitmap_vals_ = 0;
            other.bitmap_deleted_vals_ = 0;
            other.size_ = 0;
            other.capacity_ = 0;
        }

        //const Allocator needed for MoveInsertable requirement
        sparse_array(sparse_array &&other, Allocator const &const_alloc)
                : values_ptr_(nullptr),
                  bitmap_vals_(other.bitmap_vals_),
                  bitmap_deleted_vals_(other.bitmap_deleted_vals_),
                  size_(0),
                  capacity_(other.capacity_),
                  is_last_array_(other.is_last_array_) {
            // TODO: review
            // TODO: is this only applied if the allocator is different from this' allocator?
            tsl_sh_assert(other.capacity_ >= other.size_);
            if (capacity_ == 0) {
                return;
            }

            auto alloc = const_cast<Allocator &>(const_alloc);
            values_ptr_ = alloc.allocate(capacity_);
            tsl_sh_assert(values_ptr_ !=
                          nullptr);  // allocate should throw if there is a failure
            try {
                for (size_type i = 0; i < other.size_; i++) {
                    construct_value(alloc, values_ptr_ + i, std::move(other.values_ptr_[i]));
                    size_++;
                }
            } catch (...) {
                clear(alloc);
                throw;
            }
        }

        sparse_array &operator=(const sparse_array &) = delete;

        sparse_array &operator=(sparse_array &&other) noexcept {
            // TODO: review
            this->values_ptr_ = other.values_ptr_;
            this->bitmap_vals_ = other.bitmap_vals_;
            this->bitmap_deleted_vals_ = other.bitmap_deleted_vals_;
            this->size_ = other.size_;
            this->capacity_ = other.capacity_;
            other.values_ptr_ = nullptr;
            other.bitmap_vals_ = 0;
            other.bitmap_deleted_vals_ = 0;
            other.size_ = 0;
            other.capacity_ = 0;
            return *this;
        }


        ~sparse_array() noexcept {
            // TODO: review
            // The code that manages the sparse_array must have called clear before
            // destruction. See documentation of sparse_array for more details.
            tsl_sh_assert(capacity_ == 0 && size_ == 0 && values_ptr_ == nullptr);
        }

        /**
         * Member functions
         */

        iterator begin() noexcept { return values_ptr_; }

        iterator end() noexcept { return values_ptr_ + size_; }

        const_iterator begin() const noexcept { return cbegin(); }

        const_iterator end() const noexcept { return cend(); }

        const_iterator cbegin() const noexcept { return values_ptr_; }

        const_iterator cend() const noexcept { return values_ptr_ + size_; }

        [[nodiscard]] bool empty() const noexcept { return size_ == 0; }

        [[nodiscard]] size_type size() const noexcept { return size_; }

        void clear(allocator_type &alloc) noexcept {
            destroy_and_deallocate_values(alloc);
            values_ptr_ = nullptr;
            bitmap_vals_ = 0;
            bitmap_deleted_vals_ = 0;
            size_ = 0;
            capacity_ = 0;
        }

        [[nodiscard]] bool last() const noexcept { return is_last_array_; }

        void set_as_last() noexcept { is_last_array_ = true; }

        [[nodiscard]] bool has_value(size_type index) const noexcept {
            tsl_sh_assert(index < BITMAP_NB_BITS);
            return (bitmap_vals_ & (bitmap_type(1) << index)) != 0;
        }

        [[nodiscard]] bool has_deleted_value(size_type index) const noexcept {
            tsl_sh_assert(index < BITMAP_NB_BITS);
            return (bitmap_deleted_vals_ & (bitmap_type(1) << index)) != 0;
        }

        iterator value(size_type index) noexcept {
            tsl_sh_assert(has_value(index));
            return values_ptr_ + index_to_offset(index);
        }

        const_iterator value(size_type index) const noexcept {
            tsl_sh_assert(has_value(index));
            return values_ptr_ + index_to_offset(index);
        }

        /**
         * Emplace a new value. There must be no value at index before.
         * @tparam Args
         * @param alloc
         * @param index
         * @param value_args
         * @return Iterator pointing to the newly inserted value.
         */
        template<typename... Args>
        iterator insert(allocator_type &alloc, size_type index, Args &&... value_args) {
            tsl_sh_assert(!has_value(index));

            const size_type offset = index_to_offset(index);
            emplace_at_offset(alloc, offset, std::forward<Args>(value_args)...);

            bitmap_vals_ = (bitmap_vals_ | (bitmap_type(1) << index));
            bitmap_deleted_vals_ =
                    (bitmap_deleted_vals_ & ~(bitmap_type(1) << index));

            size_++;

            tsl_sh_assert(has_value(index));
            tsl_sh_assert(!has_deleted_value(index));

            return values_ptr_ + offset;
        }

        iterator erase(allocator_type &alloc, iterator position) {
            const auto offset = std::distance(begin(), position);
            return erase(alloc, position, offset_to_index(offset));
        }

        /**
         * Erases a new value. There must be a value at index before.
         * @param alloc
         * @param position
         * @param index
         * @return
         */
        iterator erase(allocator_type &alloc, iterator position, size_type index) {
            tsl_sh_assert(has_value(index));
            tsl_sh_assert(!has_deleted_value(index));

            auto offset = std::distance(begin(), position);
            tsl_sh_assert(offset_to_index(offset) == index);
            erase_at_offset(alloc, offset);

            bitmap_vals_ = (bitmap_vals_ & ~(bitmap_type(1) << index));
            bitmap_deleted_vals_ = (bitmap_deleted_vals_ | (bitmap_type(1) << index));

            size_--;

            tsl_sh_assert(!has_value(index));
            tsl_sh_assert(has_deleted_value(index));

            return values_ptr_ + offset;
        }

        void swap(sparse_array &other) {
            using std::swap;

            swap(values_ptr_, other.values_ptr_);
            swap(bitmap_vals_, other.bitmap_vals_);
            swap(bitmap_deleted_vals_, other.bitmap_deleted_vals_);
            swap(size_, other.size_);
            swap(capacity_, other.capacity_);
            swap(is_last_array_, other.is_last_array_);
        }


    private:
        [[nodiscard]] size_type index_to_offset(size_type index) const noexcept {
            tsl_sh_assert(index < BITMAP_NB_BITS);
            return std::popcount(bitmap_vals_ &
                                 ((bitmap_type(1) << index) - bitmap_type(1)));
        }

        /**
          * Emplace a new value at the given offset.
          * @tparam Args
          * @param alloc
          * @param index
          * @param value_args
          * @return Iterator pointing to the newly inserted value.
          */
        template<typename... Args, typename U = value_type>
        requires std::is_nothrow_move_constructible_v<U>
        void emplace_at_offset(allocator_type &alloc, size_type offset,
                               Args &&... value_args) {
            // TODO: for now, we require values to be nothrow_move_constructible
            if (size_ < capacity_) {
                insert_at_offset_no_realloc(alloc, offset,
                                            std::forward<Args>(value_args)...);
            } else {
                insert_at_offset_realloc(alloc, offset, next_capacity(),
                                         std::forward<Args>(value_args)...);
            }
        }

        template<typename... Args, typename U = value_type>
        requires std::is_nothrow_move_constructible_v<U>
        void insert_at_offset_no_realloc(allocator_type &alloc, size_type offset,
                                         Args &&... value_args) {
            tsl_sh_assert(offset <= size_);
            tsl_sh_assert(size_ < capacity_);

            for (size_type i = size_; i > offset; i--) {
                construct_value(alloc, values_ptr_ + i, std::move(values_ptr_[i - 1]));
                destroy_value(alloc, values_ptr_ + i - 1);
            }

            try {
                construct_value(alloc, values_ptr_ + offset,
                                std::forward<Args>(value_args)...);
            } catch (...) {
                for (size_type i = offset; i < size_; i++) {
                    construct_value(alloc, values_ptr_ + i, std::move(values_ptr_[i + 1]));
                    destroy_value(alloc, values_ptr_ + i + 1);
                }
                throw;
            }
        }

        template<typename... Args, typename U = value_type>
        requires std::is_nothrow_move_constructible_v<U>
        void insert_at_offset_realloc(allocator_type &alloc, size_type offset,
                                      size_type new_capacity, Args &&... value_args) {
            tsl_sh_assert(new_capacity > size_);

            pointer new_value_ptr = alloc.allocate(new_capacity);
            // Allocate should throw if there is a failure
            tsl_sh_assert(new_value_ptr != nullptr);

            try {
                construct_value(alloc, new_value_ptr + offset,
                                std::forward<Args>(value_args)...);
            } catch (...) {
                alloc.deallocate(new_value_ptr, new_capacity);
                throw;
            }

            // Should not throw from here
            for (size_type i = 0; i < offset; i++)
                construct_value(alloc, new_value_ptr + i, std::move(values_ptr_[i]));

            for (size_type i = offset; i < size_; i++)
                construct_value(alloc, new_value_ptr + i + 1, std::move(values_ptr_[i]));

            destroy_and_deallocate_values(alloc);

            values_ptr_ = new_value_ptr;
            capacity_ = new_capacity;
        }

        /**
         * Erasure
         *
         * Two situations:
         * - Either we are in a situation where
         * std::is_nothrow_move_constructible<value_type>::value is true. Simply
         * destroy the value and left-shift move the value on the right of offset.
         * - Otherwise we are in a situation where
         * std::is_nothrow_move_constructible<value_type>::value is false. Copy all
         * the values except the one at offset into a new heap area. On success, we
         * set m_values to this new area. Even if slower, it's the only way to
         * preserve to strong exception guarantee.
         */
        template<typename... Args, typename U = value_type>
        requires std::is_nothrow_move_constructible_v<U>
        void erase_at_offset(allocator_type &alloc, size_type offset) noexcept {
            tsl_sh_assert(offset < size_);

            destroy_value(alloc, values_ptr_ + offset);

            for (size_type i = offset + 1; i < size_; i++) {
                construct_value(alloc, values_ptr_ + i - 1, std::move(values_ptr_[i]));
                destroy_value(alloc, values_ptr_ + i);
            }
        }

        [[nodiscard]] size_type next_capacity() const noexcept {
            return capacity_ + CAPACITY_GROWTH_STEP;
        }


        template<typename... Args>
        static void construct_value(allocator_type &alloc, pointer const &value_ptr,
                                    Args &&... value_args) {
            std::allocator_traits<allocator_type>::construct(
                    alloc, std::to_address(value_ptr), std::forward<Args>(value_args)...);
        }

        static void destroy_value(allocator_type &alloc, pointer const &value_ptr) noexcept {
            std::allocator_traits<allocator_type>::destroy(alloc, std::to_address(value_ptr));
        }

        void destroy_and_deallocate_values(allocator_type &alloc) noexcept {
            for (size_type i = 0; i < size_; i++)
                destroy_value(alloc, values_ptr_ + i);

            alloc.deallocate(values_ptr_, capacity_);
            capacity_ = 0;
        }
    };

}
#endif //TSL_SPARSE_MAP_TESTS_SPARSE_ARRAY_HPP
