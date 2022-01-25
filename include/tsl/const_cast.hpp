#ifndef TSL_SPARSE_MAP_TESTS_CONST_CAST_HPP
#define TSL_SPARSE_MAP_TESTS_CONST_CAST_HPP

namespace tsl {
    /* Replacement for const_cast in sparse_array.
 * Can be overloaded for specific fancy pointers.
 * This is just a workaround.
 * The clean way would be to change the implementation to stop using const_cast.
 */
    template<typename T>
    struct Remove_Const {
        template<typename V>
        static T remove(V iter) {
            return const_cast<T>(iter);
        }
    };
}

#endif //TSL_SPARSE_MAP_TESTS_CONST_CAST_HPP
