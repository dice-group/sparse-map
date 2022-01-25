#ifndef TSL_SPARSE_MAP_TESTS_TEMPLATE_PARAMETERS_HPP
#define TSL_SPARSE_MAP_TESTS_TEMPLATE_PARAMETERS_HPP

namespace tsl::sh {

        enum class probing {
            linear, quadratic
        };

        enum class exception_safety {
            basic, strong
        };

        enum class sparsity {
            high, medium, low
        };
    }

#endif //TSL_SPARSE_MAP_TESTS_TEMPLATE_PARAMETERS_HPP
