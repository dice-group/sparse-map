#ifndef TSL_SPARSE_MAP_TESTS_BOOST_OFFSET_POINTER_H
#define TSL_SPARSE_MAP_TESTS_BOOST_OFFSET_POINTER_H

#include "Dice/sparse-map/sparse_hash.h" //needed, so the basic template is already included
#include <boost/interprocess/offset_ptr.hpp>

namespace Dice::sparse_map {
/* Template specialisation for a "const_cast" of a boost offset_ptr.
 * @tparam PT PointedType
 * @tparam DT DifferenceType
 * @tparam OT OffsetType
 * @tparam OA OffsetAlignment
 */
template <typename PT, typename DT, typename OT, std::size_t OA>
struct Remove_Const<boost::interprocess::offset_ptr<PT, DT, OT, OA>> {
  template <typename T>
  static boost::interprocess::offset_ptr<PT, DT, OT, OA>
  remove(T const &const_iter) {
    return boost::interprocess::const_pointer_cast<PT, DT, OT, OA>(const_iter);
  }
};
} // namespace Dice

#endif // TSL_SPARSE_MAP_TESTS_BOOST_OFFSET_POINTER_H
