/***************************************************************************
 * s390x Z Vector (zvector) register definition for xsimd
 * Requires: GCC/Clang with -mzvector -march=z13 or newer
 ***************************************************************************/

#ifndef XSIMD_ZVECTOR_REGISTER_HPP
#define XSIMD_ZVECTOR_REGISTER_HPP

#include "xsimd_register.hpp"

#if defined(__VEC__) && defined(__s390x__)

#include <vecintrin.h>

namespace xsimd
{
    /**
     * @ingroup architectures
     *
     * s390x Z Vector (IBM SIMD) - 128-bit vector registers
     * Supported on z13 and later (z14, z15, z16, z17, ...)
     */
    struct zvector
    {
        static constexpr bool supported() noexcept { return true; }
        static constexpr bool available() noexcept { return true; }
        static constexpr std::size_t alignment() noexcept { return 16; }
        static constexpr bool requires_alignment() noexcept { return true; }
        static constexpr char const* name() noexcept { return "zvector"; }
    };

    namespace types
    {
        // int8_t / uint8_t  — 16 lanes
        XSIMD_DECLARE_SIMD_REGISTER(int8_t,   zvector, __vector signed char);
        XSIMD_DECLARE_SIMD_REGISTER(uint8_t,  zvector, __vector unsigned char);

        // int16_t / uint16_t — 8 lanes
        XSIMD_DECLARE_SIMD_REGISTER(int16_t,  zvector, __vector signed short);
        XSIMD_DECLARE_SIMD_REGISTER(uint16_t, zvector, __vector unsigned short);

        // int32_t / uint32_t — 4 lanes
        XSIMD_DECLARE_SIMD_REGISTER(int32_t,  zvector, __vector signed int);
        XSIMD_DECLARE_SIMD_REGISTER(uint32_t, zvector, __vector unsigned int);

        // int64_t / uint64_t — 2 lanes
        XSIMD_DECLARE_SIMD_REGISTER(int64_t,  zvector, __vector signed long long);
        XSIMD_DECLARE_SIMD_REGISTER(uint64_t, zvector, __vector unsigned long long);

        // float — 4 lanes
        XSIMD_DECLARE_SIMD_REGISTER(float,    zvector, __vector float);

        // double — 2 lanes
        XSIMD_DECLARE_SIMD_REGISTER(double,   zvector, __vector double);

        // bool registers (same underlying type as their value counterparts)
        XSIMD_DECLARE_SIMD_REGISTER(bool,     zvector, __vector unsigned char);
    } // namespace types

} // namespace xsimd

#else
// zvector not available — zvector::supported() returns false
namespace xsimd
{
    struct zvector
    {
        static constexpr bool supported() noexcept { return false; }
        static constexpr bool available() noexcept { return false; }
        static constexpr std::size_t alignment() noexcept { return 16; }
        static constexpr bool requires_alignment() noexcept { return false; }
        static constexpr char const* name() noexcept { return "zvector(unavailable)"; }
    };
} // namespace xsimd

#endif // __VEC__ && __s390x__

#endif // XSIMD_ZVECTOR_REGISTER_HPP
