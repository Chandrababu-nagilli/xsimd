/***************************************************************************
 * s390x Z Vector kernel implementation for xsimd
 * Covers: load/store, arithmetic, comparison, bitwise, shuffle, min/max
 ***************************************************************************/

#ifndef XSIMD_ZVECTOR_HPP
#define XSIMD_ZVECTOR_HPP

#include "../types/xsimd_zvector_register.hpp"

#if defined(__VEC__) && defined(__s390x__)

#include <vecintrin.h>
#include <cstring>  // memcpy

namespace xsimd
{
    namespace kernel
    {
        using namespace types;

        // ============================================================
        // broadcast
        // ============================================================

        template <class A, class T>
        XSIMD_INLINE batch<T, A> broadcast(T val, requires_arch<zvector>) noexcept
        {
            return batch<T, A>(vec_splats(val));
        }

        // Specialise for bool (broadcast as all-ones or all-zeros byte)
        template <class A>
        XSIMD_INLINE batch_bool<uint8_t, A> broadcast(bool val, requires_arch<zvector>) noexcept
        {
            uint8_t v = val ? 0xFF : 0x00;
            return batch_bool<uint8_t, A>((__vector unsigned char)vec_splats(v));
        }

        // ============================================================
        // set  (element-by-element constructor)
        // ============================================================

        template <class A, class T, class... Args>
        XSIMD_INLINE batch<T, A> set(batch<T, A> const&, requires_arch<zvector>,
                                     Args... args) noexcept
        {
            // Build a temporary array then load — portable across all element widths
            alignas(16) T tmp[] = { static_cast<T>(args)... };
            return batch<T, A>::load_aligned(tmp);
        }

        // ============================================================
        // load_aligned / load_unaligned
        // ============================================================

        template <class A, class T>
        XSIMD_INLINE batch<T, A> load_aligned(T const* mem, convert<T>,
                                               requires_arch<zvector>) noexcept
        {
            using reg_t = typename simd_register<T, zvector>::register_type;
            return batch<T, A>(*(const reg_t*)mem);
        }

        template <class A, class T>
        XSIMD_INLINE batch<T, A> load_unaligned(T const* mem, convert<T>,
                                                 requires_arch<zvector>) noexcept
        {
            using reg_t = typename simd_register<T, zvector>::register_type;
            reg_t result;
            __builtin_memcpy(&result, mem, sizeof(reg_t));
            return batch<T, A>(result);
        }

        // ============================================================
        // store_aligned / store_unaligned
        // ============================================================

        template <class A, class T>
        XSIMD_INLINE void store_aligned(T* mem, batch<T, A> const& val,
                                         requires_arch<zvector>) noexcept
        {
            using reg_t = typename simd_register<T, zvector>::register_type;
            *(reg_t*)mem = val.data;
        }

        template <class A, class T>
        XSIMD_INLINE void store_unaligned(T* mem, batch<T, A> const& val,
                                           requires_arch<zvector>) noexcept
        {
            using reg_t = typename simd_register<T, zvector>::register_type;
            __builtin_memcpy(mem, &val.data, sizeof(reg_t));
        }

        // ============================================================
        // Arithmetic: add, sub, mul, div, neg
        // ============================================================

#define XSIMD_ZVECTOR_ARITH(op_name, vec_op)                                \
        template <class A, class T>                                          \
        XSIMD_INLINE batch<T, A> op_name(batch<T, A> const& lhs,            \
                                          batch<T, A> const& rhs,           \
                                          requires_arch<zvector>) noexcept   \
        {                                                                    \
            return batch<T, A>(vec_op(lhs.data, rhs.data));                 \
        }

        XSIMD_ZVECTOR_ARITH(add, vec_add)
        XSIMD_ZVECTOR_ARITH(sub, vec_sub)
        XSIMD_ZVECTOR_ARITH(mul, vec_mul)

#undef XSIMD_ZVECTOR_ARITH

        // Division — float and double only (integer division is scalar)
        template <class A>
        XSIMD_INLINE batch<float, A> div(batch<float, A> const& lhs,
                                          batch<float, A> const& rhs,
                                          requires_arch<zvector>) noexcept
        {
            return batch<float, A>(vec_div(lhs.data, rhs.data));
        }

        template <class A>
        XSIMD_INLINE batch<double, A> div(batch<double, A> const& lhs,
                                           batch<double, A> const& rhs,
                                           requires_arch<zvector>) noexcept
        {
            return batch<double, A>(vec_div(lhs.data, rhs.data));
        }

        // Integer division — scalar fallback
        template <class A, class T>
        XSIMD_INLINE batch<T, A> div(batch<T, A> const& lhs,
                                      batch<T, A> const& rhs,
                                      requires_arch<zvector>) noexcept
        {
            constexpr int N = batch<T, A>::size;
            alignas(16) T l[N], r[N], res[N];
            lhs.store_aligned(l);
            rhs.store_aligned(r);
            for (int i = 0; i < N; ++i) res[i] = l[i] / r[i];
            return batch<T, A>::load_aligned(res);
        }

        template <class A, class T>
        XSIMD_INLINE batch<T, A> neg(batch<T, A> const& val,
                                      requires_arch<zvector>) noexcept
        {
            return batch<T, A>(vec_neg(val.data));
        }

        // ============================================================
        // min / max / abs
        // ============================================================

        template <class A, class T>
        XSIMD_INLINE batch<T, A> min(batch<T, A> const& lhs,
                                      batch<T, A> const& rhs,
                                      requires_arch<zvector>) noexcept
        {
            return batch<T, A>(vec_min(lhs.data, rhs.data));
        }

        template <class A, class T>
        XSIMD_INLINE batch<T, A> max(batch<T, A> const& lhs,
                                      batch<T, A> const& rhs,
                                      requires_arch<zvector>) noexcept
        {
            return batch<T, A>(vec_max(lhs.data, rhs.data));
        }

        template <class A, class T>
        XSIMD_INLINE batch<T, A> abs(batch<T, A> const& val,
                                      requires_arch<zvector>) noexcept
        {
            return batch<T, A>(vec_abs(val.data));
        }

        // ============================================================
        // FMA — fused multiply-add
        // ============================================================

        template <class A>
        XSIMD_INLINE batch<float, A> fma(batch<float, A> const& a,
                                          batch<float, A> const& b,
                                          batch<float, A> const& c,
                                          requires_arch<zvector>) noexcept
        {
            return batch<float, A>(vec_madd(a.data, b.data, c.data));
        }

        template <class A>
        XSIMD_INLINE batch<double, A> fma(batch<double, A> const& a,
                                           batch<double, A> const& b,
                                           batch<double, A> const& c,
                                           requires_arch<zvector>) noexcept
        {
            return batch<double, A>(vec_madd(a.data, b.data, c.data));
        }

        // ============================================================
        // Bitwise: and, or, xor, not, andnot
        // ============================================================

#define XSIMD_ZVECTOR_BITWISE(op_name, vec_op)                              \
        template <class A, class T>                                          \
        XSIMD_INLINE batch<T, A> op_name(batch<T, A> const& lhs,            \
                                          batch<T, A> const& rhs,           \
                                          requires_arch<zvector>) noexcept   \
        {                                                                    \
            return batch<T, A>(                                              \
                (typename simd_register<T,zvector>::register_type)          \
                vec_op((__vector unsigned char)lhs.data,                    \
                       (__vector unsigned char)rhs.data));                  \
        }

        XSIMD_ZVECTOR_BITWISE(bitwise_and, vec_and)
        XSIMD_ZVECTOR_BITWISE(bitwise_or,  vec_or)
        XSIMD_ZVECTOR_BITWISE(bitwise_xor, vec_xor)
        XSIMD_ZVECTOR_BITWISE(bitwise_andnot, vec_andc)

#undef XSIMD_ZVECTOR_BITWISE

        template <class A, class T>
        XSIMD_INLINE batch<T, A> bitwise_not(batch<T, A> const& val,
                                              requires_arch<zvector>) noexcept
        {
            return batch<T, A>(
                (typename simd_register<T, zvector>::register_type)
                vec_nor((__vector unsigned char)val.data,
                        (__vector unsigned char)val.data));
        }

        // ============================================================
        // Comparison → batch_bool
        // ============================================================

#define XSIMD_ZVECTOR_CMP(op_name, vec_op)                                  \
        template <class A, class T>                                          \
        XSIMD_INLINE batch_bool<T, A> op_name(batch<T, A> const& lhs,       \
                                               batch<T, A> const& rhs,      \
                                               requires_arch<zvector>) noexcept \
        {                                                                    \
            return batch_bool<T, A>(                                         \
                (typename get_bool_simd_register_t<T,zvector>::register_type)\
                vec_op(lhs.data, rhs.data));                                 \
        }

        XSIMD_ZVECTOR_CMP(eq,  vec_cmpeq)
        XSIMD_ZVECTOR_CMP(neq, vec_cmpne)
        XSIMD_ZVECTOR_CMP(lt,  vec_cmplt)
        XSIMD_ZVECTOR_CMP(le,  vec_cmple)
        XSIMD_ZVECTOR_CMP(gt,  vec_cmpgt)
        XSIMD_ZVECTOR_CMP(ge,  vec_cmpge)

#undef XSIMD_ZVECTOR_CMP

        // ============================================================
        // select (blend)
        // ============================================================

        template <class A, class T>
        XSIMD_INLINE batch<T, A> select(batch_bool<T, A> const& cond,
                                         batch<T, A> const& lhs,
                                         batch<T, A> const& rhs,
                                         requires_arch<zvector>) noexcept
        {
            return batch<T, A>(vec_sel(rhs.data, lhs.data, cond.data));
        }

        // ============================================================
        // bitwise_cast / reinterpret
        // ============================================================

        template <class A, class T, class U>
        XSIMD_INLINE batch<T, A> bitwise_cast(batch<U, A> const& val,
                                               batch<T, A> const&,
                                               requires_arch<zvector>) noexcept
        {
            typename simd_register<T, zvector>::register_type result;
            __builtin_memcpy(&result, &val.data, sizeof(result));
            return batch<T, A>(result);
        }

        // ============================================================
        // logical shift (integer types)
        // ============================================================

        template <class A, class T>
        XSIMD_INLINE batch<T, A> bitwise_lshift(batch<T, A> const& lhs,
                                                  int shift,
                                                  requires_arch<zvector>) noexcept
        {
            using unsigned_t = typename std::make_unsigned<T>::type;
            auto shift_vec = vec_splats(static_cast<unsigned_t>(shift));
            return batch<T, A>(vec_sl(lhs.data, shift_vec));
        }

        template <class A, class T>
        XSIMD_INLINE batch<T, A> bitwise_rshift(batch<T, A> const& lhs,
                                                  int shift,
                                                  requires_arch<zvector>) noexcept
        {
            using unsigned_t = typename std::make_unsigned<T>::type;
            auto shift_vec = vec_splats(static_cast<unsigned_t>(shift));
            return batch<T, A>(vec_sra(lhs.data, shift_vec));   // arithmetic shift
        }

        // ============================================================
        // sqrt
        // ============================================================

        template <class A>
        XSIMD_INLINE batch<float, A> sqrt(batch<float, A> const& val,
                                           requires_arch<zvector>) noexcept
        {
            return batch<float, A>(vec_sqrt(val.data));
        }

        template <class A>
        XSIMD_INLINE batch<double, A> sqrt(batch<double, A> const& val,
                                            requires_arch<zvector>) noexcept
        {
            return batch<double, A>(vec_sqrt(val.data));
        }

        // ============================================================
        // from_bool / to_bool helpers
        // ============================================================

        template <class A, class T>
        XSIMD_INLINE batch<T, A> from_bool(batch_bool<T, A> const& val,
                                            requires_arch<zvector>) noexcept
        {
            // Convert mask (all-ones / all-zeros per lane) to 0 or 1
            return batch<T, A>(val) & batch<T, A>(1);
        }

        // ============================================================
        // get (scalar extraction — debug use only)
        // ============================================================

        template <class A, class T>
        XSIMD_INLINE T get(batch<T, A> const& val, std::size_t i,
                            requires_arch<zvector>) noexcept
        {
            alignas(16) T tmp[batch<T, A>::size];
            val.store_aligned(tmp);
            return tmp[i];
        }

        // ============================================================
        // store (batch_bool → bool array)
        // ============================================================

        template <class A, class T>
        XSIMD_INLINE void store(batch_bool<T, A> const& val, bool* mem,
                                 requires_arch<zvector>) noexcept
        {
            constexpr int N = batch_bool<T, A>::size;
            alignas(16) typename batch_bool<T, A>::value_type tmp[N];
            val.store_aligned(tmp);
            for (int i = 0; i < N; ++i)
                mem[i] = static_cast<bool>(tmp[i]);
        }

        // ============================================================
        // hadd (horizontal add)
        // ============================================================

        template <class A, class T>
        XSIMD_INLINE T hadd(batch<T, A> const& val,
                             requires_arch<zvector>) noexcept
        {
            constexpr int N = batch<T, A>::size;
            alignas(16) T tmp[N];
            val.store_aligned(tmp);
            T result = tmp[0];
            for (int i = 1; i < N; ++i)
                result += tmp[i];
            return result;
        }

    } // namespace kernel
} // namespace xsimd

#endif // __VEC__ && __s390x__
#endif // XSIMD_ZVECTOR_HPP
