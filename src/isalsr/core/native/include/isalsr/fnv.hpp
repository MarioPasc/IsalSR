#pragma once
// FNV-1a 64-bit hash (Fowler–Noll–Vo, public domain).
// Offset basis and prime from the FNV spec (Noll et al., 1994).
// This header is the single source of truth: both the C++ binding and the
// Python reference in tests/unit/test_native_build.py must produce identical
// results for the same byte sequence.

#include <cstddef>
#include <cstdint>
#include <span>

namespace isalsr {

inline constexpr uint64_t FNV1A64_OFFSET = UINT64_C(0xcbf29ce484222325);
inline constexpr uint64_t FNV1A64_PRIME  = UINT64_C(0x00000100000001b3);

/// Compute the FNV-1a 64-bit hash of @p data.
///
/// The hash is byte-level and endian-neutral: the same byte sequence always
/// produces the same value regardless of CPU architecture.
[[nodiscard]] inline uint64_t fnv1a64(const uint8_t* data, std::size_t n) noexcept {
    uint64_t h = FNV1A64_OFFSET;
    for (std::size_t i = 0; i < n; ++i) {
        h ^= static_cast<uint64_t>(data[i]);
        h *= FNV1A64_PRIME;
    }
    return h;
}

} // namespace isalsr
