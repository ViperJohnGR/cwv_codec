#pragma once

#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

std::string getFilenameFromPath(const std::string& path);
std::string getExtensionFromPath(const std::string& path);
std::string removeExtensionFromPath(const std::string& path);
std::string printBytes(std::uint64_t bytes);
int openFile(FILE** f, const char* path, const char* mode);

struct BitPack {
    std::vector<std::uint8_t> bytes; // packed bitstream (MSB-first per byte)
    std::uint8_t bit_width = 0;      // bits used per value (0 if input empty)
    std::uint32_t count = 0;
};

// Packs with an explicitly provided bit width (MSB-first).
// Useful when you want per-block fixed bit widths (e.g., CWV blocks).
template <class T>
BitPack packBitsFixed(const std::vector<T>& input, std::uint8_t bit_width)
{
    static_assert(std::is_unsigned_v<T>, "T must be unsigned");
    static_assert(std::is_integral_v<T>, "T must be an integral type");
    static_assert(sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4, "T must be uint8_t, uint16_t, or uint32_t");

    if (input.empty()) return {};
    if (bit_width == 0 || bit_width > 31)
        throw std::invalid_argument("packBitsFixed: bit_width must be in [1,31]");
    if (bit_width > std::numeric_limits<T>::digits)
        throw std::invalid_argument("packBitsFixed: bit_width doesn't fit in T");

    BitPack out;
    out.bit_width = bit_width;
    out.count = static_cast<std::uint32_t>(input.size());

    const std::uint32_t bw = bit_width;
    const std::uint64_t mask = (bw == 64) ? ~0ULL : ((1ULL << bw) - 1ULL);

    const std::size_t total_bits = std::size_t(bw) * input.size();
    const std::size_t total_bytes = (total_bits + 7) / 8;
    out.bytes.reserve(total_bytes);

    std::uint64_t buffer = 0;
    std::uint32_t bit_count = 0;

    for (T v : input) {
        const std::uint64_t vv = std::uint64_t(std::uint32_t(v)) & mask;
        buffer = (buffer << bw) | vv;
        bit_count += bw;

        while (bit_count >= 8) {
            const std::uint32_t shift = bit_count - 8;
            std::uint8_t byte = static_cast<std::uint8_t>((buffer >> shift) & 0xFFu);
            out.bytes.push_back(byte);
            bit_count -= 8;
            if (bit_count == 0) buffer = 0;
            else buffer &= ((1ULL << bit_count) - 1ULL);
        }
    }

    if (bit_count > 0) {
        std::uint8_t last = static_cast<std::uint8_t>((buffer << (8 - bit_count)) & 0xFFu);
        out.bytes.push_back(last);
    }

    return out;
}

inline size_t calculateBitPackedSize(size_t n, size_t bits)
{
    return (n * bits + 7U) / 8U;
}
