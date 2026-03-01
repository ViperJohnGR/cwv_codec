#pragma once

#include <bit>
#include <limits>
#include <stdexcept>
#include <string>
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

template <class T>
BitPack packBits(const std::vector<T>& input)
{
    static_assert(std::is_unsigned_v<T>, "T must be unsigned");
    static_assert(std::is_integral_v<T>, "T must be an integral type");
    static_assert(sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4, "T must be uint8_t, uint16_t, or uint32_t");

    BitPack out;

    if (input.empty()) {
        out.bit_width = 0;
        return out;
    }

    // 1) Detect max
    std::uint32_t maxv = 0;
    for (T v : input)
        if (std::uint32_t(v) > maxv) maxv = std::uint32_t(v);

    // 2) Compute needed bit width (at least 1)
    auto bw = std::bit_width(maxv);
    if (bw > 31) {
        throw std::invalid_argument("bitpack_msb_first: bit width > 31 not supported");
    }
    out.bit_width = bw;

    // 3) Pack MSB-first
    const std::size_t total_bits = std::size_t(bw) * input.size();
    const std::size_t total_bytes = (total_bits + 7) / 8;
    out.bytes.reserve(total_bytes);

    std::uint64_t buffer = 0;     // holds yet-to-be-emitted bits (in its low 'bit_count' bits)
    std::uint32_t bit_count = 0;  // number of valid bits currently in buffer

    const std::uint64_t mask = (bw == 64) ? ~0ULL : ((1ULL << bw) - 1ULL); // bw<=31 here

    for (T v : input) {
        const std::uint64_t vv = std::uint64_t(std::uint32_t(v)) & mask;

        // Append 'bw' bits to the right end of the stream:
        // existing bits shift up; new value occupies the lowest bw bits.
        buffer = (buffer << bw) | vv;
        bit_count += bw;

        // While we can emit a full byte, take the TOP 8 bits (MSB-first output).
        while (bit_count >= 8) {
            const std::uint32_t shift = bit_count - 8;
            std::uint8_t byte = static_cast<std::uint8_t>((buffer >> shift) & 0xFFu);
            out.bytes.push_back(byte);

            bit_count -= 8;

            // Keep only remaining 'bit_count' low bits
            if (bit_count == 0) {
                buffer = 0;
            }
            else {
                buffer &= ((1ULL << bit_count) - 1ULL);
            }
        }
    }

    // Emit leftover bits (pad with zeros on the right / LSB side of the last byte)
    if (bit_count > 0) {
        std::uint8_t last = static_cast<std::uint8_t>((buffer << (8 - bit_count)) & 0xFFu);
        out.bytes.push_back(last);
    }

    out.count = (uint32_t)input.size();
    return out;
}

template <class T>
std::vector<T> unpackBits(const std::vector<std::uint8_t>& bytes, std::uint8_t bit_width, std::size_t count)
{
    static_assert(std::is_unsigned_v<T>, "T must be unsigned");
    static_assert(std::is_integral_v<T>, "T must be an integral type");
    static_assert(sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4, "T must be uint8_t, uint16_t, or uint32_t");

    if (count == 0) return {};

    if (bit_width == 0 || bit_width > 31) {
        throw std::invalid_argument("bitunpack_msb_first: bit_width must be in [1, 31]");
    }

    // Ensure bit_width fits in T
    if (bit_width > std::numeric_limits<T>::digits) {
        throw std::invalid_argument("bitunpack_msb_first: bit_width doesn't fit in output type T");
    }

    const std::size_t total_bits_needed = std::size_t(bit_width) * count;
    const std::size_t total_bytes_needed = (total_bits_needed + 7) / 8;

    if (bytes.size() < total_bytes_needed) {
        throw std::invalid_argument("bitunpack_msb_first: not enough input bytes for requested count/bit_width");
    }

    std::vector<T> out;
    out.reserve(count);

    const std::uint32_t bw = bit_width;
    const std::uint64_t mask = (bw == 64) ? ~0ULL : ((1ULL << bw) - 1ULL); // bw<=31 here

    std::uint64_t buffer = 0;     // holds accumulated bits (in its low 'bit_count' bits)
    std::uint32_t bit_count = 0;  // number of valid bits currently in buffer
    std::size_t byte_index = 0;

    while (out.size() < count) {
        // Fill until we have enough bits for one value
        while (bit_count < bw) {
            if (byte_index >= bytes.size()) {
                throw std::invalid_argument("bitunpack_msb_first: ran out of bytes while unpacking");
            }
            buffer = (buffer << 8) | std::uint64_t(bytes[byte_index++]);
            bit_count += 8;
        }

        // Extract the top bw bits from buffer (MSB-first stream)
        const std::uint32_t shift = bit_count - bw;
        const std::uint64_t v = (buffer >> shift) & mask;
        out.push_back(static_cast<T>(v));

        // Remove those bits from buffer; keep remaining low bits
        bit_count -= bw;
        if (bit_count == 0) {
            buffer = 0;
        }
        else {
            buffer &= ((1ULL << bit_count) - 1ULL);
        }
    }

    return out;
}

inline size_t calculateBitPackedSize(size_t n, size_t bits)
{
    return (n * bits + 7U) / 8U;
}
