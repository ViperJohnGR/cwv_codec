#pragma once

#include "audioStream.hpp"

#include <vector>

// CWV block-based format (CWV)
//
// File layout (little-endian):
//   Header:
//     char    magic[3]      = "CWV"
//     u8      channels
//     u32     sampleRate
//     s64     totalPCMFrameCount
//     u32     blockSize      (fixed block size in PCM frames)
//     u32     numberOfBlocks
//     u8      quantFlags
//              - bit 7 = adaptive per-block quantization width present in packInfo
//              - bits 6:0 = nominal quant bits (1..8)
//
//   Then for each block:
//     u8      packInfo
//              - non-adaptive files:
//                  high nibble = predictor (0 = 1st-order, 1 = 2nd-order)
//                  low  nibble = residual bit width for the block payload (0 = implicit zero residuals)
//              - adaptive files:
//                  high nibble bit 3    = predictor (0 = 1st-order, 1 = 2nd-order)
//                  high nibble bits 2:0 = block quantBits - 1 (0..7 -> 1..8 bits)
//                  low  nibble          = residual bit width for the block payload (0 = implicit zero residuals)
//     u8      gainCode[channels]  (packed gain in dB, per-channel)
//     u8      residualPeak        (maximum absolute predictor residual in quantized-code units)
//     ...     seedSamples         (first-frame samples, packed with the block quant bits)
//     ...     audioData           (bit-packed interleaved scaled-DPCM payload for the rest of the block)
//
// Notes:
// - This format revision is intentionally not backward compatible with the previous
//   exact-delta / zigzag residual stream.
// - Each block uses a per-channel gain, computed as the gain required to normalize
//   the block peak for that channel.
// - Adaptive files keep roughly the same bitrate by choosing a per-block quantization
//   width near the requested nominal width.
// - The first frame (one sample per channel) is stored as absolute values (seedSamples).
// - The rest of the samples are stored as predictor residual buckets. For example,
//   a 4-bit residual stream carries 16 scaled residual levels across that block's
//   residualPeak range instead of exact +/- step deltas.

struct CWVHeader
{
    char magic[3]{};
    uint8_t channels = 0;
    uint32_t sampleRate = 0;
    sf_count_t totalPCMFrameCount = 0;
    uint32_t blockSize = 0; // fixed-size files only
    uint32_t numberOfBlocks = 0;
    uint8_t quantBits = 0;  // nominal quant bits only
    bool adaptiveQuantization = false;
};

// Produces a complete CWV file buffer (header + blocks).
std::vector<uint8_t> encodeCWV(audioStream& audio, uint32_t blockSizeFrames, uint8_t bitsPerSample, bool saveCompressed);

// Decodes a CWV file buffer into interleaved float samples. Optionally returns header.
int decodeCWV(const std::vector<uint8_t>& input, std::vector<float>& outputBuffer, CWVHeader* outHeader = nullptr);
