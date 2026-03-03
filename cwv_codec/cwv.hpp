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
//              - bit 7 = block-local quantization metadata present in packInfo
//              - bits 6:0 = nominal quant bits (1..8)
//
//   Then for each block:
//     u8      packInfo
//              - high nibble = predictor (0 = none, 1 = 1st-order, 2 = 2nd-order)
//              - low  nibble = block quant bits minus 1 (0..7 -> 1..8 bits)
//     u16     residualPeakQ[channels]
//              - block-local peak residual scale per channel, mapped to [0, 8]
//     ...     audioData
//              - bit-packed interleaved residual codes for every sample in the block
//
// Notes:
// - This revision is intentionally not backward compatible with the previous
//   gain-normalized seed/residual format.
// - Stereo is coded as independent channels; no side/mid transform is used.
// - Predictor state carries across block boundaries, which reduces block-edge
//   discontinuities compared with restarting prediction every block.
// - Residuals are encoded in the sample domain with companded quantization,
//   which preserves low-level detail better than uniform block-normalized PCM.

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
