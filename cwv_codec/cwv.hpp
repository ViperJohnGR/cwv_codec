#pragma once

#include "audioStream.hpp"

#include <cstdint>
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
//     u8      quantBits
//              - fixed quant bits for the whole file (2..8)
//
//   Then for each block:
//     u8      packInfo
//              - high nibble = predictor
//                  0 = none
//                  1 = previous sample
//                  2 = 2nd-order extrapolation
//                  3 = weighted 2-tap extrapolation
//                  4 = 3rd-order extrapolation
//                  5 = damped slope, prev1 + 0.25 * (prev1 - prev2)
//                  6 = damped slope, prev1 + 0.75 * (prev1 - prev2)
//                  7 = smoothed, 0.75 * prev1 + 0.25 * prev2
//                  8 = leaky previous sample, 0.9375 * prev1
//                  9 = slope-limited extrapolation
//              - low  nibble = stored residual quantizer mode
//                  0 = legacy CWV block: legacy predictor clamp, mu-law mu = 127
//                  1 = sample-clamped predictors, mu-law mu = 127
//                  2 = sample-clamped predictors, weaker mu-law mu = 15
//                  3 = sample-clamped predictors, stronger mu-law mu = 255
//                  4 = sample-clamped predictors, linear residual quantizer
//     u16     residualPeakQ[channels]
//              - block-local peak residual scale per channel, mapped to [0, 8]
//     s16     seedSampleQ[min(3, framesInBlock)][channels]
//              - first samples in the block, interleaved by frame then channel
//     ...     audioData
//              - bit-packed interleaved residual codes for the remaining samples in the block
//
// Notes:
// - All streams are coded per channel with the same tools; no stereo-only transform is used.
// - Each block carries its own predictor seed samples and can be decoded independently.
// - Residuals are encoded in the sample domain with companded quantization,
//   which preserves low-level detail better than uniform block-normalized PCM.

struct CWVHeader
{
    char magic[3]{};
    uint8_t channels = 0;
    uint32_t sampleRate = 0;
    int64_t totalPCMFrameCount = 0;
    uint32_t blockSize = 0; // fixed-size files only
    uint32_t numberOfBlocks = 0;
    uint8_t quantBits = 0;
};

// Produces a complete CWV file buffer (header + blocks).
std::vector<uint8_t> encodeCWV(audioStream& audio, uint32_t blockSizeFrames, uint8_t bitsPerSample);

// Decodes a CWV file buffer into interleaved float samples. Optionally returns header.
int decodeCWV(const std::vector<uint8_t>& input, std::vector<float>& outputBuffer, CWVHeader* outHeader = nullptr);
