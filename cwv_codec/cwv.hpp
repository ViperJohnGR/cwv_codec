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
//     u32     blockSize      (in PCM frames, 0 = variable-size block plan follows)
//     u32     numberOfBlocks
//     u8      quantFlags
//              - bit 7 = adaptive per-block quantization width present in packInfo
//              - bit 6 = variable block sizes (header-side change table follows)
//              - bits 5:0 = nominal quant bits (1..8)
//
//     if variable block sizes:
//       u16    initialBlockSize
//       u32    blockSizeChangeCount
//       repeat blockSizeChangeCount times:
//         u16  deltaFramesFromPreviousChange
//         u16  newBlockSize
//
//   Then for each block:
//     u8      packInfo
//              - non-adaptive files:
//                  high nibble = predictor (0 = 1st-order, 1 = 2nd-order)
//                  low  nibble = bitWidth for the block's DPCM payload
//              - adaptive files:
//                  high nibble bit 3    = predictor (0 = 1st-order, 1 = 2nd-order)
//                  high nibble bits 2:0 = block quantBits - 1 (0..7 -> 1..8 bits)
//                  low  nibble          = bitWidth for the block's DPCM payload
//     u8      gainCode[channels]  (packed gain in dB, per-channel)
//     ...     seedSamples         (first-frame samples, packed with the block quant bits)
//     ...     audioData           (bit-packed interleaved DPCM+zigzag payload for the rest of the block)
//
// Notes:
// - Fixed-size files keep the legacy header layout.
// - Variable-size files keep block sizes out of the block payload; only size changes are
//   stored in the header-side change table using frame deltas.
// - Each block uses a per-channel gain, computed as the gain required to normalize
//   the block peak for that channel.
// - Adaptive files keep roughly the same bitrate by choosing a per-block quantization
//   width near the requested nominal width.
// - The first frame (one sample per channel) is stored as absolute values (seedSamples).
// - The rest of the samples are stored as per-channel DPCM diffs (mod 2^quantBits)
//   followed by zigzag, then bit-packed using the smallest bitWidth that fits that
//   block's payload.

struct CWVHeader
{
    char magic[3]{};
    uint8_t channels = 0;
    uint32_t sampleRate = 0;
    sf_count_t totalPCMFrameCount = 0;
    uint32_t blockSize = 0; // fixed-size files only; 0 means variable block sizes
    uint32_t numberOfBlocks = 0;
    uint8_t quantBits = 0;  // nominal quant bits only
    bool adaptiveQuantization = false;
    bool variableBlockSize = false;
    uint32_t initialBlockSize = 0;
    uint32_t blockSizeChangeCount = 0;
};

// Produces a complete CWV file buffer (header + blocks).
std::vector<uint8_t> encodeCWV(audioStream& audio, uint32_t blockSizeFrames, uint8_t bitsPerSample, bool saveCompressed, uint8_t autoBlockQuality = 5);

// Decodes a CWV file buffer into interleaved float samples. Optionally returns header.
int decodeCWV(const std::vector<uint8_t>& input, std::vector<float>& outputBuffer, CWVHeader* outHeader = nullptr);
