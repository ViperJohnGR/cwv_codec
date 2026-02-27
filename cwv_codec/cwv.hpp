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
//     u32     blockSize      (in PCM frames)
//     u32     numberOfBlocks
//     u8      quantBits
//              - legacy files: 1..8 = fixed quantization bits for every block
//              - adaptive files: bit 7 set, low 7 bits = nominal/target quant bits
//
//   Then for each block:
//     u8      packInfo
//              - legacy files:
//                  high nibble = predictor (0 = 1st-order, 1 = 2nd-order)
//                  low  nibble = bitWidth for the block's DPCM payload
//              - adaptive files:
//                  high nibble bit 3   = predictor (0 = 1st-order, 1 = 2nd-order)
//                  high nibble bits 2:0 = block quantBits - 1 (0..7 -> 1..8 bits)
//                  low  nibble          = bitWidth for the block's DPCM payload
//     u8      gainCode[channels]  (packed gain in dB, per-channel)
//     ...     seedSamples         (first-frame samples, packed with the block quant bits)
//     ...     audioData           (bit-packed interleaved DPCM+zigzag payload for the rest of the block)
//
// Notes:
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
    char magic[3];
    uint8_t channels;
    uint32_t sampleRate;
    sf_count_t totalPCMFrameCount;
    uint32_t blockSize;
    uint32_t numberOfBlocks;
    uint8_t quantBits; // nominal quant bits (adaptive files store the flag in-file only)
};

// Produces a complete CWV file buffer (header + blocks).
std::vector<uint8_t> encodeCWV(audioStream& audio, uint32_t blockSizeFrames, uint8_t bitsPerSample, bool saveCompressed);

// Decodes a CWV file buffer into interleaved float samples. Optionally returns header.
int decodeCWV(const std::vector<uint8_t>& input, std::vector<float>& outputBuffer, CWVHeader* outHeader = nullptr);
