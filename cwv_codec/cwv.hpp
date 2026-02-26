#pragma once

#include "audioStream.hpp"
#include "helpers.hpp"

#include <vector>

/*
class cwv
{
    unsigned char channels;
    unsigned int sampleRate;
    unsigned long long totalPCMFrameCount;
    unsigned char bitsPerSample;
    float gainStep;
    
    std::vector<uint8_t> cwvStream;

    //struct gainInfo {};

public:
    cwv(audioStream stream, uint8_t targetBits);
};
*/

// CWV block-based format (CWV3)
//
// File layout (little-endian):
//   Header:
//     char    magic[4]      = "CWV3"
//     u8      channels
//     u32     sampleRate
//     s64     totalPCMFrameCount
//     u32     blockSize      (in PCM frames)
//     u32     numberOfBlocks
//     u8      quantBits      (quantization bits per sample, 1..8)
//
//   Then for each block:
//     u8      bitWidth       (bit-packed width for the block's DPCM payload)
//     u8      gainCode[channels]  (packed gain in dB, per-channel)
//     ...     seedSamples    (first-frame samples, packed with quantBits)
//     ...     audioData      (bit-packed interleaved DPCM+zigzag payload for the rest of the block)
//
// Notes:
// - Each block uses a per-channel gain, computed as the gain required to normalize
//   the block peak for that channel.
// - Samples are quantized to quantBits (constant for the whole file).
// - The first frame (one sample per channel) is stored as absolute values (seedSamples).
// - The rest of the samples are stored as per-channel DPCM diffs (mod 2^quantBits)
//   followed by zigzag, then bit-packed using the smallest bitWidth that fits that
//   block's payload.

struct CWVHeader
{
    char magic[4];
    uint8_t channels;
    uint32_t sampleRate;
    sf_count_t totalPCMFrameCount;
    uint32_t blockSize;
    uint32_t numberOfBlocks;
    uint8_t quantBits;
};

// Produces a complete CWV file buffer (header + blocks).
std::vector<uint8_t> encodeCWV(audioStream& audio, uint32_t blockSizeFrames, uint8_t bitsPerSample, bool saveCompressed);

// Decodes a CWV file buffer into interleaved float samples. Optionally returns header.
int decodeCWV(const std::vector<uint8_t>& input, std::vector<float>& outputBuffer, CWVHeader* outHeader = nullptr);
