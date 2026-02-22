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

struct gainInfo
{
    uint32_t numInfos;
    uint8_t endsBitSize;
    std::vector<uint8_t> ends;
    std::vector<uint8_t> gains;
};


BitPack encodeStream(audioStream& audio, std::vector<gainInfo>& gainInfos, int bitsPerSample, float gainStep, bool saveCompressed);
int decodeStream(const std::vector<uint8_t>& input, std::vector<float>& outputBuffer);
