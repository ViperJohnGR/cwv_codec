#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct audioStream
{
    uint8_t channels;
    int sampleRate;
    int64_t totalPCMFrameCount;
    std::vector<float> sampleData;

public:
    audioStream();
    audioStream(const std::string& path);

    bool normalize();
    bool applyGain(float gain);
    bool applyLowPass(float cutoffHz);
};
