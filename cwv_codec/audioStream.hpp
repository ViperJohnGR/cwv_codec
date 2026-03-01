#pragma once

#include "sndfile.hh"
#include <string>
#include <vector>


struct audioStream
{
    uint8_t channels;
    int sampleRate;
    sf_count_t totalPCMFrameCount;
    std::vector<float> sampleData;

public:
    audioStream();
    audioStream(const std::string& path);

    bool normalize();
    bool applyLowPass(float cutoffHz);
};
