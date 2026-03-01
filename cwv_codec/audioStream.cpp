#include "audioStream.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace
{

constexpr double kPi = 3.14159265358979323846;
constexpr float kButterworthQ = 0.7071067811865475f;

struct BiquadCoefficients
{
    double b0 = 1.0;
    double b1 = 0.0;
    double b2 = 0.0;
    double a1 = 0.0;
    double a2 = 0.0;
};

BiquadCoefficients makeLowPassBiquad(float cutoffHz, int sampleRate, float q)
{
    const double omega = 2.0 * kPi * static_cast<double>(cutoffHz) / static_cast<double>(sampleRate);
    const double cosw = std::cos(omega);
    const double sinw = std::sin(omega);
    const double alpha = sinw / (2.0 * static_cast<double>(q));

    const double b0 = (1.0 - cosw) * 0.5;
    const double b1 = 1.0 - cosw;
    const double b2 = (1.0 - cosw) * 0.5;
    const double a0 = 1.0 + alpha;
    const double a1 = -2.0 * cosw;
    const double a2 = 1.0 - alpha;

    BiquadCoefficients coeffs;
    coeffs.b0 = b0 / a0;
    coeffs.b1 = b1 / a0;
    coeffs.b2 = b2 / a0;
    coeffs.a1 = a1 / a0;
    coeffs.a2 = a2 / a0;
    return coeffs;
}

} // namespace


audioStream::audioStream()
{
    channels = 0;
    sampleRate = 0;
    totalPCMFrameCount = 0;
}

audioStream::audioStream(const std::string& path)
{
    SF_INFO info{};
    SNDFILE* file = sf_open(path.c_str(), SFM_READ, &info);
    if (file == nullptr)
    {
        channels = 0;
        sampleRate = 0;
        totalPCMFrameCount = 0;
        printf("Cannot open '%s'. %s\n", path.c_str(), sf_strerror(NULL));
        return;
    }

    if (info.channels < 1 || info.channels > 255)
    {
        channels = 0;
        sampleRate = 0;
        totalPCMFrameCount = 0;
        printf("Error. Invalid number of channels (%d).\n", info.channels);
        return;
    }

    sampleData.resize(info.frames * info.channels);
    sf_readf_float(file, &sampleData[0], info.frames);
    sf_close(file);

    channels = static_cast<uint8_t>(info.channels);
    sampleRate = info.samplerate;
    totalPCMFrameCount = info.frames;
}

bool audioStream::normalize()
{
    if (channels < 1 || sampleRate <= 0 || totalPCMFrameCount <= 0)
    {
        printf("Error! Cannot normalize an invalid audio stream.\n");
        return false;
    }

    float peak = 0.0f;
    for (const float sample : sampleData)
        peak = std::max(peak, std::abs(sample));

    if (peak <= 0.0f)
    {
        printf("Warning: input is silent; skipping normalization.\n");
        return true;
    }

    if (peak == 1.0f)
        return true;

    const float gain = 1.0f / peak;
    for (float& sample : sampleData)
        sample *= gain;

    printf("Normalized input by %.6fx (peak %.6f -> 1.000000).\n", gain, peak);
    return true;
}

bool audioStream::applyGain(float gain)
{
    if (channels < 1 || sampleRate <= 0 || totalPCMFrameCount <= 0)
    {
        printf("Error! Cannot apply gain to an invalid audio stream.\n");
        return false;
    }

    if (gain == 1.0f)
        return true;

    for (float& sample : sampleData)
        sample *= gain;

    printf("Applied gain of %.6fx.\n", gain);
    return true;
}

bool audioStream::applyLowPass(float cutoffHz)
{
    if (channels < 1 || sampleRate <= 0 || totalPCMFrameCount <= 0)
    {
        printf("Error! Cannot apply low-pass filter to an invalid audio stream.\n");
        return false;
    }

    if (cutoffHz <= 0.0f)
    {
        printf("Error! lowpass cutoff must be > 0 Hz.\n");
        return false;
    }

    const float nyquist = static_cast<float>(sampleRate) * 0.5f;
    const float clampedCutoff = std::clamp(cutoffHz, 1.0f, nyquist * 0.999f);

    if (clampedCutoff != cutoffHz)
        printf("Warning: lowpass cutoff %.2f Hz adjusted to %.2f Hz for sample rate %d Hz.\n", cutoffHz, clampedCutoff, sampleRate);

    const BiquadCoefficients coeffs = makeLowPassBiquad(clampedCutoff, sampleRate, kButterworthQ);

    std::vector<double> z1(channels, 0.0);
    std::vector<double> z2(channels, 0.0);

    const uint64_t frames = static_cast<uint64_t>(totalPCMFrameCount);
    for (uint64_t frame = 0; frame < frames; ++frame)
    {
        const size_t base = static_cast<size_t>(frame) * channels;
        for (uint8_t ch = 0; ch < channels; ++ch)
        {
            const double in = static_cast<double>(sampleData[base + ch]);
            const double out = coeffs.b0 * in + z1[ch];
            z1[ch] = coeffs.b1 * in - coeffs.a1 * out + z2[ch];
            z2[ch] = coeffs.b2 * in - coeffs.a2 * out;
            sampleData[base + ch] = static_cast<float>(out);
        }
    }

    return true;
}
