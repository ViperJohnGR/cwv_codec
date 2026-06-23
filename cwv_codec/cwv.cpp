#include "cwv.hpp"
#include "helpers.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <thread>
#include <vector>

namespace {

constexpr float kResidualPeakRange = 8.0f;
constexpr float kResidualPeakMin = 1.0e-9f;
constexpr float kMuLaw = 127.0f;
constexpr float kSampleClamp = 1.0f;
constexpr float kSilentPeakEpsilon = 1e-12f;
constexpr uint8_t kMinQuantBits = 2;
constexpr uint8_t kPredictorCount = 10;
constexpr uint8_t kMaxPredictor = kPredictorCount - 1u;
constexpr uint8_t kQuantizerModeCount = 4;
constexpr uint8_t kMaxQuantizerMode = kQuantizerModeCount - 1u;
constexpr uint8_t kMaxStoredQuantizerMode = kQuantizerModeCount;
constexpr uint8_t kAnalysisJobCount = kPredictorCount * kQuantizerModeCount;
constexpr uint32_t kSeedFramesPerBlock = 3;


struct PlannedBlock
{
    uint32_t startFrame = 0;
    uint32_t frames = 0;
};

struct BlockCandidate
{
    uint8_t predictor = 0;
    uint8_t quantizerMode = 0;
    uint8_t quantBits = 0;
    std::vector<uint16_t> peakQ;
    double distortion = 0.0;
    std::vector<float> endPrev1;
    std::vector<float> endPrev2;
    std::vector<float> endPrev3;
};

template <class T>
void appendLE(std::vector<uint8_t>& out, const T& v)
{
    static_assert(std::is_trivially_copyable_v<T>);
    uint8_t buf[sizeof(T)];
    std::memcpy(buf, &v, sizeof(T));
    if constexpr (std::endian::native == std::endian::big)
        std::reverse(buf, buf + sizeof(T));
    out.insert(out.end(), buf, buf + sizeof(T));
}

template <class T>
bool readLE(const std::vector<uint8_t>& in, size_t& offset, T& outV)
{
    static_assert(std::is_trivially_copyable_v<T>);
    if (offset + sizeof(T) > in.size())
        return false;

    uint8_t buf[sizeof(T)];
    std::memcpy(buf, in.data() + offset, sizeof(T));
    if constexpr (std::endian::native == std::endian::big)
        std::reverse(buf, buf + sizeof(T));
    std::memcpy(&outV, buf, sizeof(T));
    offset += sizeof(T);
    return true;
}

struct PackedBitReader
{
    const uint8_t* data = nullptr;
    size_t size = 0;
    size_t byteIndex = 0;
    uint64_t buffer = 0;
    uint32_t bitCount = 0;

    bool read(uint8_t bitWidth, uint8_t& out)
    {
        static constexpr uint16_t kMask[9] = { 0u, 0x01u, 0x03u, 0x07u, 0x0Fu, 0x1Fu, 0x3Fu, 0x7Fu, 0xFFu };

        while (bitCount < bitWidth)
        {
            if (byteIndex >= size)
                return false;

            buffer = (buffer << 8) | static_cast<uint64_t>(data[byteIndex++]);
            bitCount += 8;
        }

        const uint32_t shift = bitCount - bitWidth;
        out = static_cast<uint8_t>((buffer >> shift) & kMask[bitWidth]);

        bitCount -= bitWidth;
        if (bitCount == 0)
            buffer = 0;
        else
            buffer &= ((uint64_t{ 1 } << bitCount) - 1u);

        return true;
    }
};

float clampPredict(float sample)
{
    return std::clamp(sample, -kSampleClamp, kSampleClamp);
}

float predictSample(uint8_t predictor, float prev1, float prev2, float prev3)
{
    switch (predictor)
    {
    case 0: // no prediction
        return 0.0f;
    case 1: // previous sample
        return prev1;
    case 2: // full 2nd-order extrapolation
        return clampPredict(2.0f * prev1 - prev2);
    case 3: // weighted 2-tap extrapolation
        return clampPredict(1.5f * prev1 - 0.5f * prev2);
    case 4: // 3rd-order extrapolation
        return clampPredict(3.0f * prev1 - 3.0f * prev2 + prev3);
    case 5: // damped slope
        return clampPredict(prev1 + 0.25f * (prev1 - prev2));
    case 6: // stronger damped slope
        return clampPredict(prev1 + 0.75f * (prev1 - prev2));
    case 7: // smoothed previous samples
        return clampPredict(0.75f * prev1 + 0.25f * prev2);
    case 8: // leaky previous sample
        return clampPredict(0.9375f * prev1);
    case 9: // slope-limited extrapolation
    {
        const float slope = prev1 - prev2;
        const float prevSlope = prev2 - prev3;
        const float limit = std::max(0.03125f, 1.5f * std::fabs(prevSlope));
        return clampPredict(prev1 + std::clamp(slope, -limit, limit));
    }
    default:
        return 0.0f;
    }
}

uint16_t quantizeResidualPeak(float peak)
{
    if (!(peak > kSilentPeakEpsilon))
        return 0;

    peak = std::clamp(peak, kResidualPeakMin, kResidualPeakRange);
    static const float logMin = std::log(kResidualPeakMin);
    static const float logRange = std::log(kResidualPeakRange) - logMin;
    const float normalized = (std::log(peak) - logMin) / logRange;
    const uint32_t quantized = 1u + static_cast<uint32_t>(std::lround(normalized * 65534.0f));
    return static_cast<uint16_t>(std::min<uint32_t>(quantized, 65535u));
}

float dequantizeResidualPeak(uint16_t peakQ)
{
    if (peakQ == 0)
        return 0.0f;

    static const float logMin = std::log(kResidualPeakMin);
    static const float logRange = std::log(kResidualPeakRange) - logMin;
    const float normalized = static_cast<float>(peakQ - 1u) / 65534.0f;
    return std::exp(logMin + normalized * logRange);
}

float quantizerMu(uint8_t quantizerMode)
{
    switch (quantizerMode)
    {
    case 1:
        return 15.0f;
    case 2:
        return 255.0f;
    default:
        return kMuLaw;
    }
}

bool quantizerIsLinear(uint8_t quantizerMode)
{
    return quantizerMode == 3u;
}

float compandResidual(float x, uint8_t quantizerMode)
{
    x = std::clamp(x, -1.0f, 1.0f);
    if (quantizerIsLinear(quantizerMode))
        return x;

    const float ax = std::fabs(x);
    if (ax <= 0.0f)
        return 0.0f;

    const float mu = quantizerMu(quantizerMode);
    const float denom = std::log1p(mu);
    return std::copysign(std::log1p(mu * ax) / denom, x);
}

float expandResidual(float y, uint8_t quantizerMode)
{
    y = std::clamp(y, -1.0f, 1.0f);
    if (quantizerIsLinear(quantizerMode))
        return y;

    const float ay = std::fabs(y);
    if (ay <= 0.0f)
        return 0.0f;

    const float mu = quantizerMu(quantizerMode);
    const float x = (std::pow(1.0f + mu, ay) - 1.0f) / mu;
    return std::copysign(x, y);
}

int32_t decodeSignedResidualCode(uint8_t code, uint8_t quantBits)
{
    const uint32_t codeRange = 1u << quantBits;
    const uint32_t mask = codeRange - 1u;
    const uint32_t signBit = codeRange >> 1u;
    const uint32_t value = static_cast<uint32_t>(code) & mask;
    int32_t quantized = (value & signBit) != 0u
        ? static_cast<int32_t>(value) - static_cast<int32_t>(codeRange)
        : static_cast<int32_t>(value);
    const int32_t maxMagnitude = static_cast<int32_t>(signBit) - 1;
    return std::clamp(quantized, -maxMagnitude, maxMagnitude);
}

int32_t quantizeResidualMagnitudeReference(float normalizedMagnitude, uint32_t maxMagnitude, uint8_t quantizerMode)
{
    const float companded = compandResidual(normalizedMagnitude, quantizerMode);
    return std::clamp(
        static_cast<int32_t>(std::lround(companded * static_cast<float>(maxMagnitude))),
        0,
        static_cast<int32_t>(maxMagnitude));
}

const std::array<std::array<std::array<float, 127>, 9>, kQuantizerModeCount>& getResidualEncodeThresholds()
{
    // The encoder is monotonic for |residual / peak| in [0, 1]. Find the exact
    // float at which each magnitude code begins for each quantizer mode, once,
    // then use comparisons in the hot loop. Searching float bit patterns
    // preserves the quantizer mapping without log1p() per sample.
    static const std::array<std::array<std::array<float, 127>, 9>, kQuantizerModeCount> thresholds = []
    {
        std::array<std::array<std::array<float, 127>, 9>, kQuantizerModeCount> table{};
        constexpr uint32_t oneBits = std::bit_cast<uint32_t>(1.0f);

        for (uint8_t quantizerMode = 0; quantizerMode < kQuantizerModeCount; ++quantizerMode)
        {
            for (uint8_t quantBits = kMinQuantBits; quantBits <= 8; ++quantBits)
            {
                const uint32_t maxMagnitude = (1u << (quantBits - 1u)) - 1u;
                for (uint32_t q = 1; q <= maxMagnitude; ++q)
                {
                    uint32_t low = 0u;
                    uint32_t high = oneBits;
                    while (low < high)
                    {
                        const uint32_t mid = low + ((high - low) >> 1u);
                        const float x = std::bit_cast<float>(mid);
                        if (quantizeResidualMagnitudeReference(x, maxMagnitude, quantizerMode) >= static_cast<int32_t>(q))
                            high = mid;
                        else
                            low = mid + 1u;
                    }
                    table[quantizerMode][quantBits][q - 1u] = std::bit_cast<float>(low);
                }
            }
        }
        return table;
    }();

    return thresholds;
}

inline uint8_t encodeResidualCodeFast(float residual,
    float residualPeak,
    uint8_t quantBits,
    const float* thresholds,
    uint32_t maxMagnitude)
{
    if (residualPeak <= kSilentPeakEpsilon)
        return 0;

    const float normalizedMagnitude = std::min(std::fabs(residual / residualPeak), 1.0f);
    const uint32_t magnitude = static_cast<uint32_t>(
        std::upper_bound(thresholds, thresholds + maxMagnitude, normalizedMagnitude) - thresholds);

    if (residual >= 0.0f || magnitude == 0u)
        return static_cast<uint8_t>(magnitude);

    return static_cast<uint8_t>((1u << quantBits) - magnitude);
}

uint8_t encodeResidualCode(float residual, uint8_t quantBits, uint8_t quantizerMode, float residualPeak)
{
    if (quantBits < kMinQuantBits || quantBits > 8 || quantizerMode >= kQuantizerModeCount)
        return 0;

    const uint32_t maxMagnitude = (1u << (quantBits - 1u)) - 1u;
    return encodeResidualCodeFast(
        residual,
        residualPeak,
        quantBits,
        getResidualEncodeThresholds()[quantizerMode][quantBits].data(),
        maxMagnitude);
}

const std::array<std::array<std::array<float, 256>, 9>, kQuantizerModeCount>& getResidualDecodeLut()
{
    static const std::array<std::array<std::array<float, 256>, 9>, kQuantizerModeCount> lut = []
    {
        std::array<std::array<std::array<float, 256>, 9>, kQuantizerModeCount> table{};
        for (uint8_t quantizerMode = 0; quantizerMode < kQuantizerModeCount; ++quantizerMode)
        {
            for (uint8_t quantBits = kMinQuantBits; quantBits <= 8; ++quantBits)
            {
                const uint32_t maxCode = (1u << quantBits) - 1u;
                const int32_t maxMagnitude = static_cast<int32_t>((1u << (quantBits - 1u)) - 1u);
                for (uint32_t code = 0; code <= maxCode; ++code)
                {
                    const int32_t quantized = decodeSignedResidualCode(static_cast<uint8_t>(code), quantBits);
                    table[quantizerMode][quantBits][code] = expandResidual(
                        static_cast<float>(quantized) / static_cast<float>(maxMagnitude),
                        quantizerMode);
                }
            }
        }
        return table;
    }();

    return lut;
}

inline float clampSampleFast(float sample)
{
    if (sample < -kSampleClamp)
        return -kSampleClamp;
    if (sample > kSampleClamp)
        return kSampleClamp;
    return sample;
}

uint32_t seedFramesForBlock(uint32_t framesInBlock)
{
    return std::min<uint32_t>(kSeedFramesPerBlock, framesInBlock);
}

int16_t quantizeSeedSample(float sample)
{
    sample = clampSampleFast(sample);
    const float scaled = sample < 0.0f ? sample * 32768.0f : sample * 32767.0f;
    const int32_t quantized = static_cast<int32_t>(std::lround(scaled));
    return static_cast<int16_t>(std::clamp<int32_t>(quantized, -32768, 32767));
}

float dequantizeSeedSample(int16_t sampleQ)
{
    if (sampleQ < 0)
        return static_cast<float>(sampleQ) / 32768.0f;
    return static_cast<float>(sampleQ) / 32767.0f;
}

void resetStateFromBlockSeeds(const audioStream& audio,
    uint32_t startFrame,
    uint32_t framesInBlock,
    std::vector<float>& statePrev1,
    std::vector<float>& statePrev2,
    std::vector<float>& statePrev3)
{
    statePrev1.assign(audio.channels, 0.0f);
    statePrev2.assign(audio.channels, 0.0f);
    statePrev3.assign(audio.channels, 0.0f);

    const uint32_t seedFrames = seedFramesForBlock(framesInBlock);
    const uint64_t startSample = static_cast<uint64_t>(startFrame) * audio.channels;
    for (uint32_t f = 0; f < seedFrames; ++f)
    {
        const uint64_t base = startSample + static_cast<uint64_t>(f) * audio.channels;
        for (uint8_t ch = 0; ch < audio.channels; ++ch)
        {
            const int16_t sampleQ = quantizeSeedSample(audio.sampleData[base + ch]);
            const float sample = dequantizeSeedSample(sampleQ);
            statePrev3[ch] = statePrev2[ch];
            statePrev2[ch] = statePrev1[ch];
            statePrev1[ch] = sample;
        }
    }
}

void appendBlockSeeds(std::vector<uint8_t>& out,
    const audioStream& audio,
    uint32_t startFrame,
    uint32_t framesInBlock)
{
    const uint32_t seedFrames = seedFramesForBlock(framesInBlock);
    const uint64_t startSample = static_cast<uint64_t>(startFrame) * audio.channels;
    for (uint32_t f = 0; f < seedFrames; ++f)
    {
        const uint64_t base = startSample + static_cast<uint64_t>(f) * audio.channels;
        for (uint8_t ch = 0; ch < audio.channels; ++ch)
            appendLE(out, quantizeSeedSample(audio.sampleData[base + ch]));
    }
}

inline float clampPredictFast(float sample)
{
    if (sample < -kSampleClamp)
        return -kSampleClamp;
    if (sample > kSampleClamp)
        return kSampleClamp;
    return sample;
}

inline float clampLegacyPredictFast(float sample)
{
    if (sample < -kResidualPeakRange)
        return -kResidualPeakRange;
    if (sample > kResidualPeakRange)
        return kResidualPeakRange;
    return sample;
}

template <bool LegacyPredictClamp>
inline float clampPredictForMode(float sample)
{
    if constexpr (LegacyPredictClamp)
        return clampLegacyPredictFast(sample);
    else
        return clampPredictFast(sample);
}

template <uint8_t Predictor, bool LegacyPredictClamp = false>
inline float predictSampleFast(float prev1, float prev2, float prev3)
{
    if constexpr (Predictor == 0)
        return 0.0f;
    else if constexpr (Predictor == 1)
        return prev1;
    else if constexpr (Predictor == 2)
        return clampPredictForMode<LegacyPredictClamp>(2.0f * prev1 - prev2);
    else if constexpr (Predictor == 3)
        return clampPredictForMode<LegacyPredictClamp>(1.5f * prev1 - 0.5f * prev2);
    else if constexpr (Predictor == 4)
        return clampPredictForMode<LegacyPredictClamp>(3.0f * prev1 - 3.0f * prev2 + prev3);
    else if constexpr (Predictor == 5)
        return clampPredictFast(prev1 + 0.25f * (prev1 - prev2));
    else if constexpr (Predictor == 6)
        return clampPredictFast(prev1 + 0.75f * (prev1 - prev2));
    else if constexpr (Predictor == 7)
        return clampPredictFast(0.75f * prev1 + 0.25f * prev2);
    else if constexpr (Predictor == 8)
        return clampPredictFast(0.9375f * prev1);
    else
    {
        const float slope = prev1 - prev2;
        const float prevSlope = prev2 - prev3;
        const float limit = std::max(0.03125f, 1.5f * std::fabs(prevSlope));
        return clampPredictFast(prev1 + std::clamp(slope, -limit, limit));
    }
}

template <uint8_t Predictor, bool LegacyPredictClamp = false>
bool decodeBlockSamples(PackedBitReader& payloadReader,
    uint8_t blockQuantBits,
    uint8_t quantizerMode,
    const float* residualPeak,
    uint8_t channels,
    uint32_t framesInBlock,
    float* prev1,
    float* prev2,
    float* prev3,
    float* output)
{
    const float* residualDecodeLut = getResidualDecodeLut()[quantizerMode][blockQuantBits].data();
    size_t outIndex = 0;

    for (uint32_t f = 0; f < framesInBlock; ++f)
    {
        for (uint8_t ch = 0; ch < channels; ++ch)
        {
            uint8_t code = 0;
            if (!payloadReader.read(blockQuantBits, code))
                return false;

            const float pred = predictSampleFast<Predictor, LegacyPredictClamp>(prev1[ch], prev2[ch], prev3[ch]);
            const float sample = clampSampleFast(pred + residualDecodeLut[code] * residualPeak[ch]);
            output[outIndex++] = sample;
            prev3[ch] = prev2[ch];
            prev2[ch] = prev1[ch];
            prev1[ch] = sample;
        }
    }

    return true;
}

template <uint8_t Predictor>
bool decodeBlockSamplesWithClampMode(PackedBitReader& payloadReader,
    uint8_t blockQuantBits,
    uint8_t quantizerMode,
    bool legacyPredictorClamp,
    const float* residualPeak,
    uint8_t channels,
    uint32_t framesInBlock,
    float* prev1,
    float* prev2,
    float* prev3,
    float* output)
{
    if (legacyPredictorClamp)
    {
        return decodeBlockSamples<Predictor, true>(
            payloadReader,
            blockQuantBits,
            quantizerMode,
            residualPeak,
            channels,
            framesInBlock,
            prev1,
            prev2,
            prev3,
            output);
    }

    return decodeBlockSamples<Predictor, false>(
        payloadReader,
        blockQuantBits,
        quantizerMode,
        residualPeak,
        channels,
        framesInBlock,
        prev1,
        prev2,
        prev3,
        output);
}

std::vector<PlannedBlock> buildFixedBlockPlan(uint64_t totalFrames, uint32_t blockSizeFrames)
{
    std::vector<PlannedBlock> plan;
    if (blockSizeFrames == 0)
        return plan;

    const uint32_t numberOfBlocks = static_cast<uint32_t>((totalFrames + blockSizeFrames - 1u) / blockSizeFrames);
    plan.reserve(numberOfBlocks);

    uint64_t frame = 0;
    while (frame < totalFrames)
    {
        const uint32_t framesInBlock = static_cast<uint32_t>(std::min<uint64_t>(blockSizeFrames, totalFrames - frame));
        plan.push_back({ static_cast<uint32_t>(frame), framesInBlock });
        frame += framesInBlock;
    }

    return plan;
}

struct ChannelScaleResult
{
    double distortion = std::numeric_limits<double>::infinity();
    float endPrev1 = 0.0f;
    float endPrev2 = 0.0f;
    float endPrev3 = 0.0f;
    float maxResidual = 0.0f;
};

template <uint8_t Predictor>
ChannelScaleResult evaluateChannelScale(const audioStream& audio,
    uint32_t startFrame,
    uint32_t framesInBlock,
    uint32_t firstFrameInBlock,
    uint8_t channel,
    uint8_t quantBits,
    uint8_t quantizerMode,
    float residualPeak,
    float startPrev1,
    float startPrev2,
    float startPrev3)
{
    ChannelScaleResult result{};
    result.distortion = 0.0;

    float prev1 = startPrev1;
    float prev2 = startPrev2;
    float prev3 = startPrev3;
    const float* input = audio.sampleData.data()
        + static_cast<size_t>(startFrame + firstFrameInBlock) * audio.channels
        + channel;
    const size_t stride = audio.channels;
    const float* residualDecodeLut = getResidualDecodeLut()[quantizerMode][quantBits].data();
    const float* encodeThresholds = getResidualEncodeThresholds()[quantizerMode][quantBits].data();
    const uint32_t maxMagnitude = (1u << (quantBits - 1u)) - 1u;

    for (uint32_t f = firstFrameInBlock; f < framesInBlock; ++f, input += stride)
    {
        const float inputSample = clampSampleFast(*input);
        const float pred = predictSampleFast<Predictor>(prev1, prev2, prev3);
        const float residual = inputSample - pred;
        result.maxResidual = std::max(result.maxResidual, std::fabs(residual));

        const uint8_t code = encodeResidualCodeFast(
            residual, residualPeak, quantBits, encodeThresholds, maxMagnitude);
        const float reconstructed = clampSampleFast(
            pred + residualDecodeLut[code] * residualPeak);
        const double error = static_cast<double>(reconstructed) - static_cast<double>(inputSample);
        result.distortion += error * error;

        prev3 = prev2;
        prev2 = prev1;
        prev1 = reconstructed;
    }

    result.endPrev1 = prev1;
    result.endPrev2 = prev2;
    result.endPrev3 = prev3;
    return result;
}

float sortedQuantile(const std::vector<float>& sortedValues, float quantile)
{
    if (sortedValues.empty())
        return 0.0f;

    const float position = quantile * static_cast<float>(sortedValues.size() - 1u);
    const size_t lower = static_cast<size_t>(position);
    const size_t upper = std::min(lower + 1u, sortedValues.size() - 1u);
    const float fraction = position - static_cast<float>(lower);
    return sortedValues[lower] + (sortedValues[upper] - sortedValues[lower]) * fraction;
}

void addPeakCandidate(std::vector<uint16_t>& candidates, float peak)
{
    const uint16_t peakQ = quantizeResidualPeak(peak);
    if (std::find(candidates.begin(), candidates.end(), peakQ) == candidates.end())
        candidates.push_back(peakQ);
}

template <uint8_t Predictor>
BlockCandidate evaluateBlockCandidate(const audioStream& audio,
    uint32_t startFrame,
    uint32_t framesInBlock,
    uint8_t quantBits,
    uint8_t quantizerMode,
    const std::vector<float>& startPrev1,
    const std::vector<float>& startPrev2,
    const std::vector<float>& startPrev3)
{
    BlockCandidate candidate{};
    candidate.predictor = Predictor;
    candidate.quantizerMode = quantizerMode;
    candidate.quantBits = quantBits;
    candidate.peakQ.assign(audio.channels, 0u);
    candidate.endPrev1 = startPrev1;
    candidate.endPrev2 = startPrev2;
    candidate.endPrev3 = startPrev3;
    candidate.distortion = 0.0;

    std::vector<float> analysisPrev1 = startPrev1;
    std::vector<float> analysisPrev2 = startPrev2;
    std::vector<float> analysisPrev3 = startPrev3;
    const uint32_t firstFrameInBlock = seedFramesForBlock(framesInBlock);
    const uint32_t residualFrames = framesInBlock - firstFrameInBlock;

    std::vector<std::vector<float>> residualMagnitudes(audio.channels);
    for (auto& values : residualMagnitudes)
        values.reserve(residualFrames);

    const uint64_t startSample = static_cast<uint64_t>(startFrame) * audio.channels;
    for (uint32_t f = firstFrameInBlock; f < framesInBlock; ++f)
    {
        const uint64_t base = startSample + static_cast<uint64_t>(f) * audio.channels;
        for (uint8_t ch = 0; ch < audio.channels; ++ch)
        {
            const float sample = std::clamp(audio.sampleData[base + ch], -kSampleClamp, kSampleClamp);
            const float pred = predictSampleFast<Predictor>(analysisPrev1[ch], analysisPrev2[ch], analysisPrev3[ch]);
            residualMagnitudes[ch].push_back(std::fabs(sample - pred));
            analysisPrev3[ch] = analysisPrev2[ch];
            analysisPrev2[ch] = analysisPrev1[ch];
            analysisPrev1[ch] = sample;
        }
    }

    static constexpr std::array<float, 7> kScaleQuantiles = {
        0.75f, 0.85f, 0.90f, 0.95f, 0.98f, 0.995f, 1.0f
    };

    for (uint8_t ch = 0; ch < audio.channels; ++ch)
    {
        std::vector<float>& magnitudes = residualMagnitudes[ch];
        std::sort(magnitudes.begin(), magnitudes.end());

        std::vector<uint16_t> peakCandidates;
        peakCandidates.reserve(24);
        addPeakCandidate(peakCandidates, 0.0f);
        for (const float quantile : kScaleQuantiles)
            addPeakCandidate(peakCandidates, sortedQuantile(magnitudes, quantile));

        const float openLoopMax = magnitudes.empty() ? 0.0f : magnitudes.back();
        addPeakCandidate(peakCandidates, openLoopMax * 1.10f);
        addPeakCandidate(peakCandidates, openLoopMax * 1.25f);

        uint16_t bestPeakQ = 0;
        ChannelScaleResult bestResult{};
        size_t evaluatedCount = 0;

        for (uint32_t refinement = 0; refinement < 3; ++refinement)
        {
            while (evaluatedCount < peakCandidates.size())
            {
                const uint16_t peakQ = peakCandidates[evaluatedCount++];
                const float decodedPeak = dequantizeResidualPeak(peakQ);
                const ChannelScaleResult result = evaluateChannelScale<Predictor>(
                    audio,
                    startFrame,
                    framesInBlock,
                    firstFrameInBlock,
                    ch,
                    quantBits,
                    quantizerMode,
                    decodedPeak,
                    startPrev1[ch],
                    startPrev2[ch],
                    startPrev3[ch]);

                if (result.distortion < bestResult.distortion)
                {
                    bestResult = result;
                    bestPeakQ = peakQ;
                }
            }

            if (refinement + 1u < 3u)
            {
                const float bestPeak = dequantizeResidualPeak(bestPeakQ);
                addPeakCandidate(peakCandidates, bestPeak * 0.90f);
                addPeakCandidate(peakCandidates, bestPeak * 1.10f);
                addPeakCandidate(peakCandidates, bestResult.maxResidual * 0.90f);
                addPeakCandidate(peakCandidates, bestResult.maxResidual);
                addPeakCandidate(peakCandidates, bestResult.maxResidual * 1.10f);
            }
        }

        candidate.peakQ[ch] = bestPeakQ;
        candidate.distortion += bestResult.distortion;
        candidate.endPrev1[ch] = bestResult.endPrev1;
        candidate.endPrev2[ch] = bestResult.endPrev2;
        candidate.endPrev3[ch] = bestResult.endPrev3;
    }

    return candidate;
}

void evaluateBlockCandidateByPredictor(uint8_t predictor,
    const audioStream& audio,
    uint32_t startFrame,
    uint32_t framesInBlock,
    uint8_t quantBits,
    uint8_t quantizerMode,
    const std::vector<float>& startPrev1,
    const std::vector<float>& startPrev2,
    const std::vector<float>& startPrev3,
    BlockCandidate& output)
{
    switch (predictor)
    {
    case 0:
        output = evaluateBlockCandidate<0>(audio, startFrame, framesInBlock, quantBits, quantizerMode, startPrev1, startPrev2, startPrev3);
        break;
    case 1:
        output = evaluateBlockCandidate<1>(audio, startFrame, framesInBlock, quantBits, quantizerMode, startPrev1, startPrev2, startPrev3);
        break;
    case 2:
        output = evaluateBlockCandidate<2>(audio, startFrame, framesInBlock, quantBits, quantizerMode, startPrev1, startPrev2, startPrev3);
        break;
    case 3:
        output = evaluateBlockCandidate<3>(audio, startFrame, framesInBlock, quantBits, quantizerMode, startPrev1, startPrev2, startPrev3);
        break;
    case 4:
        output = evaluateBlockCandidate<4>(audio, startFrame, framesInBlock, quantBits, quantizerMode, startPrev1, startPrev2, startPrev3);
        break;
    case 5:
        output = evaluateBlockCandidate<5>(audio, startFrame, framesInBlock, quantBits, quantizerMode, startPrev1, startPrev2, startPrev3);
        break;
    case 6:
        output = evaluateBlockCandidate<6>(audio, startFrame, framesInBlock, quantBits, quantizerMode, startPrev1, startPrev2, startPrev3);
        break;
    case 7:
        output = evaluateBlockCandidate<7>(audio, startFrame, framesInBlock, quantBits, quantizerMode, startPrev1, startPrev2, startPrev3);
        break;
    case 8:
        output = evaluateBlockCandidate<8>(audio, startFrame, framesInBlock, quantBits, quantizerMode, startPrev1, startPrev2, startPrev3);
        break;
    default:
        output = evaluateBlockCandidate<9>(audio, startFrame, framesInBlock, quantBits, quantizerMode, startPrev1, startPrev2, startPrev3);
        break;
    }
}

class BlockAnalysisPool
{
public:
    BlockAnalysisPool(const audioStream& audio, uint8_t quantBits, unsigned workerCount)
        : audio_(audio),
          quantBits_(quantBits),
          workerCount_(workerCount),
          startBarrier_(static_cast<std::ptrdiff_t>(workerCount + 1u)),
          finishBarrier_(static_cast<std::ptrdiff_t>(workerCount + 1u))
    {
        workers_.reserve(workerCount_);
        for (unsigned i = 0; i < workerCount_; ++i)
            workers_.emplace_back([this] { workerMain(); });
    }

    ~BlockAnalysisPool()
    {
        if (workerCount_ == 0)
            return;

        stopping_.store(true, std::memory_order_relaxed);
        startBarrier_.arrive_and_wait();
        for (std::thread& worker : workers_)
            worker.join();
    }

    BlockCandidate analyze(uint32_t startFrame,
        uint32_t framesInBlock,
        const std::vector<float>& startPrev1,
        const std::vector<float>& startPrev2,
        const std::vector<float>& startPrev3)
    {
        startFrame_ = startFrame;
        framesInBlock_ = framesInBlock;
        startPrev1_ = &startPrev1;
        startPrev2_ = &startPrev2;
        startPrev3_ = &startPrev3;

        if (workerCount_ == 0)
        {
            for (uint8_t job = 0; job < kAnalysisJobCount; ++job)
                evaluateOne(job);
        }
        else
        {
            nextJob_.store(0u, std::memory_order_relaxed);
            startBarrier_.arrive_and_wait();
            runAvailableJobs();
            finishBarrier_.arrive_and_wait();
        }

        BlockCandidate best = candidates_[0];
        for (uint8_t job = 1; job < kAnalysisJobCount; ++job)
        {
            if (candidates_[job].distortion < best.distortion)
                best = candidates_[job];
        }
        return best;
    }

private:
    void workerMain()
    {
        for (;;)
        {
            startBarrier_.arrive_and_wait();
            if (stopping_.load(std::memory_order_relaxed))
                return;

            runAvailableJobs();
            finishBarrier_.arrive_and_wait();
        }
    }

    void runAvailableJobs()
    {
        for (;;)
        {
            const unsigned job = nextJob_.fetch_add(1u, std::memory_order_relaxed);
            if (job >= kAnalysisJobCount)
                return;
            evaluateOne(static_cast<uint8_t>(job));
        }
    }

    void evaluateOne(uint8_t job)
    {
        const uint8_t predictor = static_cast<uint8_t>(job / kQuantizerModeCount);
        const uint8_t quantizerMode = static_cast<uint8_t>(job % kQuantizerModeCount);
        evaluateBlockCandidateByPredictor(
            predictor,
            audio_,
            startFrame_,
            framesInBlock_,
            quantBits_,
            quantizerMode,
            *startPrev1_,
            *startPrev2_,
            *startPrev3_,
            candidates_[job]);
    }

    const audioStream& audio_;
    uint8_t quantBits_ = 0;
    unsigned workerCount_ = 0;
    std::barrier<> startBarrier_;
    std::barrier<> finishBarrier_;
    std::vector<std::thread> workers_;
    std::atomic<unsigned> nextJob_{ 0u };
    std::atomic<bool> stopping_{ false };
    uint32_t startFrame_ = 0;
    uint32_t framesInBlock_ = 0;
    const std::vector<float>* startPrev1_ = nullptr;
    const std::vector<float>* startPrev2_ = nullptr;
    const std::vector<float>* startPrev3_ = nullptr;
    std::array<BlockCandidate, kAnalysisJobCount> candidates_{};
};

void encodeBlockPayload(const audioStream& audio,
    uint32_t startFrame,
    uint32_t framesInBlock,
    uint32_t firstFrameInBlock,
    uint8_t quantBits,
    uint8_t predictor,
    uint8_t quantizerMode,
    const std::vector<uint16_t>& peakQ,
    std::vector<float>& statePrev1,
    std::vector<float>& statePrev2,
    std::vector<float>& statePrev3,
    std::vector<uint8_t>& codes)
{
    const uint32_t residualFrames = framesInBlock - firstFrameInBlock;
    const uint32_t samplesInBlock = residualFrames * audio.channels;
    const uint64_t startSample = static_cast<uint64_t>(startFrame) * audio.channels;

    codes.clear();
    codes.reserve(samplesInBlock);

    std::vector<float> decodedPeak(audio.channels, 0.0f);
    for (uint8_t ch = 0; ch < audio.channels; ++ch)
        decodedPeak[ch] = dequantizeResidualPeak(peakQ[ch]);
    const float* residualDecodeLut = getResidualDecodeLut()[quantizerMode][quantBits].data();

    for (uint32_t f = firstFrameInBlock; f < framesInBlock; ++f)
    {
        const uint64_t base = startSample + static_cast<uint64_t>(f) * audio.channels;
        for (uint8_t ch = 0; ch < audio.channels; ++ch)
        {
            const float inputSample = std::clamp(audio.sampleData[base + ch], -kSampleClamp, kSampleClamp);
            const float pred = predictSample(predictor, statePrev1[ch], statePrev2[ch], statePrev3[ch]);
            const uint8_t code = encodeResidualCode(inputSample - pred, quantBits, quantizerMode, decodedPeak[ch]);
            const float reconResidual = residualDecodeLut[code] * decodedPeak[ch];
            const float outputSample = std::clamp(pred + reconResidual, -kSampleClamp, kSampleClamp);

            codes.push_back(code);

            statePrev3[ch] = statePrev2[ch];
            statePrev2[ch] = statePrev1[ch];
            statePrev1[ch] = outputSample;
        }
    }
}

} // namespace

std::vector<uint8_t> encodeCWV(audioStream& audio, uint32_t blockSizeFrames, uint8_t bitsPerSample)
{
    if (audio.channels < 1 || audio.sampleRate <= 0 || audio.totalPCMFrameCount <= 0)
    {
        printf("Error! Invalid audioStream metadata.\n");
        return {};
    }
    if (bitsPerSample < kMinQuantBits || bitsPerSample > 8)
    {
        printf("Error! bitsPerSample must be in [2, 8].\n");
        return {};
    }
    if (blockSizeFrames == 0)
    {
        printf("Error! blockSizeFrames must be > 0. Fixed block sizing is required.\n");
        return {};
    }

    const uint8_t channels = audio.channels;
    const uint64_t totalFrames = static_cast<uint64_t>(audio.totalPCMFrameCount);
    std::vector<PlannedBlock> plan = buildFixedBlockPlan(totalFrames, blockSizeFrames);
    if (plan.empty())
    {
        printf("Error! Failed to build a block plan.\n");
        return {};
    }

    const uint32_t numberOfBlocks = static_cast<uint32_t>(plan.size());
    printf("Using fixed-size block plan with %u blocks.\n", numberOfBlocks);
    printf("Analyzing %u blocks...\n", numberOfBlocks);
    const auto analysisStart = std::chrono::steady_clock::now();

    std::vector<BlockCandidate> selected(numberOfBlocks);
    std::vector<float> runningPrev1(channels, 0.0f);
    std::vector<float> runningPrev2(channels, 0.0f);
    std::vector<float> runningPrev3(channels, 0.0f);

    const unsigned hardwareThreads = std::thread::hardware_concurrency();
    const unsigned analysisWorkers = numberOfBlocks >= 64u && hardwareThreads > 1u
        ? std::min(4u, hardwareThreads - 1u)
        : 0u;
    if (analysisWorkers > 0u)
        printf("Analysis threads: %u\n", analysisWorkers + 1u);

    BlockAnalysisPool analysisPool(audio, bitsPerSample, analysisWorkers);

    unsigned lastPercent = 101;
    for (uint32_t b = 0; b < numberOfBlocks; ++b)
    {
        const unsigned percent = static_cast<unsigned>(((static_cast<uint64_t>(b) + 1u) * 100u) / numberOfBlocks);
        if (percent != lastPercent)
        {
            printf("\rAnalysis progress: %3u%% (%u/%u)", percent, b + 1u, numberOfBlocks);
            fflush(stdout);
            lastPercent = percent;
        }

        const uint32_t startFrame = plan[b].startFrame;
        const uint32_t framesInBlock = plan[b].frames;

        resetStateFromBlockSeeds(audio, startFrame, framesInBlock, runningPrev1, runningPrev2, runningPrev3);

        BlockCandidate best = analysisPool.analyze(
            startFrame,
            framesInBlock,
            runningPrev1,
            runningPrev2,
            runningPrev3);

        selected[b] = best;
    }
    printf("\n");
    const auto analysisEnd = std::chrono::steady_clock::now();
    printf("Analysis time: %.3f s\n", std::chrono::duration<double>(analysisEnd - analysisStart).count());

    uint64_t payloadBytes = 0;
    for (const PlannedBlock& block : plan)
    {
        const uint32_t seedFrames = seedFramesForBlock(block.frames);
        const uint32_t residualFrames = block.frames - seedFrames;
        payloadBytes += 1u
            + static_cast<uint64_t>(channels) * sizeof(uint16_t)
            + static_cast<uint64_t>(seedFrames) * channels * sizeof(int16_t)
            + calculateBitPackedSize(static_cast<size_t>(residualFrames) * channels, bitsPerSample);
    }

    std::vector<uint8_t> out;
    out.reserve(64 + static_cast<size_t>(payloadBytes));

    out.insert(out.end(), { 'C', 'W', 'V' });
    appendLE(out, audio.channels);
    appendLE(out, static_cast<uint32_t>(audio.sampleRate));
    appendLE(out, static_cast<int64_t>(audio.totalPCMFrameCount));
    appendLE(out, static_cast<uint32_t>(blockSizeFrames));
    appendLE(out, static_cast<uint32_t>(numberOfBlocks));

    appendLE(out, bitsPerSample);

    std::vector<uint8_t> codes;
    runningPrev1.assign(channels, 0.0f);
    runningPrev2.assign(channels, 0.0f);
    runningPrev3.assign(channels, 0.0f);

    printf("Encoding %u blocks...\n", numberOfBlocks);
    uint32_t lastEncodeProgressPercent = std::numeric_limits<uint32_t>::max();
    auto printEncodeProgress = [&](uint32_t completedBlocks)
    {
        const uint32_t percent = (completedBlocks * 100u) / numberOfBlocks;
        if (percent == lastEncodeProgressPercent)
            return;
        printf("\rEncoding progress: %3u%% (%u/%u blocks)", percent, completedBlocks, numberOfBlocks);
        fflush(stdout);
        lastEncodeProgressPercent = percent;
    };

    printEncodeProgress(0);
    for (uint32_t b = 0; b < numberOfBlocks; ++b)
    {
        const uint32_t startFrame = plan[b].startFrame;
        const uint32_t framesInBlock = plan[b].frames;
        const BlockCandidate& choice = selected[b];

        const uint8_t storedQuantizerMode = static_cast<uint8_t>((choice.quantizerMode + 1u) & 0x0Fu);
        const uint8_t packInfo = static_cast<uint8_t>(((choice.predictor & 0x0Fu) << 4) | storedQuantizerMode);
        out.push_back(packInfo);
        for (uint8_t ch = 0; ch < channels; ++ch)
            appendLE(out, choice.peakQ[ch]);

        appendBlockSeeds(out, audio, startFrame, framesInBlock);
        const uint32_t firstFrameInBlock = seedFramesForBlock(framesInBlock);
        resetStateFromBlockSeeds(audio, startFrame, framesInBlock, runningPrev1, runningPrev2, runningPrev3);

        encodeBlockPayload(audio,
            startFrame,
            framesInBlock,
            firstFrameInBlock,
            choice.quantBits,
            choice.predictor,
            choice.quantizerMode,
            choice.peakQ,
            runningPrev1,
            runningPrev2,
            runningPrev3,
            codes);

        const BitPack packed = packBitsFixed<uint8_t>(codes, choice.quantBits);
        out.insert(out.end(), packed.bytes.begin(), packed.bytes.end());

        printEncodeProgress(b + 1u);
    }

    if (numberOfBlocks > 0)
        printf("\n");

    printf("Encoding done. Output size: %s\n", printBytes(out.size()).c_str());
    return out;
}

int decodeCWV(const std::vector<uint8_t>& input, std::vector<float>& outputBuffer, CWVHeader* outHeader)
{
    if (input.size() < 3)
        return 1;

    size_t offset = 0;

    CWVHeader hdr{};
    hdr.magic[0] = input[offset + 0];
    hdr.magic[1] = input[offset + 1];
    hdr.magic[2] = input[offset + 2];
    offset += 3;

    if (!(hdr.magic[0] == 'C' && hdr.magic[1] == 'W' && hdr.magic[2] == 'V'))
    {
        printf("Error! Not a CWV file (bad magic).\n");
        return 1;
    }

    if (!readLE(input, offset, hdr.channels)) return 1;
    if (!readLE(input, offset, hdr.sampleRate)) return 1;
    if (!readLE(input, offset, hdr.totalPCMFrameCount)) return 1;
    if (!readLE(input, offset, hdr.blockSize)) return 1;
    if (!readLE(input, offset, hdr.numberOfBlocks)) return 1;

    if (!readLE(input, offset, hdr.quantBits)) return 1;

    if (hdr.channels < 1 || hdr.numberOfBlocks == 0 || hdr.sampleRate == 0 || hdr.totalPCMFrameCount <= 0)
    {
        printf("Error! Invalid CWV header.\n");
        return 1;
    }
    if (hdr.blockSize == 0)
    {
        printf("Error! Invalid fixed block size.\n");
        return 1;
    }
    if (hdr.quantBits < kMinQuantBits || hdr.quantBits > 8)
    {
        printf("Error! Invalid quantBits (%u).\n", hdr.quantBits);
        return 1;
    }

    std::vector<PlannedBlock> plan = buildFixedBlockPlan(static_cast<uint64_t>(hdr.totalPCMFrameCount), hdr.blockSize);
    if (plan.size() != hdr.numberOfBlocks)
    {
        printf("Error! Fixed block plan block count mismatch (header=%u, decoded=%zu).\n", hdr.numberOfBlocks, plan.size());
        return 1;
    }

    const uint64_t totalFrames = static_cast<uint64_t>(hdr.totalPCMFrameCount);
    const uint64_t totalSamples = totalFrames * hdr.channels;
    outputBuffer.assign(static_cast<size_t>(totalSamples), 0.0f);

    std::vector<float> prev1(hdr.channels, 0.0f);
    std::vector<float> prev2(hdr.channels, 0.0f);
    std::vector<float> prev3(hdr.channels, 0.0f);
    std::vector<float> residualPeak(hdr.channels, 0.0f);

    for (uint32_t b = 0; b < hdr.numberOfBlocks; ++b)
    {
        const uint32_t startFrame = plan[b].startFrame;
        const uint32_t framesInBlock = plan[b].frames;
        const uint64_t startSample = static_cast<uint64_t>(startFrame) * hdr.channels;

        if (offset + 1 > input.size())
        {
            printf("Error! Truncated CWV block header.\n");
            return 1;
        }
        const uint8_t packInfo = input[offset++];
        const uint8_t predictor = static_cast<uint8_t>(packInfo >> 4);
        const uint8_t storedQuantizerMode = static_cast<uint8_t>(packInfo & 0x0Fu);
        const bool legacyPredictorClamp = storedQuantizerMode == 0u;
        const uint8_t quantizerMode = legacyPredictorClamp ? 0u : static_cast<uint8_t>(storedQuantizerMode - 1u);
        const uint8_t blockQuantBits = hdr.quantBits;

        if (predictor > kMaxPredictor)
        {
            printf("Error! Invalid predictor type (%u).\n", predictor);
            return 1;
        }
        if (storedQuantizerMode > kMaxStoredQuantizerMode)
        {
            printf("Error! Invalid stored quantizer mode (%u).\n", storedQuantizerMode);
            return 1;
        }
        if (blockQuantBits < kMinQuantBits || blockQuantBits > 8)
        {
            printf("Error! Invalid block quantBits (%u).\n", blockQuantBits);
            return 1;
        }

        for (uint8_t ch = 0; ch < hdr.channels; ++ch)
        {
            uint16_t peakQ = 0;
            if (!readLE(input, offset, peakQ))
            {
                printf("Error! Truncated CWV residual scales.\n");
                return 1;
            }
            residualPeak[ch] = dequantizeResidualPeak(peakQ);
        }

        prev1.assign(hdr.channels, 0.0f);
        prev2.assign(hdr.channels, 0.0f);
        prev3.assign(hdr.channels, 0.0f);

        const uint32_t firstFrameInBlock = seedFramesForBlock(framesInBlock);
        float* const blockOutput = outputBuffer.data() + static_cast<size_t>(startSample);
        for (uint32_t f = 0; f < firstFrameInBlock; ++f)
        {
            for (uint8_t ch = 0; ch < hdr.channels; ++ch)
            {
                int16_t sampleQ = 0;
                if (!readLE(input, offset, sampleQ))
                {
                    printf("Error! Truncated CWV block seeds.\n");
                    return 1;
                }
                const float sample = dequantizeSeedSample(sampleQ);
                blockOutput[static_cast<size_t>(f) * hdr.channels + ch] = sample;
                prev3[ch] = prev2[ch];
                prev2[ch] = prev1[ch];
                prev1[ch] = sample;
            }
        }

        const uint32_t residualSamplesInBlock = (framesInBlock - firstFrameInBlock) * hdr.channels;
        const size_t payloadBytes = calculateBitPackedSize(residualSamplesInBlock, blockQuantBits);
        if (offset + payloadBytes > input.size())
        {
            printf("Error! Truncated CWV block payload.\n");
            return 1;
        }

        PackedBitReader payloadReader{ input.data() + offset, payloadBytes };
        offset += payloadBytes;

        float* const residualOutput = blockOutput + static_cast<size_t>(firstFrameInBlock) * hdr.channels;
        float* const prev1Data = prev1.data();
        float* const prev2Data = prev2.data();
        float* const prev3Data = prev3.data();
        const float* const residualPeakData = residualPeak.data();

        bool decodeOk = false;
        switch (predictor)
        {
        case 0:
            decodeOk = decodeBlockSamplesWithClampMode<0>(payloadReader, blockQuantBits, quantizerMode, legacyPredictorClamp, residualPeakData, hdr.channels, framesInBlock - firstFrameInBlock, prev1Data, prev2Data, prev3Data, residualOutput);
            break;
        case 1:
            decodeOk = decodeBlockSamplesWithClampMode<1>(payloadReader, blockQuantBits, quantizerMode, legacyPredictorClamp, residualPeakData, hdr.channels, framesInBlock - firstFrameInBlock, prev1Data, prev2Data, prev3Data, residualOutput);
            break;
        case 2:
            decodeOk = decodeBlockSamplesWithClampMode<2>(payloadReader, blockQuantBits, quantizerMode, legacyPredictorClamp, residualPeakData, hdr.channels, framesInBlock - firstFrameInBlock, prev1Data, prev2Data, prev3Data, residualOutput);
            break;
        case 3:
            decodeOk = decodeBlockSamplesWithClampMode<3>(payloadReader, blockQuantBits, quantizerMode, legacyPredictorClamp, residualPeakData, hdr.channels, framesInBlock - firstFrameInBlock, prev1Data, prev2Data, prev3Data, residualOutput);
            break;
        case 4:
            decodeOk = decodeBlockSamplesWithClampMode<4>(payloadReader, blockQuantBits, quantizerMode, legacyPredictorClamp, residualPeakData, hdr.channels, framesInBlock - firstFrameInBlock, prev1Data, prev2Data, prev3Data, residualOutput);
            break;
        case 5:
            decodeOk = decodeBlockSamplesWithClampMode<5>(payloadReader, blockQuantBits, quantizerMode, legacyPredictorClamp, residualPeakData, hdr.channels, framesInBlock - firstFrameInBlock, prev1Data, prev2Data, prev3Data, residualOutput);
            break;
        case 6:
            decodeOk = decodeBlockSamplesWithClampMode<6>(payloadReader, blockQuantBits, quantizerMode, legacyPredictorClamp, residualPeakData, hdr.channels, framesInBlock - firstFrameInBlock, prev1Data, prev2Data, prev3Data, residualOutput);
            break;
        case 7:
            decodeOk = decodeBlockSamplesWithClampMode<7>(payloadReader, blockQuantBits, quantizerMode, legacyPredictorClamp, residualPeakData, hdr.channels, framesInBlock - firstFrameInBlock, prev1Data, prev2Data, prev3Data, residualOutput);
            break;
        case 8:
            decodeOk = decodeBlockSamplesWithClampMode<8>(payloadReader, blockQuantBits, quantizerMode, legacyPredictorClamp, residualPeakData, hdr.channels, framesInBlock - firstFrameInBlock, prev1Data, prev2Data, prev3Data, residualOutput);
            break;
        case 9:
            decodeOk = decodeBlockSamplesWithClampMode<9>(payloadReader, blockQuantBits, quantizerMode, legacyPredictorClamp, residualPeakData, hdr.channels, framesInBlock - firstFrameInBlock, prev1Data, prev2Data, prev3Data, residualOutput);
            break;
        default:
            decodeOk = false;
            break;
        }

        if (!decodeOk)
        {
            printf("Error! Truncated CWV block payload.\n");
            return 1;
        }
    }

    if (outHeader)
        *outHeader = hdr;

    return 0;
}
