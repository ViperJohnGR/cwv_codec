#include "cwv.hpp"
#include "helpers.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

namespace {

constexpr float kResidualPeakRange = 8.0f;
constexpr float kResidualPeakMin = 1.0e-9f;
constexpr float kMuLaw = 127.0f;
constexpr float kSampleClamp = 1.0f;
constexpr float kSilentPeakEpsilon = 1e-12f;
constexpr uint8_t kMinQuantBits = 2;

struct PlannedBlock
{
    uint32_t startFrame = 0;
    uint32_t frames = 0;
};

struct BlockCandidate
{
    uint8_t predictor = 0;
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

float predictSample(uint8_t predictor, float prev1, float prev2, float prev3)
{
    switch (predictor)
    {
    case 0:
        return 0.0f;
    case 1:
        return prev1;
    case 2:
        return std::clamp(2.0f * prev1 - prev2, -kResidualPeakRange, kResidualPeakRange);
    case 3:
        return std::clamp(1.5f * prev1 - 0.5f * prev2, -kResidualPeakRange, kResidualPeakRange);
    case 4:
        return std::clamp(3.0f * prev1 - 3.0f * prev2 + prev3, -kResidualPeakRange, kResidualPeakRange);
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

float compandMuLaw(float x)
{
    x = std::clamp(x, -1.0f, 1.0f);
    const float ax = std::fabs(x);
    if (ax <= 0.0f)
        return 0.0f;

    const float denom = std::log1p(kMuLaw);
    return std::copysign(std::log1p(kMuLaw * ax) / denom, x);
}

float expandMuLaw(float y)
{
    y = std::clamp(y, -1.0f, 1.0f);
    const float ay = std::fabs(y);
    if (ay <= 0.0f)
        return 0.0f;

    const float x = (std::pow(1.0f + kMuLaw, ay) - 1.0f) / kMuLaw;
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

uint8_t encodeResidualCode(float residual, uint8_t quantBits, float residualPeak)
{
    if (quantBits < kMinQuantBits || quantBits > 8 || residualPeak <= kSilentPeakEpsilon)
        return 0;

    const int32_t maxMagnitude = static_cast<int32_t>((1u << (quantBits - 1u)) - 1u);
    const float normalized = std::clamp(residual / residualPeak, -1.0f, 1.0f);
    const float companded = compandMuLaw(normalized);
    const int32_t quantized = std::clamp(
        static_cast<int32_t>(std::lround(companded * static_cast<float>(maxMagnitude))),
        -maxMagnitude,
        maxMagnitude);
    const int32_t codeRange = static_cast<int32_t>(1u << quantBits);
    return static_cast<uint8_t>(quantized < 0 ? quantized + codeRange : quantized);
}

const std::array<std::array<float, 256>, 9>& getResidualDecodeLut()
{
    static const std::array<std::array<float, 256>, 9> lut = []
    {
        std::array<std::array<float, 256>, 9> table{};
        for (uint8_t quantBits = kMinQuantBits; quantBits <= 8; ++quantBits)
        {
            const uint32_t maxCode = (1u << quantBits) - 1u;
            const int32_t maxMagnitude = static_cast<int32_t>((1u << (quantBits - 1u)) - 1u);
            for (uint32_t code = 0; code <= maxCode; ++code)
            {
                const int32_t quantized = decodeSignedResidualCode(static_cast<uint8_t>(code), quantBits);
                table[quantBits][code] = expandMuLaw(
                    static_cast<float>(quantized) / static_cast<float>(maxMagnitude));
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

inline float clampPredictFast(float sample)
{
    if (sample < -kResidualPeakRange)
        return -kResidualPeakRange;
    if (sample > kResidualPeakRange)
        return kResidualPeakRange;
    return sample;
}

template <uint8_t Predictor>
inline float predictSampleFast(float prev1, float prev2, float prev3)
{
    if constexpr (Predictor == 0)
        return 0.0f;
    else if constexpr (Predictor == 1)
        return prev1;
    else if constexpr (Predictor == 2)
        return clampPredictFast(2.0f * prev1 - prev2);
    else if constexpr (Predictor == 3)
        return clampPredictFast(1.5f * prev1 - 0.5f * prev2);
    else
        return clampPredictFast(3.0f * prev1 - 3.0f * prev2 + prev3);
}

template <uint8_t Predictor>
bool decodeBlockSamples(PackedBitReader& payloadReader,
    uint8_t blockQuantBits,
    const float* residualPeak,
    uint8_t channels,
    uint32_t framesInBlock,
    float* prev1,
    float* prev2,
    float* prev3,
    float* output)
{
    const float* residualDecodeLut = getResidualDecodeLut()[blockQuantBits].data();
    size_t outIndex = 0;

    for (uint32_t f = 0; f < framesInBlock; ++f)
    {
        for (uint8_t ch = 0; ch < channels; ++ch)
        {
            uint8_t code = 0;
            if (!payloadReader.read(blockQuantBits, code))
                return false;

            const float pred = predictSampleFast<Predictor>(prev1[ch], prev2[ch], prev3[ch]);
            const float sample = clampSampleFast(pred + residualDecodeLut[code] * residualPeak[ch]);
            output[outIndex++] = sample;
            prev3[ch] = prev2[ch];
            prev2[ch] = prev1[ch];
            prev1[ch] = sample;
        }
    }

    return true;
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

ChannelScaleResult evaluateChannelScale(const audioStream& audio,
    uint32_t startFrame,
    uint32_t framesInBlock,
    uint8_t channel,
    uint8_t quantBits,
    uint8_t predictor,
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
    const uint64_t startSample = static_cast<uint64_t>(startFrame) * audio.channels;
    const float* residualDecodeLut = getResidualDecodeLut()[quantBits].data();

    for (uint32_t f = 0; f < framesInBlock; ++f)
    {
        const uint64_t sampleIndex = startSample + static_cast<uint64_t>(f) * audio.channels + channel;
        const float inputSample = std::clamp(audio.sampleData[sampleIndex], -kSampleClamp, kSampleClamp);
        const float pred = predictSample(predictor, prev1, prev2, prev3);
        const float residual = inputSample - pred;
        result.maxResidual = std::max(result.maxResidual, std::fabs(residual));

        const uint8_t code = encodeResidualCode(residual, quantBits, residualPeak);
        const float reconstructed = std::clamp(
            pred + residualDecodeLut[code] * residualPeak,
            -kSampleClamp,
            kSampleClamp);
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

BlockCandidate evaluateBlockCandidate(const audioStream& audio,
    uint32_t startFrame,
    uint32_t framesInBlock,
    uint8_t quantBits,
    uint8_t predictor,
    const std::vector<float>& startPrev1,
    const std::vector<float>& startPrev2,
    const std::vector<float>& startPrev3)
{
    BlockCandidate candidate{};
    candidate.predictor = predictor;
    candidate.quantBits = quantBits;
    candidate.peakQ.assign(audio.channels, 0u);
    candidate.endPrev1 = startPrev1;
    candidate.endPrev2 = startPrev2;
    candidate.endPrev3 = startPrev3;
    candidate.distortion = 0.0;

    std::vector<float> analysisPrev1 = startPrev1;
    std::vector<float> analysisPrev2 = startPrev2;
    std::vector<float> analysisPrev3 = startPrev3;
    std::vector<std::vector<float>> residualMagnitudes(audio.channels);
    for (auto& values : residualMagnitudes)
        values.reserve(framesInBlock);

    const uint64_t startSample = static_cast<uint64_t>(startFrame) * audio.channels;
    for (uint32_t f = 0; f < framesInBlock; ++f)
    {
        const uint64_t base = startSample + static_cast<uint64_t>(f) * audio.channels;
        for (uint8_t ch = 0; ch < audio.channels; ++ch)
        {
            const float sample = std::clamp(audio.sampleData[base + ch], -kSampleClamp, kSampleClamp);
            const float pred = predictSample(predictor, analysisPrev1[ch], analysisPrev2[ch], analysisPrev3[ch]);
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
                const ChannelScaleResult result = evaluateChannelScale(
                    audio,
                    startFrame,
                    framesInBlock,
                    ch,
                    quantBits,
                    predictor,
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

void encodeBlockPayload(const audioStream& audio,
    uint32_t startFrame,
    uint32_t framesInBlock,
    uint8_t quantBits,
    uint8_t predictor,
    const std::vector<uint16_t>& peakQ,
    std::vector<float>& statePrev1,
    std::vector<float>& statePrev2,
    std::vector<float>& statePrev3,
    std::vector<uint8_t>& codes)
{
    const uint32_t samplesInBlock = framesInBlock * audio.channels;
    const uint64_t startSample = static_cast<uint64_t>(startFrame) * audio.channels;

    codes.clear();
    codes.reserve(samplesInBlock);

    std::vector<float> decodedPeak(audio.channels, 0.0f);
    for (uint8_t ch = 0; ch < audio.channels; ++ch)
        decodedPeak[ch] = dequantizeResidualPeak(peakQ[ch]);
    const float* residualDecodeLut = getResidualDecodeLut()[quantBits].data();

    for (uint32_t f = 0; f < framesInBlock; ++f)
    {
        const uint64_t base = startSample + static_cast<uint64_t>(f) * audio.channels;
        for (uint8_t ch = 0; ch < audio.channels; ++ch)
        {
            const float inputSample = std::clamp(audio.sampleData[base + ch], -kSampleClamp, kSampleClamp);
            const float pred = predictSample(predictor, statePrev1[ch], statePrev2[ch], statePrev3[ch]);
            const uint8_t code = encodeResidualCode(inputSample - pred, quantBits, decodedPeak[ch]);
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

    std::vector<BlockCandidate> selected(numberOfBlocks);
    std::vector<float> runningPrev1(channels, 0.0f);
    std::vector<float> runningPrev2(channels, 0.0f);
    std::vector<float> runningPrev3(channels, 0.0f);

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

        BlockCandidate best = evaluateBlockCandidate(audio, startFrame, framesInBlock, bitsPerSample, 0u, runningPrev1, runningPrev2, runningPrev3);
        for (uint8_t predictor = 1; predictor <= 4; ++predictor)
        {
            const BlockCandidate candidate = evaluateBlockCandidate(audio, startFrame, framesInBlock, bitsPerSample, predictor, runningPrev1, runningPrev2, runningPrev3);
            if (candidate.distortion < best.distortion)
                best = candidate;
        }

        selected[b] = best;
        runningPrev1 = best.endPrev1;
        runningPrev2 = best.endPrev2;
        runningPrev3 = best.endPrev3;
    }
    printf("\n");

    uint64_t payloadBytes = 0;
    for (const PlannedBlock& block : plan)
        payloadBytes += 1u + static_cast<uint64_t>(channels) * sizeof(uint16_t) + calculateBitPackedSize(static_cast<size_t>(block.frames) * channels, bitsPerSample);

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

        const uint8_t packInfo = static_cast<uint8_t>((choice.predictor & 0x0Fu) << 4);
        out.push_back(packInfo);
        for (uint8_t ch = 0; ch < channels; ++ch)
            appendLE(out, choice.peakQ[ch]);

        encodeBlockPayload(audio,
            startFrame,
            framesInBlock,
            choice.quantBits,
            choice.predictor,
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
        const uint8_t blockQuantBits = hdr.quantBits;

        if (predictor > 4)
        {
            printf("Error! Invalid predictor type (%u).\n", predictor);
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

        const uint32_t samplesInBlock = framesInBlock * hdr.channels;
        const size_t payloadBytes = calculateBitPackedSize(samplesInBlock, blockQuantBits);
        if (offset + payloadBytes > input.size())
        {
            printf("Error! Truncated CWV block payload.\n");
            return 1;
        }

        PackedBitReader payloadReader{ input.data() + offset, payloadBytes };
        offset += payloadBytes;

        float* const blockOutput = outputBuffer.data() + static_cast<size_t>(startSample);
        float* const prev1Data = prev1.data();
        float* const prev2Data = prev2.data();
        float* const prev3Data = prev3.data();
        const float* const residualPeakData = residualPeak.data();

        bool decodeOk = false;
        switch (predictor)
        {
        case 0:
            decodeOk = decodeBlockSamples<0>(payloadReader, blockQuantBits, residualPeakData, hdr.channels, framesInBlock, prev1Data, prev2Data, prev3Data, blockOutput);
            break;
        case 1:
            decodeOk = decodeBlockSamples<1>(payloadReader, blockQuantBits, residualPeakData, hdr.channels, framesInBlock, prev1Data, prev2Data, prev3Data, blockOutput);
            break;
        case 2:
            decodeOk = decodeBlockSamples<2>(payloadReader, blockQuantBits, residualPeakData, hdr.channels, framesInBlock, prev1Data, prev2Data, prev3Data, blockOutput);
            break;
        case 3:
            decodeOk = decodeBlockSamples<3>(payloadReader, blockQuantBits, residualPeakData, hdr.channels, framesInBlock, prev1Data, prev2Data, prev3Data, blockOutput);
            break;
        case 4:
            decodeOk = decodeBlockSamples<4>(payloadReader, blockQuantBits, residualPeakData, hdr.channels, framesInBlock, prev1Data, prev2Data, prev3Data, blockOutput);
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
