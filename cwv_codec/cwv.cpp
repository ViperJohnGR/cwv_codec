#include "cwv.hpp"
#include "helpers.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

namespace {

constexpr float kMaxGain = 100.0f;
constexpr bool kEnableAdaptiveBlockQuantization = true;
constexpr uint8_t kAdaptiveQuantRadius = 1; // explore nominalBits +/- 1
constexpr uint8_t kFlagAdaptiveQuantization = 0x80u;

uint8_t packGainDb(float gain)
{
    gain = std::clamp(gain, 1.0f, kMaxGain);
    const float maxDb = 20.0f * std::log10(kMaxGain);
    const float db = 20.0f * std::log10(gain);
    const float x = db / maxDb;
    return static_cast<uint8_t>(std::lround(x * 255.0f));
}

float unpackGainDb(uint8_t code)
{
    const float maxDb = 20.0f * std::log10(kMaxGain);
    const float db = (static_cast<float>(code) / 255.0f) * maxDb;
    return std::pow(10.0f, db / 20.0f);
}

uint8_t computePredictedCode(uint8_t predictor, uint32_t frameIndex, uint8_t prev1, uint8_t prev2, uint8_t bitsPerSample)
{
    const int qMax = static_cast<int>((1u << bitsPerSample) - 1u);

    if (predictor == 0 || frameIndex <= 1u)
        return prev1;

    const int predicted = 2 * static_cast<int>(prev1) - static_cast<int>(prev2);
    return static_cast<uint8_t>(std::clamp(predicted, 0, qMax));
}

uint8_t computeResidualPeakFromSource(const std::vector<uint8_t>& q, uint32_t framesInBlock, uint8_t channels, uint8_t bitsPerSample, uint8_t predictor)
{
    if (framesInBlock <= 1 || channels == 0)
        return 0;

    int maxAbsResidual = 0;
    for (uint8_t ch = 0; ch < channels; ++ch)
    {
        uint8_t prev2 = q[ch];
        uint8_t prev1 = q[ch];

        for (uint32_t f = 1; f < framesInBlock; ++f)
        {
            const uint32_t idx = f * channels + ch;
            const uint8_t cur = q[idx];
            const uint8_t pred = computePredictedCode(predictor, f, prev1, prev2, bitsPerSample);
            const int residual = static_cast<int>(cur) - static_cast<int>(pred);
            maxAbsResidual = std::max(maxAbsResidual, std::abs(residual));

            prev2 = prev1;
            prev1 = cur;
        }
    }

    return static_cast<uint8_t>(std::min(maxAbsResidual, 255));
}

uint8_t encodeScaledResidual(int residual, uint8_t packWidth, uint8_t residualPeak)
{
    if (packWidth < 2 || packWidth > 8)
        return 0;

    if (residualPeak == 0)
        return static_cast<uint8_t>(1u << (packWidth - 1u));

    const int levelAbsMax = 1 << (packWidth - 1u);
    const double scaled = static_cast<double>(residual) * static_cast<double>(levelAbsMax) / static_cast<double>(residualPeak);
    const int level = std::clamp(static_cast<int>(std::lround(scaled)), -levelAbsMax, levelAbsMax - 1);
    return static_cast<uint8_t>(level + levelAbsMax);
}

int decodeScaledResidual(uint8_t code, uint8_t packWidth, uint8_t residualPeak)
{
    if (packWidth < 2 || packWidth > 8 || residualPeak == 0)
        return 0;

    const int levelAbsMax = 1 << (packWidth - 1u);
    const int level = static_cast<int>(code) - levelAbsMax;
    if (level == 0)
        return 0;

    return static_cast<int>(std::lround(static_cast<double>(level) * static_cast<double>(residualPeak) / static_cast<double>(levelAbsMax)));
}

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

uint8_t quantizeNormalizedSample(float s, uint8_t bitsPerSample)
{
    const float qScale = static_cast<float>(std::pow(2.0, bitsPerSample - 1.0) - 0.5);
    const uint32_t qMax = (1u << bitsPerSample) - 1u;
    s = std::clamp(s, -1.0f, 1.0f);
    const int qi = static_cast<int>(std::lround((s + 1.0f) * qScale));
    return static_cast<uint8_t>(std::clamp(qi, 0, static_cast<int>(qMax)));
}

float dequantizeNormalizedCode(uint8_t q, uint8_t bitsPerSample)
{
    const float denom = static_cast<float>((1u << bitsPerSample) - 1u);
    return (static_cast<float>(q) * 2.0f / denom) - 1.0f;
}

struct DecodeTables
{
    float inverseGain[256]{};
    float dequant[9][256]{};
};

const DecodeTables& getDecodeTables()
{
    static const DecodeTables tables = []
    {
        DecodeTables t{};
        for (uint32_t code = 0; code < 256; ++code)
            t.inverseGain[code] = 1.0f / unpackGainDb(static_cast<uint8_t>(code));

        for (uint8_t bits = 1; bits <= 8; ++bits)
            for (uint32_t code = 0; code < 256; ++code)
                t.dequant[bits][code] = dequantizeNormalizedCode(static_cast<uint8_t>(code), bits);

        return t;
    }();

    return tables;
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

struct BlockCandidate
{
    uint8_t quantBits = 0;
    uint8_t predictor = 0;
    uint8_t packWidth = 0;
    uint8_t residualPeak = 0;
    uint32_t encodedBytes = 0;
    double distortion = 0.0;
};

struct PlannedBlock
{
    uint32_t startFrame = 0;
    uint32_t frames = 0;
};

void computeBlockGain(const audioStream& audio, uint64_t startSample, uint32_t framesInBlock, uint8_t channels,
    std::vector<float>& peak, std::vector<float>& gain, std::vector<uint8_t>& gainCode)
{
    peak.assign(channels, 0.0f);
    gain.assign(channels, 1.0f);
    gainCode.assign(channels, 0u);

    for (uint32_t f = 0; f < framesInBlock; ++f)
    {
        const uint64_t base = startSample + static_cast<uint64_t>(f) * channels;
        for (uint8_t ch = 0; ch < channels; ++ch)
            peak[ch] = std::max(peak[ch], std::fabs(audio.sampleData[base + ch]));
    }

    for (uint8_t ch = 0; ch < channels; ++ch)
    {
        float g = 1.0f;
        if (peak[ch] > 1e-12f)
            g = std::clamp(1.0f / peak[ch], std::numeric_limits<float>::min(), kMaxGain);
        gainCode[ch] = packGainDb(g);
        gain[ch] = unpackGainDb(gainCode[ch]);
    }
}

BlockCandidate evaluateScaledBlockCandidate(const audioStream& audio, uint32_t startFrame, uint32_t framesInBlock, uint8_t blockQuantBits,
    uint8_t predictor, uint8_t packWidth, const std::vector<float>& gain, const std::vector<uint8_t>& q)
{
    BlockCandidate candidate{};
    candidate.quantBits = blockQuantBits;
    candidate.predictor = predictor;
    candidate.packWidth = packWidth;

    const uint8_t channels = audio.channels;
    const uint32_t samplesInBlock = framesInBlock * channels;
    const uint32_t residualCount = (samplesInBlock > channels) ? (samplesInBlock - channels) : 0;
    const uint32_t qMax = (1u << blockQuantBits) - 1u;
    const uint64_t startSample = static_cast<uint64_t>(startFrame) * channels;

    candidate.residualPeak = computeResidualPeakFromSource(q, framesInBlock, channels, blockQuantBits, predictor);
    if (candidate.residualPeak == 0)
        candidate.packWidth = 0;

    candidate.encodedBytes = 1u
        + channels
        + 1u
        + static_cast<uint32_t>(calculateBitPackedSize(channels, blockQuantBits))
        + ((candidate.packWidth == 0 || residualCount == 0) ? 0u : static_cast<uint32_t>(calculateBitPackedSize(residualCount, candidate.packWidth)));

    std::vector<uint8_t> prev1(channels, 0u);
    std::vector<uint8_t> prev2(channels, 0u);
    candidate.distortion = 0.0;

    for (uint8_t ch = 0; ch < channels; ++ch)
    {
        const uint8_t seedCode = q[ch];
        prev1[ch] = seedCode;
        prev2[ch] = seedCode;
        const float reconstructed = dequantizeNormalizedCode(seedCode, blockQuantBits) / gain[ch];
        const double err = static_cast<double>(reconstructed) - static_cast<double>(audio.sampleData[startSample + ch]);
        candidate.distortion += err * err;
    }

    for (uint32_t f = 1; f < framesInBlock; ++f)
    {
        const uint64_t inBase = startSample + static_cast<uint64_t>(f) * channels;
        for (uint8_t ch = 0; ch < channels; ++ch)
        {
            const uint32_t idx = f * channels + ch;
            const uint8_t pred = computePredictedCode(candidate.predictor, f, prev1[ch], prev2[ch], blockQuantBits);
            const int residual = static_cast<int>(q[idx]) - static_cast<int>(pred);

            uint8_t residualCode = 0;
            int reconstructedResidual = 0;
            if (candidate.packWidth != 0)
            {
                residualCode = encodeScaledResidual(residual, candidate.packWidth, candidate.residualPeak);
                reconstructedResidual = decodeScaledResidual(residualCode, candidate.packWidth, candidate.residualPeak);
            }

            const uint8_t reconstructedCode = static_cast<uint8_t>(std::clamp(static_cast<int>(pred) + reconstructedResidual, 0, static_cast<int>(qMax)));
            prev2[ch] = prev1[ch];
            prev1[ch] = reconstructedCode;

            const float reconstructed = dequantizeNormalizedCode(reconstructedCode, blockQuantBits) / gain[ch];
            const double err = static_cast<double>(reconstructed) - static_cast<double>(audio.sampleData[inBase + ch]);
            candidate.distortion += err * err;
        }
    }

    return candidate;
}

BlockCandidate evaluateNominalBlock(const audioStream& audio, uint32_t startFrame, uint32_t framesInBlock, uint8_t bitsPerSample,
    std::vector<float>& peak, std::vector<float>& gain, std::vector<uint8_t>& gainCode, std::vector<uint8_t>& q)
{
    BlockCandidate bestCandidate{};
    bool haveBest = false;

    const uint8_t channels = audio.channels;
    const uint32_t samplesInBlock = framesInBlock * channels;
    const uint64_t startSample = static_cast<uint64_t>(startFrame) * channels;
    const uint8_t nominalPackWidth = (framesInBlock <= 1) ? 0u : std::max<uint8_t>(2u, bitsPerSample);

    computeBlockGain(audio, startSample, framesInBlock, channels, peak, gain, gainCode);

    q.resize(samplesInBlock);
    for (uint32_t f = 0; f < framesInBlock; ++f)
    {
        const uint64_t inBase = startSample + static_cast<uint64_t>(f) * channels;
        const uint32_t outBase = f * channels;
        for (uint8_t ch = 0; ch < channels; ++ch)
        {
            const float normalized = std::clamp(audio.sampleData[inBase + ch] * gain[ch], -1.0f, 1.0f);
            q[outBase + ch] = quantizeNormalizedSample(normalized, bitsPerSample);
        }
    }

    for (uint8_t predictor = 0; predictor <= 1; ++predictor)
    {
        const BlockCandidate candidate = evaluateScaledBlockCandidate(audio, startFrame, framesInBlock, bitsPerSample, predictor, nominalPackWidth, gain, q);
        if (!haveBest
            || candidate.distortion < bestCandidate.distortion
            || (candidate.distortion == bestCandidate.distortion && candidate.encodedBytes < bestCandidate.encodedBytes))
        {
            bestCandidate = candidate;
            haveBest = true;
        }
    }

    return bestCandidate;
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

} // namespace


std::vector<uint8_t> encodeCWV(audioStream& audio, uint32_t blockSizeFrames, uint8_t bitsPerSample, bool saveCompressed)
{
    if (audio.channels < 1 || audio.sampleRate <= 0 || audio.totalPCMFrameCount <= 0)
    {
        printf("Error! Invalid audioStream metadata.\n");
        return {};
    }
    if (bitsPerSample < 1 || bitsPerSample > 8)
    {
        printf("Error! bitsPerSample must be in [1, 8].\n");
        return {};
    }

    if (blockSizeFrames == 0)
    {
        printf("Error! blockSizeFrames must be > 0. Fixed block sizing is required.\n");
        return {};
    }

    const bool useAdaptiveBlockQuantization = kEnableAdaptiveBlockQuantization;
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

    std::vector<float> peak(channels, 0.0f);
    std::vector<float> gain(channels, 1.0f);
    std::vector<uint8_t> gainCode(channels, 0u);
    std::vector<uint8_t> q;
    q.reserve(static_cast<size_t>(plan.front().frames) * channels);

    std::vector<std::vector<BlockCandidate>> blockCandidates(numberOfBlocks);
    std::vector<BlockCandidate> selected(numberOfBlocks);
    uint64_t targetPayloadBytes = 0;

    printf("Analyzing %u blocks...\n", numberOfBlocks);
    unsigned lastPercent = 101;
    for (uint32_t b = 0; b < numberOfBlocks; ++b)
    {
        unsigned percent = (unsigned)(((uint64_t)(b + 1) * 100) / numberOfBlocks);
        const uint32_t startFrame = plan[b].startFrame;
        const uint32_t framesInBlock = plan[b].frames;
        const uint32_t samplesInBlock = framesInBlock * channels;
        const uint64_t startSample = static_cast<uint64_t>(startFrame) * channels;

        if (percent != lastPercent)
        {
            printf("\rProgress: %3u%% (%u/%u)", percent, b + 1, numberOfBlocks);
            fflush(stdout);
            lastPercent = percent;
        }

        computeBlockGain(audio, startSample, framesInBlock, channels, peak, gain, gainCode);

        const uint8_t minQuantBits = useAdaptiveBlockQuantization
            ? static_cast<uint8_t>(std::max<int>(1, static_cast<int>(bitsPerSample) - static_cast<int>(kAdaptiveQuantRadius)))
            : bitsPerSample;
        const uint8_t maxQuantBits = useAdaptiveBlockQuantization
            ? static_cast<uint8_t>(std::min<int>(8, static_cast<int>(bitsPerSample) + static_cast<int>(kAdaptiveQuantRadius)))
            : bitsPerSample;

        blockCandidates[b].reserve(useAdaptiveBlockQuantization ? static_cast<size_t>((maxQuantBits - minQuantBits + 1u) * 16u) : 16u);

        bool haveNominalSelection = false;
        for (uint8_t blockQuantBits = (useAdaptiveBlockQuantization ? minQuantBits : bitsPerSample);
             blockQuantBits <= (useAdaptiveBlockQuantization ? maxQuantBits : bitsPerSample);
             ++blockQuantBits)
        {
            q.resize(samplesInBlock);
            for (uint32_t f = 0; f < framesInBlock; ++f)
            {
                const uint64_t inBase = startSample + static_cast<uint64_t>(f) * channels;
                const uint32_t outBase = f * channels;
                for (uint8_t ch = 0; ch < channels; ++ch)
                {
                    const float normalized = std::clamp(audio.sampleData[inBase + ch] * gain[ch], -1.0f, 1.0f);
                    q[outBase + ch] = quantizeNormalizedSample(normalized, blockQuantBits);
                }
            }

            const uint8_t nominalPackWidth = (framesInBlock <= 1) ? 0u : static_cast<uint8_t>(std::max<int>(2, blockQuantBits));
            const uint8_t maxPackWidth = (framesInBlock <= 1) ? 0u : static_cast<uint8_t>(std::max<int>(2, blockQuantBits));

            for (uint8_t predictor = 0; predictor <= 1; ++predictor)
            {
                if (framesInBlock <= 1)
                {
                    const BlockCandidate candidate = evaluateScaledBlockCandidate(audio, startFrame, framesInBlock, blockQuantBits, predictor, 0u, gain, q);
                    blockCandidates[b].push_back(candidate);

                    if (blockQuantBits == bitsPerSample
                        && (!haveNominalSelection
                            || candidate.distortion < selected[b].distortion
                            || (candidate.distortion == selected[b].distortion && candidate.encodedBytes < selected[b].encodedBytes)))
                    {
                        selected[b] = candidate;
                        haveNominalSelection = true;
                    }
                    continue;
                }

                for (uint8_t packWidth = 2; packWidth <= maxPackWidth; ++packWidth)
                {
                    const BlockCandidate candidate = evaluateScaledBlockCandidate(audio, startFrame, framesInBlock, blockQuantBits, predictor, packWidth, gain, q);
                    blockCandidates[b].push_back(candidate);

                    if (blockQuantBits == bitsPerSample
                        && (packWidth == nominalPackWidth || candidate.packWidth == 0)
                        && (!haveNominalSelection
                            || candidate.distortion < selected[b].distortion
                            || (candidate.distortion == selected[b].distortion && candidate.encodedBytes < selected[b].encodedBytes)))
                    {
                        selected[b] = candidate;
                        haveNominalSelection = true;
                    }

                    if (candidate.packWidth == 0)
                        break;
                }
            }

            if (!useAdaptiveBlockQuantization)
                break;
        }

        if (!haveNominalSelection)
        {
            printf("\nError! Failed to select a nominal block candidate.\n");
            return {};
        }

        targetPayloadBytes += selected[b].encodedBytes;
    }
    printf("\n");

    if (useAdaptiveBlockQuantization)
    {
        auto chooseWithLambda = [&](double lambda, std::vector<BlockCandidate>& outSelection) -> uint64_t
        {
            uint64_t totalBytes = 0;
            outSelection.resize(numberOfBlocks);

            for (uint32_t b = 0; b < numberOfBlocks; ++b)
            {
                const auto& candidates = blockCandidates[b];
                const BlockCandidate* best = &candidates.front();
                double bestCost = best->distortion + lambda * static_cast<double>(best->encodedBytes);

                for (size_t i = 1; i < candidates.size(); ++i)
                {
                    const double cost = candidates[i].distortion + lambda * static_cast<double>(candidates[i].encodedBytes);
                    if (cost < bestCost
                        || (cost == bestCost && candidates[i].encodedBytes < best->encodedBytes)
                        || (cost == bestCost && candidates[i].encodedBytes == best->encodedBytes && candidates[i].distortion < best->distortion))
                    {
                        best = &candidates[i];
                        bestCost = cost;
                    }
                }

                outSelection[b] = *best;
                totalBytes += best->encodedBytes;
            }

            return totalBytes;
        };

        std::vector<BlockCandidate> bestSelection(numberOfBlocks);
        uint64_t bestSelectionBytes = chooseWithLambda(0.0, bestSelection);

        if (bestSelectionBytes > targetPayloadBytes)
        {
            double lo = 0.0;
            double hi = 1.0;
            std::vector<BlockCandidate> trial(numberOfBlocks);

            while (chooseWithLambda(hi, trial) > targetPayloadBytes)
                hi *= 2.0;

            for (int iter = 0; iter < 56; ++iter)
            {
                const double mid = 0.5 * (lo + hi);
                const uint64_t totalBytes = chooseWithLambda(mid, trial);
                if (totalBytes > targetPayloadBytes)
                    lo = mid;
                else
                    hi = mid;
            }

            bestSelectionBytes = chooseWithLambda(hi, bestSelection);
        }

        while (bestSelectionBytes < targetPayloadBytes)
        {
            const uint64_t remainingBudget = targetPayloadBytes - bestSelectionBytes;
            double bestBenefitPerByte = 0.0;
            int bestBlock = -1;
            BlockCandidate bestUpgrade{};

            for (uint32_t b = 0; b < numberOfBlocks; ++b)
            {
                const BlockCandidate& current = bestSelection[b];
                for (const BlockCandidate& candidate : blockCandidates[b])
                {
                    if (candidate.encodedBytes <= current.encodedBytes || candidate.distortion >= current.distortion)
                        continue;

                    const uint64_t extraBytes = static_cast<uint64_t>(candidate.encodedBytes - current.encodedBytes);
                    if (extraBytes > remainingBudget)
                        continue;

                    const double benefitPerByte = (current.distortion - candidate.distortion) / static_cast<double>(extraBytes);
                    if (benefitPerByte > bestBenefitPerByte)
                    {
                        bestBenefitPerByte = benefitPerByte;
                        bestBlock = static_cast<int>(b);
                        bestUpgrade = candidate;
                    }
                }
            }

            if (bestBlock < 0)
                break;

            bestSelectionBytes += static_cast<uint64_t>(bestUpgrade.encodedBytes - bestSelection[bestBlock].encodedBytes);
            bestSelection[bestBlock] = bestUpgrade;
        }

        selected = std::move(bestSelection);

        printf("Adaptive quantization target payload: %s, selected payload: %s\n",
            printBytes(targetPayloadBytes).c_str(),
            printBytes(bestSelectionBytes).c_str());
    }

    std::vector<uint8_t> out;
    out.reserve(64 + static_cast<size_t>(targetPayloadBytes));

    out.insert(out.end(), { 'C','W','V' });
    appendLE(out, audio.channels);
    appendLE(out, static_cast<uint32_t>(audio.sampleRate));
    appendLE(out, static_cast<sf_count_t>(audio.totalPCMFrameCount));
    appendLE(out, static_cast<uint32_t>(blockSizeFrames));
    appendLE(out, static_cast<uint32_t>(numberOfBlocks));

    uint8_t rawQuantFlags = static_cast<uint8_t>(bitsPerSample & 0x7Fu);
    if (useAdaptiveBlockQuantization)
        rawQuantFlags |= kFlagAdaptiveQuantization;
    appendLE(out, rawQuantFlags);

    FILE* cmprFile = nullptr;
    if (saveCompressed)
        openFile(&cmprFile, "compressed", "wb");

    std::vector<uint8_t> seed(channels, 0);
    std::vector<uint8_t> residual;

    std::vector<float> norm;
    if (cmprFile != nullptr)
        norm.reserve(static_cast<size_t>(plan.front().frames) * channels);

    printf("Encoding %u blocks...\n", numberOfBlocks);

    uint32_t lastEncodeProgressPercent = std::numeric_limits<uint32_t>::max();
    auto printEncodeProgress = [&](uint32_t completedBlocks)
    {
        if (numberOfBlocks == 0)
            return;

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
        const uint32_t samplesInBlock = framesInBlock * channels;
        const uint64_t startSample = static_cast<uint64_t>(startFrame) * channels;
        const uint8_t blockQuantBits = selected[b].quantBits;
        const uint8_t predictor = selected[b].predictor;
        const uint32_t seedCount = channels;
        const uint32_t residualCount = (samplesInBlock > seedCount) ? (samplesInBlock - seedCount) : 0;
        computeBlockGain(audio, startSample, framesInBlock, channels, peak, gain, gainCode);

        q.resize(samplesInBlock);
        for (uint32_t f = 0; f < framesInBlock; ++f)
        {
            const uint64_t inBase = startSample + static_cast<uint64_t>(f) * channels;
            const uint32_t outBase = f * channels;
            for (uint8_t ch = 0; ch < channels; ++ch)
            {
                const float normalized = std::clamp(audio.sampleData[inBase + ch] * gain[ch], -1.0f, 1.0f);
                q[outBase + ch] = quantizeNormalizedSample(normalized, blockQuantBits);
            }
        }

        std::copy_n(q.begin(), seedCount, seed.begin());

        const uint8_t packWidth = selected[b].packWidth;
        const uint8_t residualPeak = selected[b].residualPeak;
        residual.resize((packWidth == 0) ? 0u : residualCount);
        if (packWidth != 0 && residualCount > 0)
        {
            std::vector<uint8_t> prev1State(channels, 0u);
            std::vector<uint8_t> prev2State(channels, 0u);
            for (uint8_t ch = 0; ch < channels; ++ch)
            {
                prev1State[ch] = q[ch];
                prev2State[ch] = q[ch];
            }

            for (uint32_t f = 1; f < framesInBlock; ++f)
            {
                for (uint8_t ch = 0; ch < channels; ++ch)
                {
                    const uint32_t idx = f * channels + ch;
                    const uint8_t pred = computePredictedCode(predictor, f, prev1State[ch], prev2State[ch], blockQuantBits);
                    const int residualValue = static_cast<int>(q[idx]) - static_cast<int>(pred);
                    const uint8_t residualCode = encodeScaledResidual(residualValue, packWidth, residualPeak);
                    const int reconstructedResidual = decodeScaledResidual(residualCode, packWidth, residualPeak);
                    const uint8_t reconstructedCode = static_cast<uint8_t>(std::clamp(static_cast<int>(pred) + reconstructedResidual, 0, static_cast<int>((1u << blockQuantBits) - 1u)));
                    const uint32_t rIdx = (f - 1u) * channels + ch;
                    residual[rIdx] = residualCode;

                    prev2State[ch] = prev1State[ch];
                    prev1State[ch] = reconstructedCode;
                }
            }
        }

        uint8_t packInfo = 0;
        if (useAdaptiveBlockQuantization)
        {
            const uint8_t blockMode = static_cast<uint8_t>(((predictor & 0x01u) << 3) | ((blockQuantBits - 1u) & 0x07u));
            packInfo = static_cast<uint8_t>((blockMode << 4) | (packWidth & 0x0Fu));
        }
        else
        {
            packInfo = static_cast<uint8_t>((predictor << 4) | (packWidth & 0x0F));
        }

        out.push_back(packInfo);
        for (uint8_t ch = 0; ch < channels; ++ch)
            out.push_back(gainCode[ch]);
        out.push_back(residualPeak);

        const BitPack seedPacked = packBitsFixed<uint8_t>(seed, blockQuantBits);
        out.insert(out.end(), seedPacked.bytes.begin(), seedPacked.bytes.end());

        if (packWidth != 0 && residualCount > 0)
        {
            const BitPack residualPacked = packBitsFixed<uint8_t>(residual, packWidth);
            out.insert(out.end(), residualPacked.bytes.begin(), residualPacked.bytes.end());
        }

        if (cmprFile != nullptr)
        {
            norm.resize(samplesInBlock);
            for (uint32_t f = 0; f < framesInBlock; ++f)
            {
                const uint64_t inBase = startSample + static_cast<uint64_t>(f) * channels;
                const uint32_t outBase = f * channels;
                for (uint8_t ch = 0; ch < channels; ++ch)
                    norm[outBase + ch] = std::clamp(audio.sampleData[inBase + ch] * gain[ch], -1.0f, 1.0f);
            }
            fwrite(norm.data(), sizeof(float), samplesInBlock, cmprFile);
        }

        printEncodeProgress(b + 1u);
    }

    if (numberOfBlocks > 0)
        printf("\n");

    if (cmprFile != nullptr)
        fclose(cmprFile);

    printf("Encoding done. Output size: %s\n", printBytes(out.size()).c_str());
    return out;
}


int decodeCWV(const std::vector<uint8_t>& input, std::vector<float>& outputBuffer, CWVHeader* outHeader)
{
    if (input.size() < 3)
        return 1;

    size_t offset = 0;

    CWVHeader hdr{};
    if (offset + 3 > input.size())
        return 1;

    hdr.magic[0] = input[offset + 0];
    hdr.magic[1] = input[offset + 1];
    hdr.magic[2] = input[offset + 2];
    offset += 3;

    const bool isCWV = (hdr.magic[0] == 'C' && hdr.magic[1] == 'W' && hdr.magic[2] == 'V');
    if (!isCWV)
    {
        printf("Error! Not a CWV file (bad magic).\n");
        return 1;
    }

    if (!readLE(input, offset, hdr.channels)) return 1;
    if (!readLE(input, offset, hdr.sampleRate)) return 1;
    if (!readLE(input, offset, hdr.totalPCMFrameCount)) return 1;
    if (!readLE(input, offset, hdr.blockSize)) return 1;
    if (!readLE(input, offset, hdr.numberOfBlocks)) return 1;

    uint8_t rawQuantFlags = 0;
    if (!readLE(input, offset, rawQuantFlags)) return 1;

    hdr.adaptiveQuantization = (rawQuantFlags & kFlagAdaptiveQuantization) != 0;
    hdr.quantBits = static_cast<uint8_t>(rawQuantFlags & 0x7Fu);

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

    if (hdr.quantBits < 1 || hdr.quantBits > 8)
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

    const DecodeTables& tables = getDecodeTables();
    const uint8_t channels = hdr.channels;
    std::vector<float> channelScale(channels, 1.0f);
    std::vector<uint8_t> prev1(channels, 0u);
    std::vector<uint8_t> prev2(channels, 0u);

    for (uint32_t b = 0; b < hdr.numberOfBlocks; ++b)
    {
        const uint32_t startFrame = plan[b].startFrame;
        const uint32_t framesInBlock = plan[b].frames;
        const uint64_t startSample = static_cast<uint64_t>(startFrame) * channels;

        if (offset + 1 > input.size())
            return 1;
        const uint8_t packInfo = input[offset++];

        uint8_t predictor = 0;
        uint8_t blockQuantBits = hdr.quantBits;
        const uint8_t packWidth = static_cast<uint8_t>(packInfo & 0x0F);

        if (hdr.adaptiveQuantization)
        {
            const uint8_t blockMode = static_cast<uint8_t>(packInfo >> 4);
            predictor = static_cast<uint8_t>(blockMode >> 3);
            blockQuantBits = static_cast<uint8_t>((blockMode & 0x07u) + 1u);
        }
        else
        {
            predictor = static_cast<uint8_t>(packInfo >> 4);
        }

        if (packWidth > 8)
        {
            printf("Error! Invalid block packWidth (%u).\n", packWidth);
            return 1;
        }
        if (predictor > 1)
        {
            printf("Error! Invalid predictor type (%u).\n", predictor);
            return 1;
        }
        if (blockQuantBits < 1 || blockQuantBits > 8)
        {
            printf("Error! Invalid block quantBits (%u).\n", blockQuantBits);
            return 1;
        }

        if (offset + channels > input.size())
            return 1;
        for (uint8_t ch = 0; ch < channels; ++ch)
            channelScale[ch] = tables.inverseGain[input[offset++]];

        if (offset + 1 > input.size())
        {
            printf("Error! Truncated CWV residual scale.\n");
            return 1;
        }
        const uint8_t residualPeak = input[offset++];

        const size_t seedBytes = calculateBitPackedSize(channels, blockQuantBits);
        if (offset + seedBytes > input.size())
        {
            printf("Error! Truncated CWV seed samples.\n");
            return 1;
        }

        PackedBitReader seedReader{ input.data() + offset, seedBytes };
        offset += seedBytes;

        const float* dequant = tables.dequant[blockQuantBits];
        const size_t firstFrameBase = static_cast<size_t>(startSample);
        for (uint8_t ch = 0; ch < channels; ++ch)
        {
            uint8_t code = 0;
            if (!seedReader.read(blockQuantBits, code))
            {
                printf("Error! Truncated CWV seed samples.\n");
                return 1;
            }

            prev1[ch] = code;
            prev2[ch] = code;
            outputBuffer[firstFrameBase + ch] = dequant[code] * channelScale[ch];
        }

        if (framesInBlock <= 1)
            continue;

        if (packWidth == 0 && residualPeak != 0)
        {
            printf("Error! Invalid CWV residual block scale/width combination.\n");
            return 1;
        }

        const uint32_t residualCount = (framesInBlock - 1u) * channels;
        const size_t payloadBytes = (packWidth == 0) ? 0u : calculateBitPackedSize(residualCount, packWidth);
        if (offset + payloadBytes > input.size())
        {
            printf("Error! Truncated CWV block payload.\n");
            return 1;
        }

        PackedBitReader payloadReader{ input.data() + offset, payloadBytes };
        offset += payloadBytes;

        const int qMax = static_cast<int>((1u << blockQuantBits) - 1u);
        for (uint32_t f = 1; f < framesInBlock; ++f)
        {
            const size_t outBase = static_cast<size_t>(startSample + static_cast<uint64_t>(f) * channels);
            for (uint8_t ch = 0; ch < channels; ++ch)
            {
                uint8_t residualCode = 0;
                if (packWidth != 0)
                {
                    if (!payloadReader.read(packWidth, residualCode))
                    {
                        printf("Error! Truncated CWV block payload.\n");
                        return 1;
                    }
                }

                const uint8_t pred = computePredictedCode(predictor, f, prev1[ch], prev2[ch], blockQuantBits);
                const int residual = decodeScaledResidual(residualCode, packWidth, residualPeak);
                const uint8_t code = static_cast<uint8_t>(std::clamp(static_cast<int>(pred) + residual, 0, qMax));
                prev2[ch] = prev1[ch];
                prev1[ch] = code;
                outputBuffer[outBase + ch] = dequant[code] * channelScale[ch];
            }
        }
    }

    if (outHeader)
        *outHeader = hdr;

    return 0;
}
