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

constexpr bool kBlockQuantBitsInPackInfo = true;
constexpr uint8_t kFlagBlockQuantMetadata = 0x80u;
constexpr float kResidualPeakRange = 8.0f;
constexpr float kMuLaw = 127.0f;
constexpr float kSampleClamp = 1.0f;
constexpr float kSilentPeakEpsilon = 1e-12f;

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

float predictSample(uint8_t predictor, float prev1, float prev2)
{
    switch (predictor)
    {
    case 0:
        return 0.0f;
    case 1:
        return prev1;
    case 2:
        return std::clamp(2.0f * prev1 - prev2, -kResidualPeakRange, kResidualPeakRange);
    default:
        return 0.0f;
    }
}

uint16_t quantizeResidualPeak(float peak)
{
    peak = std::clamp(peak, 0.0f, kResidualPeakRange);
    return static_cast<uint16_t>(std::lround((peak / kResidualPeakRange) * 65535.0f));
}

float dequantizeResidualPeak(uint16_t peakQ)
{
    return (static_cast<float>(peakQ) / 65535.0f) * kResidualPeakRange;
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

uint8_t encodeResidualCode(float residual, uint8_t quantBits, float residualPeak)
{
    if (quantBits < 1 || quantBits > 8)
        return 0;

    const uint32_t maxCode = (1u << quantBits) - 1u;
    if (residualPeak <= kSilentPeakEpsilon || maxCode == 0)
        return 0;

    const float normalized = std::clamp(residual / residualPeak, -1.0f, 1.0f);
    const float companded = compandMuLaw(normalized);
    const float mapped = (companded * 0.5f + 0.5f) * static_cast<float>(maxCode);
    return static_cast<uint8_t>(std::clamp<int>(static_cast<int>(std::lround(mapped)), 0, static_cast<int>(maxCode)));
}

float decodeResidualCode(uint8_t code, uint8_t quantBits, float residualPeak)
{
    if (quantBits < 1 || quantBits > 8)
        return 0.0f;

    const uint32_t maxCode = (1u << quantBits) - 1u;
    if (residualPeak <= kSilentPeakEpsilon || maxCode == 0)
        return 0.0f;

    const float companded = (static_cast<float>(code) * 2.0f / static_cast<float>(maxCode)) - 1.0f;
    return expandMuLaw(companded) * residualPeak;
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

BlockCandidate evaluateBlockCandidate(const audioStream& audio,
    uint32_t startFrame,
    uint32_t framesInBlock,
    uint8_t quantBits,
    uint8_t predictor,
    const std::vector<float>& startPrev1,
    const std::vector<float>& startPrev2)
{
    BlockCandidate candidate{};
    candidate.predictor = predictor;
    candidate.quantBits = quantBits;
    candidate.peakQ.assign(audio.channels, 0u);
    candidate.endPrev1 = startPrev1;
    candidate.endPrev2 = startPrev2;

    std::vector<float> analysisPrev1 = startPrev1;
    std::vector<float> analysisPrev2 = startPrev2;
    std::vector<float> residualPeak(audio.channels, 0.0f);

    const uint64_t startSample = static_cast<uint64_t>(startFrame) * audio.channels;
    for (uint32_t f = 0; f < framesInBlock; ++f)
    {
        const uint64_t base = startSample + static_cast<uint64_t>(f) * audio.channels;
        for (uint8_t ch = 0; ch < audio.channels; ++ch)
        {
            const float sample = std::clamp(audio.sampleData[base + ch], -kSampleClamp, kSampleClamp);
            const float pred = predictSample(predictor, analysisPrev1[ch], analysisPrev2[ch]);
            residualPeak[ch] = std::max(residualPeak[ch], std::fabs(sample - pred));
            analysisPrev2[ch] = analysisPrev1[ch];
            analysisPrev1[ch] = sample;
        }
    }

    std::vector<float> decodedPeak(audio.channels, 0.0f);
    for (uint8_t ch = 0; ch < audio.channels; ++ch)
    {
        candidate.peakQ[ch] = quantizeResidualPeak(residualPeak[ch]);
        decodedPeak[ch] = dequantizeResidualPeak(candidate.peakQ[ch]);
    }

    std::vector<float> reconPrev1 = startPrev1;
    std::vector<float> reconPrev2 = startPrev2;
    candidate.distortion = 0.0;

    for (uint32_t f = 0; f < framesInBlock; ++f)
    {
        const uint64_t base = startSample + static_cast<uint64_t>(f) * audio.channels;
        for (uint8_t ch = 0; ch < audio.channels; ++ch)
        {
            const float inputSample = std::clamp(audio.sampleData[base + ch], -kSampleClamp, kSampleClamp);
            const float pred = predictSample(predictor, reconPrev1[ch], reconPrev2[ch]);
            const uint8_t code = encodeResidualCode(inputSample - pred, quantBits, decodedPeak[ch]);
            const float reconResidual = decodeResidualCode(code, quantBits, decodedPeak[ch]);
            const float reconstructed = std::clamp(pred + reconResidual, -kSampleClamp, kSampleClamp);
            const double err = static_cast<double>(reconstructed) - static_cast<double>(inputSample);
            candidate.distortion += err * err;
            reconPrev2[ch] = reconPrev1[ch];
            reconPrev1[ch] = reconstructed;
        }
    }

    candidate.endPrev1 = std::move(reconPrev1);
    candidate.endPrev2 = std::move(reconPrev2);
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
    std::vector<uint8_t>& codes,
    std::vector<float>* reconstructed = nullptr)
{
    const uint32_t samplesInBlock = framesInBlock * audio.channels;
    const uint64_t startSample = static_cast<uint64_t>(startFrame) * audio.channels;

    codes.clear();
    codes.reserve(samplesInBlock);

    std::vector<float> decodedPeak(audio.channels, 0.0f);
    for (uint8_t ch = 0; ch < audio.channels; ++ch)
        decodedPeak[ch] = dequantizeResidualPeak(peakQ[ch]);

    if (reconstructed != nullptr)
    {
        reconstructed->clear();
        reconstructed->reserve(samplesInBlock);
    }

    for (uint32_t f = 0; f < framesInBlock; ++f)
    {
        const uint64_t base = startSample + static_cast<uint64_t>(f) * audio.channels;
        for (uint8_t ch = 0; ch < audio.channels; ++ch)
        {
            const float inputSample = std::clamp(audio.sampleData[base + ch], -kSampleClamp, kSampleClamp);
            const float pred = predictSample(predictor, statePrev1[ch], statePrev2[ch]);
            const uint8_t code = encodeResidualCode(inputSample - pred, quantBits, decodedPeak[ch]);
            const float reconResidual = decodeResidualCode(code, quantBits, decodedPeak[ch]);
            const float outputSample = std::clamp(pred + reconResidual, -kSampleClamp, kSampleClamp);

            codes.push_back(code);
            if (reconstructed != nullptr)
                reconstructed->push_back(outputSample);

            statePrev2[ch] = statePrev1[ch];
            statePrev1[ch] = outputSample;
        }
    }
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

        BlockCandidate best = evaluateBlockCandidate(audio, startFrame, framesInBlock, bitsPerSample, 0u, runningPrev1, runningPrev2);
        for (uint8_t predictor = 1; predictor <= 2; ++predictor)
        {
            const BlockCandidate candidate = evaluateBlockCandidate(audio, startFrame, framesInBlock, bitsPerSample, predictor, runningPrev1, runningPrev2);
            if (candidate.distortion < best.distortion)
                best = candidate;
        }

        selected[b] = best;
        runningPrev1 = best.endPrev1;
        runningPrev2 = best.endPrev2;
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
    appendLE(out, static_cast<sf_count_t>(audio.totalPCMFrameCount));
    appendLE(out, static_cast<uint32_t>(blockSizeFrames));
    appendLE(out, static_cast<uint32_t>(numberOfBlocks));

    uint8_t rawQuantFlags = static_cast<uint8_t>(bitsPerSample & 0x7Fu);
    if (kBlockQuantBitsInPackInfo)
        rawQuantFlags |= kFlagBlockQuantMetadata;
    appendLE(out, rawQuantFlags);

    FILE* cmprFile = nullptr;
    if (saveCompressed)
        openFile(&cmprFile, "compressed", "wb");

    std::vector<uint8_t> codes;
    std::vector<float> debugReconstructed;
    runningPrev1.assign(channels, 0.0f);
    runningPrev2.assign(channels, 0.0f);

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

        const uint8_t packInfo = static_cast<uint8_t>(((choice.predictor & 0x0Fu) << 4) | ((choice.quantBits - 1u) & 0x0Fu));
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
            codes,
            (cmprFile != nullptr) ? &debugReconstructed : nullptr);

        const BitPack packed = packBitsFixed<uint8_t>(codes, choice.quantBits);
        out.insert(out.end(), packed.bytes.begin(), packed.bytes.end());

        if (cmprFile != nullptr && !debugReconstructed.empty())
            fwrite(debugReconstructed.data(), sizeof(float), debugReconstructed.size(), cmprFile);

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

    uint8_t rawQuantFlags = 0;
    if (!readLE(input, offset, rawQuantFlags)) return 1;

    hdr.adaptiveQuantization = (rawQuantFlags & kFlagBlockQuantMetadata) != 0;
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

    std::vector<float> prev1(hdr.channels, 0.0f);
    std::vector<float> prev2(hdr.channels, 0.0f);
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
        const uint8_t blockQuantBits = hdr.adaptiveQuantization
            ? static_cast<uint8_t>((packInfo & 0x0Fu) + 1u)
            : hdr.quantBits;

        if (predictor > 2)
        {
            printf("Error! Invalid predictor type (%u).\n", predictor);
            return 1;
        }
        if (blockQuantBits < 1 || blockQuantBits > 8)
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

        for (uint32_t f = 0; f < framesInBlock; ++f)
        {
            const uint64_t outBase = startSample + static_cast<uint64_t>(f) * hdr.channels;
            for (uint8_t ch = 0; ch < hdr.channels; ++ch)
            {
                uint8_t code = 0;
                if (!payloadReader.read(blockQuantBits, code))
                {
                    printf("Error! Truncated CWV block payload.\n");
                    return 1;
                }

                const float pred = predictSample(predictor, prev1[ch], prev2[ch]);
                const float residual = decodeResidualCode(code, blockQuantBits, residualPeak[ch]);
                const float sample = std::clamp(pred + residual, -kSampleClamp, kSampleClamp);
                outputBuffer[static_cast<size_t>(outBase + ch)] = sample;
                prev2[ch] = prev1[ch];
                prev1[ch] = sample;
            }
        }
    }

    if (outHeader)
        *outHeader = hdr;

    return 0;
}
