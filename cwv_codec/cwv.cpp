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

uint8_t diff_u8_mod(uint32_t a, uint32_t b, uint8_t bitsPerSample)
{
    if (!(bitsPerSample >= 1 && bitsPerSample <= 8))
        return 0;

    const uint32_t mask = (1u << bitsPerSample) - 1u;
    a &= mask;
    b &= mask;
    return static_cast<uint8_t>((a - b) & mask);
}

uint8_t decode_u8_mod(uint32_t b, uint32_t diff, uint8_t bitsPerSample)
{
    if (!(bitsPerSample >= 1 && bitsPerSample <= 8))
        return 0;

    const uint32_t mask = (1u << bitsPerSample) - 1u;
    b &= mask;
    diff &= mask;
    return static_cast<uint8_t>((b + diff) & mask);
}

constexpr std::uint8_t zigzag_encode_int8(std::int8_t value, std::uint8_t bits)
{
    if (bits == 0 || bits > 8) return 0;

    const std::uint16_t mask = (bits == 8) ? 0xFFu : static_cast<std::uint16_t>((1u << bits) - 1u);
    std::uint16_t u = static_cast<std::uint16_t>(static_cast<std::uint8_t>(value)) & mask;

    const std::uint16_t signbit = static_cast<std::uint16_t>(1u << (bits - 1u));
    if (u & signbit) u |= static_cast<std::uint16_t>(~mask); // sign-extend

    const std::int16_t s = static_cast<std::int16_t>(u);
    const std::uint16_t zz = (static_cast<std::uint16_t>(s) << 1)
        ^ static_cast<std::uint16_t>(-static_cast<std::int16_t>(s < 0));

    return static_cast<std::uint8_t>(zz & mask);
}

constexpr std::int8_t zigzag_decode_int8(std::uint8_t zigzag, std::uint8_t bits)
{
    if (bits == 0 || bits > 8) return 0;

    const std::uint16_t mask = (bits == 8) ? 0xFFu : static_cast<std::uint16_t>((1u << bits) - 1u);
    const std::uint16_t zz = static_cast<std::uint16_t>(zigzag) & mask;

    std::uint16_t raw = (zz >> 1) ^ static_cast<std::uint16_t>(-static_cast<std::int16_t>(zz & 1u));
    raw &= mask;

    const std::uint16_t signbit = static_cast<std::uint16_t>(1u << (bits - 1u));
    if (raw & signbit) raw |= static_cast<std::uint16_t>(~mask); // sign-extend

    return static_cast<std::int8_t>(static_cast<std::int16_t>(raw));
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

uint32_t computeBestPredictorAndWidth(const std::vector<uint8_t>& q, uint32_t framesInBlock, uint8_t channels, uint8_t bitsPerSample, uint8_t& outPredictor, uint8_t& outPackWidth)
{
    const uint32_t samplesInBlock = framesInBlock * channels;
    const uint32_t residualCount = (samplesInBlock > channels) ? (samplesInBlock - channels) : 0;

    if (residualCount == 0)
    {
        outPredictor = 0;
        outPackWidth = 1;
        return 1u + channels + static_cast<uint32_t>(calculateBitPackedSize(channels, bitsPerSample));
    }

    const uint32_t mask = (1u << bitsPerSample) - 1u;

    auto measurePredictor = [&](uint8_t predictor) -> uint8_t
    {
        uint32_t maxv = 0;
        for (uint8_t ch = 0; ch < channels; ++ch)
        {
            uint8_t prev2 = q[ch];
            uint8_t prev1 = q[ch];

            for (uint32_t f = 1; f < framesInBlock; ++f)
            {
                const uint32_t idx = f * channels + ch;
                const uint8_t cur = q[idx];
                const uint8_t pred =
                    (predictor == 0 || f == 1)
                    ? prev1
                    : static_cast<uint8_t>((2u * prev1 - prev2) & mask);

                const uint8_t diff = diff_u8_mod(cur, pred, bitsPerSample);
                const uint8_t zz = zigzag_encode_int8(static_cast<int8_t>(diff), bitsPerSample);
                maxv = std::max<uint32_t>(maxv, zz);

                prev2 = prev1;
                prev1 = cur;
            }
        }

        return static_cast<uint8_t>(std::max<uint32_t>(1, std::bit_width(maxv)));
    };

    const uint8_t width0 = measurePredictor(0);
    const uint8_t width1 = measurePredictor(1);

    outPredictor = (width1 < width0) ? 1u : 0u;
    outPackWidth = (outPredictor == 0) ? width0 : width1;

    return 1u
        + channels
        + static_cast<uint32_t>(calculateBitPackedSize(channels, bitsPerSample))
        + static_cast<uint32_t>(calculateBitPackedSize(residualCount, outPackWidth));
}

struct BlockCandidate
{
    uint8_t quantBits = 0;
    uint8_t predictor = 0;
    uint8_t packWidth = 1;
    uint32_t encodedBytes = 0;
    double distortion = 0.0;
};

int openBinaryWrite(FILE** f, const char* path)
{
#if defined(_WIN32)
    return fopen_s(f, path, "wb");
#else
    *f = std::fopen(path, "wb");
    return (*f != nullptr) ? 0 : 1;
#endif
}

} // namespace


std::vector<uint8_t> encodeCWV(audioStream& audio, uint32_t blockSizeFrames, uint8_t bitsPerSample, bool saveCompressed)
{
    if (audio.channels < 1 || audio.sampleRate <= 0 || audio.totalPCMFrameCount <= 0)
    {
        printf("Error! Invalid audioStream metadata.\n");
        return {};
    }
    if (blockSizeFrames == 0)
    {
        printf("Error! blockSizeFrames must be > 0.\n");
        return {};
    }
    if (bitsPerSample < 1 || bitsPerSample > 8)
    {
        printf("Error! bitsPerSample must be in [1,8].\n");
        return {};
    }

    const auto totalFrames = static_cast<uint64_t>(audio.totalPCMFrameCount);
    const auto totalSamples = static_cast<uint64_t>(audio.totalPCMFrameCount) * audio.channels;
    if (audio.sampleData.size() != totalSamples)
    {
        printf("Error! audio.sampleData.size() is %llu. expected %llu\n",
            static_cast<unsigned long long>(audio.sampleData.size()),
            static_cast<unsigned long long>(totalSamples));
        return {};
    }

    const uint32_t numberOfBlocks = static_cast<uint32_t>((totalFrames + blockSizeFrames - 1) / blockSizeFrames);
    const bool useAdaptiveBlockQuantization = kEnableAdaptiveBlockQuantization;
    const uint8_t minQuantBits = static_cast<uint8_t>(std::max<int>(1, bitsPerSample - static_cast<int>(kAdaptiveQuantRadius)));
    const uint8_t maxQuantBits = static_cast<uint8_t>(std::min<int>(8, bitsPerSample + static_cast<int>(kAdaptiveQuantRadius)));

    // Scratch buffers shared across analysis and emission.
    const uint8_t channels = audio.channels;
    std::vector<float> peak(channels, 0.0f);
    std::vector<float> gain(channels, 1.0f);
    std::vector<uint8_t> gainCode(channels, 0);
    std::vector<uint8_t> q;
    q.reserve(static_cast<size_t>(blockSizeFrames) * channels);

    std::vector<std::vector<BlockCandidate>> blockCandidates(numberOfBlocks);
    std::vector<BlockCandidate> selected(numberOfBlocks);
    uint64_t targetPayloadBytes = 0;

    printf("Analyzing %u blocks...\n", numberOfBlocks);

    for (uint32_t b = 0; b < numberOfBlocks; ++b)
    {
        const uint64_t startFrame = static_cast<uint64_t>(b) * blockSizeFrames;
        const uint32_t framesInBlock = static_cast<uint32_t>(std::min<uint64_t>(blockSizeFrames, totalFrames - startFrame));
        const uint32_t samplesInBlock = framesInBlock * channels;
        const uint64_t startSample = startFrame * channels;

        std::fill(peak.begin(), peak.end(), 0.0f);
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

            // Important: quantize the gain here too, so the encoder uses the exact same
            // scale factor the decoder will reconstruct from gainCode.
            gainCode[ch] = packGainDb(g);
            gain[ch] = unpackGainDb(gainCode[ch]);
        }

        blockCandidates[b].reserve(useAdaptiveBlockQuantization ? static_cast<size_t>(maxQuantBits - minQuantBits + 1u) : 1u);

        for (uint8_t blockQuantBits = (useAdaptiveBlockQuantization ? minQuantBits : bitsPerSample);
             blockQuantBits <= (useAdaptiveBlockQuantization ? maxQuantBits : bitsPerSample);
             ++blockQuantBits)
        {
            q.resize(samplesInBlock);

            double distortion = 0.0;
            for (uint32_t f = 0; f < framesInBlock; ++f)
            {
                const uint64_t inBase = startSample + static_cast<uint64_t>(f) * channels;
                const uint32_t outBase = f * channels;
                for (uint8_t ch = 0; ch < channels; ++ch)
                {
                    const float normalized = std::clamp(audio.sampleData[inBase + ch] * gain[ch], -1.0f, 1.0f);
                    const uint8_t code = quantizeNormalizedSample(normalized, blockQuantBits);
                    q[outBase + ch] = code;

                    const float reconstructed = dequantizeNormalizedCode(code, blockQuantBits) / gain[ch];
                    const double err = static_cast<double>(reconstructed) - static_cast<double>(audio.sampleData[inBase + ch]);
                    distortion += err * err;
                }
            }

            BlockCandidate candidate{};
            candidate.quantBits = blockQuantBits;
            candidate.distortion = distortion;
            candidate.encodedBytes = computeBestPredictorAndWidth(q, framesInBlock, channels, blockQuantBits, candidate.predictor, candidate.packWidth);
            blockCandidates[b].push_back(candidate);

            if (blockQuantBits == bitsPerSample)
            {
                selected[b] = candidate;
                targetPayloadBytes += candidate.encodedBytes;
            }

            if (!useAdaptiveBlockQuantization)
                break;
        }
    }

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

        // Spend any leftover budget on the most beneficial upgrades.
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

    // Rough reserve: header + chosen payload
    std::vector<uint8_t> out;
    out.reserve(64 + static_cast<size_t>(targetPayloadBytes));

    // Header
    out.insert(out.end(), { 'C','W','V' });
    appendLE(out, audio.channels);
    appendLE(out, static_cast<uint32_t>(audio.sampleRate));
    appendLE(out, static_cast<sf_count_t>(audio.totalPCMFrameCount));
    appendLE(out, static_cast<uint32_t>(blockSizeFrames));
    appendLE(out, static_cast<uint32_t>(numberOfBlocks));
    appendLE(out, static_cast<uint8_t>(useAdaptiveBlockQuantization ? (bitsPerSample | 0x80u) : bitsPerSample));

    // For debug: save normalized audio (float) as a raw file, like the old code.
    FILE* cmprFile = nullptr;
    if (saveCompressed)
        openBinaryWrite(&cmprFile, "compressed");

    std::vector<uint8_t> seed(channels, 0);
    std::vector<uint8_t> residual;
    if (blockSizeFrames > 0)
        residual.reserve(static_cast<size_t>(blockSizeFrames - 1u) * channels);

    std::vector<float> norm;
    if (cmprFile != nullptr)
        norm.reserve(static_cast<size_t>(blockSizeFrames) * channels);

    printf("Encoding %u blocks...\n", numberOfBlocks);

    for (uint32_t b = 0; b < numberOfBlocks; ++b)
    {
        const uint64_t startFrame = static_cast<uint64_t>(b) * blockSizeFrames;
        const uint32_t framesInBlock = static_cast<uint32_t>(std::min<uint64_t>(blockSizeFrames, totalFrames - startFrame));
        const uint32_t samplesInBlock = framesInBlock * channels;
        const uint64_t startSample = startFrame * channels;
        const uint8_t blockQuantBits = selected[b].quantBits;
        const uint8_t predictor = selected[b].predictor;
        const uint32_t seedCount = channels;
        const uint32_t residualCount = (samplesInBlock > seedCount) ? (samplesInBlock - seedCount) : 0;
        const uint32_t mask = (1u << blockQuantBits) - 1u;

        std::fill(peak.begin(), peak.end(), 0.0f);
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

        uint8_t packWidth = 1;
        residual.resize(residualCount);
        if (residualCount > 0)
        {
            uint32_t maxv = 0;
            for (uint8_t ch = 0; ch < channels; ++ch)
            {
                uint8_t prev2 = q[ch];
                uint8_t prev1 = q[ch];

                for (uint32_t f = 1; f < framesInBlock; ++f)
                {
                    const uint32_t idx = f * channels + ch;
                    const uint8_t cur = q[idx];
                    const uint8_t pred =
                        (predictor == 0 || f == 1)
                        ? prev1
                        : static_cast<uint8_t>((2u * prev1 - prev2) & mask);

                    const uint8_t diff = diff_u8_mod(cur, pred, blockQuantBits);
                    const uint8_t zz = zigzag_encode_int8(static_cast<int8_t>(diff), blockQuantBits);
                    const uint32_t rIdx = (f - 1u) * channels + ch;
                    residual[rIdx] = zz;
                    maxv = std::max<uint32_t>(maxv, zz);

                    prev2 = prev1;
                    prev1 = cur;
                }
            }

            packWidth = static_cast<uint8_t>(std::max<uint32_t>(1, std::bit_width(maxv)));
        }

        // Block header.
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

        const BitPack seedPacked = packBitsFixed<uint8_t>(seed, blockQuantBits);
        out.insert(out.end(), seedPacked.bytes.begin(), seedPacked.bytes.end());

        if (residualCount > 0)
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
    }

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
    // File magic is exactly 3 bytes: "CWV".
    hdr.magic[0] = input[offset + 0];
    hdr.magic[1] = input[offset + 1];
    hdr.magic[2] = input[offset + 2];
    offset += 3;

    const bool isCWV = (hdr.magic[0] == 'C' && hdr.magic[1] == 'W' && hdr.magic[2] == 'V');
    if (!isCWV)
    {
        printf("Error! Not a CWV file (bad magic).\\n");
        return 1;
    }

    if (!readLE(input, offset, hdr.channels)) return 1;
    if (!readLE(input, offset, hdr.sampleRate)) return 1;
    if (!readLE(input, offset, hdr.totalPCMFrameCount)) return 1;
    if (!readLE(input, offset, hdr.blockSize)) return 1;
    if (!readLE(input, offset, hdr.numberOfBlocks)) return 1;

    uint8_t rawQuantBits = 0;
    if (!readLE(input, offset, rawQuantBits)) return 1;
    const bool adaptiveBlockQuantization = (rawQuantBits & 0x80u) != 0;
    hdr.quantBits = static_cast<uint8_t>(rawQuantBits & 0x7Fu);

    if (hdr.channels < 1 || hdr.blockSize == 0 || hdr.numberOfBlocks == 0 || hdr.sampleRate == 0 || hdr.totalPCMFrameCount <= 0)
    {
        printf("Error! Invalid CWV header.\n");
        return 1;
    }

    if (hdr.quantBits < 1 || hdr.quantBits > 8)
    {
        printf("Error! Invalid quantBits (%u).\n", hdr.quantBits);
        return 1;
    }

    const uint64_t totalFrames = static_cast<uint64_t>(hdr.totalPCMFrameCount);
    const uint64_t totalSamples = totalFrames * hdr.channels;
    outputBuffer.assign(static_cast<size_t>(totalSamples), 0.0f);

    const uint8_t channels = hdr.channels;
    std::vector<float> gain(channels, 1.0f);
    std::vector<uint8_t> seedPacked;
    std::vector<uint8_t> payloadPacked;
    std::vector<uint8_t> seed;
    std::vector<uint8_t> residual;
    std::vector<uint8_t> q;
    const uint64_t maxSamplesInBlock = static_cast<uint64_t>(hdr.blockSize) * channels;
    const uint64_t maxResidual = (maxSamplesInBlock > channels) ? (maxSamplesInBlock - channels) : 0;
    const uint32_t maxResidualU32 = static_cast<uint32_t>(std::min<uint64_t>(maxResidual, std::numeric_limits<uint32_t>::max()));
    seedPacked.reserve(calculateBitPackedSize(channels, 8));
    payloadPacked.reserve(calculateBitPackedSize(maxResidualU32, 8));
    seed.reserve(channels);
    residual.reserve(maxResidualU32);
    q.reserve(static_cast<size_t>(maxSamplesInBlock));

    for (uint32_t b = 0; b < hdr.numberOfBlocks; ++b)
    {
        const uint64_t startFrame = static_cast<uint64_t>(b) * hdr.blockSize;
        if (startFrame >= totalFrames)
        {
            printf("Error! Block index out of range.\n");
            return 1;
        }
        const uint32_t framesInBlock = static_cast<uint32_t>(std::min<uint64_t>(hdr.blockSize, totalFrames - startFrame));
        const uint32_t samplesInBlock = framesInBlock * channels;
        const uint64_t startSample = startFrame * channels;

        if (offset + 1 > input.size())
            return 1;
        const uint8_t packInfo = input[offset++];

        uint8_t predictor = 0;
        uint8_t blockQuantBits = hdr.quantBits;
        const uint8_t packWidth = static_cast<uint8_t>(packInfo & 0x0F);

        if (adaptiveBlockQuantization)
        {
            const uint8_t blockMode = static_cast<uint8_t>(packInfo >> 4);
            predictor = static_cast<uint8_t>(blockMode >> 3);
            blockQuantBits = static_cast<uint8_t>((blockMode & 0x07u) + 1u);
        }
        else
        {
            predictor = static_cast<uint8_t>(packInfo >> 4);
        }

        if (packWidth < 1 || packWidth > 8)
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
            gain[ch] = unpackGainDb(input[offset++]);

        const size_t seedBytes = calculateBitPackedSize(channels, blockQuantBits);
        if (offset + seedBytes > input.size())
        {
            printf("Error! Truncated CWV seed samples.\n");
            return 1;
        }
        seedPacked.assign(input.begin() + offset, input.begin() + offset + seedBytes);
        offset += seedBytes;

        seed = unpackBits<uint8_t>(seedPacked, blockQuantBits, channels);

        const uint32_t seedCount = channels;
        const uint32_t residualCount = (samplesInBlock > seedCount) ? (samplesInBlock - seedCount) : 0;

        const size_t payloadBytes = calculateBitPackedSize(residualCount, packWidth);
        if (offset + payloadBytes > input.size())
        {
            printf("Error! Truncated CWV block payload.\n");
            return 1;
        }

        if (payloadBytes > 0)
        {
            payloadPacked.assign(input.begin() + offset, input.begin() + offset + payloadBytes);
            offset += payloadBytes;
        }
        else
        {
            payloadPacked.clear();
        }

        q.assign(samplesInBlock, 0);
        for (uint8_t ch = 0; ch < channels; ++ch)
            q[ch] = seed[ch];

        if (residualCount > 0)
        {
            residual = unpackBits<uint8_t>(payloadPacked, packWidth, residualCount);
            for (uint32_t i = 0; i < residualCount; ++i)
                q[seedCount + i] = residual[i];

            for (uint32_t i = seedCount; i < samplesInBlock; ++i)
                q[i] = static_cast<uint8_t>(zigzag_decode_int8(q[i], blockQuantBits));

            if (predictor == 0)
            {
                for (uint8_t ch = 0; ch < channels; ++ch)
                    for (uint32_t i = ch + channels; i < samplesInBlock; i += channels)
                        q[i] = decode_u8_mod(q[i - channels], q[i], blockQuantBits);
            }
            else
            {
                const uint32_t mask = (1u << blockQuantBits) - 1u;
                for (uint8_t ch = 0; ch < channels; ++ch)
                {
                    for (uint32_t f = 1; f < framesInBlock; ++f)
                    {
                        const uint32_t idx = f * channels + ch;
                        const uint8_t diff = q[idx];

                        const uint8_t pred = (f == 1)
                            ? q[idx - channels]
                            : static_cast<uint8_t>((2u * q[idx - channels] - q[idx - 2u * channels]) & mask);

                        q[idx] = decode_u8_mod(pred, diff, blockQuantBits);
                    }
                }
            }
        }

        for (uint32_t f = 0; f < framesInBlock; ++f)
        {
            const uint64_t outBase = startSample + static_cast<uint64_t>(f) * channels;
            const uint32_t inBase = f * channels;
            for (uint8_t ch = 0; ch < channels; ++ch)
            {
                float s = dequantizeNormalizedCode(q[inBase + ch], blockQuantBits);
                s /= gain[ch];
                outputBuffer[static_cast<size_t>(outBase + ch)] = s;
            }
        }
    }

    if (outHeader)
        *outHeader = hdr;

    return 0;
}
