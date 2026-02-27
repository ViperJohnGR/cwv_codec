#include "cwv.hpp"
#include "helpers.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace {

constexpr float kMaxGain = 100.0f;

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

    // Rough reserve: header + per-block headers + packed data
    std::vector<uint8_t> out;
    out.reserve(64 + numberOfBlocks * 8);

    // Header
    out.insert(out.end(), { 'C','W','V','3' });
    appendLE(out, audio.channels);
    appendLE(out, static_cast<uint32_t>(audio.sampleRate));
    appendLE(out, static_cast<sf_count_t>(audio.totalPCMFrameCount));
    appendLE(out, static_cast<uint32_t>(blockSizeFrames));
    appendLE(out, static_cast<uint32_t>(numberOfBlocks));
    appendLE(out, static_cast<uint8_t>(bitsPerSample));

    // For debug: save normalized audio (float) as a raw file, like the old code.
    FILE* cmprFile = nullptr;
    if (saveCompressed)
        fopen_s(&cmprFile, "compressed", "wb");

    const float qScale = static_cast<float>(std::pow(2.0, bitsPerSample - 1.0) - 0.5);
    const uint32_t qMax = (1u << bitsPerSample) - 1u;

    printf("Encoding %u blocks...\n", numberOfBlocks);

    // Reuse per-block scratch buffers to avoid per-block heap churn.
    const uint8_t channels = audio.channels;
    std::vector<float> peak(channels, 0.0f);
    std::vector<float> gain(channels, 1.0f);
    std::vector<uint8_t> gainCode(channels, 0);

    std::vector<uint8_t> q;
    q.reserve(static_cast<size_t>(blockSizeFrames) * channels);

    std::vector<uint8_t> seed(channels, 0);
    std::vector<uint8_t> residual;
    if (blockSizeFrames > 0)
        residual.reserve(static_cast<size_t>(blockSizeFrames - 1u) * channels);

    std::vector<float> norm;
    if (cmprFile != nullptr)
        norm.reserve(static_cast<size_t>(blockSizeFrames) * channels);

    for (uint32_t b = 0; b < numberOfBlocks; ++b)
    {
        const uint64_t startFrame = static_cast<uint64_t>(b) * blockSizeFrames;
        const uint32_t framesInBlock = static_cast<uint32_t>(std::min<uint64_t>(blockSizeFrames, totalFrames - startFrame));
        const uint32_t samplesInBlock = framesInBlock * channels;
        const uint64_t startSample = startFrame * channels;

        // Per-channel peak for normalization (avoid modulo in the hot loop)
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
            gain[ch] = g;
            gainCode[ch] = packGainDb(g);
        }

        // Quantize + DPCM (per-channel) + zigzag inside the block.
        q.resize(samplesInBlock);
        for (uint32_t f = 0; f < framesInBlock; ++f)
        {
            const uint64_t inBase = startSample + static_cast<uint64_t>(f) * channels;
            const uint32_t outBase = f * channels;
            for (uint8_t ch = 0; ch < channels; ++ch)
            {
                float s = audio.sampleData[inBase + ch] * gain[ch];
                s = std::clamp(s, -1.0f, 1.0f);
                const int qi = static_cast<int>(std::lround((s + 1.0f) * qScale));
                q[outBase + ch] = static_cast<uint8_t>(std::clamp(qi, 0, static_cast<int>(qMax)));
            }
        }

        for (uint8_t ch = 0; ch < channels; ++ch)
        {
            uint8_t prev = q[ch];
            for (uint32_t i = ch + channels; i < samplesInBlock; i += channels)
            {
                const uint8_t cur = q[i];
                q[i] = zigzag_encode_int8(static_cast<int8_t>(diff_u8_mod(cur, prev, bitsPerSample)), bitsPerSample);
                prev = cur;
            }
        }

        // Split into seed (first frame) and residual payload (rest of block)
        const uint32_t seedCount = channels;
        const uint32_t residualCount = (samplesInBlock > seedCount) ? (samplesInBlock - seedCount) : 0;

        // Reuse seed/residual buffers.
        std::copy_n(q.begin(), seedCount, seed.begin());
        residual.resize(residualCount);
        if (residualCount > 0)
            std::copy_n(q.begin() + seedCount, residualCount, residual.begin());

        // Pick the smallest packing bit width that fits this residual payload
        uint32_t maxv = 0;
        for (uint8_t v : residual)
            maxv = std::max<uint32_t>(maxv, v);
        const uint8_t packWidth = static_cast<uint8_t>(std::max<uint32_t>(1, std::bit_width(maxv)));

        // Block header
        out.push_back(packWidth);
        for (uint8_t ch = 0; ch < channels; ++ch)
            out.push_back(gainCode[ch]);

        // Seed samples (packed with quant bits)
        const BitPack seedPacked = packBitsFixed<uint8_t>(seed, bitsPerSample);
        out.insert(out.end(), seedPacked.bytes.begin(), seedPacked.bytes.end());

        // Residual payload (packed with packWidth)
        if (residualCount > 0)
        {
            const BitPack residualPacked = packBitsFixed<uint8_t>(residual, packWidth);
            out.insert(out.end(), residualPacked.bytes.begin(), residualPacked.bytes.end());
        }

        // Optional debug dump: write normalized float samples
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
    if (input.size() < 4)
        return 1;

    size_t offset = 0;

    CWVHeader hdr{};
    if (offset + 4 > input.size())
        return 1;
    std::memcpy(hdr.magic, input.data(), 4);
    offset += 4;

    if (!(hdr.magic[0] == 'C' && hdr.magic[1] == 'W' && hdr.magic[2] == 'V' && hdr.magic[3] == '3'))
    {
        printf("Error! Not a CWV3 file (bad magic).\n");
        return 1;
    }

    if (!readLE(input, offset, hdr.channels)) return 1;
    if (!readLE(input, offset, hdr.sampleRate)) return 1;
    if (!readLE(input, offset, hdr.totalPCMFrameCount)) return 1;
    if (!readLE(input, offset, hdr.blockSize)) return 1;
    if (!readLE(input, offset, hdr.numberOfBlocks)) return 1;
    if (!readLE(input, offset, hdr.quantBits)) return 1;

    if (hdr.channels < 1 || hdr.blockSize == 0 || hdr.numberOfBlocks == 0 || hdr.sampleRate == 0 || hdr.totalPCMFrameCount <= 0)
    {
        printf("Error! Invalid CWV3 header.\n");
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

    // Reuse per-block scratch buffers to avoid per-block heap churn.
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
    seedPacked.reserve(calculateBitPackedSize(channels, hdr.quantBits));
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
        const uint8_t packWidth = input[offset++];
        if (packWidth < 1 || packWidth > 8)
        {
            printf("Error! Invalid block packWidth (%u).\n", packWidth);
            return 1;
        }

        // Gain codes per channel
        if (offset + channels > input.size())
            return 1;
        for (uint8_t ch = 0; ch < channels; ++ch)
            gain[ch] = unpackGainDb(input[offset++]);

        // Seed samples: first frame packed with quantBits
        const size_t seedBytes = calculateBitPackedSize(channels, hdr.quantBits);
        if (offset + seedBytes > input.size())
        {
            printf("Error! Truncated CWV3 seed samples.\n");
            return 1;
        }
        seedPacked.assign(input.begin() + offset, input.begin() + offset + seedBytes);
        offset += seedBytes;

        seed = unpackBits<uint8_t>(seedPacked, hdr.quantBits, channels);

        const uint32_t seedCount = channels;
        const uint32_t residualCount = (samplesInBlock > seedCount) ? (samplesInBlock - seedCount) : 0;

        const size_t payloadBytes = calculateBitPackedSize(residualCount, packWidth);
        if (offset + payloadBytes > input.size())
        {
            printf("Error! Truncated CWV3 block payload.\n");
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

            // Undo zigzag on residuals (encoded with quantBits)
            for (uint32_t i = seedCount; i < samplesInBlock; ++i)
                q[i] = static_cast<uint8_t>(zigzag_decode_int8(q[i], hdr.quantBits));

            // Undo DPCM (mod 2^quantBits)
            for (uint8_t ch = 0; ch < channels; ++ch)
                for (uint32_t i = ch + channels; i < samplesInBlock; i += channels)
                    q[i] = decode_u8_mod(q[i - channels], q[i], hdr.quantBits);
        }

        // De-quantize and undo per-channel gain
        const float denom = static_cast<float>((1u << hdr.quantBits) - 1u);
        for (uint32_t f = 0; f < framesInBlock; ++f)
        {
            const uint64_t outBase = startSample + static_cast<uint64_t>(f) * channels;
            const uint32_t inBase = f * channels;
            for (uint8_t ch = 0; ch < channels; ++ch)
            {
                float s = (static_cast<float>(q[inBase + ch]) * 2.0f / denom) - 1.0f;
                s /= gain[ch];
                outputBuffer[static_cast<size_t>(outBase + ch)] = s;
            }
        }
    }

    if (outHeader)
        *outHeader = hdr;

    return 0;
}
