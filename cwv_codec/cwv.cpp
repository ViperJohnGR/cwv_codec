#include "cwv.hpp"
#include "helpers.hpp"

#include <algorithm>
#include <cmath>

/*
cwv::cwv(audioStream stream, uint8_t targetBits)
{
	//a
}*/

constexpr float maxGain = 100.0f;

uint8_t packGainDb(float gain)
{
    gain = std::clamp(gain, 1.0f, maxGain);
    float maxDb = 20.0f * std::log10(maxGain);          // 40 dB if maxGain=100
    float db = 20.0f * std::log10(gain);             // 0..maxDb
    float x = db / maxDb;                           // 0..1
    return static_cast<uint8_t>(std::lround(x * 255.0f));
}

float unpackGainDb(uint8_t code)
{
    float maxDb = 20.0f * std::log10(maxGain);
    float db = (static_cast<float>(code) / 255.0f) * maxDb;
    return std::pow(10.0f, db / 20.0f);
}

uint8_t diff_u8_mod(uint32_t a, uint32_t b, uint8_t bitsPerSample)
{
    if (!(bitsPerSample >= 1 && bitsPerSample <= 8))
        return 0;

    const uint32_t mask = (1u << bitsPerSample) - 1u;  // e.g. bits=5 -> 0x1F
    a &= mask;                                          // ensure in-range
    b &= mask;

    // Modular subtraction (wraps at 2^bitsPerSample)
    return static_cast<uint8_t>((a - b) & mask);
}

uint8_t decode_u8_mod(uint32_t b, uint32_t diff, uint8_t bitsPerSample)
{
    if (!(bitsPerSample >= 1 && bitsPerSample <= 8))
        return 0;

    const uint32_t mask = (1u << bitsPerSample) - 1u; // e.g. bits=5 -> 0x1F
    b &= mask;
    diff &= mask;

    // Modular addition (wraps at 2^bitsPerSample)
    return static_cast<uint8_t>((b + diff) & mask);
}

constexpr std::uint8_t zigzag_encode_int8(std::int8_t value, std::uint8_t bits) {
    if (bits == 0 || bits > 8) return 0;

    const std::uint16_t mask = (bits == 8) ? 0xFFu : static_cast<std::uint16_t>((1u << bits) - 1u);
    std::uint16_t u = static_cast<std::uint16_t>(static_cast<std::uint8_t>(value)) & mask;

    const std::uint16_t signbit = static_cast<std::uint16_t>(1u << (bits - 1u));
    if (u & signbit) u |= static_cast<std::uint16_t>(~mask); // sign-extend from `bits`

    const std::int16_t s = static_cast<std::int16_t>(u);
    const std::uint16_t zz = (static_cast<std::uint16_t>(s) << 1)
        ^ static_cast<std::uint16_t>(-static_cast<std::int16_t>(s < 0));

    return static_cast<std::uint8_t>(zz & mask);
}

constexpr std::int8_t zigzag_decode_int8(std::uint8_t zigzag, std::uint8_t bits) {
    if (bits == 0 || bits > 8) return 0;

    const std::uint16_t mask = (bits == 8) ? 0xFFu : static_cast<std::uint16_t>((1u << bits) - 1u);
    const std::uint16_t zz = static_cast<std::uint16_t>(zigzag) & mask;

    std::uint16_t raw = (zz >> 1) ^ static_cast<std::uint16_t>(-static_cast<std::int16_t>(zz & 1u));
    raw &= mask;

    const std::uint16_t signbit = static_cast<std::uint16_t>(1u << (bits - 1u));
    if (raw & signbit) raw |= static_cast<std::uint16_t>(~mask); // sign-extend back to int8

    return static_cast<std::int8_t>(static_cast<std::int16_t>(raw));
}

BitPack encodeStream(audioStream &audio, std::vector<gainInfo> &gainInfos, int bitsPerSample, float gainStep, bool saveCompressed)
{
    if (audio.sampleData.size() != audio.totalPCMFrameCount * audio.channels)
    {
        printf("Error! audio.sampleData.size() is %llu. audio.totalPCMFrameCount(%llu) audio.channels(%u)\n", audio.sampleData.size(), audio.totalPCMFrameCount, audio.channels);
        return {};
    }

    gainInfos.clear();
    gainInfos.resize(audio.channels);

    std::vector<std::vector<uint32_t>> ends(audio.channels, {});

	auto totalSamples = audio.totalPCMFrameCount * audio.channels;
	std::vector<float> currentGain(audio.channels);

    for (int i = 0; i < audio.channels; i++)
    {
        if (fabs(audio.sampleData[i]) < 0.0001)
            currentGain[i] = 1.0;
        else
            currentGain[i] = std::clamp(1.0f / fabs(audio.sampleData[i]), std::numeric_limits<float>::min(), maxGain);

        audio.sampleData[i] *= currentGain[i];

        gainInfos[i].numInfos++;
        ends[i].push_back(0);
        gainInfos[i].gains.push_back(packGainDb(currentGain[i]));
    }

    int progress = 0;

    printf("Encoding... 0%%\r");

    std::vector<uint32_t> currentEndsDelta(audio.channels, 0);

    for (int i = audio.channels; i < totalSamples; i += audio.channels)
    {
        int currentProgress = static_cast<int>((i / (float)totalSamples) * 100.0f);
        for (int channel = 0; channel < audio.channels; channel++)
        {
            if (currentGain[channel] > maxGain)
                currentGain[channel] = maxGain;
            if ((fabs(audio.sampleData[i + channel]) * currentGain[channel]) > 1.0)
            {
                currentGain[channel] = std::clamp(1.0f / fabs(audio.sampleData[i + channel]), std::numeric_limits<float>::min(), maxGain);
                audio.sampleData[i + channel] *= currentGain[channel];

                gainInfos[channel].numInfos++;
                ends[channel].push_back(currentEndsDelta[channel]);
                currentEndsDelta[channel] = 0;
                gainInfos[channel].gains.push_back(packGainDb(currentGain[channel]));
            }
            else
            {
                audio.sampleData[i + channel] *= currentGain[channel];
                if (currentGain[channel] < maxGain)
                    currentGain[channel] += gainStep;
                currentEndsDelta[channel]++;
            }
        }

        if (currentProgress != progress)
        {
            printf("Encoding... %d%%\r", currentProgress);
            progress = currentProgress;
        }
    }
    printf("Encoding... 100%%\n");

    for (int i = 0; i < audio.channels; i++)
    {
        printf("numGainInfo[%d] = %u\n", i, gainInfos[i].numInfos);
        //printf("numGainInfo[%d] * sizeof(struct gainInfo) = %llu bytes\n", i, gainInfos[i].numInfos * sizeof(gainInfo));
    }
    puts("");


    if (saveCompressed)
    {
        FILE* cmprFile = nullptr;
        fopen_s(&cmprFile, "compressed", "wb");
        if (cmprFile != NULL)
        {
            fwrite(&audio.sampleData[0], sizeof(float), totalSamples, cmprFile);
            fclose(cmprFile);
        }
    }

    printf("Writing output buffer... 0%%\r");

    std::vector<uint8_t> outputBuffer(totalSamples);
    auto pow2toBitsPerSample = pow(2.0, bitsPerSample - 1.0) - 0.5;

    for (int i = 0; i < totalSamples; i++)
    {
        int currentProgress = static_cast<int>((i / (float)totalSamples) * 100.0);
        outputBuffer[i] = static_cast<uint8_t>(round((audio.sampleData[i] + 1.0) * pow2toBitsPerSample));
        if (currentProgress != progress)
        {
            printf("Writing output buffer... %d%%\r", currentProgress);
            progress = currentProgress;
        }
    }
    printf("Writing output buffer... 100%%\n");

    for (int ch = 0; ch < audio.channels; ++ch)
    {
        uint8_t prev = outputBuffer[ch]; // keep first absolute sample

        for (int i = ch + audio.channels; i < totalSamples; i += audio.channels)
        {
            uint8_t cur = outputBuffer[i];                       // absolute (still)
            outputBuffer[i] = zigzag_encode_int8((int8_t)diff_u8_mod(cur, prev, bitsPerSample), bitsPerSample); // store delta in-place
            prev = cur;                                          // advance prev (absolute)
        }
    }

    for (uint8_t i = 0; i < audio.channels; i++)
    {
        auto endsPacked = packBits<uint32_t>(ends[i]);
        gainInfos[i].ends = endsPacked.bytes;
        gainInfos[i].endsBitSize = endsPacked.bit_width;
        printf("Packed ends bit width on channel %u = %u\n", i, endsPacked.bit_width);
    }

    return packBits<uint8_t>(outputBuffer);
}


int decodeStream(const std::vector<uint8_t> &input, std::vector<float> &outputBuffer)
{
    //header = u8_channels + s32_sampleRate + s64_PCMFrameCount + u8_bitSize + float_gainStep
    int headerLength = sizeof(uint8_t) + sizeof(uint32_t) + sizeof(sf_count_t) + sizeof(uint8_t) + sizeof(float);


    const uint8_t* pInput = &input[0];

    uint8_t channels = *pInput;
    int sampleRate = *((int*)(pInput + 1));
    sf_count_t totalPCMFrameCount = *((sf_count_t*)(pInput + 1 + sizeof(uint32_t)));
    uint8_t bitsPerSample = *(pInput + 1 + sizeof(uint32_t) + sizeof(sf_count_t));
    float gainStep = *(float*)(pInput + headerLength - sizeof(float));
    auto totalSamples = totalPCMFrameCount * channels;

    //const auto* audio = (&input[0]) + headerLength;
    std::vector<uint8_t> audio = std::vector<std::uint8_t>(input.begin() + headerLength, input.end());

    int progress = 0;
    printf("Reading input... 0%%\r");
 
    auto unpackedAudio = unpackBits<uint8_t>(audio, bitsPerSample, totalSamples);

    for (int i = channels; i < unpackedAudio.size(); i++)
    {
        unpackedAudio[i] = (uint8_t)zigzag_decode_int8(unpackedAudio[i], bitsPerSample);
    }

    for (int ch = 0; ch < channels; ++ch)
        for (int i = ch + channels; i < totalSamples; i += channels)
            unpackedAudio[i] = decode_u8_mod(unpackedAudio[i - channels], unpackedAudio[i], bitsPerSample);

    printf("Reading input... 100%%\nDecoding... 0%%\r");

    outputBuffer.resize(totalSamples);

    uint32_t* pGainInfo = (uint32_t*)(pInput + headerLength + calculateBitPackedSize(totalSamples, bitsPerSample));
    uint8_t* pGainInfo8Bit = NULL;

    std::vector<std::vector<uint8_t>> gainInfosPacked(channels, {});
    std::vector<gainInfo> gainInfos(channels, {});

    for (int i=0;i < channels;i++)
    {
        gainInfos[i].numInfos = *pGainInfo++;
        pGainInfo8Bit = (uint8_t*)pGainInfo;
        gainInfos[i].endsBitSize = *pGainInfo8Bit++;
        uint32_t gainInfosPackedSize = (uint32_t)calculateBitPackedSize(gainInfos[i].numInfos, gainInfos[i].endsBitSize);
        for (uint32_t j = 0; j < gainInfosPackedSize; j++)
            gainInfosPacked[i].push_back(*pGainInfo8Bit++);

        for (uint32_t j = 0; j < gainInfos[i].numInfos; j++)
        {
            gainInfos[i].gains.push_back(*pGainInfo8Bit++);
        }
        pGainInfo = (uint32_t*)pGainInfo8Bit;
    }

    std::vector<std::vector<uint32_t>> unpackedEnds(channels, {});

    for (int i = 0; i < channels; i++)
        unpackedEnds[i] = unpackBits<uint32_t>(gainInfosPacked[i], gainInfos[i].endsBitSize, gainInfos[i].numInfos);

    const float pow2toBitsPerSample = (float)pow(2, bitsPerSample) - 1.0f;

    for (int k = 0; k < channels; k++)
    {
        uint32_t currentGainInfo = 0;
        uint32_t currentGainEndsDeltaCnt = 0;
        float currentGain = unpackGainDb(gainInfos[k].gains[currentGainInfo]);
        for (int i = 0; i < totalSamples; i += channels)
        {
            int currentProgress = (int)((i / (float)totalSamples) * (100.0f / channels) + ((100.0f / channels) * k));
            if (currentGainInfo < gainInfos[k].numInfos && currentGainEndsDeltaCnt == unpackedEnds[k][currentGainInfo])
            {
                currentGain = unpackGainDb(gainInfos[k].gains[currentGainInfo]);
                outputBuffer[i + k] = unpackedAudio[i + k] * 2.0f / pow2toBitsPerSample - 1.0f;
                outputBuffer[i + k] /= currentGain;
                currentGainInfo++;
                currentGainEndsDeltaCnt = 0;
            }
            else
            {
                //printf("currentGain = %f\n", currentGain);
                outputBuffer[i + k] = unpackedAudio[i + k] * 2.0f / pow2toBitsPerSample - 1.0f;
                outputBuffer[i + k] /= currentGain;
                currentGain += gainStep;
                currentGainEndsDeltaCnt++;
            }

            if (currentProgress != progress)
            {
                printf("Decoding... %d%%\r", currentProgress);
                progress = currentProgress;
            }
        }
    }
    printf("Decoding... 100%%\n");

    return 0;
}
