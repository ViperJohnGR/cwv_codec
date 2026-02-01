#include "cwv.hpp"
#include "helpers.hpp"

/*
cwv::cwv(audioStream stream, uint8_t targetBits)
{
	//a
}*/


BitPack encodeStream(audioStream &audio, std::vector<gainInfo> &gainInfos, int bitsPerSample, float gainStep, bool saveCompressed)
{
    if (audio.totalPCMFrameCount < audio.channels)
    {
        printf("Error! audio.totalPCMFrameCount(%lld) < audio.channels(%d)\n", audio.totalPCMFrameCount, audio.channels);
        return {};
    }

    gainInfos.clear();
    gainInfos.resize(audio.channels);

    std::vector<std::vector<uint32_t>> ends(audio.channels, {});

	auto totalSamples = audio.totalPCMFrameCount * audio.channels;
	float maxGain = 100.0;
	std::vector<float> currentGain(audio.channels);

    for (int i = 0; i < audio.channels; i++)
    {
        if (fabs(audio.sampleData[i]) < 0.0001)
            currentGain[i] = 1.0;
        else
            currentGain[i] = (float)(1.0 / fabs(audio.sampleData[i]));

        gainInfos[i].numInfos++;
        ends[i].push_back(0);
        gainInfos[i].gains.push_back(currentGain[i]);
    }

    int progress = 0;

    printf("Encoding... 0%%\r");

    std::vector<uint32_t> currentEndsDelta(audio.channels, 0);

    for (int i = audio.channels; i < totalSamples; i += audio.channels)
    {
        int currentProgress = static_cast<int>((i / (float)totalSamples) * 100.0f);
        for (int channel = 0; channel < audio.channels; channel++)
        {
            if (fabs(currentGain[channel]) > maxGain)
                currentGain[channel] = maxGain;
            if ((fabs(audio.sampleData[i + channel]) * currentGain[channel]) > 1.0)
            {
                currentGain[channel] = 1.0f / (float)fabs(audio.sampleData[i + channel]);
                audio.sampleData[i + channel] *= currentGain[channel];

                gainInfos[channel].numInfos++;
                ends[channel].push_back(currentEndsDelta[channel]);
                currentEndsDelta[channel] = 0;
                gainInfos[channel].gains.push_back(currentGain[channel]);
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

    std::vector<uint8_t> outputBuffer;
    for (int i = 0; i < totalSamples; i++)
    {
        int currentProgress = static_cast<int>((i / (float)totalSamples) * 100.0);
        outputBuffer.push_back(static_cast<uint8_t>(round((audio.sampleData[i] + 1.0) * (pow(2.0, bitsPerSample - 1.0) - 0.5))));
        if (currentProgress != progress)
        {
            printf("Writing output buffer... %d%%\r", currentProgress);
            progress = currentProgress;
        }
    }
    printf("Writing output buffer... 100%%\n");

    for (auto i = 0; i < audio.channels; i++)
    {
        auto endsPacked = packBits<uint32_t>(ends[i]);
        gainInfos[i].ends = endsPacked.bytes;
        gainInfos[i].endsBitSize = endsPacked.bit_width;
        printf("Packed ends size channel %d = %d\n", i, endsPacked.bit_width);
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

    std::vector<uint8_t> finalBuffer;
    //const auto* audio = (&input[0]) + headerLength;
    std::vector<uint8_t> audio = std::vector<std::uint8_t>(input.begin() + headerLength, input.end());

    int progress = 0;
    printf("Reading input... 0%%\r");
 
    auto unpackedAudio = unpackBits<uint8_t>(audio, bitsPerSample, totalSamples);

    printf("Reading input... 100%%\nDecoding... 0%%\r");

    outputBuffer.resize(totalSamples);

    uint32_t* pGainInfo = (uint32_t*)(pInput + headerLength + (int)ceil(totalSamples * (bitsPerSample / 8.0)));
    uint8_t* pGainInfo8Bit = NULL;

    std::vector<std::vector<uint8_t>> gainInfosPacked(channels, {});
    std::vector<gainInfo> gainInfos(channels, {});

    for (int i=0;i < channels;i++)
    {
        gainInfos[i].numInfos = *pGainInfo++;
        pGainInfo8Bit = (uint8_t*)pGainInfo;
        gainInfos[i].endsBitSize = *pGainInfo8Bit++;
        uint32_t gainInfosPackedSize = (uint32_t)ceil(gainInfos[i].numInfos * (gainInfos[i].endsBitSize / 8.0));
        for (uint32_t j = 0; j < gainInfosPackedSize; j++)
            gainInfosPacked[i].push_back(*pGainInfo8Bit++);
        pGainInfo = (uint32_t*)pGainInfo8Bit;
        for (uint32_t j = 0; j < gainInfos[i].numInfos; j++)
        {
            gainInfos[i].gains.push_back(*reinterpret_cast<float*>(pGainInfo));
            pGainInfo++;
        }
    }

    std::vector<std::vector<uint32_t>> unpackedEnds(channels, {});

    for (int i = 0; i < channels; i++)
        unpackedEnds[i] = unpackBits<uint32_t>(gainInfosPacked[i], gainInfos[i].endsBitSize, gainInfos[i].numInfos);

    for (int k = 0; k < channels; k++)
    {
        uint32_t currentGainInfo = 0;
        uint32_t currentGainEndsDeltaCnt = 0;
        float currentGain = gainInfos[k].gains[currentGainInfo];
        for (int i = 0; i < totalSamples; i += channels)
        {
            int currentProgress = (int)((i / (float)totalSamples) * (100.0f / channels) + ((100.0f / channels) * k));
            if (currentGainInfo < gainInfos[k].numInfos && currentGainEndsDeltaCnt == unpackedEnds[k][currentGainInfo])
            {
                currentGain = gainInfos[k].gains[currentGainInfo];
                outputBuffer[i + k] = (float)(unpackedAudio[i + k] * 2.0f / (pow(2, bitsPerSample) - 1.0f) - 1.0f);
                outputBuffer[i + k] /= currentGain;
                currentGainInfo++;
                currentGainEndsDeltaCnt = 0;
            }
            else
            {
                //printf("currentGain = %f\n", currentGain);
                outputBuffer[i + k] = (float)(unpackedAudio[i + k] * 2.0f / (pow(2, bitsPerSample) - 1.0f) - 1.0f);
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
