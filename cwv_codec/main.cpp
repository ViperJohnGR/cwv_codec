#include "audioStream.hpp"
#include "cwv.hpp"
#include "helpers.hpp"

#include <cstdio>


int writeEncodedStream(const BitPack& audio, uint8_t channels, int sampleRate, sf_count_t totalPCMFrameCount, const std::vector<gainInfo> &gainInfos, const std::string& filename, float gainStep)
{
    FILE* fp = nullptr;
    fopen_s(&fp, filename.c_str(), "wb");
    if (fp != NULL)
    {
        printf("Writing '%s'...\n", filename.c_str());
        fwrite(&channels, sizeof(uint8_t), 1, fp);
        fwrite(&sampleRate, sizeof(uint32_t), 1, fp);
        fwrite(&totalPCMFrameCount, sizeof(sf_count_t), 1, fp);
        fwrite(&audio.bit_width, sizeof(uint8_t), 1, fp);
        fwrite(&gainStep, sizeof(float), 1, fp);

        size_t totalSamples = (uint32_t)ceil(totalPCMFrameCount * channels * (audio.bit_width / 8.0));
        size_t bytesWritten = fwrite(&audio.bytes[0], 1, totalSamples, fp);
        if (bytesWritten != totalSamples)
        {
            printf("bytesWritten = %llu out of %llu\n", bytesWritten, totalSamples);
            fclose(fp);
            return 1;
        }

        size_t finalGainsSize = 0;
        for (const auto &info : gainInfos)
        {
            finalGainsSize += fwrite(&info.numInfos, sizeof(uint32_t), 1, fp) * sizeof(uint32_t);
            finalGainsSize += fwrite(&info.endsBitSize, sizeof(uint8_t), 1, fp);
            finalGainsSize += fwrite(&info.ends[0], sizeof(uint8_t), info.ends.size(), fp);
            finalGainsSize += fwrite(&info.gains[0], sizeof(float), info.gains.size(), fp) * sizeof(float);
        }

        printf("Audio size written to file: %s\n", printBytes(bytesWritten).c_str());
        printf("Gains block size written to file: %s\n", printBytes(finalGainsSize).c_str());

        fclose(fp);
    }
    else
    {
        printf("Error opening output file '%s'\n", filename.c_str());
        return 1;
    }

    return 0;
}



int main(int argc, char** argv)
{
    if (argc < 2)
        return printf("Usage: %s input", getFilenameFromPath(argv[0]).c_str());

    float gain = 0.0009765625;
    int bits = 4;
    bool expectbits = false;
    bool expectgain = false;
    bool saveCompressed = false;


    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-bits") == 0)
            expectbits = true;
        else if (strcmp(argv[i], "-sc") == 0)
            saveCompressed = true;
        else if (strcmp(argv[i], "-gain") == 0)
            expectgain = true;
        else if (expectbits)
        {
            bits = atoi(argv[i]);
            expectbits = 0;
        }
        else if (expectgain)
        {
            gain = (float)atof(argv[i]);
            expectgain = 0;
        }
        else if (getExtensionFromPath(argv[i]) != "cwv") //TODO: add error handling for invalid input files (unsupported extension, corrupted files etc)
        {
            printf("Using gain %.9g\n", gain);
            printf("Reading input...\n");

            audioStream inStream(argv[i]);
            if (inStream.channels < 1)
                return printf("Error! inStream.channels is %d\n", inStream.channels);

            std::vector<gainInfo> gainInfos;

            BitPack output = encodeStream(inStream, gainInfos, bits, gain, saveCompressed);

            std::string outputName = removeExtensionFromPath(argv[i]);
            outputName += ".cwv";

            if (writeEncodedStream(output, inStream.channels, inStream.sampleRate, inStream.totalPCMFrameCount, gainInfos, outputName, gain) > 0)
                puts("Error writing output!");
        }
        else
        {
            FILE* fp = nullptr;
            fopen_s(&fp, argv[i], "rb");
            if (fp != NULL)
            {
                fseek(fp, 0, SEEK_END);
                unsigned long filesize = ftell(fp);
                fseek(fp, 0, SEEK_SET);
                std::vector<uint8_t> inBuf(filesize, 0);
                if (fread(&inBuf[0], 1, filesize, fp) == filesize)
                {
                    std::vector<float> output;
                    decodeStream(inBuf, output);
                    SNDFILE* outFile;
                    SF_INFO outFileInfo = { 0 };

                    uint8_t* pInput = &inBuf[0];
                    outFileInfo.channels = *pInput;
                    sf_count_t frames = *((sf_count_t*)(pInput + 1 + sizeof(int)));
                    outFileInfo.frames = frames;
                    outFileInfo.samplerate = *((int*)(pInput + 1));
                    outFileInfo.format = SF_FORMAT_WAV | SF_FORMAT_FLOAT;

                    outFile = sf_open("output.wav", SFM_WRITE, &outFileInfo);
                    if (outFile == NULL)
                    {
                        printf("Cannot open '%s'.\n%s\n", "output.wav", sf_strerror(NULL));
                        return 1;
                    }
                    sf_writef_float(outFile, &output[0], frames);
                    sf_close(outFile);
                }
            }
        }

    }

    return 0;
}
