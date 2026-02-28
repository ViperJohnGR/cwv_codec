#include "audioStream.hpp"
#include "cwv.hpp"
#include "helpers.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char** argv)
{
    if (argc < 2)
        return printf("Usage: %s input [-bits N] [-block FRAMES|auto] [-lowpass HZ] [-sc]\n", getFilenameFromPath(argv[0]).c_str());

    int bits = 6;
    uint32_t blockSize = 0;
    float lowpassHz = 0.0f;
    bool expectbits = false;
    bool expectblock = false;
    bool expectlowpass = false;
    bool saveCompressed = false;


    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-bits") == 0)
            expectbits = true;
        else if (strcmp(argv[i], "-block") == 0)
            expectblock = true;
        else if (strcmp(argv[i], "-sc") == 0)
            saveCompressed = true;
        else if (strcmp(argv[i], "-lowpass") == 0)
            expectlowpass = true;
        else if (expectbits)
        {
            bits = atoi(argv[i]);
            expectbits = 0;
        }
        else if (expectblock)
        {
            if (strcmp(argv[i], "auto") == 0)
                blockSize = 0;
            else
                blockSize = (uint32_t)std::max(1, atoi(argv[i]));
            expectblock = 0;
        }
        else if (expectlowpass)
        {
            lowpassHz = std::max(0.0f, static_cast<float>(atof(argv[i])));
            expectlowpass = 0;
        }
        else if (getExtensionFromPath(argv[i]) != "cwv")
        {
            printf("Reading input...\n");
            printf("bitsPerSample = %d\n", bits);
            if (blockSize == 0)
                printf("blockSize (frames) = auto\n");
            else
                printf("blockSize (frames) = %u\n", blockSize);
            if (lowpassHz > 0.0f)
                printf("lowpass = %.2f Hz\n", lowpassHz);

            audioStream inStream(argv[i]);
            if (inStream.channels < 1)
                return printf("Error! inStream.channels is %d\n", inStream.channels);

            if (lowpassHz > 0.0f && !inStream.applyLowPass(lowpassHz))
                return 1;

            printf("Starting encoder...\n");
            const auto encodeStart = std::chrono::steady_clock::now();
            std::vector<uint8_t> outBuf = encodeCWV(inStream, blockSize, static_cast<uint8_t>(bits), saveCompressed);
            const auto encodeEnd = std::chrono::steady_clock::now();
            const auto encodeMs = std::chrono::duration_cast<std::chrono::milliseconds>(encodeEnd - encodeStart);
            printf("Encoder time: %lld ms (%.3f s)\n", static_cast<long long>(encodeMs.count()), std::chrono::duration<double>(encodeEnd - encodeStart).count());
            if (outBuf.empty())
                return 1;

            std::string outputName = removeExtensionFromPath(argv[i]);
            outputName += ".cwv";

            FILE* fp = nullptr;
            fopen_s(&fp, outputName.c_str(), "wb");
            if (fp == NULL)
                return printf("Error opening output file '%s'\n", outputName.c_str());

            printf("Writing '%s'...\n", outputName.c_str());
            const size_t wrote = fwrite(outBuf.data(), 1, outBuf.size(), fp);
            fclose(fp);

            if (wrote != outBuf.size())
                return printf("Error! wrote %llu of %llu bytes\n", (unsigned long long)wrote, (unsigned long long)outBuf.size());

            printf("Wrote %s\n", printBytes(wrote).c_str());
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

                    CWVHeader hdr{};
                    const auto decodeStart = std::chrono::steady_clock::now();
                    const int decodeResult = decodeCWV(inBuf, output, &hdr);
                    const auto decodeEnd = std::chrono::steady_clock::now();
                    const auto decodeMs = std::chrono::duration_cast<std::chrono::milliseconds>(decodeEnd - decodeStart);
                    printf("Decoder time: %lld ms (%.3f s)\n",
                        static_cast<long long>(decodeMs.count()),
                        std::chrono::duration<double>(decodeEnd - decodeStart).count());
                    if (decodeResult != 0)
                        return 1;

                    SNDFILE* outFile;
                    SF_INFO outFileInfo = { 0 };

                    outFileInfo.channels = hdr.channels;
                    outFileInfo.frames = hdr.totalPCMFrameCount;
                    outFileInfo.samplerate = (int)hdr.sampleRate;
                    outFileInfo.format = SF_FORMAT_WAV | SF_FORMAT_FLOAT;

                    outFile = sf_open("output.wav", SFM_WRITE, &outFileInfo);
                    if (outFile == NULL)
                    {
                        printf("Cannot open '%s'.\n%s\n", "output.wav", sf_strerror(NULL));
                        return 1;
                    }
                    sf_writef_float(outFile, &output[0], hdr.totalPCMFrameCount);
                    sf_close(outFile);
                }
            }
        }

    }

    return 0;
}
