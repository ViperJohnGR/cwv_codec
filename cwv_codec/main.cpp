#define DR_WAV_IMPLEMENTATION

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
        return printf("Usage: %s input [-bits N] [-block FRAMES] [-lowpass HZ] [-normalize] [-gain FLOAT]\n", getFilenameFromPath(argv[0]).c_str());

    int bits = 4;
    uint32_t blockSize = 128;
    float lowpassHz = 0.0f;
    float gain = 1.0f;
    bool expectbits = false;
    bool expectblock = false;
    bool expectfilename = false;
    bool expectlowpass = false;
    bool expectgain = false;
    bool normalize = false;

    std::string outputFilename = "output.wav";


    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-bits") == 0)
            expectbits = true;
        else if (strcmp(argv[i], "-block") == 0)
            expectblock = true;
        else if (strcmp(argv[i], "-normalize") == 0)
            normalize = true;
        else if (strcmp(argv[i], "-gain") == 0)
            expectgain = true;
        else if (strcmp(argv[i], "-lowpass") == 0)
            expectlowpass = true;
        else if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "-output") == 0)
            expectfilename = true;
        else if (expectbits)
        {
            bits = atoi(argv[i]);
            expectbits = false;
        }
        else if (expectblock)
        {
            const int parsedBlockSize = atoi(argv[i]);
            if (parsedBlockSize <= 0)
                return printf("Error: block size must be a positive integer number of frames.\n");

            blockSize = static_cast<uint32_t>(parsedBlockSize);
            expectblock = false;
        }
        else if (expectfilename)
        {
            outputFilename = getExtensionFromPath(argv[i]).empty() ? (argv[i] + std::string(".wav")) : argv[i];
            expectfilename = false;
        }
        else if (expectlowpass)
        {
            lowpassHz = std::max(0.0f, static_cast<float>(atof(argv[i])));
            expectlowpass = false;
        }
        else if (expectgain)
        {
            gain = static_cast<float>(atof(argv[i]));
            expectgain = false;
        }
        else if (getExtensionFromPath(argv[i]) != "cwv")
        {
            printf("Reading input...\n");
            printf("bitsPerSample = %d\n", bits);
            printf("blockSize (frames) = %u\n", blockSize);
            if (lowpassHz > 0.0f)
                printf("lowpass = %.2f Hz\n", lowpassHz);
            if (normalize)
                printf("normalize = on\n");
            if (gain != 1.0f)
                printf("gain = %.6f\n", gain);

            audioStream inStream(argv[i]);
            if (inStream.channels < 1)
                return printf("Error! inStream.channels is %d\n", inStream.channels);

            if (lowpassHz > 0.0f && !inStream.applyLowPass(lowpassHz))
                printf("Failed to apply lowpass at %.2f Hz.\n", lowpassHz);

            if (normalize && !inStream.normalize())
                printf("Failed to apply normalization.\n");

            if (gain != 1.0f && !inStream.applyGain(gain))
                printf("Failed to apply gain %.6f\n", gain);

            printf("Starting encoder...\n");
            const auto encodeStart = std::chrono::steady_clock::now();
            std::vector<uint8_t> outBuf = encodeCWV(inStream, blockSize, static_cast<uint8_t>(bits));
            const auto encodeEnd = std::chrono::steady_clock::now();
            const auto encodeMs = std::chrono::duration_cast<std::chrono::milliseconds>(encodeEnd - encodeStart);
            printf("Encoder time: %lld ms (%.3f s)\n", static_cast<long long>(encodeMs.count()), std::chrono::duration<double>(encodeEnd - encodeStart).count());
            if (outBuf.empty())
                return 1;

            std::string outputName = removeExtensionFromPath(argv[i]);
            outputName += ".cwv";

            FILE* fp = nullptr;
            openFile(&fp, outputName.c_str(), "wb");
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
            openFile(&fp, argv[i], "rb");
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

#ifdef SNDFILE_HH
                    SNDFILE* outFile;
                    SF_INFO outFileInfo = { 0 };

                    outFileInfo.channels = hdr.channels;
                    outFileInfo.frames = hdr.totalPCMFrameCount;
                    outFileInfo.samplerate = (int)hdr.sampleRate;
                    outFileInfo.format = SF_FORMAT_WAV | SF_FORMAT_FLOAT;

                    outFile = sf_open(outputFilename.c_str(), SFM_WRITE, &outFileInfo);
                    if (outFile == NULL)
                    {
                        printf("Cannot open '%s'.\n%s\n", outputFilename.c_str(), sf_strerror(NULL));
                        return 1;
                    }
                    sf_writef_float(outFile, &output[0], hdr.totalPCMFrameCount);
                    sf_close(outFile);
#else
                    drwav_data_format format{};
                    format.container = drwav_container_riff;     // normal .wav
                    format.format = DR_WAVE_FORMAT_IEEE_FLOAT;   // float WAV
                    format.channels = hdr.channels;
                    format.sampleRate = hdr.sampleRate;
                    format.bitsPerSample = 32;

                    const drwav_uint64 frameCount = static_cast<drwav_uint64>(output.size() / format.channels);

                    drwav wav{};
                    if (!drwav_init_file_write(&wav, outputFilename.c_str(), &format, nullptr))
                        printf("Failed to open WAV for writing\n");

                    const drwav_uint64 framesWritten = drwav_write_pcm_frames(&wav, frameCount, output.data());
                    drwav_uninit(&wav);
                    if (framesWritten != frameCount)
                        printf("Failed to write all PCM frames\n");
#endif
                }
            }
            else
                printf("Error! Could not open file '%s'\n", argv[i]);
        }

    }

    return 0;
}
