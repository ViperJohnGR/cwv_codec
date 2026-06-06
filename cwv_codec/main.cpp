#define DR_WAV_IMPLEMENTATION

#include "audioStream.hpp"
#include "cwv.hpp"
#include "helpers.hpp"

#ifdef _WIN32
    #include "sndfile.hh"
#else
    #include "dr_wav.h"
#endif

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace {

struct ProgramOptions
{
    int bits = 4;
    uint32_t blockSize = 128;
    float lowpassHz = 0.0f;
    float gain = 1.0f;
    bool normalize = false;
    std::string outputFilename;
    std::vector<std::string> inputs;
};

bool stringsEqualIgnoreCase(const std::string& a, const std::string& b)
{
    if (a.size() != b.size())
        return false;

    for (size_t i = 0; i < a.size(); ++i)
    {
        if (std::tolower(static_cast<unsigned char>(a[i])) != std::tolower(static_cast<unsigned char>(b[i])))
            return false;
    }

    return true;
}

bool isCWVPath(const std::string& path)
{
    return stringsEqualIgnoreCase(getExtensionFromPath(path), "cwv");
}

bool parseIntArgument(const char* value, const char* optionName, int minValue, int maxValue, int& out)
{
    if (value == nullptr || value[0] == '\0')
    {
        printf("Error: %s requires a value.\n", optionName);
        return false;
    }

    char* end = nullptr;
    errno = 0;
    const long parsed = std::strtol(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0')
    {
        printf("Error: %s expects an integer value.\n", optionName);
        return false;
    }
    if (parsed < minValue || parsed > maxValue)
    {
        printf("Error: %s must be in [%d, %d].\n", optionName, minValue, maxValue);
        return false;
    }

    out = static_cast<int>(parsed);
    return true;
}

bool parseFloatArgument(const char* value, const char* optionName, float minValue, float maxValue, float& out)
{
    if (value == nullptr || value[0] == '\0')
    {
        printf("Error: %s requires a value.\n", optionName);
        return false;
    }

    char* end = nullptr;
    errno = 0;
    const float parsed = std::strtof(value, &end);
    if (errno != 0 || end == value || *end != '\0')
    {
        printf("Error: %s expects a numeric value.\n", optionName);
        return false;
    }
    if (parsed < minValue || parsed > maxValue)
    {
        printf("Error: %s must be in [%.3f, %.3f].\n", optionName, minValue, maxValue);
        return false;
    }

    out = parsed;
    return true;
}

bool parseCommandLine(int argc, char** argv, ProgramOptions& options)
{
    for (int i = 1; i < argc; ++i)
    {
        const char* arg = argv[i];

        if (std::strcmp(arg, "-bits") == 0)
        {
            if (i + 1 >= argc)
            {
                printf("Error: -bits requires a value.\n");
                return false;
            }

            int parsedBits = 0;
            if (!parseIntArgument(argv[++i], "-bits", 1, 8, parsedBits))
                return false;
            options.bits = parsedBits;
        }
        else if (std::strcmp(arg, "-block") == 0)
        {
            if (i + 1 >= argc)
            {
                printf("Error: -block requires a value.\n");
                return false;
            }

            int parsedBlockSize = 0;
            if (!parseIntArgument(argv[++i], "-block", 1, std::numeric_limits<int>::max(), parsedBlockSize))
                return false;
            options.blockSize = static_cast<uint32_t>(parsedBlockSize);
        }
        else if (std::strcmp(arg, "-lowpass") == 0)
        {
            if (i + 1 >= argc)
            {
                printf("Error: -lowpass requires a value.\n");
                return false;
            }

            if (!parseFloatArgument(argv[++i], "-lowpass", 0.0f, std::numeric_limits<float>::max(), options.lowpassHz))
                return false;
        }
        else if (std::strcmp(arg, "-gain") == 0)
        {
            if (i + 1 >= argc)
            {
                printf("Error: -gain requires a value.\n");
                return false;
            }

            if (!parseFloatArgument(argv[++i], "-gain", -std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), options.gain))
                return false;
        }
        else if (std::strcmp(arg, "-normalize") == 0)
        {
            options.normalize = true;
        }
        else if (std::strcmp(arg, "-o") == 0 || std::strcmp(arg, "-output") == 0)
        {
            if (i + 1 >= argc)
            {
                printf("Error: %s requires a filename.\n", arg);
                return false;
            }

            options.outputFilename = argv[++i];
            if (options.outputFilename.empty())
            {
                printf("Error: %s requires a non-empty filename.\n", arg);
                return false;
            }
        }
        else if (std::strcmp(arg, "-h") == 0 || std::strcmp(arg, "--help") == 0)
        {
            return false;
        }
        else if (arg[0] == '-')
        {
            printf("Error: unknown option '%s'.\n", arg);
            return false;
        }
        else
        {
            options.inputs.emplace_back(arg);
        }
    }

    if (options.inputs.empty())
    {
        printf("Error: no input files were provided.\n");
        return false;
    }

    if (!options.outputFilename.empty() && options.inputs.size() > 1)
    {
        printf("Error: -o can only be used when encoding or decoding a single input file.\n");
        return false;
    }

    return true;
}

std::string makeOutputPath(const std::string& inputPath, const ProgramOptions& options)
{
    if (!options.outputFilename.empty())
    {
        if (!getExtensionFromPath(options.outputFilename).empty())
            return options.outputFilename;

        return options.outputFilename + (isCWVPath(inputPath) ? ".wav" : ".cwv");
    }
    /*
    std::string outputPath = removeExtensionFromPath(inputPath);
    outputPath += isCWVPath(inputPath) ? ".wav" : ".cwv";
    */
    std::string outputPath = removeExtensionFromPath(inputPath);
    if (isCWVPath(inputPath))
        outputPath = "output.wav";
    else
        outputPath += ".cwv";
    return outputPath;
}

int encodeInput(const std::string& inputPath, const std::string& outputPath, const ProgramOptions& options)
{
    printf("Reading input...\n");
    printf("bitsPerSample = %d\n", options.bits);
    printf("blockSize (frames) = %u\n", options.blockSize);
    if (options.lowpassHz > 0.0f)
        printf("lowpass = %.2f Hz\n", options.lowpassHz);
    if (options.normalize)
        printf("normalize = on\n");
    if (options.gain != 1.0f)
        printf("gain = %.6f\n", options.gain);

    audioStream inStream(inputPath);
    if (inStream.channels < 1)
        return printf("Error! inStream.channels is %d\n", inStream.channels);

    if (options.lowpassHz > 0.0f && !inStream.applyLowPass(options.lowpassHz))
        printf("Failed to apply lowpass at %.2f Hz.\n", options.lowpassHz);

    if (options.normalize && !inStream.normalize())
        printf("Failed to apply normalization.\n");

    if (options.gain != 1.0f && !inStream.applyGain(options.gain))
        printf("Failed to apply gain %.6f\n", options.gain);

    printf("Starting encoder...\n");
    const auto encodeStart = std::chrono::steady_clock::now();
    std::vector<uint8_t> outBuf = encodeCWV(inStream, options.blockSize, static_cast<uint8_t>(options.bits));
    const auto encodeEnd = std::chrono::steady_clock::now();
    const auto encodeMs = std::chrono::duration_cast<std::chrono::milliseconds>(encodeEnd - encodeStart);
    printf("Encoder time: %lld ms (%.3f s)\n", static_cast<long long>(encodeMs.count()), std::chrono::duration<double>(encodeEnd - encodeStart).count());
    if (outBuf.empty())
        return 1;

    FILE* fp = nullptr;
    openFile(&fp, outputPath.c_str(), "wb");
    if (fp == nullptr)
        return printf("Error opening output file '%s'\n", outputPath.c_str());

    printf("Writing '%s'...\n", outputPath.c_str());
    const size_t wrote = fwrite(outBuf.data(), 1, outBuf.size(), fp);
    fclose(fp);

    if (wrote != outBuf.size())
        return printf("Error! wrote %llu of %llu bytes\n", (unsigned long long)wrote, (unsigned long long)outBuf.size());

    printf("Wrote %s\n", printBytes(wrote).c_str());
    return 0;
}

bool readFileToBuffer(const std::string& inputPath, std::vector<uint8_t>& buffer)
{
    FILE* fp = nullptr;
    openFile(&fp, inputPath.c_str(), "rb");
    if (fp == nullptr)
    {
        printf("Error! Could not open file '%s'\n", inputPath.c_str());
        return false;
    }

    if (std::fseek(fp, 0, SEEK_END) != 0)
    {
        fclose(fp);
        printf("Error! Could not seek file '%s'\n", inputPath.c_str());
        return false;
    }

    const long fileSizeLong = std::ftell(fp);
    if (fileSizeLong < 0)
    {
        fclose(fp);
        printf("Error! Could not determine file size for '%s'\n", inputPath.c_str());
        return false;
    }

    if (std::fseek(fp, 0, SEEK_SET) != 0)
    {
        fclose(fp);
        printf("Error! Could not rewind file '%s'\n", inputPath.c_str());
        return false;
    }

    const size_t fileSize = static_cast<size_t>(fileSizeLong);
    buffer.assign(fileSize, 0u);

    if (fileSize > 0)
    {
        const size_t bytesRead = std::fread(buffer.data(), 1, fileSize, fp);
        fclose(fp);
        if (bytesRead != fileSize)
        {
            printf("Error! Read %zu of %zu bytes from '%s'\n", bytesRead, fileSize, inputPath.c_str());
            return false;
        }
    }
    else
    {
        fclose(fp);
    }

    return true;
}

int decodeInput(const std::string& inputPath, const std::string& outputPath)
{
    std::vector<uint8_t> inBuf;
    if (!readFileToBuffer(inputPath, inBuf))
        return 1;

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
    outFileInfo.samplerate = static_cast<int>(hdr.sampleRate);
    outFileInfo.format = SF_FORMAT_WAV | SF_FORMAT_FLOAT;

    outFile = sf_open(outputPath.c_str(), SFM_WRITE, &outFileInfo);
    if (outFile == nullptr)
    {
        printf("Cannot open '%s'.\n%s\n", outputPath.c_str(), sf_strerror(nullptr));
        return 1;
    }
    sf_writef_float(outFile, output.data(), hdr.totalPCMFrameCount);
    sf_close(outFile);
#else
    drwav_data_format format{};
    format.container = drwav_container_riff;
    format.format = DR_WAVE_FORMAT_IEEE_FLOAT;
    format.channels = hdr.channels;
    format.sampleRate = hdr.sampleRate;
    format.bitsPerSample = 32;

    const drwav_uint64 frameCount = static_cast<drwav_uint64>(output.size() / format.channels);

    drwav wav{};
    if (!drwav_init_file_write(&wav, outputPath.c_str(), &format, nullptr))
    {
        printf("Failed to open WAV for writing\n");
        return 1;
    }

    const drwav_uint64 framesWritten = drwav_write_pcm_frames(&wav, frameCount, output.data());
    drwav_uninit(&wav);
    if (framesWritten != frameCount)
    {
        printf("Failed to write all PCM frames\n");
        return 1;
    }
#endif

    printf("Wrote %s\n", outputPath.c_str());
    return 0;
}

void printUsage(const char* argv0)
{
    printf("Usage: %s input [-bits N] [-block FRAMES] [-lowpass HZ] [-normalize] [-gain FLOAT] [-o OUTPUT]\n",
        getFilenameFromPath(argv0).c_str());
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        printUsage(argv[0]);
        return 1;
    }

    ProgramOptions options;
    if (!parseCommandLine(argc, argv, options))
    {
        printUsage(argv[0]);
        return 1;
    }

    for (const std::string& inputPath : options.inputs)
    {
        const std::string outputPath = makeOutputPath(inputPath, options);
        const int result = isCWVPath(inputPath)
            ? decodeInput(inputPath, outputPath)
            : encodeInput(inputPath, outputPath, options);
        if (result != 0)
            return result;
    }

    return 0;
}
