#include "audioStream.hpp"



audioStream::audioStream()
{
    channels = 0;
    sampleRate = 0;
    totalPCMFrameCount = 0;
}

audioStream::audioStream(const std::string& path)
{
    SF_INFO info{};
    SNDFILE* file = sf_open(path.c_str(), SFM_READ, &info);
    if (file == nullptr)
    {
        channels = 0;
        sampleRate = 0;
        totalPCMFrameCount = 0;
        printf("Cannot open '%s'. %s\n", path.c_str(), sf_strerror(NULL));
        return;
    }

    if (info.channels < 1 || info.channels > 255)
    {
        channels = 0;
        sampleRate = 0;
        totalPCMFrameCount = 0;
        printf("Error. Invalid number of channels (%d).\n", info.channels);
        return;
    }

    sampleData.resize(info.frames * info.channels);
    sf_readf_float(file, &sampleData[0], info.frames);
    sf_close(file);

    channels = static_cast<uint8_t>(info.channels);
    sampleRate = info.samplerate;
    totalPCMFrameCount = info.frames;
}

