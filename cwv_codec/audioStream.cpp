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
        printf("Cannot open '%s'. %s\n", path.c_str(), sf_strerror(NULL));

    sampleData.resize(info.frames * info.channels);
    sf_readf_float(file, &sampleData[0], info.frames);
    sf_close(file);

    channels = static_cast<uint8_t>(info.channels); //TODO: throw error if channels > 255 || channels < 1
    sampleRate = info.samplerate;
    totalPCMFrameCount = info.frames;
}

