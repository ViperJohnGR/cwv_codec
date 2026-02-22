#include "helpers.hpp" 


std::string getFilenameFromPath(const std::string& path)
{
    return path.substr(path.find_last_of("/\\") + 1);
}

std::string getExtensionFromPath(const std::string& path)
{
    size_t pos = path.find_last_of('.');
    if (pos != std::string::npos)
        return path.substr(pos + 1);

    return "";
}

std::string removeExtensionFromPath(const std::string& path)
{
    size_t dot = path.find_last_of('.');

    if (dot == std::string::npos)
        return path;

    return path.substr(0, dot);
}

std::string printBytes(std::uint64_t bytes)
{
    char buffer[255] = {};

    const double KB = 1024.0;
    const double MB = 1024.0 * KB;
    const double GB = 1024.0 * MB;

    if (bytes >= (std::uint64_t)GB) {
        snprintf(buffer, 255, "%.2f GB", bytes / GB);
    }
    else if (bytes >= (std::uint64_t)MB) {
        snprintf(buffer, 255, "%.2f MB", bytes / MB);
    }
    else {
        // For anything smaller than 1 MB, show KB (including < 1 KB)
        snprintf(buffer, 255, "%.2f KB", bytes / KB);
    }

    return buffer;
}
