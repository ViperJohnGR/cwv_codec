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
