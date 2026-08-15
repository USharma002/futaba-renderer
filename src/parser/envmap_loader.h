#pragma once

#include "types.cuh"
#include <string>
#include <vector>

FUTABA_NAMESPACE_BEGIN

bool loadEnvMapHDR(const std::string& filename,
                   const std::string& baseDir,
                   std::vector<Color3f>& pixels,
                   int& width,
                   int& height,
                   std::string& errorOut);

bool loadEnvMapEXR(const std::string& filename,
                   const std::string& baseDir,
                   std::vector<Color3f>& pixels,
                   int& width,
                   int& height,
                   std::string& errorOut);

FUTABA_NAMESPACE_END
