#pragma once

#include "types.cuh"
#include <vector>

namespace futaba {

// Block: Tile-based rendering infrastructure (Mitsuba pattern)
// Used for parallel rendering and memory management
struct Block {
  int x, y;          // Top-left corner
  int width, height; // Dimensions
  int tileIndex;     // Unique ID

  Block(int x_ = 0, int y_ = 0, int w = 32, int h = 32, int idx = 0)
      : x(x_), y(y_), width(w), height(h), tileIndex(idx) {}

  // Check if pixel (px, py) is in this block
  bool contains(int px, int py) const {
    return px >= x && px < x + width && py >= y && py < y + height;
  }
};

// BlockScheduler: Manages tile-based rendering queue
class BlockScheduler {
private:
  std::vector<Block> m_blocks;
  int m_tileSize;
  int m_imageWidth, m_imageHeight;

public:
  BlockScheduler(int imgWidth, int imgHeight, int tileSize = 32)
      : m_tileSize(tileSize), m_imageWidth(imgWidth), m_imageHeight(imgHeight) {
    generateBlocks();
  }

  // Generate tile grid
  void generateBlocks() {
    m_blocks.clear();
    int tileIdx = 0;
    for (int y = 0; y < m_imageHeight; y += m_tileSize) {
      for (int x = 0; x < m_imageWidth; x += m_tileSize) {
        int w = std::min(m_tileSize, m_imageWidth - x);
        int h = std::min(m_tileSize, m_imageHeight - y);
        m_blocks.emplace_back(x, y, w, h, tileIdx++);
      }
    }
  }

  const std::vector<Block> &getBlocks() const { return m_blocks; }
  int getBlockCount() const { return (int)m_blocks.size(); }

  // TODO: Implement smart scheduling (progressive refinement)
  // void scheduleProgressive() { ... }

  // TODO: Implement adaptive scheduling (prioritize high-variance regions)
  // void scheduleAdaptive(const Bitmap& variance) { ... }
};

} // namespace futaba
