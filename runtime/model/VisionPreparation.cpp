#include "model/VisionPreparation.hpp"
#include "model/Bfloat16.hpp"
#include "model/WeightImages.hpp"
#include "model/WeightLayout.hpp"
#include "model/WeightStore.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <functional>
#include <span>

namespace splash::model::vision {
namespace {

uint64_t rowBytes(const SourceTensor &tensor, uint32_t columns) {
  if (tensor.dtype == "Q8_0") {
    // A GGUF vector is one block-aligned row, represented by scalar rows
    // in the vision image. Budget it conservatively as one block per row.
    if (columns == 1) return 34;
    if (columns % 32) throw WeightStoreError("vision Q8_0 row is not block aligned");
    return uint64_t(columns / 32) * 34;
  }
  return uint64_t(columns) * (tensor.dtype == "F32" ? 4 : 2);
}

// Converts count values of tensor to BF16 bits. False when a value is not
// exactly a BF16, or a Q8_0 scale is non-finite.
bool convert(const SourceTensor &tensor, const uint8_t *source, uint16_t *destination, uint64_t count) {
  if (tensor.dtype == "BF16") {
    std::memcpy(destination, source, count * kBFloat16Bytes);
    return true;
  }
  if (tensor.dtype == "Q8_0") {
    // GGML block_q8_0: little-endian half scale, then 32 signed int8 values.
    // Half times int8 is exact in F32. The vision operator consumes BF16,
    // rounded once to nearest with ties to even, including signed zero.
    for (uint64_t block = 0; block < count / 32; ++block) {
      _Float16 half;
      std::memcpy(&half, source + block * 34, 2);
      const float scale = static_cast<float>(half);
      if (!std::isfinite(scale)) return false;
      for (uint32_t i = 0; i < 32; ++i) {
        int8_t code;
        std::memcpy(&code, source + block * 34 + 2 + i, 1);
        const uint32_t bits = std::bit_cast<uint32_t>(scale * static_cast<float>(code));
        destination[block * 32 + i] = static_cast<uint16_t>((bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16);
      }
    }
    return true;
  }
  for (uint64_t i = 0; i < count; ++i) {
    float value;
    if (tensor.dtype == "F32") {
      std::memcpy(&value, source + i * 4, 4);
    } else {
      _Float16 half;
      std::memcpy(&half, source + i * 2, 2);
      value = static_cast<float>(half);
    }
    const auto bits = exactBfloat16(value);
    if (!bits) return false;
    destination[i] = *bits;
  }
  return true;
}

// Source bytes and patch values of one task.
struct Staging {
  std::vector<uint8_t> source = std::vector<uint8_t>(kLoadStepBytes / loadThreads());
  std::vector<uint16_t> values = std::vector<uint16_t>(kLoadStepBytes / loadThreads() / kBFloat16Bytes);
};

// The tasks that write a section in batches of whole rows converted to BF16.
// The image's patch embedding orders a row [channel, frame, patch-row,
// patch-col]; MLX stores [frame, patch-row, patch-col, channel] and GGUF one
// [channel, patch-row, patch-col] tensor per frame. Padded columns are zero;
// padded rows are not written.
void addSectionTasks(uint8_t *image, const Section &s, uint32_t pixels,
                     std::vector<std::function<void(Staging &)>> &tasks) {
  const auto frames = static_cast<uint32_t>(s.inputs.size());
  const uint32_t columns = s.columns / frames;
  uint64_t widest = 0;
  for (const auto &in : s.inputs) widest = std::max(widest, rowBytes(in.tensor, columns));
  const uint64_t stagedRowBytes = widest + (s.patch ? uint64_t(s.columns) * kBFloat16Bytes : 0);
  // Split one load step across all workers. Each worker retains at most a
  // step share for source and patch storage, bounding aggregate staging at
  // twice kLoadStepBytes, regardless of core count or Q8 expansion.
  const uint64_t stepBytes = kLoadStepBytes / loadThreads();
  if (!stagedRowBytes || stagedRowBytes > stepBytes)
    throw WeightStoreError("vision row exceeds the staging budget");
  uint32_t batchRows = static_cast<uint32_t>(std::min<uint64_t>(stepBytes / stagedRowBytes, s.rows));
  const bool blockRows = columns == 1 && std::any_of(s.inputs.begin(), s.inputs.end(),
      [](const Input &in) { return in.tensor.dtype == "Q8_0"; });
  if (blockRows) {
    if (s.patch || s.storedColumns != 1 || s.rows % 32)
      throw WeightStoreError("vision Q8_0 vector is not block aligned");
    batchRows -= batchRows % 32;
    if (!batchRows) throw WeightStoreError("vision Q8_0 vector exceeds the staging budget");
  }
  for (uint32_t row = 0; row < s.rows; row += batchRows) {
    const uint32_t count = std::min(batchRows, s.rows - row);
    auto *output = reinterpret_cast<uint16_t *>(image + s.offset) + uint64_t(row) * s.storedColumns;
    tasks.push_back([&s, pixels, frames, columns, row, count, output](Staging &staging) {
      // Converts count rows of one input to rows of stride BF16 values.
      const auto convertRows = [&](const Input &in, uint16_t *to, uint32_t stride) {
        if (in.tensor.dtype == "Q8_0" && columns == 1) {
          const uint64_t bytes = uint64_t(count / 32) * 34;
          in.tensor.read(uint64_t(row / 32) * 34, std::span(staging.source).first(bytes));
          if (!convert(in.tensor, staging.source.data(), to, count))
            throw WeightStoreError("vision tensor " + in.name + " in " + in.tensor.file->path().string() +
                                   " has a non-finite Q8_0 scale");
          return;
        }
        const uint64_t bytes = rowBytes(in.tensor, columns);
        in.tensor.read(row * bytes, std::span(staging.source).first(count * bytes));
        for (uint32_t r = 0; r < count; ++r)
          if (!convert(in.tensor, staging.source.data() + r * bytes, to + uint64_t(r) * stride, columns))
            throw WeightStoreError("vision tensor " + in.name + " in " + in.tensor.file->path().string() +
                                   (in.tensor.dtype == "Q8_0" ? " has a non-finite Q8_0 scale" : " is not exactly representable in BF16"));
      };
      if (!s.patch) {
        convertRows(s.inputs.front(), output, s.storedColumns);
      } else {
        uint16_t *values = staging.values.data();
        for (uint32_t frame = 0; frame < frames; ++frame)
          convertRows(s.inputs[frame], values + uint64_t(frame) * count * columns, columns);
        for (uint32_t r = 0; r < count; ++r)
          for (uint32_t c = 0; c < s.columns; ++c) {
            const uint32_t channel = c / (2 * pixels), frame = c / pixels % 2, pixel = c % pixels;
            output[uint64_t(r) * s.storedColumns + c] =
                frames == 1 ? values[uint64_t(r) * columns + (uint64_t(frame) * pixels + pixel) * 3 + channel]
                            : values[(uint64_t(frame) * count + r) * columns + uint64_t(channel) * pixels + pixel];
          }
      }
      for (uint32_t r = 0; r < count; ++r)
        std::fill_n(output + uint64_t(r) * s.storedColumns + s.columns, s.storedColumns - s.columns, uint16_t{0});
    });
  }
}

} // namespace

void writeVision(std::span<uint8_t> destination, const Plan &plan) {
  if (destination.size() != plan.bytes) throw WeightStoreError("vision image destination size differs");
  const auto header = weightFileHeader(kVisionMagic, plan.depth, 0);
  std::vector<std::pair<uint64_t, uint64_t>> extents{{0, header.size()}};
  for (const auto &s : plan.sections)
    extents.emplace_back(s.offset, uint64_t(s.rows) * s.storedColumns * kBFloat16Bytes);
  zeroUnwritten(destination, std::move(extents));
  std::memcpy(destination.data(), header.data(), header.size());
  std::vector<std::function<void(Staging &)>> tasks;
  for (const auto &s : plan.sections)
    addSectionTasks(destination.data(), s, plan.patchSize * plan.patchSize, tasks);
  std::vector<Staging> staging(loadThreads());
  parallelFor(tasks.size(), [&](size_t index, unsigned thread) { tasks[index](staging[thread]); });
}

} // namespace splash::model::vision
