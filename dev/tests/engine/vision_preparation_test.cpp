// Writes the tiny vision source run_vision_preparation.py writes (depth 2,
// width 8, 2x2 patches) and prints its image's SHA-256.
//
//   vision-preparation safetensors|gguf DIRECTORY EXPECTED
//   vision-preparation padding DIRECTORY
//
// EXPECTED is an independently serialized file the image and the size
// estimate must equal. padding writes a padded section after an unpadded one
// and checks its padding is zero.

#include "TestFiles.hpp"
#include "model/VisionLoader.hpp"
#include "model/VisionPreparation.hpp"
#include "model/WeightLayout.hpp"
#include "model/WeightStore.hpp"

#include <algorithm>
#include <bit>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace splash;

namespace {

// A 1x8 section, then a 2x3 section stored as 2x4, written from one BF16
// source: the second section's rows are staged after the first's, and its
// padding column must still be zero.
void checkPadding(const std::filesystem::path &directory) {
  std::vector<uint16_t> values(14);
  for (size_t i = 0; i < values.size(); ++i) values[i] = uint16_t(0x3F80 + i); // 1 + i/128
  test::writeFile(directory / "source.bin",
                  {reinterpret_cast<const uint8_t *>(values.data()), values.size() * sizeof(uint16_t)});
  const model::WeightSource source(directory / "source.bin");
  const model::SourceTensor first{&source, "BF16", {1, 8}, 0, 16}, padded{&source, "BF16", {2, 3}, 16, 12};
  model::vision::Plan plan{.patchSize = 2, .bytes = 3 * model::kWeightFileAlignment};
  plan.sections.push_back({.rows = 1, .columns = 8, .storedRows = 1, .storedColumns = 8,
                           .offset = model::kWeightFileAlignment, .inputs = {{"first", first}}});
  plan.sections.push_back({.rows = 2, .columns = 3, .storedRows = 2, .storedColumns = 4,
                           .offset = 2 * model::kWeightFileAlignment, .inputs = {{"padded", padded}}});
  // Written over nonzero memory, as an image's buffer may hold.
  std::vector<uint8_t> bytes(plan.bytes, 0xFF);
  model::vision::writeVision(bytes, plan);
  std::vector<uint16_t> section(8);
  std::memcpy(section.data(), bytes.data() + plan.sections[1].offset, section.size() * sizeof(uint16_t));
  const std::vector<uint16_t> expected{values[8], values[9], values[10], 0, values[11], values[12], values[13], 0};
  if (section != expected) throw std::runtime_error("a padded section's padding columns are not zero");
  if (std::any_of(bytes.begin() + 16, bytes.begin() + plan.sections[0].offset, [](uint8_t b) { return b; }))
    throw std::runtime_error("the bytes between the header and the first section are not zero");
}

// A 3.4 MB Q8 tensor spans several row batches, with padding after every
// row. Exact power-of-two scales make this oracle independent of BF16 rounding.
void checkQ8Batches(const std::filesystem::path &directory) {
  constexpr uint32_t rows = 50000, columns = 64, storedColumns = 80;
  std::vector<uint8_t> raw(uint64_t(rows) * 2 * 34);
  std::vector<uint16_t> expected(uint64_t(rows) * storedColumns, 0);
  for (uint64_t block = 0; block < uint64_t(rows) * 2; ++block) {
    const _Float16 scale = block % 2 ? -0.5 : 0.125;
    std::memcpy(raw.data() + block * 34, &scale, 2);
    for (uint32_t i = 0; i < 32; ++i) {
      const int8_t code = static_cast<int8_t>(int((block * 13 + i * 17) % 256) - 128);
      std::memcpy(raw.data() + block * 34 + 2 + i, &code, 1);
      const float value = float(scale) * float(code);
      expected[(block / 2) * storedColumns + (block % 2) * 32 + i] =
          static_cast<uint16_t>(std::bit_cast<uint32_t>(value) >> 16);
    }
  }
  test::writeFile(directory / "q8.bin", raw);
  const model::WeightSource source(directory / "q8.bin");
  const model::SourceTensor tensor{&source, "Q8_0", {columns, rows}, 0, raw.size()};
  model::vision::Plan plan{.bytes = model::alignWeightOffset(
      model::kWeightFileAlignment + uint64_t(rows) * storedColumns * 2)};
  plan.sections.push_back({.rows = rows, .columns = columns, .storedRows = rows,
                          .storedColumns = storedColumns, .offset = model::kWeightFileAlignment,
                          .inputs = {{"q8", tensor}}});
  const uint64_t vectorOffset = plan.bytes;
  plan.bytes += model::kWeightFileAlignment;
  constexpr uint32_t vectorRows = 50016;
  const model::SourceTensor vector{&source, "Q8_0", {vectorRows}, 0, uint64_t(vectorRows / 32) * 34};
  plan.bytes = model::alignWeightOffset(vectorOffset + uint64_t(vectorRows) * 2);
  plan.sections.push_back({.rows = vectorRows, .columns = 1, .storedRows = vectorRows, .storedColumns = 1,
                          .offset = vectorOffset, .inputs = {{"q8-vector", vector}}});
  std::vector<uint8_t> bytes(plan.bytes, 0xFF);
  model::vision::writeVision(bytes, plan);
  const auto *vectorValues = reinterpret_cast<const uint16_t *>(bytes.data() + vectorOffset);
  for (uint32_t i = 0; i < vectorRows; ++i)
    if (vectorValues[i] != expected[uint64_t(i / columns) * storedColumns + i % columns])
      throw std::runtime_error("Q8_0 vector differs from the independent oracle");
  if (std::memcmp(bytes.data() + model::kWeightFileAlignment, expected.data(), expected.size() * 2))
    throw std::runtime_error("Q8_0 batches differ from the independent padded row oracle");
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc == 3 && std::string_view(argv[1]) == "q8-batches") {
      checkQ8Batches(argv[2]);
      std::cout << "Q8_0 batches and padding PASS\n";
      return 0;
    }
    if (argc == 3 && std::string_view(argv[1]) == "padding") {
      checkPadding(argv[2]);
      std::cout << "padding zero\n";
      return 0;
    }
    if (argc != 4)
      throw std::runtime_error("usage: vision-preparation safetensors|gguf DIRECTORY EXPECTED | padding DIRECTORY");
    const std::string format = argv[1];
    if (format != "safetensors" && format != "gguf") throw std::runtime_error("invalid vision-preparation arguments");
    const auto source = format == "safetensors" ? model::VisionSource::Safetensors
                                        : model::VisionSource::Gguf;
    ops::VisionLayout layout;
    layout.depth = 2;
    layout.hiddenSize = 8;
    layout.patchSize = 2;
    layout.patchDimension = 24;
    layout.intermediateSize = 10;
    layout.paddedIntermediateSize = 16;
    layout.mergedHiddenSize = 32;
    layout.outputHiddenSize = 8;
    layout.heads = 2;
    layout.headDimension = 4;
    layout.positionGridSide = 2;
    const model::ImagePlan image = model::VisionLoader(argv[2], source, layout).image();
    std::vector<uint8_t> bytes(image.bytes);
    image.write(bytes, {});
    const auto expected = test::readFile(argv[3]);
    const auto differs = std::mismatch(bytes.begin(), bytes.end(), expected.begin(), expected.end());
    if (differs.first != bytes.end() || differs.second != expected.end())
      throw std::runtime_error(
          "the image differs from the oracle from byte " +
          std::to_string(differs.first - bytes.begin()) + " (" +
          std::to_string(bytes.size()) + " written, " +
          std::to_string(expected.size()) + " expected)");
    if (model::visionImageBytes(layout) != expected.size())
      throw std::runtime_error("vision size estimate differs from the oracle");
    std::cout << model::weightDigest(bytes) << '\n';
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
