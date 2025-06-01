#include "pipeline_stateful.hpp"
#include "utils.hpp"

namespace ov {
namespace genai {

StatefulLLMPipeline::StatefulLLMPipeline(
    const std::filesystem::path& models_path,
    const ov::genai::Tokenizer& tokenizer,
    const std::string& device,
    ov::AnyMap properties) {

    if (device == "NPU") {
        patch_npu_properties(properties);
    }

    // existing logic...
}

StatefulLLMPipeline::StatefulLLMPipeline(
    const std::shared_ptr<ov::Model>& model,
    const ov::genai::Tokenizer& tokenizer,
    const std::string& device,
    ov::AnyMap properties,
    const GenerationConfig& generation_config) {

    if (device == "NPU") {
        patch_npu_properties(properties);
    }

    // existing logic...
}

} // namespace genai
} // namespace ov

