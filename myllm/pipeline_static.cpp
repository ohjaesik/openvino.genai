// This file contains the modifications needed to align the rest of the pipeline with your modified `pipeline.cpp`.
// The key change is applying `patch_npu_properties()` before NPU pipeline creation.


#include "pipeline_static.hpp"
#include "utils.hpp"

namespace static_llm {

std::unique_ptr<ov::genai::PipelineBase> LLMPipelineFactory::create(
    const std::filesystem::path& models_path,
    const ov::genai::Tokenizer& tokenizer,
    ov::AnyMap properties) {

    // Apply NPU-safe defaults if targeting NPU
    if (properties.count("DEVICE") && properties["DEVICE"].as<std::string>() == "NPU") {
        patch_npu_properties(properties);
    }

    auto [plugin_config, scheduler_config] = ov::genai::utils::extract_scheduler_config(properties);
    auto model = ov::genai::utils::singleton_core().read_model(models_path / "openvino_model.xml", {}, plugin_config);
    auto generation_config = ov::genai::utils::from_config_json_if_exists(models_path);

    return std::make_unique<StaticLLMPipeline>(model, tokenizer, plugin_config, generation_config);
}

std::unique_ptr<ov::genai::PipelineBase> LLMPipelineFactory::create(
    const std::shared_ptr<ov::Model>& model,
    const ov::genai::Tokenizer& tokenizer,
    ov::AnyMap properties,
    const ov::genai::GenerationConfig& generation_config) {

    // Apply NPU-safe defaults if targeting NPU
    if (properties.count("DEVICE") && properties["DEVICE"].as<std::string>() == "NPU") {
        patch_npu_properties(properties);
    }

    auto [plugin_config, _] = ov::genai::utils::extract_scheduler_config(properties);
    return std::make_unique<StaticLLMPipeline>(model, tokenizer, plugin_config, generation_config);
}

