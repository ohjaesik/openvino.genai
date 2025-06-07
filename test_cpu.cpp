
#include "./myllm/pipeline_stateful.hpp"
#include "openvino/core/core.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/genai/llm_pipeline.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "openvino/genai/generation_config.hpp"

#include <iostream>
#include <fstream>
#include <filesystem>

using namespace ov;
using ov::genai::Tokenizer;

constexpr size_t MAX_PROMPT_LEN = 32;
constexpr size_t MIN_RESPONSE_LEN = 32;

bool is_npu_available() {
    try {
        Core core;
        auto devices = core.get_available_devices();
        for (const auto& d : devices) {
            if (d.find("NPU") != std::string::npos) {
                return true;
            }
        }
    } catch (...) {}
    return false;
}

AnyMap configure_properties_for_device(const std::string& device) {
    AnyMap properties;
    if (device.find("NPU") != std::string::npos) {
        properties["STATIC_PIPELINE"] = true;
        properties["MAX_PROMPT_LEN"] = static_cast<int>(MAX_PROMPT_LEN);
        properties["MIN_RESPONSE_LEN"] = static_cast<int>(MIN_RESPONSE_LEN);
        properties["NUM_STREAMS"] = 1;
        properties["CACHE_DIR"] = "./ov_cache";
        properties["PERFORMANCE_HINT"] = "LATENCY";
    } else {
        properties["CACHE_DIR"] = "./ov_cache";
    }
    return properties;
}

std::shared_ptr<Model> load_and_reshape_model(const std::filesystem::path& model_path, const std::string& device) {
    Core core;
    auto model = core.read_model(model_path.string());
    if (device.find("NPU") != std::string::npos || device.find("MULTI") != std::string::npos || device.find("AUTO") != std::string::npos) {
        model->reshape({
            {"input_ids", PartialShape{1, MAX_PROMPT_LEN}},
            {"attention_mask", PartialShape{1, MAX_PROMPT_LEN}}
        });
    }
    return model;
}

ov::genai::StatefulLLMPipeline build_pipeline(const std::filesystem::path& model_path,
                                              const Tokenizer& tokenizer,
                                              const std::string& device) {
    auto properties = configure_properties_for_device(device);
    auto model = load_and_reshape_model(model_path / "openvino_model.xml", device);

    ov::genai::GenerationConfig gen_config;
    gen_config.max_length = MAX_PROMPT_LEN + MIN_RESPONSE_LEN;
    gen_config.eos_token_id = tokenizer.get_eos_token_id();
    gen_config.stop_token_ids = {gen_config.eos_token_id}; 
    gen_config.num_return_sequences = 1;
    gen_config.do_sample = false;

    return ov::genai::StatefulLLMPipeline(model, tokenizer, device, properties, gen_config);
}

int main() {
    std::string model_dir = "./open_llama_7b_v2-fp16-ov";

    std::string selected_device = is_npu_available() ? "MULTI:NPU,CPU" : "CPU";

    std::cout << "\n[Running on device: " << selected_device << "]\n" << std::endl;

    Tokenizer tokenizer(model_dir);
    auto pipeline = build_pipeline(model_dir, tokenizer, selected_device);

    std::string prompt = "What is the capital of France?";
    auto result = pipeline.generate({prompt}, std::nullopt, std::monostate{});

    for (const auto& output : result.texts) {
        std::cout << "[Answer]: " << output << std::endl;
    }

    return 0;
}
