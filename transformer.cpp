#include <torch/script.h>
#include <iostream>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

class TransformerLLM {
private:
    torch::jit::script::Module model;

public:
    TransformerLLM(const std::string& model_path) {
        try {
            model = torch::jit::load(model_path);
            std::cout << "Modelo cargado correctamente.\n";
        } catch (const c10::Error& e) {
            std::cerr << "Error cargando el modelo: " << e.what() << std::endl;
        }
    }

    /*
    std::vector<int64_t> generate_text(const std::vector<int64_t>& input_tokens, int max_length) {
        torch::Tensor input_tensor = torch::tensor(input_tokens, torch::kInt64).unsqueeze(0);
        std::vector<int64_t> generated_text = input_tokens;

        for (int i = 0; i < max_length; ++i) {
            auto output = model.forward({input_tensor}).toTensor();
            int64_t next_token = output.argmax(-1).squeeze().item<int64_t>();

            generated_text.push_back(next_token);
            input_tensor = torch::cat({input_tensor, torch::tensor({next_token})}, 1);
        }
        return generated_text;
    }

    std::vector<int64_t> generate_text(const std::vector<int64_t>& input_tokens, int max_length) {
        torch::Tensor input_tensor = torch::tensor(input_tokens, torch::kInt64).unsqueeze(0);
        std::vector<int64_t> generated_text = input_tokens;
    
        for (int i = 0; i < max_length; ++i) {
            auto output = model.forward({input_tensor}).toTensor();
            auto token_tensor = output.argmax(-1);  // Obtiene el índice del token más probable
            int64_t next_token = token_tensor.flatten().item<int64_t>();  // Convierte correctamente a escalar
    
            generated_text.push_back(next_token);
            input_tensor = torch::cat({input_tensor, torch::tensor({next_token}, torch::kInt64).unsqueeze(0)}, 1);
        }
        return generated_text;
    }

    std::vector<int64_t> generate_text(const std::vector<int64_t>& input_tokens, int max_length) {
        torch::Tensor input_tensor = torch::tensor(input_tokens, torch::kInt64).unsqueeze(0);
        std::vector<int64_t> generated_text = input_tokens;
    
        for (int i = 0; i < max_length; ++i) {
            auto output = model.forward({input_tensor}).toTensor();
            auto token_tensor = output.argmax(-1);  // Obtiene el índice del token más probable
            
            std::cout << "Dimensiones del tensor generado: " << token_tensor.sizes() << std::endl;
            
            int64_t next_token = token_tensor[0].item<int64_t>();  // Extrae el primer valor de forma segura
    
            generated_text.push_back(next_token);
            input_tensor = torch::cat({input_tensor, torch::tensor({next_token}, torch::kInt64).unsqueeze(0)}, 1);
        }
        return generated_text;
    }*/

    std::vector<int64_t> generate_text(const std::vector<int64_t>& input_tokens, int max_length) {
        torch::Tensor input_tensor = torch::tensor(input_tokens, torch::kInt64).unsqueeze(0);
        std::vector<int64_t> generated_text = input_tokens;
    
        for (int i = 0; i < max_length; ++i) {
            auto output = model.forward({input_tensor}).toTensor();
            auto token_tensor = output.argmax(-1);  // Obtiene el índice del token más probable
    
            std::cout << "Dimensiones del tensor generado: " << token_tensor.sizes() << std::endl;
    
            int64_t next_token = token_tensor[0][-1].item<int64_t>();  // Obtiene solo el último token
    
            generated_text.push_back(next_token);
            input_tensor = torch::cat({input_tensor, torch::tensor({next_token}, torch::kInt64).unsqueeze(0)}, 1);
        }
        return generated_text;
    }
    
    
    
};

// 🔹 Exponemos la clase a Python con Pybind11
PYBIND11_MODULE(transformer_llm, m) {
    py::class_<TransformerLLM>(m, "TransformerLLM")
        .def(py::init<const std::string&>())
        .def("generate_text", &TransformerLLM::generate_text);
}
