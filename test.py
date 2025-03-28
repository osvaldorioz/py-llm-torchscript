import transformer_llm

# 🔹 Cargar modelo en C++
llm = transformer_llm.TransformerLLM("mini_transformer.pt")

# 🔹 Generar texto
input_str = input("Ingresa varios tokens (ejemplo: 5, 42, 17, 88, 77): ")
lg = input("Ingresa longitud: ")

try:
    # 🔹 Convertir el texto en una lista
    array = eval(input_str)

    # 🔹 Mostrar el array
    print("Array leído:", array)
    input_tokens = array  
    max_length = eval(lg)
    generated_tokens = llm.generate_text(input_tokens, max_length)

    print("Texto generado (tokens):", generated_tokens)
except ValueError:
    print("Existe un problema con los valores ingresados")


