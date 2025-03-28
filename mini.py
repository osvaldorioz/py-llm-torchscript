import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 🔹 Definimos un modelo Transformer básico
class MiniTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_layers):
        super(MiniTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, num_heads, hidden_dim),
            num_layers
        )
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=-1)

# 🔹 Hiperparámetros
vocab_size = 1000
embed_dim = 64
num_heads = 4
hidden_dim = 128
num_layers = 2

# 🔹 Creamos el modelo
model = MiniTransformer(vocab_size, embed_dim, num_heads, hidden_dim, num_layers)

# 🔹 Guardamos el modelo como TorchScript para usarlo en C++
dummy_input = torch.randint(0, vocab_size, (1, 10))  # 10 tokens de entrada
traced_model = torch.jit.trace(model, dummy_input)
traced_model.save("mini_transformer.pt")
print("Modelo guardado como TorchScript.")
