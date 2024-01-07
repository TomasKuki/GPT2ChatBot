import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config

# Defina a configuração personalizada do GPT-2
custom_config_dict = {
    "activation_function": "gelu_new",
    "model_type": "gpt2",
    "n_ctx": 1024,
    "n_embd": 768,
    "n_head": 12,
    "n_layer": 12,
    "n_positions": 1024,
    "vocab_size": 50257  # Este número pode variar dependendo do tokenizer utilizado
}

# Crie uma classe de modelo GPT-2 personalizada
class MyGPT2Model(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)

# Carregue o modelo GPT-2 personalizado
model = MyGPT2Model(config=GPT2Config.from_dict(custom_config_dict))
model_path = "./model"  # Substitua pelo caminho real do modelo
model = MyGPT2Model.from_pretrained(model_path)
model.eval()

# Carregue o tokenizador
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Função para manter uma conversa
def conversa(model, tokenizer):
    print("Bot: Hello! How can I help you? (Type 'goodbye' to close)")

    while True:
        # Obtenha a entrada do utilizador
        user_input = input("You: ")

        # Verifique se o utilizador deseja sair
        if user_input.lower() == 'goodbye':
            print("Bot: goodbye until next time!")
            break

        # Tokenize a entrada do utilizador
        input_ids = tokenizer.encode(user_input, return_tensors='pt')

        # Check if input_ids is not None and not empty
        if input_ids is not None and input_ids.numel() > 0:
            # Extract the attention mask from the tokenizer output
            attention_mask = torch.ones_like(input_ids)

            # Gere uma resposta do modelo
            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    max_length=50,
                    num_beams=5,
                    no_repeat_ngram_size=2,
                    top_k=10,
                    do_sample=True,  # Set do_sample to True
                    top_p=0.9,# Optionally remove top_p or set it to a different value
                    temperature=0.2,  # Adjust the temperature
                    attention_mask=attention_mask,  # Pass the attention mask
                    pad_token_id=tokenizer.eos_token_id  # Set pad_token_id explicitly
                )
            
            # Decode e imprima a resposta
            bot_response = tokenizer.decode(output[0], skip_special_tokens=True)
            print(f"Bot: {bot_response}")
        else:
            print("Bot: I'm sorry, but I couldn't understand your input.")

# Inicie a conversa
conversa(model, tokenizer)