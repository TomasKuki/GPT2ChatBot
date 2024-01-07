import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

print(torch.cuda.is_available())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(0))

# Define the custom GPT-2 configuration
custom_config_dict = {
    "activation_function": "gelu_new", # função de ativação a ser utilizada
    "model_type": "gpt2", # tipo de modelo 
    "n_ctx": 1024, # tamanho do contexto ou o numero maximo de tokens a ser utilizado por sequencia
    "n_embd": 768, # tamanho dos vetores de representacao que o modelo usa para cada token
    "n_head": 12,# numero de heads a ser utilizado  no modelo de atencao 
    "n_layer": 12,# numero de camadas para a rede neural
    "n_positions": 1024, # numero maximo de posicoes que o modelo pode utilizar
    "vocab_size": 50257 # tamanho do vocabulario que o modelo pode reconhecer numero de tokens ou palavras que o modelo pode reconhecer
}

# Create a simple dataset class for titles and phrases
class QADataset(Dataset):
    def __init__(self, file_path):  # definir o contrutor que le o dataset
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        self.titles = [line.split(',')[0] for line in lines]
        self.phrases = [line.split(',')[1] for line in lines]

        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2') #inicializa o tokenizador gpt2

    def __len__(self):  # devolve o numero de exemplos no dataset
        return len(self.titles)

    def __getitem__(self, idx):  # devolve um conjunto de dados com base no indice
        title = self.titles[idx]
        phrase = self.phrases[idx]
        
        # Concatenate title and phrase to form input
        input_text = title + ', ' + phrase
        
        # Tokenize input text
        tokenized_input = self.tokenizer.encode(input_text, return_tensors='pt')

        return tokenized_input

# Create the model GPT-2 custom
model = GPT2LMHeadModel(config=GPT2Config(**custom_config_dict))  # criacao do modelo com a configuração personalizada

# Load the dataset
file_path = 'chatDataSet2.txt' 
dataset = QADataset(file_path) ## atribuir o dataset
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Train the model
num_epochs = 32 # numero de epochs a ser utilizado no treino
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # escolha do dispositivo a ser utilizado /grafica pu /cpu
print(f'Using {device}')
model.to(device) #atribuir o dispositivo
model.train() # treinar o modelo

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)  # configurar o otimizador "AdamW"

# Initialize an empty list to store the training loss values
training_losses = []

# Training loop
for epoch in range(num_epochs):
    epoch_loss = 0.0  # variavel para aculumar o valor das loss
    for batch_idx, batch in enumerate(tqdm(dataloader)):  # inicia um loop sobre o dataloader que fornece batches de dados # tqdm é um wrapper que adicionar uma barra de progresso 
        optimizer.zero_grad()  # serve para zerar os gradientes aculumados no otimizador

        # Adjust batch size and indices
        input_ids_batch = batch.squeeze(0).to(device)  # remove dimensoes desnecessarias e move o lote para o dispositivo utilizado /grafica ou /cpu
        labels = input_ids_batch.clone() # cria uma copia para ser utilizada como rotulo

        outputs = model(input_ids_batch, labels=labels)  # insere os dados e os rotulos no modelo
        loss = outputs.loss # extrai as perdas da do modelo
        loss.backward() # determina como mudar os parametros para diminuir a perda
        optimizer.step() # atualiza os parametros do modelo com base nos gradiantes calculados "etapa onde o modelo aprende"

        # Accumulate the loss for the epoch
        epoch_loss += loss.item()

    # Calculate average loss for the epoch
    avg_epoch_loss = epoch_loss / len(dataloader)
    
    # Log the average loss for the epoch
    print(f"Epoch {epoch + 1}, Average Loss: {avg_epoch_loss}")
    
    # Append the average loss to the list for plotting
    training_losses.append(avg_epoch_loss)

    # Clear GPU memory
    torch.cuda.empty_cache()

# Plot the training loss curve
plt.plot(training_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Save the trained model
model.save_pretrained("./model")
