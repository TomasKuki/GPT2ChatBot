
# GPT2ChatBot

A project to prepare a dataset train and use a gpt2 model to simulate a chat bot


## Installation

To install the project i recomend the use of a venv and the Installation of the requirements txt file.

```bash
    python -m venv env  # Crie o ambiente virtual
    source env/bin/activate  # Ative o ambiente virtual no macOS ou Linux
    env\Scripts\activate  # Ative o ambiente virtual no Windows
    pip install -r requirements.txt
```
    
## Create Dataset

To Utilize this project you can use the given Dataset "chatDataSet2.txt" or "chatDataSet.txt" or create your own.

To create your own you can use insert ebooks in the ./ebooks folder and use the "readBooks.py" script to create a txt file with the text from all the ebooks

Read Ebooks and create the txt file:
```bash
  python readBooks.py
```
Modify the txt file to be ready for use:

1- insert the word "phrase" on the first line
![1Line](https://github.com/TomasKuki/GPT2ChatBot/blob/main/documentation/1Line.png?raw=true)

2-run the "dataManipulation2.py" script to insert a "," at each end of line
```bash
  python dataManipulation2.py
```

## Train the model

The next step to Utilize this project is train the model you can also use the model given in the ./model folder but its a very small and unrealistic model who was train in a very small dataset so its not going to perform like it should.

To train the model you need to run the "gpt2Train.py" script
```bash
  python gpt2Train.py
```
This is going to take time depending on the dataset and computer used.
In the end the script shows a graphic of the train losses to show the improvent of the model its also possible to see the model losses changing when each epoch ends.

In the training process you can Utilize the token from the "huggingface" or the files give in the ./token folder to do this you need to change this line:

(Utilize the token from the "huggingface")
```bash
  self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
```

to this 

(Utilize the token the ./token folder)
```bash
  self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
```

## Use the model

The next step to Utilize this project is using the model for this you only need to run the "gpt2Use.py" script

```bash
  python gpt2Use.py
```
this is going to start the chat! to end the chat you just type
```bash
  goodbye
```



## Check if you can use the graphics card

"./checkCuda.py"

```javascript
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU disponível. Usando GPU.")
else:
    device = torch.device("cpu")
    print("GPU não disponível. Usando CPU.")

```

## Install the requirements.txt

"./requirements.txt"

```javascript
pip install -r requirements.txt

```


## Acknowledgements

 - [Free Ebooks Used](https://www.gutenberg.org/)


## Authors

- [@TomasKuki](https://www.github.com/TomasKuki)

