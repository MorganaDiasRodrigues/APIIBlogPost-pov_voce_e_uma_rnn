# Gerando nomes brasileiros com RNNs

## Vamos fazer a previsão do próximo caractere baseado em uma lista de nomes brasileiros! :)

#### Imports


```python
import torch
import torch.nn as nn
import torch.optim as optim
# seed, para reproducibilidade
torch.manual_seed(0)
```




    <torch._C.Generator at 0x18d425e53d0>



#### Carregando os dados


```python
def carrega_dados(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        nomes = file.read().strip().split('\n')
    return nomes
```


```python
nomes = carrega_dados('nomes_pt.txt')
print(f'5 primeiros nomes: {nomes[:5]}')
```

    5 primeiros nomes: ['Ana', 'Lucas', 'Maria', 'João', 'Sofia']
    

#### Adicionando os tokens de Início e Fim dos nomes

Para ajudar o modelo a entender onde começou e onde terminou o nome.


```python
nomes = ['b/' + nome + '/e' for nome in nomes]
print(nomes[:5])
```

    ['b/Ana/e', 'b/Lucas/e', 'b/Maria/e', 'b/João/e', 'b/Sofia/e']
    

#### Criando o vocabulário, o encoder e o decoder


```python
vocab = sorted(set(''.join(nomes))) # Aqui, o vocabulário é o conjunto de todos os caracteres únicos presentes nos nomes
print(f'Tamanho do vocabulário: {len(vocab)} caracteres únicos')
encoder = {ch: i for i, ch in enumerate(vocab)} # Aqui, criamos um dicionário que mapeia cada caractere para um índice
decoder = {i: ch for i, ch in enumerate(vocab)} # Aqui, criamos um dicionário que mapeia cada índice para um caractere
```

    Tamanho do vocabulário: 47 caracteres únicos
    


```python
print(f'A letra L está codificada como o número {encoder["L"]}')
```

    A letra L está codificada como o número 11
    

### Funções auxiliares para o pré-processamento dos dados

#### Função para mapear cada caractere codificado para um tensor


```python
def char_tensor(string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        tensor[c] = encoder[string[c]]
    return tensor
```


```python
print(f'Nome Lucas codificado e em tensor: {char_tensor("Lucas")}')
```

    Nome Lucas codificado e em tensor: tensor([11, 37, 21, 19, 35])
    

#### Aplicando OHE


```python
def one_hot_encoding(tensor):
    return torch.nn.functional.one_hot(tensor, num_classes=len(vocab)).type(torch.float32) # Vai retornar um tensor com o tamanho do vocabulário, com 1s nas posições correspondentes aos caracteres presentes no tensor de entrada

```


```python
ohe_exemplo = one_hot_encoding(char_tensor('Lucas'))
print(f'One-hot encoding do nome Lucas: {ohe_exemplo} \n')
# Veja que em algumas posições, há o número 1. Signifa que ali havia alguma letra correspondente ao nome 'Lucas'
print(f'Tamanho do one-hot encoding: {ohe_exemplo.size()}')
```

    One-hot encoding do nome Lucas: tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]) 
    
    Tamanho do one-hot encoding: torch.Size([5, 47])
    

# Modelo RNN para previsão do próximo caractere


```python
class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CharRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True) # Olha aqui nossa diva! O shape dela será (tamanho_do_vocab, 100, tamanho_do_vocab)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        out, hidden = self.rnn(input, hidden)
        out = self.fc(out[:, -1, :]) # Pega o último output da sequência
        return out, hidden # Sempre retornar hidden para a próxima iteração

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, hidden_size) # Vamos inicializar o hidden com zeros.
```

#### Instanciando o modelo


```python
hidden_size = 100
model = CharRNN(len(vocab), hidden_size, len(vocab))
```

#### Definindo nossa função de perda e nosso otimizador


```python
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)
```

#### Função de treino


```python
def train(input_line_tensor, target_char_tensor, device):
    if device == 'cuda':
        input_line_tensor = input_line_tensor.cuda()
        target_char_tensor = target_char_tensor.cuda()
        model.to(device)
    hidden = model.init_hidden(1)
    hidden = hidden.to(device)
    model.zero_grad()
    loss = 0

    for i in range(input_line_tensor.size(0) - 1):
        input = input_line_tensor[i:i+1].unsqueeze(0)
        target = target_char_tensor[i+1]
        output, hidden = model(input, hidden)
        loss += loss_fn(output, target.view(1))

    loss.backward()
    optimizer.step()

    return loss.item() / (input_line_tensor.size(0) - 1)
```

### Loop de treinamento


```python
EPOCHS = 300 # Número de épocas que vamos treinar

if torch.cuda.is_available():
    device = 'cuda' # Usaremos a GPU se disponível

for epoch in range(EPOCHS):
    for name in nomes:
        name_tensor = one_hot_encoding(char_tensor(name))
        name_tensor.to(device)
        loss = train(name_tensor, char_tensor(name), device=device)
    if epoch % 50 == 0:
        print(f'Epoch: {epoch}, Loss: {loss:.4f}')
```

    Epoch: 0, Loss: 2.2929
    Epoch: 50, Loss: 0.6604
    Epoch: 100, Loss: 0.6915
    Epoch: 150, Loss: 1.0655
    Epoch: 200, Loss: 0.6139
    Epoch: 250, Loss: 0.6143
    

### A melhor hora: Vamos fazer inferências com nosso modelo!

#### Função para geração de nomes


```python
def generate(initial_char, max_length=12): # Initial_char é a letra que a rede vai usar para começar a gerar nomes e o max_length é o tamanho máximo do nome gerado. Eu definir 12, mas você pode mudar.
    with torch.no_grad(): 
        hidden = model.init_hidden(1).to(device)
        input_char_tensor = one_hot_encoding(char_tensor(initial_char)).to(device)
        predicted_name = initial_char

        for _ in range(max_length - 1):
            input = input_char_tensor.unsqueeze(0)
            output, hidden = model(input, hidden)

            _ , topi = output.topk(1) # Vamos pegar o índice do maior valor, que é a previsão da rede para o próximo caractere
            char_index = topi[0][0].item()
            if char_index == encoder[initial_char]:
                break
            else:
                char = decoder[char_index]
                predicted_name += char
                input_char_tensor = one_hot_encoding(char_tensor(char)).to(device)

        return predicted_name
```

### Gerando nomes 😁


```python
model = model.to(device) # Colocando o modelo na GPU
model.eval()  # E colocando ele em modo de avaliação
```




    CharRNN(
      (rnn): RNN(47, 100, batch_first=True)
      (fc): Linear(in_features=100, out_features=47, bias=True)
    )




```python
start_char = 'L'
predicted_name = generate(start_char)
# Pegar os caractres até o /
print(f"Nome gerado com o caractere '{start_char}': {predicted_name.split('/')[0]}")
```

    Nome gerado com o caractere 'L': Lívia
    


```python
start_char = 'M'
predicted_name = generate(start_char)
print(f"Nome gerado com o caractere '{start_char}': {predicted_name.split('/')[0]}")
```

    Nome gerado com o caractere 'M': Mardo
    


```python
start_char = 'D'
predicted_name = generate(start_char)
print(f"Nome gerado com o caractere '{start_char}': {predicted_name.split('/')[0]}")
```

    Nome gerado com o caractere 'D': Danciss
    


```python
start_char = 'P'
predicted_name = generate(start_char)
print(f"Nome gerado com o caractere '{start_char}': {predicted_name.split('/')[0]}")
```

    Nome gerado com o caractere 'P': Pedoísna
    


```python
start_char = 'Y'
predicted_name = generate(start_char)
print(f"Nome gerado com o caractere '{start_char}': {predicted_name.split('/')[0]}")
```

    Nome gerado com o caractere 'Y': Yasmindo
    

### Conclusão:

Por esta lista ser super curta, somente com 70 nomes, o modelo tem algumas limitações como não conseguir gerar nomes que começam com O (porque não tem nomes com essa letra na lista) além de ter pocuso exemplares de nomes diversos.

Porém o intuito aqui era apenas exemplificar o uso da RNN :)
