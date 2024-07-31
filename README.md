!![alt](img/RNN-desdobrada.png)

**Link do post no Blogger: [POV: Você é uma Rede Recorrente](https://umblogparameuprof.blogspot.com/2024/04/redes-recorrentes.html)**

Ok, ok, eu sei, eu também queria estar aqui te ensinando sobre Transformers e LLMs e coisas mágicas que todo mundo no Linkedin tá comentando. Mas, além de que eu sou apenas uma iniciante na área e não possuo total conhecimento para te ofertar, jovem Padawan, eu quero te apresentar uma arquitetura mais antiga PORÉM, mais do que essencial pra gente ter chego tão longe nas arquiteturas.

Sim, é ela mesma: a rede neural recorrente.

Neste blog post vamos conhecer essa diva icônica em termos teóricos e depois vamos botar as mãos na massa com o PyTorch!

#### O castigo do monstro: lidando com problemas sequenciais 
É, o pessoal das MLPs e CNNs também se fizeram esta mesma pergunta. Aqui, problemas sequenciais são aqueles em que precisamos observar a ordem dos dados ao realizar a tarefa. O exemplo mais clássico de problemas que devem ser modelados de forma sequencial é o processamento de linguagem natural, mas séries temporais e sequências genéticas também são outros dois exemplos. Aqui, vamos focar em exemplos da área de PLN para fins de praticidade (e porque eu gosto muito também). 😉

Imagina a seguinte situação:
Queremos fazer um levantamento de qualidade do conteúdo do nosso blog a partir dos comentários dos usuários realizando uma classificação binária para dizer se o comentário em questão é positivo ou negativo. Como sabemos, textos são entradas sequenciais e este tipo de entrada já iria quebrar qualquer MLP pois estas só conseguem lidar com entradas de tamanho fixo e não conseguiriam capturar o contexto dos comentários, iria ser um desastre total!
"Ah, mas se o problema é tamanho fixo, bora usar uma CNN da vida!"
É, CNNs podem sim aceitar entradas com tamanhos variados, mas tem um pequeno grande detalhe: dependendo do tamanho da frase, ela pode acabar perdendo o contexto pois teríamos que ficar adicionando muitas camadas convolucionais para aumentar o campo receptivo delas e eles serem capazes de capturar todo o contexto da frase. Além do escopo do nosso problema, outras tarefas de PLN como tradução de texto exigem que as redes possam aceitar  cenários onde temos uma entrada com um determinado tamanho e retornar uma saída com um tamanho diferente da entrada, e isso não ocorre nessas arquiteturas.

Em terra de sequencinhas, quem guarda hidden states é raínha
As RNRs surgiram justamente para resolver este problema que tínhamos com as entradas (e saídas) sequenciais. Isto porquê essas queridas são capazes de manter um estado que carrega as informações obtidas no passo anterior! Vamos conhecer um pouco melhor a arquitetura dela?

### Redes Neurais Recorrentes

![alt](img/RNN-legendada.png)

RNRs adicionam uma dimensão de tempo (representado aqui pelo subscrito t) e apresentam uma nova funcionalidade em sua arquitetura que nós chamamos de estado interno ou, mais comumente, os hidden states. Eles são responsáveis por guardar o estado da rede no instante t após processar a entrada X no mesmo instante. Pode parecer confuso lendo, então vamos dar uma abridinha nela usando como exemplo de entrada um comentário fictício do blog.

![alt](img/comentario-blog.png)
![alt](img/RNN-desdobrada.png)

Assim é uma RNR desdobrada no tempo. Dada a sentença "As imagens estão boas", cada palavra da sentença entra na rede no instante t, faz as computações internas e para o próximo passo no instante t+1 ela envia o estado interno da última entrada vista. Por isso elas são ditas recorrentes: a saída de cada célula da rede depende não somente da entrada mas também do estado interno da célula anterior, criando este 'loop' que chamamos de recorrência.

---

### Mãos na massa com o PyTorch 👩‍💻
Aqui nesta seção, vamos conhecer o PyTorch e o seu módulo dedicado a camadas RNN, o torch.nn.RNN

#### Quem é o PyTorch na fila do pão?
O PyTorch se descreve oficialmente como uma biblioteca otimizada para se trabalhar com tensores em aplicações de aprendizado profundo usando GPUs e CPUs, mas eu gosto de definir ele como um dos melhores (se não o melhor) e mais popular framework para desenvolvimento de modelos de aprendizado profundo. Eles oferecem uma gama de funcionalidades dentro das Python APIs deles que no geral começam com torch.[algum_módulo específico]. Por exemplo, a que vamos utilizar aqui é a torch.nn, uma API Python que oferece diversos building blocks para construir a rede que você deseja com as funções e otimizadores que você precisa.

#### torch.nn.RNN 😎
Esta camada computa, para cada elemento da sequência de entrada, a seguinte função:
![alt](img/tahn_hidden_state_torch.png)

Faremos a computação desta função no nosso Jupyter Notebook na sessão Fala menos e faz mais para melhor entendimento, e lá também vou te apresentar o que significa cada elemento dela!
 
Primeiro precisamos entender o que tem nesta camada, o que ela recebe e o que ela retorna. 

##### Como funciona a camada 

**Parâmetros**

Para instanciar um objeto torch.nn.RNN, devemos passar, pelo menos, os seguintes parâmetros:

- input_size: representa o número de features de entradas que vamos ter
- hidden_size: representa o número de features da camada interna

Os outros parâmetros já possuem um valor padrão caso você não os coloque, que são:

- num_layers: número de camadas RNN que você deseja. Por default, 1 camada
nonlinearity: qual a função não-linear que queremos usar, podendo escolher entre a Tanh (que é a default) e a 'ReLu'
- bias: um valor booleano para dizer se queres que a camada use ou não os pesos de bias. Default é que utilize, ou seja, True
- batch_first: valor booleano para indicar se queremos que os dados de entrada e de saída sejão providos no seguinte formato: (tamanho_do_batch, tamanho_da_sequência, feature) ao invés de (tamanho_da_sequência, tamanho_do_batch, feature). O tamanho da sequência geralmente se refere ao tamanho máximo que a sequência vai ter. Por exemplo, dado um vocabulário sobre nomes, o maior nome tem 12 letras. 12 será o tamanho_da_sequência. Por default, ele vem como False.
- dropout: utilizado para dizer se queremos introduzir camas de Dropout nos outputs de cada camada RNN exceto a última, sendo usada a probabilidade referenciada pelo próprio parâmetro dropout. Quando ele vem 0.0, como é o seu valor default, não introduzimos. 
- bidirectional: valor booleano para indicar se a camada deve ser bidirecional o unidirecional. Por default, é False e é unidirecional.

**Entrada, saída e estado interno**

A entrada da camada deve ser um tensor com o seguinte formato, se batch_size=False:

- Tensor(tamanho_da_sequência, batch_size, input_size)
A saída será um tensor com o seguinte formato, se batch_size=False e bidirectional=False:
- Tensor(tamanho_da_sequência, batch_size, hidden_size)
      Caso ela seja bidirecional, será:
- Tensor(tamanho_da_sequência, batch_size, 2 * hidden_size)
E o estado interno será um tensor com o seguinte formato, se bidirectional=False
Tensor(número_de_camadas, batch_size, hidden_size)

### Fala menos e faz mais (e uma pincelada de matemática, pra ficar chique 💅) 💻
Para explorar melhor como funciona a rede, vamos instanciar uma RNN com uma única camada que vai receber um único dado de entrada (e consequentemente de saída) e vai ter um hidden state de apenas uma única feature também. Ou seja, o t dela referente ao tempo, vai ir só até o 0, realizando somente uma única computação.
Instanciada a rede, vamos criar um dado de entrada e um hidden_state inicial, e passar para a camada realizar a computação da função Tahn. 
Depois, vamos acessar todos os atributos  de peso e bias da camada para passar para esta mesma função Tahn e conferir que o resultado é o mesmo.

![alt](img/torch_rnn_example.png)

#### Camada RNN do torch.nn


```python
import torch
import torch.nn as nn
# Seed para reprodução de resultados
torch.manual_seed(42)

# Instanciando a rede
rnn = nn.RNN(1, 1, 1)
# tamanho_da_sequência = 1, batch_size = 1, input_size = 1
input = torch.tensor([[[0.5]]])

# número_de_camadas * 1, batch_size, hidden_size
hidden = torch.zeros(1, 1, 1)

# Passando a entrada e o estado oculto pela RNN
output, new_hidden = rnn(input, hidden)
# Resultado
print(f'Saída: {output[0][0][0]}')
print(f'Hidden state: {new_hidden[0][0][0]}')
```

    Saída: 0.788179874420166
    Hidden state: 0.788179874420166
    

#### Acessando os pesos e bias da rede para computar a função Tahn

Como explicamos no blog, o que a camada realiza é a computação da seguinte fórmula:

$$
h_t = \tanh(x_t W_{ih}^T + b_{ih} + h_{t-1} W_{hh}^T + b_{hh})
$$


Onde os elementos de pesos e bias, representados por $W$ e $b$, colocados abaixo:


```python
# Acessando os pesos e bias da RNN
w_ih = rnn.weight_ih_l0  # Peso para a entrada para a camada oculta
w_hh = rnn.weight_hh_l0  # Peso para a camada oculta para a camada oculta
b_ih = rnn.bias_ih_l0    # Bias para a entrada para a camada oculta
b_hh = rnn.bias_hh_l0    # Bias para a camada oculta para a camada oculta
```

E o elemento $x_t$ da fórmula se refere à nossa entrada e o elemento $h_t$ se refere ao hidden_state

Como temos todos estes elementos, podemos calcular 'manualmente' a mesma função com eles e observar o mesmo resultado de saída que a cadama RNN nos retornou.


```python
hidden_manual = torch.tanh(w_ih @ input[0] + b_ih + w_hh @ hidden + b_hh)
print(f'Resultado da fórmula: {hidden_manual[0][0][0]}')
```

    Resultado da fórmula: 0.788179874420166


#### Gerando nomes com uma RNN 🤖

Agora que entendemos um pouco melhor com funcionam as camadas RNN do PyTorch, vamos ver um novo exemplo onde faremos a previsão do próximo caractere para realizar a geração de nomes brasileiros! 

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


##### Quer que desenhe? Pois eu também!

Assim irá funcionar o treinamento da rede para gerar nomes:
Vamos usar como exemplo uma única instância de nome, o nome 'Lucas'.
Nosso vocabulário terá 5 letras e será decodificado pata números e depois tensores.

![alt](img/rnn_vocab.png)
Além disso, como as redes precisam de entradas numéricas, vamos fazer a representação desse nome como um vetor one-hot-encoding, desta forma

![alt](img/rnn_vocab_ohe.png)

Essa é a representação que vai para a rede, ok?
Então, como queremos que ela faça previsão do próximo caractere, e temos somente o nome 'Lucas', queremos que ela consiga predizer que após o caractere 'L', a maior probabilidade para o próximo caractere seja para o 'u', e assim por diante. Para isso, vamos dar como entrada o vetor OHE de cada letra e através do algoritmo de backpropagation, vamos atualizar os pesos W os bias dela até que ela tenha um erro mínimo entre o caractere que ela previu e o que de fato é esperado.

![alt](img/rnn_training.png)

Depois da rede treinada, ela será capaz de gerar o nome 'Lucas' pois estará com os pesos ajustados para isso!

![alt](img/rede%20treinada.png)

### Referências:
Capítulo 48 – Redes Neurais Recorrentes. (n.d.). In Deep Learning Book em Português. Recuperado de https://www.deeplearningbook.com.br/redes-neurais-recorrentes/

Karpathy, A. (2015, 21 de maio). The Unreasonable Effectiveness of Recurrent Neural Networks Blog de Andrej Karpathy. Recuperado de https://karpathy.github.io/2015/05/21/rnn-effectiveness/

Loeber, P. (2020, 3 de setembro). PyTorch Tutorial - RNN & LSTM & GRU - Redes Neurais Recorrentes [Vídeo]. YouTube. https://www.youtube.com/watch?v=0_PgWWmauHk&t=331s

Também utilizei como referência alguns materiais do meu professor Lucas Kupssinsku que leciona a cadeira de Aprendizado Profundo I e II na PUCRS.