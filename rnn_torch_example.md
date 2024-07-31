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
    
