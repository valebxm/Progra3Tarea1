# Tensor++

Implementación en C++ de una librería básica de tensores inspirada en NumPy y PyTorch.

## Archivos
- `Tensor.h`
- `Tensor.cpp`
- `main.cpp`

## Compilación

```bash
g++ -std=c++17 -O2 main.cpp Tensor.cpp -o tensor_app
./tensor_app
```

## Contenido implementado
- Tensores de hasta 3 dimensiones
- Memoria dinámica
- Regla de cinco
- `zeros`, `ones`, `random`, `arange`
- Operadores `+`, `-`, `*`, `* escalar`
- `view`
- `unsqueeze`
- `concat`
- `dot`
- `matmul`
- Polimorfismo con `TensorTransform`, `ReLU` y `Sigmoid`
- Simulación de una red neuronal simple
