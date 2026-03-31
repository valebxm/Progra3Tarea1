#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <string>
#include <cstddef>

class TensorTransform;

class Tensor {
private:
    double* data_;
    std::vector<size_t> shape_;
    size_t total_size_;
    size_t* ref_count_;
    // para compartir memoria en view y unsqueeze

    static void validate_shape(const std::vector<size_t>& shape);
    static size_t compute_total_size(const std::vector<size_t>& shape);
    static std::vector<size_t> compute_strides(const std::vector<size_t>& shape);

    void release();

    // Constructor privado para compartir memoria sin copiar
    Tensor(double* shared_data,
           const std::vector<size_t>& new_shape,
           size_t total_size,
           size_t* shared_ref_count);

public:
    // Constructor principal
    Tensor(const std::vector<size_t>& shape, const std::vector<double>& values);

    // Constructor auxiliar
    explicit Tensor(const std::vector<size_t>& shape);

    // Regla de cinco
    Tensor(const Tensor& other);                 // copia profunda
    Tensor(Tensor&& other) noexcept;            // movimiento
    Tensor& operator=(const Tensor& other);     // asignación copia
    Tensor& operator=(Tensor&& other) noexcept;  // asignación movimiento
    ~Tensor();                                  // destructor

    // Métodos estáticos
    static Tensor zeros(const std::vector<size_t>& shape);
    static Tensor ones(const std::vector<size_t>& shape);
    static Tensor random(const std::vector<size_t>& shape, double min, double max);
    static Tensor arange(int start, int end);

    // Observadores
    const std::vector<size_t>& shape() const;
    size_t size() const;
    size_t ndim() const;

    double at_flat(size_t index) const;
    void set_flat(size_t index, double value);

    // Operaciones
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator*(double scalar) const;

    Tensor view(const std::vector<size_t>& new_shape) const;
    Tensor unsqueeze(size_t dim) const;

    static Tensor concat(const std::vector<Tensor>& tensors, size_t axis);

    Tensor apply(const TensorTransform& transform) const;

    void print(const std::string& name = "Tensor", size_t max_elements = 20) const;

    friend Tensor operator*(double scalar, const Tensor& t);
    friend Tensor dot(const Tensor& a, const Tensor& b);
    friend Tensor matmul(const Tensor& a, const Tensor& b);
};

class TensorTransform {
public:
    virtual Tensor apply(const Tensor& t) const = 0;
    virtual ~TensorTransform() = default;
};

class ReLU : public TensorTransform {
public:
    Tensor apply(const Tensor& t) const override;
};

class Sigmoid : public TensorTransform {
public:
    Tensor apply(const Tensor& t) const override;
};

Tensor dot(const Tensor& a, const Tensor& b);
Tensor matmul(const Tensor& a, const Tensor& b);

#endif
