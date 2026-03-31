#include "Tensor.h"

#include <iostream>
#include <stdexcept>
#include <random>
#include <cmath>
#include <algorithm>

// Métodos privados

void Tensor::validate_shape(const std::vector<size_t>& shape) {
    if (shape.empty() || shape.size() > 3) {
        throw std::invalid_argument("El tensor debe tener entre 1 y 3 dimensiones.");
    }

    for (size_t dim : shape) {
        if (dim == 0) {
            throw std::invalid_argument("Las dimensiones deben ser mayores que cero.");
        }
    }
}

size_t Tensor::compute_total_size(const std::vector<size_t>& shape) {
    validate_shape(shape);

    size_t total = 1;
    for (size_t dim : shape) {
        total *= dim;
    }
    return total;
}

std::vector<size_t> Tensor::compute_strides(const std::vector<size_t>& shape) {
    std::vector<size_t> strides(shape.size(), 1);
    if (!shape.empty()) {
        for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
    }
    return strides;
}

void Tensor::release() {
    if (ref_count_ != nullptr) {
        (*ref_count_)--;
        if (*ref_count_ == 0) {
            delete[] data_;
            delete ref_count_;
        }
    }

    data_ = nullptr;
    ref_count_ = nullptr;
    total_size_ = 0;
    shape_.clear();
}

// Constructor privado para compartir memoria
Tensor::Tensor(double* shared_data,
               const std::vector<size_t>& new_shape,
               size_t total_size,
               size_t* shared_ref_count)
    : data_(shared_data),
      shape_(new_shape),
      total_size_(total_size),
      ref_count_(shared_ref_count) {

    if (ref_count_ == nullptr) {
        throw std::runtime_error("Error interno: contador de referencias nulo.");
    }

    ++(*ref_count_);
}


//Constructores y destructor


Tensor::Tensor(const std::vector<size_t>& shape, const std::vector<double>& values)
    : data_(nullptr),
      shape_(shape),
      total_size_(compute_total_size(shape)),
      ref_count_(nullptr) {

    if (values.size() != total_size_) {
        throw std::invalid_argument("La cantidad de valores no coincide con el shape.");
    }

    data_ = new double[total_size_];
    ref_count_ = new size_t(1);

    for (size_t i = 0; i < total_size_; ++i) {
        data_[i] = values[i];
    }
}

Tensor::Tensor(const std::vector<size_t>& shape)
    : data_(nullptr),
      shape_(shape),
      total_size_(compute_total_size(shape)),
      ref_count_(nullptr) {

    data_ = new double[total_size_];
    ref_count_ = new size_t(1);

    for (size_t i = 0; i < total_size_; ++i) {
        data_[i] = 0.0;
    }
}

Tensor::Tensor(const Tensor& other)
    : data_(nullptr),
      shape_(other.shape_),
      total_size_(other.total_size_),
      ref_count_(nullptr) {

    // Copia profunda
    data_ = new double[total_size_];
    ref_count_ = new size_t(1);

    for (size_t i = 0; i < total_size_; ++i) {
        data_[i] = other.data_[i];
    }
}

Tensor::Tensor(Tensor&& other) noexcept
    : data_(other.data_),
      shape_(std::move(other.shape_)),
      total_size_(other.total_size_),
      ref_count_(other.ref_count_) {

    other.data_ = nullptr;
    other.ref_count_ = nullptr;
    other.total_size_ = 0;
    other.shape_.clear();
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        release();

        shape_ = other.shape_;
        total_size_ = other.total_size_;
        data_ = new double[total_size_];
        ref_count_ = new size_t(1);

        for (size_t i = 0; i < total_size_; ++i) {
            data_[i] = other.data_[i];
        }
    }
    return *this;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        release();

        data_ = other.data_;
        shape_ = std::move(other.shape_);
        total_size_ = other.total_size_;
        ref_count_ = other.ref_count_;

        other.data_ = nullptr;
        other.ref_count_ = nullptr;
        other.total_size_ = 0;
        other.shape_.clear();
    }
    return *this;
}

Tensor::~Tensor() {
    release();
}

// ==========================
// Métodos estáticos
// ==========================

Tensor Tensor::zeros(const std::vector<size_t>& shape) {
    return Tensor(shape);
}

Tensor Tensor::ones(const std::vector<size_t>& shape) {
    Tensor t(shape);
    for (size_t i = 0; i < t.total_size_; ++i) {
        t.data_[i] = 1.0;
    }
    return t;
}

Tensor Tensor::random(const std::vector<size_t>& shape, double min, double max) {
    Tensor t(shape);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(min, max);

    for (size_t i = 0; i < t.total_size_; ++i) {
        t.data_[i] = dist(gen);
    }

    return t;
}

Tensor Tensor::arange(int start, int end) {
    if (end <= start) {
        throw std::invalid_argument("En arange, end debe ser mayor que start.");
    }

    std::vector<double> values;
    for (int i = start; i < end; ++i) {
        values.push_back(static_cast<double>(i));
    }

    return Tensor({values.size()}, values);
}

// Observadores

const std::vector<size_t>& Tensor::shape() const {
    return shape_;
}

size_t Tensor::size() const {
    return total_size_;
}

size_t Tensor::ndim() const {
    return shape_.size();
}

double Tensor::at_flat(size_t index) const {
    if (index >= total_size_) {
        throw std::out_of_range("Índice fuera de rango.");
    }
    return data_[index];
}

void Tensor::set_flat(size_t index, double value) {
    if (index >= total_size_) {
        throw std::out_of_range("Índice fuera de rango.");
    }
    data_[index] = value;
}

// Operadores

Tensor Tensor::operator+(const Tensor& other) const {
    // Caso 1: mismo shape
    if (shape_ == other.shape_) {
        Tensor result(shape_);
        for (size_t i = 0; i < total_size_; ++i) {
            result.data_[i] = data_[i] + other.data_[i];
        }
        return result;
    }

    // Caso 2: broadcasting limitado para red neuronal
    // (m x n) + (1 x n)
    if (ndim() == 2 && other.ndim() == 2 &&
        other.shape_[0] == 1 && shape_[1] == other.shape_[1]) {

        Tensor result(shape_);
        size_t rows = shape_[0];
        size_t cols = shape_[1];

        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result.data_[i * cols + j] = data_[i * cols + j] + other.data_[j];
            }
        }
        return result;
    }

    // (1 x n) + (m x n)
    if (ndim() == 2 && other.ndim() == 2 &&
        shape_[0] == 1 && shape_[1] == other.shape_[1]) {

        Tensor result(other.shape_);
        size_t rows = other.shape_[0];
        size_t cols = other.shape_[1];

        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result.data_[i * cols + j] = data_[j] + other.data_[i * cols + j];
            }
        }
        return result;
    }

    throw std::invalid_argument("Suma inválida: shapes incompatibles.");
}

Tensor Tensor::operator-(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::invalid_argument("Resta inválida: shapes incompatibles.");
    }

    Tensor result(shape_);
    for (size_t i = 0; i < total_size_; ++i) {
        result.data_[i] = data_[i] - other.data_[i];
    }
    return result;
}

Tensor Tensor::operator*(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::invalid_argument("Multiplicación elemento a elemento inválida.");
    }

    Tensor result(shape_);
    for (size_t i = 0; i < total_size_; ++i) {
        result.data_[i] = data_[i] * other.data_[i];
    }
    return result;
}

Tensor Tensor::operator*(double scalar) const {
    Tensor result(shape_);
    for (size_t i = 0; i < total_size_; ++i) {
        result.data_[i] = data_[i] * scalar;
    }
    return result;
}

Tensor operator*(double scalar, const Tensor& t) {
    return t * scalar;
}

// Transformaciones de forma

Tensor Tensor::view(const std::vector<size_t>& new_shape) const {
    size_t new_total = compute_total_size(new_shape);

    if (new_total != total_size_) {
        throw std::invalid_argument("view inválido: el número total de elementos debe mantenerse.");
    }

    // comparte memoria, no copia datos
    return Tensor(data_, new_shape, total_size_, ref_count_);
}

Tensor Tensor::unsqueeze(size_t dim) const {
    if (shape_.size() >= 3) {
        throw std::invalid_argument("unsqueeze inválido: excede 3 dimensiones.");
    }

    if (dim > shape_.size()) {
        throw std::invalid_argument("Posición inválida para unsqueeze.");
    }

    std::vector<size_t> new_shape = shape_;
    new_shape.insert(new_shape.begin() + dim, 1);

    // comparte memoria, no copia datos
    return Tensor(data_, new_shape, total_size_, ref_count_);
}

// Concatenación

Tensor Tensor::concat(const std::vector<Tensor>& tensors, size_t axis) {
    if (tensors.empty()) {
        throw std::invalid_argument("No se puede concatenar una lista vacía.");
    }

    const std::vector<size_t>& base_shape = tensors[0].shape_;
    size_t dims = base_shape.size();

    if (axis >= dims) {
        throw std::invalid_argument("Eje inválido para concat.");
    }

    for (const auto& t : tensors) {
        if (t.shape_.size() != dims) {
            throw std::invalid_argument("Todos los tensores deben tener la misma cantidad de dimensiones.");
        }
        for (size_t d = 0; d < dims; ++d) {
            if (d != axis && t.shape_[d] != base_shape[d]) {
                throw std::invalid_argument("Shapes incompatibles para concat.");
            }
        }
    }

    std::vector<size_t> new_shape = base_shape;
    new_shape[axis] = 0;
    for (const auto& t : tensors) {
        new_shape[axis] += t.shape_[axis];
    }

    Tensor result(new_shape);

    std::vector<size_t> result_strides = compute_strides(new_shape);
    size_t axis_offset = 0;

    for (const auto& t : tensors) {
        std::vector<size_t> t_strides = compute_strides(t.shape_);

        for (size_t linear = 0; linear < t.total_size_; ++linear) {
            // convertir índice lineal del tensor t a índices multidimensionales
            std::vector<size_t> idx(dims, 0);
            size_t temp = linear;

            for (size_t d = 0; d < dims; ++d) {
                idx[d] = temp / t_strides[d];
                temp %= t_strides[d];
            }

            idx[axis] += axis_offset;

            size_t result_linear = 0;
            for (size_t d = 0; d < dims; ++d) {
                result_linear += idx[d] * result_strides[d];
            }

            result.data_[result_linear] = t.data_[linear];
        }

        axis_offset += t.shape_[axis];
    }

    return result;
}

// Polimorfismo

Tensor Tensor::apply(const TensorTransform& transform) const {
    return transform.apply(*this);
}

Tensor ReLU::apply(const Tensor& t) const {
    Tensor result(t.shape());
    for (size_t i = 0; i < t.size(); ++i) {
        result.set_flat(i, std::max(0.0, t.at_flat(i)));
    }
    return result;
}

Tensor Sigmoid::apply(const Tensor& t) const {
    Tensor result(t.shape());
    for (size_t i = 0; i < t.size(); ++i) {
        double x = t.at_flat(i);
        result.set_flat(i, 1.0 / (1.0 + std::exp(-x)));
    }
    return result;
}

// Funciones amigas

Tensor dot(const Tensor& a, const Tensor& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("dot inválido: distinto número de elementos.");
    }

    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        sum += a.data_[i] * b.data_[i];
    }

    return Tensor({1}, {sum});
}

Tensor matmul(const Tensor& a, const Tensor& b) {
    if (a.ndim() != 2 || b.ndim() != 2) {
        throw std::invalid_argument("matmul requiere tensores 2D.");
    }

    size_t m = a.shape_[0];
    size_t k1 = a.shape_[1];
    size_t k2 = b.shape_[0];
    size_t n = b.shape_[1];

    if (k1 != k2) {
        throw std::invalid_argument("matmul inválido: dimensiones incompatibles.");
    }

    Tensor result({m, n});

    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < k1; ++k) {
                sum += a.data_[i * k1 + k] * b.data_[k * n + j];
            }
            result.data_[i * n + j] = sum;
        }
    }

    return result;
}

// Imprimir

void Tensor::print(const std::string& name, size_t max_elements) const {
    std::cout << name << " shape = {";
    for (size_t i = 0; i < shape_.size(); ++i) {
        std::cout << shape_[i];
        if (i + 1 < shape_.size()) {
            std::cout << ", ";
        }
    }
    std::cout << "}\n";

    std::cout << "data = [";
    size_t limit = std::min(max_elements, total_size_);
    for (size_t i = 0; i < limit; ++i) {
        std::cout << data_[i];
        if (i + 1 < limit) {
            std::cout << ", ";
        }
    }
    if (total_size_ > max_elements) {
        std::cout << ", ...";
    }
    std::cout << "]\n\n";
}
