#include "Tensor.h"
#include <iostream>

int main() {
    try {
        std::cout << "==== PRUEBAS BASICAS ====\n\n";

        Tensor A = Tensor::zeros({2, 3});
        Tensor B = Tensor::ones({2, 3});
        Tensor C = A + B;
        C.print("C = A + B");

        Tensor D = Tensor::arange(0, 6);
        D.print("D = arange(0,6)");

        Tensor E = D.view({2, 3});
        E.print("E = D.view({2,3})");

        Tensor F = D.unsqueeze(0);
        F.print("F = D.unsqueeze(0)");

        Tensor G = D.unsqueeze(1);
        G.print("G = D.unsqueeze(1)");

        ReLU relu;
        Sigmoid sigmoid;

        Tensor H = Tensor::arange(-5, 5).view({2, 5});
        H.print("H");

        Tensor H1 = H.apply(relu);
        H1.print("ReLU(H)");

        Tensor H2 = H.apply(sigmoid);
        H2.print("Sigmoid(H)");

        Tensor M1({2, 3}, {1, 2, 3, 4, 5, 6});
        Tensor M2({3, 2}, {7, 8, 9, 10, 11, 12});
        Tensor MM = matmul(M1, M2);
        MM.print("matmul(M1, M2)");

        Tensor v1({3}, {1, 2, 3});
        Tensor v2({3}, {4, 5, 6});
        Tensor dp = dot(v1, v2);
        dp.print("dot(v1, v2)");

        Tensor X = Tensor::concat({Tensor::ones({2, 3}), Tensor::zeros({2, 3})}, 0);
        X.print("concat axis 0");

        std::cout << "==== RED NEURONAL ====\n\n";

        // 1. entrada 1000 x 20 x 20
        Tensor input = Tensor::random({1000, 20, 20}, -1.0, 1.0);

        // 2. view -> 1000 x 400
        Tensor x = input.view({1000, 400});

        // 3. matmul con W1 (400 x 100)
        Tensor W1 = Tensor::random({400, 100}, -0.5, 0.5);
        Tensor z1 = matmul(x, W1);

        // 4. suma con b1 (1 x 100)
        Tensor b1 = Tensor::random({1, 100}, -0.1, 0.1);
        Tensor a1 = z1 + b1;

        // 5. ReLU
        Tensor h1 = a1.apply(relu);

        // 6. matmul con W2 (100 x 10)
        Tensor W2 = Tensor::random({100, 10}, -0.5, 0.5);
        Tensor z2 = matmul(h1, W2);

        // 7. suma con b2 (1 x 10)
        Tensor b2 = Tensor::random({1, 10}, -0.1, 0.1);
        Tensor a2 = z2 + b2;

        // 8. Sigmoid
        Tensor output = a2.apply(sigmoid);

        output.print("Output final", 15);

        std::cout << "Programa ejecutado correctamente.\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
    }

    return 0;
}
