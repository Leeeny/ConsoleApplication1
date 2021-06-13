#include <iostream>
#include <iomanip>
#include <chrono>
#include "mpi.h"

class Matrix
{
    int h, w;
    double* arr;

public:
    Matrix()
    {
        h = w = 0;
        arr = 0;
    }
    Matrix(int h, int w)
    {
        this->h = h;
        this->w = w;
        arr = new double[h * w];
        for (size_t i = 0; i < h * w; i++)
            arr[i] = 0;
    }
    Matrix(double* array, int h, int w)
    {
        this->h = h;
        this->w = w;
        arr = new double[h * w];
        for (int i = 0; i < h * w; ++i)
            arr[i] = array[i];
    }
    Matrix(const Matrix& other)
    {
        h = other.h;
        w = other.w;
        arr = new double[h * w];
        for (int i = 0; i < h * w; ++i)
            arr[i] = other.arr[i];
    }
    ~Matrix()
    {
        if (arr) delete[] arr;
    }
    int width() const
    {
        return w;
    }
    int height() const
    {
        return h;
    }
    double& operator()(int row, int col);
    double operator()(int row, int col) const;

    Matrix& operator=(const Matrix& other);
};
double& Matrix::operator()(int x, int y)
{
    return arr[x + y * h];
}
double Matrix::operator()(int x, int y) const
{
    return arr[x * y * h];
}
Matrix& Matrix::operator=(const Matrix& other) {
    Matrix t(other);
    double* tt = t.arr;
    t.arr = arr;
    arr = tt;
    w = other.w;
    h = other.h;
    return *this;
}

Matrix Generate(int N) {
    double* arr = new double[N * N];
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            arr[i * N + j] = 1.0 * (rand() % 1001) / 1000;
    for (int i = 0; i < N; i++)
        arr[i * N + i] += N;
    return Matrix(arr, N, N);
}

double GetDelta(Matrix& A, Matrix& B) {
    double max = 0, temp;
    for (int j = 0; j < A.height(); j++)
        for (int i = 0; i < A.width(); i++) {
            temp = std::abs(A(i, j) - B(i, j));
            if (max < temp)
                max = temp;
        }
    return max;
}

double GetDelta1(Matrix& Orig, Matrix& A) {
    double max = 0;
    for (int j = 0; j < A.height(); j++)
        for (int i = 0; i < A.width(); i++) {
            double sum = 0;
            for (int k = 0; k < std::min(i + 1, j); k++)
                sum += A(j, k) * A(k, i);
            if (i >= j) sum += A(j, i);
            if (max < std::abs(Orig(j, i) - sum))
                max = std::abs(Orig(j, i) - sum);
        }
    return max;
}

void NotParallel(Matrix& A) {
    for (int j = 0; j < A.width(); ++j) {
        // решение СЛАУ
        for (int i = 0; i < j - 1; ++i)
            for (int k = i + 1; k < j; ++k)
                A(k, j) -= A(i, j) * A(k, i);
        // gaxpy
        for (int k = 0; k < j; ++k)
            for (int p = j; p < A.height(); ++p)
                A(p, j) -= A(k, j) * A(p, k);
        for (int k = j + 1; k < A.height(); ++k)
            A(k, j) /= A(j, j);
    }
}

void Parallel(int N) {
    int rank, size; // rank = mu, size - задач всего
    MPI_Status status;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size < 2)    return;

    if (rank == 0) { //если первая задача

        // ---------------------- Генерируем матрицу A и копируем в B ----------------------

        Matrix A = Generate(N);
        Matrix B(A);
        Matrix Orig = A;
        NotParallel(B);

        // ------------------ Считаем последовательно первые два столбца не паралльельно -------------------

        for (int j = 0; j < 2; ++j) {
            // решение СЛАУ
            for (int i = 0; i < j - 1; ++i)
                for (int k = i + 1; k < j; ++k)
                    A(k, j) -= A(i, j) * A(k, i);
            // gaxpy
            for (int k = 0; k < j; ++k)
                for (int p = j; p < A.height(); ++p)
                    A(p, j) -= A(k, j) * A(p, k);
            for (int k = j + 1; k < A.height(); ++k)
                A(k, j) /= A(j, j);
        }
        

        // ------------------ Начинаем идти по всем столбцам ---------------------
        for (int j = 2; j < N - 1; ++j) {

            // ------------------ Считаем последовательно решение СЛАУ ---------------------

            for (int i = 0; i < j - 1; ++i)
                for (int k = i + 1; k < j; ++k)
                    A(k, j) -= A(i, j) * A(k, i);

            // ---------------- Определяем какая информация должна быть у задач ------------
            int p; //кол-во активных задач
            {
                if (j / size > 0 && (N - j) / size > 0) //если высота и ширина матрицы больше, чем общее число задач
                    p = size;
                else if (j / size == 0) //если ширина матрицы меньше числа задач
                    p = j % size;
                else //если высота матрицы меньше числа задач
                    p = (N - j) % size;
            }

            int* L = new int[size]; //высота столбца для i-й задачи
            int* X = new int[size]; //ширина локальной матрицы для i-й задачи

            for (int i = 0; i < size; ++i) { //определяем размеры X и L
                if (i < p) { //если i-ая задача входит в число активных задач
                    X[i] = j / p; //то тогда ширина локальной матрицы для i-й задачи будет равана общая ширина / число задач
                    if (i < j % p) //если остаются лишние столбцы после деления, то мы раскидываем их по другим задачам
                        X[i] += 1; //тогда увеличим ширину на 1 для i-й задачи
                    L[i] = (N - j) / p; // определяем высоту столбца для i-й задачи
                    if (i < (N - j) % p) // если остается лишние элементы (остаток) в высоте после деления
                        L[i] += 1; //то увеличим высоту столбца на 1
                }
                else { //если задача не входит в число активных
                    X[i] = 0;
                    L[i] = 0;
                }
            }


            // -------------------------- Отправляем остальным задачам данные --------------

            if (p > 1) { //если число задач больше одной
                for (int i = 1, height = L[0], width = X[0]; i < size; height += L[i], width += X[i], i++) { 
                    //i - по задачам, height - по высоте, width - по ширине
                    int info[5];  // высота A_loc, ширина A_loc, позиция первого элемента x_loc, поз. последного, кол-во активных задач
                    info[0] = N - j; // высота локальной матрицы
                    info[1] = X[i]; // ширина локальной матрицы
                    info[2] = height; // начало локального столбца
                    info[3] = height + L[i]; // конец локального столбца
                    info[4] = p; // количество активных задач
                    MPI_Send(&info, 5, MPI_INT, i, 0, MPI_COMM_WORLD);
                    if (i < p) {
                        Matrix data_A_loc(info[0], info[1]); //создаем локальную матрицу
                        Matrix data_y_loc(L[i], 1); //создаем итоговый вектор
                        Matrix data_x_loc(info[1], 1); //создаем вектор, на который умножаем матрицу
                        for (int k = 0; k < info[0]; ++k)
                            for (int m = 0; m < info[1]; ++m)
                                data_A_loc(k, m) = A(k + j, m + width); //заполняем локальную матрицу
                        for (int k = info[2]; k < info[3]; ++k)
                            data_y_loc(k - info[2], 0) = A(k + j, j); //заполняем локальный итоговый вектор
                        for (int k = 0; k < info[1]; ++k)
                            data_x_loc(k, 0) = A(width + k, j); //заполняем вектор, на который умножаем матрицу

                        MPI_Send(&data_A_loc(0, 0), info[0] * info[1], MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
                        MPI_Send(&data_y_loc(0, 0), info[3] - info[2], MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
                        MPI_Send(&data_x_loc(0, 0), info[1], MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
                    }
                }
            }

            // ------------------------- Инициализация для первой задачи -------------------

            int left = p - 1, right = 1;

            Matrix A_loc(N - j, X[0]);
            Matrix y_loc(L[0], 1);
            Matrix x_loc(X[0], 1);
            for (int k = j; k < N; ++k)
                for (int m = 0; m < X[0]; ++m)
                    A_loc(k - j, m) = A(k, m);
            for (int k = 0; k < L[0]; ++k)
                y_loc(k, 0) = A(k + j, j);
            for (int k = 0; k < X[0]; ++k)
                x_loc(k, 0) = A(k, j);

            // -------------------- Работа параллельного алгоритма -------------------------
            Matrix omega(A_loc.height(), 1);
            for (int i = 0; i < A_loc.width(); ++i)
                for (int j = 0; j < A_loc.height(); ++j)
                    omega(j, 0) -= A_loc(j, i) * x_loc(i, 0); //считаем омега (gaxpy)


            for (int i = 0; i < p; ++i) {
                MPI_Send(&omega(0, 0), omega.height(), MPI_DOUBLE, right, 0, MPI_COMM_WORLD);
                MPI_Recv(&omega(0, 0), omega.height(), MPI_DOUBLE, left, 0, MPI_COMM_WORLD, &status);
                for (int k = 0; k < L[0]; ++k)
                    y_loc(k, 0) += omega(k, 0); //складываем все полученные векторы омега от других задач
            }

            // ---------------------- Сбор результатов вычислений --------------------------


            for (int k = 0; k < L[0]; ++k) //складываем первый кусочек итогового вектора Y
                A(k + j, j) = y_loc(k, 0);

            for (int i = 1, height = j + L[0]; i < p; height += L[i], i++) {
                MPI_Recv(&y_loc(0, 0), L[i], MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status);
                for (int k = 0; k < L[i]; ++k)
                    A(height + k, j) = y_loc(k, 0); //записываем остальные кусочки итогового вектора Y
            }

            for (int k = j + 1; k < N; ++k)
                A(k, j) /= A(j, j); //делим Y на диагональный элемент
            delete[] L;
            delete[] X;
        }
        // ----------------- Вычисляем последний столбец -----------------------------------

        for (int j = N - 1; j < N; ++j) {
            // решение СЛАУ
            for (int i = 0; i < j - 1; ++i)
                for (int k = i + 1; k < j; ++k)
                    A(k, j) -= A(i, j) * A(k, i);
            // gaxpy
            for (int k = 0; k < j; ++k)
                for (int p = j; p < A.height(); ++p)
                    A(p, j) -= A(k, j) * A(p, k);
            for (int k = j + 1; k < A.height(); ++k)
                A(k, j) /= A(j, j);
        }

        std::cout << "n = " << N << '\n';
        std::cout << "eps1 = " << GetDelta(A, B) << "\n";
        std::cout << "eps2 = " << GetDelta1(Orig, A) << "\n";
    }
    else { //остальные задачи, кроме первой
        for (int j = 2; j < N - 1; ++j) {
            int info[5];
            MPI_Recv(&info, 5, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
            if (rank < info[4]) { // если задача входит в число активных задач
                Matrix A_loc(info[0], info[1]);
                Matrix y_loc(info[3] - info[2], 1);
                Matrix x_loc(info[1], 1);

                // ----------------- Принимаем информацию от первой задачи -----------------

                MPI_Recv(&A_loc(0, 0), info[0] * info[1], MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
                MPI_Recv(&y_loc(0, 0), info[3] - info[2], MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
                MPI_Recv(&x_loc(0, 0), info[1], MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);

                // ---------------------------Инициализируем -------------------------------

                int p = info[4]; //число активных задач
                int left = rank - 1, right = rank + 1;
                if (right == p)  right = 0;


                Matrix omega(A_loc.height(), 1);
                for (int i = 0; i < A_loc.width(); ++i)
                    for (int j = 0; j < A_loc.height(); ++j)
                        omega(j, 0) -= A_loc(j, i) * x_loc(i, 0); //считаем омега (gaxpy)


                for (int i = 0; i < p; ++i) {
                    MPI_Send(&omega(0, 0), omega.height(), MPI_DOUBLE, right, 0, MPI_COMM_WORLD);
                    MPI_Recv(&omega(0, 0), omega.height(), MPI_DOUBLE, left, 0, MPI_COMM_WORLD, &status);
                    for (int k = info[2]; k < info[3]; ++k) {
                        y_loc(k - info[2], 0) += omega(k, 0); //складываем все полученные векторы омега от других задач
                    }
                }

                // --------------------- Отправка результата вычислений первой задаче ------------------------

                MPI_Send(&y_loc(0, 0), info[3] - info[2], MPI_DOUBLE, 0, 0, MPI_COMM_WORLD); //отправка y_loc
            }
        }
    }
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    for (int n = 8; n <= 1024; n <<= 1) {
        Parallel(n);
    }
    MPI_Finalize();

}

