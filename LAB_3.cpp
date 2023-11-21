#include <iostream>
#include <ctime>
#include <omp.h>
#include <chrono>
using namespace std;
#define NUM_THREADS 4

double** generate_A_matrix(int N) {
	srand(time(0));
	
	double** A = new double* [N];
	double sum;
	double sign;
	for (int i = 0; i < N; i++) {
		A[i] = new double[N];
	}

	for (int i = 0; i < N; i++) {
		sum = 0.0;
		for (int j = 0; j < N; j++) {
			if (i == j) {
				continue;
			}
			else {
				if (rand() % 10 < 5) {
					sign = 1.0;
				}
				else {
					sign = -1.0;
				}

				A[i][j] = sign * (rand() / 100 + 1);
				sum += abs(A[i][j]);
			}
		}

		A[i][i] = sign * (sum + rand() % 10);
	}

	return A;
}

double* generate_B_matrix(double** A, int N) {
	srand(time(0));

	double* B = new double[N];
	double* X_rand = new double[N];

	for (int i = 0; i < N; i++) {
		B[i] = 0.0;
		X_rand[i] = rand() / 1000;
	}

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			B[i] += A[i][j] * X_rand[j];
		}
	}

	return B;
}

double* jacobi_serial(double** A, double* B, int N) {
	double* X = new double[N];
	double eps = 0.0001;
	double* X_next = new double[N];
	bool flag;

	for (int i = 0; i < N; i++) {
		X[i] = B[i] / A[i][i];
	}

	do {
		for (int i = 0; i < N; i++) {
			X_next[i] = B[i] / A[i][i];
			for (int j = 0; j < N; j++) {
				if (i == j) {
					continue;
				}
				else {
					X_next[i] -= A[i][j] / A[i][i] * X[j];
				}
			}
		}

		flag = true;
		for (int i = 0; i < N; i++) {
			if (abs(X_next[i] - X[i]) > eps) {
				flag = false;
				break;
			}
		}

		for (int i = 0; i < N; i++) {
			X[i] = X_next[i];
		}

		if (flag) {
			break;
		}

	} while (1);

	return X;
}

double* jacobi_parallel(double** A, double* B, int N) {
	double* X = new double[N];
	double eps = 0.0001;
	double* X_next = new double[N];
	bool flag;


    #pragma omp parallel for num_threads(NUM_THREADS)
	for (int i = 0; i < N; i++) {
		X[i] = B[i] / A[i][i];
	}

	do {
        #pragma omp parallel for num_threads(NUM_THREADS) 
		for (int i = 0; i < N; i++) {
			X_next[i] = B[i] / A[i][i];
			for (int j = 0; j < N; j++) {
				if (i == j) {
					continue;
				}
				else {
					X_next[i] -= (A[i][j] / A[i][i]) * X[j];
				}
			}
		}

		flag = true;
		for (int i = 0; i < N; i++) {
			if (abs(X_next[i] - X[i]) > eps) {
				flag = false;
				break;
			}
		}

        #pragma omp parallel for num_threads(NUM_THREADS)
		for (int i = 0; i < N; i++) {
			X[i] = X_next[i];
		}

		if (flag) {
			break;
		}

	} while (1);

	return X;
}



int main() {
	double** A;
	double* B;
	double *X_ser, *X_par;
	int N;

	/*ifstream input;
	input.open("input.txt");

	input >> N;

	A = new double* [N];
	for (int i = 0; i < N; i++)
	{
		A[i] = new double[N];
	}
	
	B = new double[N];

	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j <= N; j++)
		{
			if (j != N)
			{
				input >> A[i][j];
			}
			else
			{
				input >> B[i];
			}
		}
	}

	input.close();*/

	cout << "Enter matrix size: ";
	cin >> N;
	cout << endl;

	A = generate_A_matrix(N);
	B = generate_B_matrix(A, N);

	//for (int i = 0; i < N; i++)
	//{
	//	for (int j = 0; j <= N; j++)
	//	{
	//		if (j != N)
	//		{
	//			cout << A[i][j] << "\t";
	//		}
	//		else
	//		{
	//			cout << B[i] << endl;
	//		}
	//	}
	//}
	//cout << endl;

	chrono::steady_clock::time_point start, stop;
	std::chrono::duration<double> duration;
	double serial_time, parallel_time;

	start = chrono::high_resolution_clock::now();
	X_ser = jacobi_serial(A, B, N);
	stop = chrono::high_resolution_clock::now();
	duration = stop - start;
	serial_time = duration.count();

	//for (int i = 0; i < N; i++)
	//{
	//	cout << X_ser[i] << " ";
	//}
	//cout << endl;

	start = chrono::high_resolution_clock::now();
	X_par = jacobi_parallel(A, B, N);
	stop = chrono::high_resolution_clock::now();
	duration = stop - start;
	parallel_time = duration.count();

	//for (int i = 0; i < N; i++)
	//{
	//	cout << X_par[i] << " ";
	//}
	//cout << endl;

	cout << "Running time of the sequential algorithm: " << serial_time << " seconds" << endl;
	cout << "Running time with parralel algorithm: " << parallel_time << " seconds" << endl;
	if (serial_time / parallel_time < 1) {
		cout << parallel_time / serial_time << " times slower" << endl;
	}
	else {
		cout << serial_time / parallel_time << " times faster" << endl;
	}

	return 0;
}
