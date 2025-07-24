#include <vector>
#include <iostream>
using namespace std;

class LossFunction {
    public:
        double MSE(vector<double> pred, vector<double> actual);
};
pair<vector<double>,double> Gradient_Descent(vector<vector<double>> x, vector<double> y, double alpha, int epochs);

class LinearRegressor {
    public:
        int epoch = 0;
        double alpha = 0;
        vector<double> weights;
        double bias = 0;
        int features;
        // Constructor to initialize the learning rate and number of epochs
        LinearRegressor(double learning_rate=0.0001, int epoch=1000);
        void train(vector<vector<double>> x, vector<double> y);
        vector<double> predict(vector<vector<double>> x, vector<double> pred);

    private:
        double gettingValues(vector<double> x );
};