#include "LinearRegression.h"
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <vector>
using namespace std;

LinearRegressor::LinearRegressor(double learning_rate, int epoch){
    //Assigning the learning rate and number of epochs
    this->alpha = learning_rate;
    this->epoch = epoch;
}
void LinearRegressor::train(vector<vector<double>> x, vector<double> y){
    int sz = x.size();
    int features = x[0].size();

    //Obtain optimised weights and bias using gradient descent
    pair<vector<double>,double> WeightsAndBias = Gradient_Descent(x,y,alpha,epoch);

    this->weights = WeightsAndBias.first;
    this->bias = WeightsAndBias.second;
    this->features = features;
}
//Gradient Descent function
pair<vector<double>,double> Gradient_Descent(vector<vector<double>> x, vector<double> y, double alpha, int epochs){
    int m = x.size();
    int n = x[0].size();
    double bias = 0;
    vector<double> weights(n, 0.0);

    for (int _=0;_<epochs; _++) {
        double updated_bias = 0.0;
        vector<double> updated_weights(n, 0.0);

        for (int i=0; i<m ; i++){
            double prediction = bias;
            for (int k = 0; k < n; k++) {
                prediction += weights[k] * x[i][k];
            }
            double error = prediction - y[i];

            for (int k=0;k<n;k++){
                updated_weights[k] += (1.0/m)*(x[i][k])*error;
            }
            updated_bias += (1.0/m)*error;
        }
        for (int i = 0; i<n; i++){
            weights[i] = weights[i] - alpha * updated_weights[i];
        }
        bias = bias - alpha * updated_bias;
    }
    return make_pair(weights, bias);
}
vector<double> LinearRegressor::predict(vector<vector<double>> x, vector<double> pred){
    int pred_size = x.size();
    for (int i = 0; i < pred_size; i++) {
        pred.push_back(gettingValues(x[i]));
    }
    return pred;
}
double LinearRegressor::gettingValues(vector<double> x) {
    double y = bias;

    for (int i = 0; i < features; i++) {
        y += weights[i] * x[i];
    }
    
    return y;
}
double LossFunction::MSE(vector<double> pred, vector<double> actual) {
    double m = actual.size();
    double loss = 0.0;

    for(int i=0; i<m; i++) {
        loss += pow(actual[i] - pred[i], 2);
    }
    double FinalLoss = loss/(2*m);
    return FinalLoss;
}