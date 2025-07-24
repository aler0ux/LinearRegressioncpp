#include <iostream>
#include <vector>
#include <iomanip>
#include "LinearRegression.h"

using namespace std;

int main() {
    LinearRegressor model(0.0001, 30000); 

    LossFunction LF;

    //Dummy data for training(linear)
    vector<vector<double>> x_train {{25, 35}, {50,60}};
    vector<double> y_train {25,50};
    vector<vector<double>> x_test {{40,50},{50,60}};
    vector<double> y_test {40,50};

    model.train(x_train, y_train);
    
    vector<double> y_pred;
    vector<double> pred = model.predict(x_test, y_pred);

    cout<<"Bias: "<<model.bias<<endl<<"Weights: "<<model.weights[0]<<"\t"<<model.weights[1]<<endl<<endl;
    cout<<"Predicted value:\t\t"<<"Actual Value: "<<endl;
    for(int i = 0; i<y_test.size(); i++){
        cout<<pred.at(i)<<"\t\t\t"<<y_test.at(i)<<endl;
    }
    double loss = LF.MSE(pred, y_test);
    
    cout << fixed << setprecision(9);
    cout<<endl<<"Loss is: "<<loss<<endl;

    return 0;
}