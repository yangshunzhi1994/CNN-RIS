#pip install correlation-pearson



from correlation_pearson.code import CorrelationPearson

X_Speed = [0.73, 0.81,  1.53, 1.97,  2.29, 2.86]

X_Energy = [1.507, 1.235, 0.654, 0.864, 0.656, 0.490]

correlation = CorrelationPearson()

print ('Correlation coefficient of speed and Energy:' + str(correlation.result(X_Speed, X_Energy)))



Y_Power = [1.0, 1.0,  1.1, 1.4,  1.5, 1.7]

Y_Energy = [0.654, 1.235, 1.507, 0.490, 0.656, 0.864]

print ('Correlation coefficient of Power and Energy:' + str(correlation.result(Y_Power, Y_Energy)))

