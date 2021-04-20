from sklearn.naive_bayes import MultinomialNB

X_train = [
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1]
]
y_train = [1, 1, 1, 0, 0, 0]

#学習
model = MultinomialNB()
model.fit(X_train, y_train)

# 検証
y_pred = model.predict([[1,0,0,1,0,1,0,0,0,0,0]])
print(y_pred)

