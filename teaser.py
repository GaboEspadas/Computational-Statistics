from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression

digits = load_digits()
train_x,test_x,train_y,test_y = train_test_split(digits.data,digits.target,train_size=0.25,random_state=0)

log_model = LogisticRegression()
log_model.fit(train_x,train_y)

log_model.predict(test_x)

score = log_model.score(test_x,test_y)