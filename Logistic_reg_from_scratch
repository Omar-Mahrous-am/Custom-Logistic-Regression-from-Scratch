import numpy as np

class LogisticRegression:
    def __init__(self, lr=.001, n_iters=1000):
        """
        Initialize the logistic regression model.

        Parameters:
        lr (float): Learning rate for gradient descent.
        n_iters (int): Number of iterations for training.
        """
        self.lr = lr
        self.n_iters = n_iters

    @staticmethod
    def sigmoid(y):
        """
        Compute the sigmoid activation.

        Parameters:
        y (array-like): Input values.

        Returns:
        array-like: Sigmoid-transformed values.
        """
        return 1 / (1 + np.exp(-y))

    def fit(self, x_train, y_train):
        """
        Train the logistic regression model using gradient descent.

        Parameters:
        x_train (ndarray): Training feature matrix of shape (n_samples, n_features).
        y_train (ndarray): Target labels of shape (n_samples,).
        """
        n_samples, n_features = x_train.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            linear_pred = np.dot(x_train, self.weights) + self.bias
            prediction = self.sigmoid(linear_pred)

            dw = (1 / n_samples) * np.dot(x_train.T, (prediction - y_train))
            db = (1 / n_samples) * np.sum(prediction - y_train)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, x_test, threshold=.5):
        """
        Predict class labels for given input data.

        Parameters:
        x_test (ndarray): Test feature matrix of shape (n_samples, n_features).
        threshold (float): Classification threshold (default is 0.5).

        Returns:
        list: Predicted class labels (0 or 1).
        """
        if not (0 <= threshold <= 1):
            raise ValueError("Threshold must be between 0 and 1.")
        linear_pred = np.dot(x_test, self.weights) + self.bias
        y_prob = self.sigmoid(linear_pred)
        y_predictions = [1 if y >= threshold else 0 for y in y_prob]
        return y_predictions

if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    
    X, y = make_classification(n_samples=500, n_features=5, n_classes=2, random_state=42)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    
    model = LogisticRegression(lr=0.01, n_iters=1000)
    model.fit(x_train, y_train)

    
    y_pred = model.predict(x_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
