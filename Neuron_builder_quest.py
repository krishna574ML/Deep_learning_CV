import numpy as np

#Perceptron Python Blueprint: Your Coding QuestWelcome back, Pioneer! 
# It's time to translate your understanding into robust Python code. 
# Below is a class structure for your Perceptron. 
# Your mission is to implement the logic within each method.T
# he Grand Design: Perceptron Classimport numpy as np # It's common and useful for this!

class Perceptron:
    """
    A simple Perceptron classifier.
    """
    # Attributes will be defined in __init__
    # self.learning_rate
    # self.n_epochs
    # self.weights
    # self.bias
    # self.activation_fn

    def __init__(self, learning_rate=0.01, n_epochs=100):
        """
        Quest Levels: 1 (Anatomy), 6 (Pace of Learning), 7 (Bias)
        Objective: Initialize the Perceptron's core properties.
        """
        # --- YOUR CODE STARTS HERE ---
        # Task 1: Store the learning rate and number of epochs.
        self.learning_rate =  learning_rate
        self.n_epochs = n_epochs

        # Task 2: Initialize weights and bias.
        # These will be set properly in the 'fit' method once we know n_features.
        # For now, you can set them to None or an empty list/default value.
        self.weights = None
        self.bias = None

        # Task 3 (Corresponds to Level 3 - Activation Function):
        # Define or assign the activation function.
        # We'll use a simple step function here.
        self.activation_fn = self._step_function
        # --- YOUR CODE ENDS HERE ---
        

    def _step_function(self, x):
        """
        Quest Level: 3 (Activation Function)
        Objective: Implement the decision-making step.
        Output: 1 if x >= 0, else 0 (or -1 if you prefer for bipolar targets)
        """
        # --- YOUR CODE STARTS HERE ---
        if x >= 0:
            return 1
        else:
            return 0 
        # --- YOUR CODE ENDS HERE ---
        

    def _weighted_sum(self, X_instance):
        """
        Quest Level: 2 (Weighted Sum)
        Objective: Calculate the net input to the neuron.
        X_instance: A single sample of input features (e.g., a numpy array or list).
        Output: The dot product of inputs and weights, plus bias.
        """
        # Ensure weights and bias are initialized (e.g., in fit method)
        # if self.weights is None or self.bias is None:
        #     raise ValueError("Weights and bias not initialized. Call fit() first.")

        weighted_sum = np.dot(X_instance , self.weights) + self.bias
        return weighted_sum

    def predict(self, X_instance):
        """
        Quest Level: 4 (First Prediction - Forward Pass)
        Objective: Make a prediction for a single instance.
        X_instance: A single sample of input features.
        Output: The class label (0 or 1, or -1 or 1).
        """
        # --- YOUR CODE STARTS HERE ---
        # Step 1: Calculate the weighted sum using self._weighted_sum()
        linear_output = self._weighted_sum(X_instance) # Corrected line

        # Step 2: Apply the activation function
        prediction = self.activation_fn(linear_output)
        return prediction
        # --- YOUR CODE ENDS HERE ---

    def fit(self, X_train, y_train):
        """
        Quest Levels: 1, 5, 6, 7, 8, 9 (All Training Aspects)
        Objective: Train the Perceptron model.
        X_train: Training vectors (samples x features).
        y_train: Target values (labels).
        """
        n_samples, n_features = X_train.shape

        # --- YOUR CODE STARTS HERE ---

        # Task 1 (Quest Level 1 & 7 - Initialize weights and bias now that we know n_features)
        # Initialize weights: Small random numbers or zeros. One weight per feature.
        self.weights = np.random.rand(n_features) * 0.01
        # Initialize bias: Typically 0 or a small random number.
        self.bias = np.random.rand(1) * 0.01

        # Task 2 (Quest Level 9 - Training Loop for multiple epochs)
        for epoch in range(self.n_epochs):
            errors_in_epoch = 0 # Optional: for tracking convergence

            # Task 3 (Quest Level 8 - Loop through each training sample)
            for idx, x_i in enumerate(X_train):
                y_true = y_train[idx]

                # Task 3a (Quest Level 4 - Make a prediction)
                prediction = self.predict(x_i)

                # Task 3b (Quest Level 5 - Calculate the error)
                error = y_true - prediction

                # Task 3c (Quest Level 6 - Update weights and bias)
                if error != 0:
                    update = self.learning_rate * error
                    self.weights += update * x_i
                    self.bias += update
                    errors_in_epoch += 1 # Optional

            # Optional: Check for convergence
            if errors_in_epoch == 0:
                print(f"Converged at epoch {epoch+1}")
                break
        # --- YOUR CODE ENDS HERE ---

#How to Embark on Your Coding Quest:Copy the Class Structure: Take the Perceptron class skeleton above and put it into your Python editor.Fill in the Blanks: Go method by method (__init__, _step_function, _weighted_sum, predict, fit). Inside each method, look for the # --- YOUR CODE STARTS HERE --- and # --- YOUR CODE ENDS HERE --- comments. This is where you'll write your Python logic based on the objectives and your understanding from the "Neuron Builder Quest."Incremental Testing (Highly Recommended!):After filling in __init__ and _step_function, try creating an instance of Perceptron and testing _step_function directly with some numbers.Once _weighted_sum is done (you'll need to temporarily set some dummy self.weights and self.bias in __init__ or directly in your test script for this part before fit is complete), test it with sample inputs.Test predict after the above are working.The fit method is the grand finale where everything comes together.Create a Simple Dataset: To test your fit and predict methods, you'll need a small, linearly separable dataset. The AND gate is perfect:# Example usage (after you've implemented the class):
X_and = np.array([[0,0], [0,1], [1,0], [1,1]])
y_and = np.array([0, 0, 0, 1]) # For AND gate

# # Create and train the perceptron
ptron = Perceptron(learning_rate=0.1, n_epochs=100)
ptron.fit(X_and, y_and)

# # Test predictions
print("Testing AND Gate:")
for xi, target in zip(X_and, y_and):
    pred = ptron.predict(xi)
    print(f"Input: {xi}, Target: {target}, Prediction: {pred}")

print(f"Final Weights: {ptron.weights}")
print(f"Final Bias: {ptron.bias}")
