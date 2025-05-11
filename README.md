# Python Perceptron Implementation: A Coding Quest

## Overview
This project contains a Python implementation of a simple Perceptron classifier, a fundamental algorithm in machine learning and the foundational building block of neural networks. The code allows you to initialize, train, and make predictions with a Perceptron model. This implementation is demonstrated by training it to learn the AND logic gate.

This README provides all the steps to set up the project, understand the code, and run the example.

## Features
-   Initialize a Perceptron with a configurable learning rate and number of training epochs.
-   Train the Perceptron on a given dataset using the Perceptron learning algorithm.
-   Make predictions on new, unseen data instances.
-   Uses a simple step function as the activation function.
-   Initializes weights and bias, which are then updated during training.
-   Includes an example of training the Perceptron for the AND gate.

## Prerequisites
-   Python 3.x
-   `pip` (Python package installer), which usually comes with Python.

## Project Setup and Installation

Follow these steps to get the project up and running on your local machine.

1.  **Create a Project Directory:**
    Open your terminal or command prompt. Create a new folder for this project and navigate into it:
    ```bash
    mkdir perceptron_quest
    cd perceptron_quest
    ```

2.  **Create the Python File (`perceptron.py`):**
    Inside the `perceptron_quest` directory, create a Python file named `perceptron.py`. Copy and paste the entire Perceptron class code and the example usage into this file:

    ```python
    import numpy as np

    # Perceptron Python Blueprint: Your Coding Quest
    # Welcome back, Pioneer!
    # It's time to translate your understanding into robust Python code.
    # Below is a class structure for your Perceptron.
    # Your mission is to implement the logic within each method.
    # The Grand Design: Perceptron Class

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
            self.learning_rate = learning_rate
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
            #     raise ValueError("Weights and bias not initialized. Call fit() first.")

            weighted_sum = np.dot(X_instance, self.weights) + self.bias
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
            linear_output = self._weighted_sum(X_instance)

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
            self.bias = np.random.rand(1) * 0.01 # Bias is a scalar, rand(1) makes it an array, could be just 0.0

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

    # Example usage (after you've implemented the class):
    # It's good practice to put example usage inside an `if __name__ == '__main__':` block
    if __name__ == '__main__':
        X_and = np.array([[0,0], [0,1], [1,0], [1,1]])
        y_and = np.array([0, 0, 0, 1]) # For AND gate

        # Create and train the perceptron
        ptron = Perceptron(learning_rate=0.1, n_epochs=100)
        ptron.fit(X_and, y_and)

        # Test predictions
        print("\nTesting AND Gate:")
        for xi, target in zip(X_and, y_and):
            pred = ptron.predict(xi)
            print(f"Input: {xi}, Target: {target}, Prediction: {pred}")

        print(f"\nFinal Weights: {ptron.weights}")
        print(f"Final Bias: {ptron.bias}")
    ```

3.  **Create and Activate a Virtual Environment:**
    It's highly recommended to use a virtual environment to manage project dependencies and avoid conflicts with global Python packages.

    * **On macOS and Linux:**
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
    * **On Windows (Command Prompt):**
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```
    * **On Windows (PowerShell):**
        ```bash
        python -m venv venv
        .\venv\Scripts\Activate.ps1
        ```
    After activation, your terminal prompt should change to indicate that the virtual environment (`venv`) is active.

4.  **Create `requirements.txt`:**
    This file lists all the Python packages your project depends on. For this Perceptron, the only external dependency is `numpy`. Create a file named `requirements.txt` in your project directory (`perceptron_quest`) and add the following line:
    ```
    numpy
    ```

5.  **Install Dependencies:**
    With your virtual environment active, install the necessary packages using `pip` and your `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```
    This command tells pip to install all the packages listed in the `requirements.txt` file.

## How to Run the Code

Once you have completed the setup and installation steps:

1.  Ensure your virtual environment is active (you should see `(venv)` or similar in your terminal prompt).
2.  Navigate to the project directory (`perceptron_quest`) if you aren't already there.
3.  Run the `perceptron.py` script using the following command:
    ```bash
    python perceptron.py
    ```

**Expected Output:**
The script will train the Perceptron on the AND gate dataset. You will see output similar to the following (the exact epoch of convergence and final weights/bias may vary slightly due to random initialization):