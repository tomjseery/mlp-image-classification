import numpy as np

from sklearn.model_selection import train_test_split, ParameterGrid, StratifiedKFold, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.metrics import accuracy_score, log_loss, precision_score, recall_score, f1_score, ConfusionMatrixDisplay

from scipy.stats import ttest_rel

from PIL import Image

import os

import matplotlib.pyplot as plt

# Please write the optimal hyperparameter values you obtain in the global variable 'optimal_hyperparm' below. This
# variable should contain the values when I look at your submission. I should not have to run your code to populate this
# variable.
optimal_hyperparam = {
    'hidden_layer_sizes': (128, 64),
    'activation': 'relu',
    'solver': 'sgd',
    'momentum': 0.9,
    'learning_rate_init': 0.005,
    'learning_rate': 'adaptive',
    'batch_size': 128
}


class COC131:
    def q1(self, filename=None):
        """
        This function should be used to load the data. To speed-up processing in later steps, lower resolution of the
        image to 32*32. The folder names in the root directory of the dataset are the class names. After loading the
        dataset, you should save it into an instance variable self.x (for samples) and self.y (for labels). Both self.x
        and self.y should be numpy arrays of dtype float.

        :param filename: this is the name of an actual random image in the dataset. You don't need this to load the
        dataset. This is used by me for testing your implementation.
        :return res1: a one-dimensional numpy array containing the flattened low-resolution image in file 'filename'.
        Flatten the image in the row major order. The dtype for the array should be float.
        :return res2: a string containing the class name for the image in file 'filename'. This string should be same as
        one of the folder names in the originally shared dataset.
        """

        data_dir = "./dataset"

        data = []
        labels = []

        res1 = None
        res2 = None

        for class_name in os.listdir(data_dir):
            # Get Absolute directory from relative
            class_folder = os.path.join(data_dir, class_name)

            if not os.path.isdir(class_folder):
                continue

            # Iterate through each image in the directory
            for img_name in os.listdir(class_folder):
                img_path = os.path.join(class_folder, img_name)

                try:
                    with Image.open(img_path) as img:
                        img = img.resize((32, 32))  # Resize to 32x32
                        img_array = np.array(img, dtype=float).flatten()  # Flatten it

                        data.append(img_array)
                        labels.append(class_name)

                        # If matching filename is found, add to responses
                        if filename is not None and img_name == filename:
                            res1 = img_array
                            res2 = class_name

                except Exception as e:
                    print(f"Could not load {img_path}: {e}")

        self.x = np.array(data, dtype=float)

        # Ensure y is encoded as float, as question specifically requires
        le = LabelEncoder()
        self.y = le.fit_transform(labels).astype(float)
        self.class_names = le.classes_ # List of unique classes
        self.label_encoder = le

        # Default return if filename wasn't found
        if res1 is None:
            res1 = np.zeros((32 * 32,), dtype=float)

        return res1, res2

    def q2(self, inp):
        """
        This function should compute the standardized data from a given 'inp' data. The function should work for a
        dataset with any number of features.

        :param inp: an array from which the standardized data is to be computed.
        :return res2: a numpy array containing the standardized data with standard deviation of 2.5. The array should
        have the same dimensions as the original data
        :return res1: sklearn object used for standardization.
        """
        scaler = StandardScaler()
        standardized = scaler.fit_transform(inp)

        res1 = scaler
        res2 = standardized * 2.5  # Ensure its multiplied by 2.5

        return res2, res1

    def standardize_train_test(self, x_train, x_test):
        """
        Standardize both the training and testing data.

        :param x_train: The training data to be standardized.
        :param x_test: The testing data to be standardized.
        :return: the standardized x_train and x_test
        """
        x_train_standardized, scaler = self.q2(x_train) # Utilize q2 to standardize x_train and get the scaler
        x_test_standardized = scaler.transform(x_test) * 2.5  # Apply the same scaling to x_test

        return x_train_standardized, x_test_standardized

    def q3(self, test_size=None, pre_split_data=None, hyperparam=None):
        """
        This function should build a MLP Classifier using the dataset loaded in function 'q1' and evaluate model
        performance. You can assume that the function 'q1' has been called prior to calling this function. This function
        should support hyperparameter optimizations.

        :param test_size: the proportion of the dataset that should be reserved for testing. This should be a fraction
        between 0 and 1.
        :param pre_split_data: Can be used to provide data already split into training and testing.
        :param hyperparam: hyperparameter values to be tested during hyperparameter optimization.
        :return: The function should return 1 model object and 3 numpy arrays which contain the loss, training accuracy
        and testing accuracy after each training iteration for the best model you found.

        TS: This code performs a 6-Stage Hyperparameter search, where in each stage, it investiages the best
        TS: value of the respective hyperparameter, with the others being left unchanged.
        TS: Each consequent stage will use this new best value in conjunction with a different hyperparameter.
        TS: Print logs are used throughout to better illustrate which hyperparameter perform better under which situations,
        TS: which are further explained in the markdown
        """

        if pre_split_data:
            x_train, x_test, y_train, y_test = pre_split_data
        else:
            x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=test_size, random_state=42,
                                                                stratify=self.y)

        # Standardize data
        x_train, x_test = self.standardize_train_test(x_train, x_test)
        
        # Trains from hyperparam if one if provided
        if hyperparam:
            print(f"Training with provided hyperparameters: {hyperparam}")
            model = MLPClassifier(**hyperparam)

            # Call the training and evaluation function
            res1, res2, res3, res4 = COC131.train_validate_loop(model, x_train, y_train, x_test, y_test)
            print(f"Model trained with provided hyperparameters")

            return res1, res2, res3, res4

        # Hyperparameter configurations
        hidden_layer_params = [
            (16,),
            (32,),
            (32, 16),
            (64, 32),
            (128, 64)
        ]
        # Adam and SGD parameters
        adam_params = {
            "solver": "adam",
            "learning_rate_init": 0.001,  # Adam is more adaptive and does well typically with lower starting lr
            "learning_rate": "constant"
        }

        sgd_params = {
            "solver": "sgd",
            "momentum": 0.9,
            "learning_rate_init": 0.005,  # SGD tends to start with a higher learning rate
            "learning_rate": "adaptive"
        }

        # Compile the default additional params for SGD and Adam to accuracy compare then using their typical peak performance metrics
        solver_params = {
            "sgd": sgd_params,
            "adam": adam_params,
        }

        # Since Adam and SGD typically thrive better on different learning rates, this will be considered here
        learning_rate_init_params = {
            "sgd": [0.005, 0.001, 0.0005],
            "adam": [0.001, 0.0005, 0.0001]
        }

        default_params = { # Params applied to every single model to ensure certain consistencies
            "early_stopping": True, # Stop training early if the validation score stops improving
            "max_iter": 1000, # Max no of iterations
            "n_iter_no_change": 10,  # How many iterations with no improvement before early stopping applies
            "validation_fraction": 0.1,  # Proportion of training data used for early stopping
            "tol": 1e-4,  # Tolerance
            "random_state": 42
        }

        best_val_score = -np.inf  # Initialize best validation score
        best_hyperparams = {}

        # Stage 1: Find the best hidden layer configuration using Adam defaults
        print("Stage 1: Testing hidden layer configurations with Adam")
        for hidden_layers in hidden_layer_params:
            model_params = {"hidden_layer_sizes": hidden_layers, **adam_params, **default_params}
            model = MLPClassifier(**model_params)
            best_val_score, best_hyperparams = COC131.train_evaluate(model, {"hidden_layer_sizes": hidden_layers},
                                                                     x_train, y_train, x_test, y_test,
                                                                     best_val_score, best_hyperparams)

        print(
            f"Conclusion: After testing different hidden layer configurations, {best_hyperparams['hidden_layer_sizes']} gave the best validation score")

        # Stage 2: Now test activation functions relu and tanh use using the best hidden layer configuration from Stage 1
        print(f"Stage 2: Testing activation functions with Adam and best hyperparameters: {best_hyperparams}")
        best_val_score = -np.inf  # Rest best val score for next validation checks
        for activation in ["relu", "tanh"]:
            model_params = {**best_hyperparams, **default_params, "activation": activation}
            model = MLPClassifier(**model_params)
            best_val_score, best_hyperparams = COC131.train_evaluate(model, {"activation": activation}, x_train,
                                                                     y_train, x_test, y_test,
                                                                     best_val_score, best_hyperparams)

        print(f"Conclusion: Between relu and tanh, {best_hyperparams['activation']} performed better")

        # Stage 3: Now test both solvers (Adam and SGD) using the hyperparameters from Stage 2
        best_val_score = -np.inf
        print(f"Stage 3: Testing solvers with best hyperparameters: {best_hyperparams}")
        for solver_specific_params in solver_params.values():
            model_params = {**best_hyperparams, **solver_specific_params, **default_params}
            model = MLPClassifier(**model_params)
            best_val_score, best_hyperparams = COC131.train_evaluate(model, solver_specific_params, x_train, y_train,
                                                                     x_test, y_test,
                                                                     best_val_score, best_hyperparams)

        print(f"Conclusion: Between Adam and SGD, {best_hyperparams['solver']} performed better")

        # Stage 4: Testing learning rate type (SGD only)
        # Here, we investigate whether SGD performs better using either constant, or adaptive learning rate (if SGD was picked)
        best_val_score = -np.inf
        print(f"Stage 4: Testing learning rates with best hyperparameters: {best_hyperparams}")

        for learning_rate in ["constant", "adaptive"]:
            model_params = {**best_hyperparams, **default_params, "learning_rate": learning_rate}
            model = MLPClassifier(**model_params)
            best_val_score, best_hyperparams = COC131.train_evaluate(model, {"learning_rate": learning_rate}, x_train,
                                                                     y_train, x_test, y_test,
                                                                     best_val_score, best_hyperparams)

        print(f"Conclusion: The learning rate strategy '{best_hyperparams['learning_rate']}' performed best on SGD")

        # Stage 5: Test the learning rate init for best solver and learning rate
        best_val_score = -np.inf
        print(f"Stage 5: Testing learning rate inits with best hyperparameters: {best_hyperparams}")
        for learning_rate_init in learning_rate_init_params[best_hyperparams["solver"]]:
            model_params = {**best_hyperparams, **default_params, "learning_rate_init": learning_rate_init}
            model = MLPClassifier(**model_params)
            best_val_score, best_hyperparams = COC131.train_evaluate(model, {"learning_rate_init": learning_rate_init},
                                                                     x_train, y_train, x_test, y_test,
                                                                     best_val_score, best_hyperparams)

        print(f"Conclusion: The learning rate initialization for {best_hyperparams['solver']} optimizer affects the convergence speed and stability, where {best_hyperparams['learning_rate_init']} was most optimal")

        # Stage 6: Test the batch size with the best learning rate from Stage 5
        print(f"Stage 6: Testing batch sizes with best hyperparameters: {best_hyperparams}")
        best_val_score = -np.inf
        for batch_size in [32, 64, 128]:
            model_params = {**best_hyperparams, "batch_size": batch_size}
            model = MLPClassifier(**model_params, **default_params)
            best_val_score, best_hyperparams = COC131.train_evaluate(model, {"batch_size": batch_size}, x_train,
                                                                     y_train, x_test, y_test,
                                                                     best_val_score, best_hyperparams)

        print(f"Conclusion: The batch size of {best_hyperparams['batch_size']} performed best")

        # After finding the best hyperparameters, train the final model
        print(f"Best Hyperparameters found: {best_hyperparams}")
        best_model = MLPClassifier(**best_hyperparams,
                                   random_state=42)  # We dont need max_iter here or early stopping as we are using partial_fit

        print("Training model with best hyperparameters")
        # Call the training and evaluation function with the best model
        res1, res2, res3, res4 = COC131.train_validate_loop(best_model, x_train, y_train, x_test, y_test)

        # Return the final model and performance metrics
        return res1, res2, res3, res4

    @staticmethod
    def record_metrics(model, x_train, y_train, x_test, y_test):
        """
        Computes and returns the loss, training accuracy, and testing accuracy for the model

        :param model: The model
        :param x_train: Training features
        :param y_train: Training labels
        :param x_test: Test features
        :param y_test: Test labels
        :return: loss value, train accuracy, test accuracy
        """
        loss = model.loss_ if hasattr(model, 'loss_') else np.nan
        train_accuracy = model.score(x_train, y_train)
        test_accuracy = model.score(x_test, y_test)
        return loss, train_accuracy, test_accuracy

    @staticmethod
    def train_evaluate(model, params, x_train, y_train, x_test, y_test, best_val_score, best_hyperparams):
        """
        Handles training, evaluation, and updating the best hyperparameters if performance improves.

        :param model: The model to train and evaluate
        :param params: The hyperparameters to update best_hyperparams
        :param x_train: Training features
        :param y_train: Training labels
        :param x_test: Test features
        :param y_test: Test labels
        :param best_val_score: The current best validation score
        :param best_hyperparams: The dictionary of the current best hyperparameters
        :return: Updated best_val_score, updated best_hyperparams
        """
        print(f"Testing hyperparam(s): {params}")
        # Train the model
        model.fit(x_train, y_train)

        # Evaluate the model
        val_score = model.score(x_test, y_test)

        # If this model performs better, update the best score and hyperparameters
        if val_score > best_val_score:
            best_val_score = val_score

            # Update the hyperparameters with the provided params
            best_hyperparams.update(params)

            print(f"New best configuration: {best_hyperparams}, Validation score: {val_score}")

        return best_val_score, best_hyperparams

    @staticmethod
    def train_validate_loop(model, x_train, y_train, x_test, y_test, n_epochs=100):
        """
        Trains the MLPClassifier using partial_fit on the model, and records loss, training, and test accuracy.

        :param model: The model
        :param x_train: The training data features
        :param y_train: The training data labels
        :param x_test: The testing data features
        :param y_test: The testing data labels
        :param n_epochs: The number of epochs for training (default is 100)
        :return: The trained model, loss curve, training accuracy curve, and testing accuracy curve
        """

        # Initialize arrays to store the metrics
        loss_curve = np.zeros(n_epochs)
        train_accs = np.zeros(n_epochs)
        test_accs = np.zeros(n_epochs)

        local_best_acc = -np.inf
        classes = np.unique(y_train)

        # Epoch 0 - need to pass classes for partial_fit
        model.partial_fit(x_train, y_train, classes=classes)
        loss, tr_acc, te_acc = COC131.record_metrics(model, x_train, y_train, x_test, y_test)
        loss_curve[0] = loss
        train_accs[0] = tr_acc
        test_accs[0] = te_acc
        local_best_acc = te_acc

        # Run the rest of the epochs as normal
        for epoch in range(1, n_epochs):
            model.partial_fit(x_train, y_train)
            loss, tr_acc, te_acc = COC131.record_metrics(model, x_train, y_train, x_test, y_test)
            loss_curve[epoch] = loss
            train_accs[epoch] = tr_acc
            test_accs[epoch] = te_acc

            # Track the best test accuracy
            if te_acc > local_best_acc:
                local_best_acc = te_acc

        # Return the final trained model and performance metrics
        return model, loss_curve, train_accs, test_accs

    def q4(self):
        """
        This function should study the impact of alpha on the performance and parameters of the model. For each value of
        alpha in the list below, train a separate MLPClassifier from scratch. Other hyperparameters for the model can
        be set to the best values you found in 'q3'. You can assume that the function 'q1' has been called
        prior to calling this function.

        :return: res should be the data you visualized.
        """

        alpha_values = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 50, 100]

        n = len(alpha_values)

        # Initialize metrics arrays
        metrics = {
            'alpha': np.array(alpha_values),
            'train_accs': np.zeros(n),
            'test_accs': np.zeros(n),
            'acc_gaps': np.zeros(n),  # Difference between train and test accuracy
            'train_losses': np.zeros(n),
            'test_losses': np.zeros(n),
            'test_precisions': np.zeros(n),
            'test_recalls': np.zeros(n),
            'test_f1s': np.zeros(n),
            'all_weights': np.zeros(n, dtype=object),
            'all_biases': np.zeros(n, dtype=object),
            'weight_norms': np.zeros(n),
            'weight_means': np.zeros(n),
            'weight_stds': np.zeros(n),
            'bias_norms': np.zeros(n),
            'bias_means': np.zeros(n),
            'bias_stds': np.zeros(n)
        }

        x_train, x_test, y_train, y_test = train_test_split(
            self.x, self.y,
            test_size=0.3,
            stratify=self.y,
            random_state=42
        )

        # Standardize data
        x_train, x_test = self.standardize_train_test(x_train, x_test)

        # Copy best hyperparameters
        params = optimal_hyperparam.copy()
        params['max_iter'] = 1000  # Ensures full convergence

        for i, alpha in enumerate(alpha_values):
            print(f"\nTesting alpha value {i + 1}/{n}: alpha={alpha}")
            params['alpha'] = alpha
            clf = MLPClassifier(random_state=42, **params)
            clf.fit(x_train, y_train)

            # Predictions and probabilities
            y_pred_train = clf.predict(x_train)
            y_pred_test = clf.predict(x_test)
            proba_train = clf.predict_proba(x_train)
            proba_test = clf.predict_proba(x_test)

            # Use above preditions and proababilities to derive Performance metrics
            metrics['train_accs'][i] = accuracy_score(y_train, y_pred_train)
            metrics['test_accs'][i] = accuracy_score(y_test, y_pred_test)
            metrics['acc_gaps'][i] = metrics['train_accs'][i] - metrics['test_accs'][i] # Computer overtraining from subtracting training accuracy from validation accuracy
            metrics['train_losses'][i] = log_loss(y_train, proba_train)
            metrics['test_losses'][i] = log_loss(y_test, proba_test)
            # Use zero division to prevent division by zero issues by returning 0 for those cases
            metrics['test_precisions'][i] = precision_score(y_test, y_pred_test, average='weighted', zero_division=0)
            metrics['test_recalls'][i] = recall_score(y_test, y_pred_test, average='weighted', zero_division=0)
            metrics['test_f1s'][i] = f1_score(y_test, y_pred_test, average='weighted', zero_division=0)

            # Parameter norms and distribution stats
            # hpstack concatenates all values into one long vector
            W = np.hstack([w.ravel() for w in
                           clf.coefs_])  # Flatten each weight within the coefs_ constant (which contains all the weights)
            b = np.hstack(clf.intercepts_)  # clf.intercepts_ = constant for list of biases, no need to flatten
            metrics['all_weights'][i] = W
            metrics['all_biases'][i] = b
            # Use numpy methods to calculate the norms and stds of weights and biases
            metrics['weight_norms'][i] = np.linalg.norm(W)
            metrics['weight_means'][i] = np.mean(W)
            metrics['weight_stds'][i] = np.std(W)
            metrics['bias_norms'][i] = np.linalg.norm(b)
            metrics['bias_means'][i] = np.mean(b)
            metrics['bias_stds'][i] = np.std(b)

        return metrics

    def q5(self):
        """
        This function should perform hypothesis testing to study the impact of using CV with and without Stratification
        on the performance of MLPClassifier. Set other model hyperparameters to the best values obtained in the previous
        questions. Use 5-fold cross validation for this question. You can assume that the function 'q1' has been called
        prior to calling this function.

        :return: The function should return 4 items - the final testing accuracy for both methods of CV, p-value of the
        test and a string representing the result of hypothesis testing. The string can have only two possible values -
        'Splitting method impacted performance' and 'Splitting method had no effect'.
        """
        # First, we standardize
        x_scaled, _ = self.q2(self.x)

        # Prepare classifier with the optimal hyperparameters
        clf = MLPClassifier(**optimal_hyperparam, random_state=42)

        # Cross-validation scores
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        strat_scores = cross_val_score(clf, x_scaled, self.y, cv=skf, scoring='accuracy', n_jobs=-1)
        unstrat_scores = cross_val_score(clf, x_scaled, self.y, cv=kf, scoring='accuracy', n_jobs=-1)

        # Calculate mean accuracies
        res1 = np.mean(strat_scores)
        res2 = np.mean(unstrat_scores)

        # Paired t-test
        _, p_value = ttest_rel(strat_scores, unstrat_scores)  # Get p-value only

        res3 = p_value

        # Hypothesis testing result based on p-value
        if res3 < 0.05: # If theres less than 5% chance that the difference occurred by random chance, conclude as positive
            res4 = 'Splitting method impacted performance'
        else: # If its more than 5% chance it occurred by randomness, conclude as negative
            res4 = 'Splitting method had no effect'

        return res1, res2, res3, res4

    def q6(self):
        """
        This function should perform unsupervised learning using LocallyLinearEmbedding in Sklearn. You can assume that
        the function 'q1' has been called prior to calling this function.

        :return: The function should return the data you visualize.
        """
        x_scaled, _ = self.q2(self.x)

        # Perform Locally Linear Embedding (LLE) to reduce to 2D space
        lle = LocallyLinearEmbedding(
            n_components=2,
            n_neighbors=100,  # 100 Neighbours was concluded as the most valuable data
            method='standard', # Standard seemed to produce more spread out results compared to modified
            eigen_solver='auto',
            random_state=42
        )

        x_lle = lle.fit_transform(x_scaled)

        return {
            'x_lle': x_lle,
            'y': self.y
        }

    @staticmethod
    def plot_single(y, x=None, label="Metric", color='blue',
                    title="Line Graph", xlabel="X", ylabel="Y",
                    xscale='linear', yscale="linear"):
        """
        Generic single-metric line plot function

        :param x: List or array of x-values
        :param y: List or array of y-values
        :param label: Label for the line (used in the legend)
        :param color: Line color
        :param title: Plot title
        :param xlabel: Label for the x-axis
        :param ylabel: Label for the y-axis
        :param x_scale: Specifies the scale for the plot (e.g. log or linear)
        """
        if x is None:
            x = np.arange(len(y))

        plt.figure(figsize=(8, 5))
        plt.plot(x, y, label=label, color=color, marker='o')
        plt.xscale(xscale)
        plt.yscale(yscale)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_comparison(y1, y2, x=None, label1="Metric 1", label2="Metric 2",
                        color1='blue', color2='red',
                        title="Line Comparison of Two Metrics",
                        xlabel="X", ylabel="Y", xscale='linear', yscale="linear"):
        """
        Generic line plot for comparing two metrics

        :param y1: First y-series.
        :param y2: Second y-series.
        :param x: Shared x-axis values (must be provided).
        :param label1: Label for the first line.
        :param label2: Label for the second line.
        :param color1: Line color for the first line.
        :param color2: Line color for the second line.
        :param title: Plot title.
        :param xlabel: X-axis label.
        :param ylabel: Y-axis label.
        :param x_scale: Specifies the scale for the plot (e.g. log or linear)
        """
        # Get the maximum length of the two ys to ensure to covers the entire space of values
        if x is None:
            max_len = max(len(y1), len(y2))
            x = np.arange(max_len)
        plt.figure(figsize=(10, 6))
        plt.plot(x, y1, label=label1, color=color1, linestyle='-', marker='o')
        plt.plot(x, y2, label=label2, color=color2, linestyle='--', marker='x')
        plt.xscale(xscale)
        plt.yscale(yscale)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(loc='best')
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_boxplot_comparison(x1, x2, label1="Label 1", label2="Label 2",
                                title="Boxplot Comparison", ylabel="Value"):
        """
        Create a side-by-side boxplot comparing two datasets.

        :param x1: First dataset (1D array-like).
        :param x2: Second dataset (1D array-like).
        :param label1: Label for the first dataset.
        :param label2: Label for the second dataset.
        :param title: Plot title.
        :param ylabel: Label for the y-axis.
        """
        plt.figure(figsize=(8, 5))
        plt.boxplot([x1, x2], labels=[label1, label2])
        plt.title(title)
        plt.ylabel(ylabel)
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.show()

    @staticmethod
    def plot_boxplot(data, labels=None,
                     title="Boxplot", ylabel="Value"):
        """
        Generic boxplot for visualizing collections of data

        :param data: List of 1D array-like values
        :param labels: List of labels for each set of values (optional).
        :param title: Title of the plot.
        :param ylabel: Label for the y-axis.
        """
        plt.figure(figsize=(8, 5))
        plt.boxplot(data, labels=labels)
        plt.title(title)
        plt.ylabel(ylabel)
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.show()

    @staticmethod
    def plot_embedding(x, y, class_names,
                       title="Cluster Visualization",
                       xlabel="Component 1", ylabel="Component 2",
                       marker='o', alpha=0.6, cmap='tab10', s=8):
        """
        Plot 2D data points grouped by class.

        :param x: 2D array of embedded coordinates (e.g. LLE output).
        :param class_names: Each unique class name in the dataset, array.
        :param title: Title of the plot.
        :param xlabel: Label for the x-axis.
        :param ylabel: Label for the y-axis.
        :param marker: Marker style for the scatter plot.
        :param alpha: Transparency of the points.
        :param cmap: Colormap for different classes.
        :param s: Marker size.
        """
        # Plot entire scatter
        plt.figure(figsize=(8, 8))
        sc = plt.scatter(
            x[:, 0], x[:, 1],
            c=y,
            cmap=cmap,
            marker=marker,
            alpha=alpha,
            s=s,
            edgecolors='w'
        )

        # Colorbar with class names
        cbar = plt.colorbar(sc, ticks=range(len(class_names)))
        cbar.ax.set_yticklabels(class_names, fontsize='small')
        cbar.set_label('Class', rotation=270, labelpad=15)

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_scatter(x, label=None, color='blue', marker='o',
                     title="Scatter Plot", xlabel="X", ylabel="Y"):
        """
        Scatter-plot a single dataset on 2D axes.

        :param x: 2D array-like values to plot.
        :param label: Optional label.
        :param color: Marker color.
        :param marker: Marker style.
        :param title: Plot title.
        :param xlabel: Label for the x-axis.
        :param ylabel: Label for the y-axis.
        """
        plt.figure(figsize=(8, 5))
        plt.scatter(x[:, 0], x[:, 1], label=label, color=color, marker=marker, alpha=0.6)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if label:
            plt.legend(loc='best')
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_scatter_comparison(x1, x2, label1=None, label2=None,
                                color1='blue', color2='red',
                                marker1='o', marker2='x',
                                title="Scatter Plot Comparison",
                                xlabel="X", ylabel="Y"):
        """
        Generic scatter plot comparing two sets of values on the same axes.

        :param x1: First set
        :param x2: Second set
        :param label1: Optional legend label for the first dataset.
        :param label2: Optional legend label for the second dataset.
        :param color1: Marker color for the first dataset.
        :param color2: Marker color for the second dataset.
        :param marker1: Marker style for the first dataset.
        :param marker2: Marker style for the second dataset.
        :param title: Plot title.
        :param xlabel: Label for the x-axis.
        :param ylabel: Label for the y-axis.
        """
        plt.figure(figsize=(8, 5))

        # Plot the first dataset with its specific color and marker
        plt.scatter(x1[:, 0], x1[:, 1], label=label1, color=color1, marker=marker1, alpha=0.6)

        # Plot the second dataset with its specific color and marker
        plt.scatter(x2[:, 0], x2[:, 1], label=label2, color=color2, marker=marker2, alpha=0.6)

        # Adding title and labels
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        # Adding legend if labels are provided
        if label1 or label2:
            plt.legend(loc='best')

        # Show grid and plot
        plt.grid(True)
        plt.show()

    @staticmethod
    def normalize(inp):
        """
        Normalize the data using MinMaxScaler to the range [0, 1].

        :param inp: A numpy array with the data to be normalized.
        :return: The normalized data and the scaler used for transformation.
        """
        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(inp)

        return normalized_data, scaler

    @staticmethod
    def plot_bar(categories, values,
                 title="Bar Plot",
                 xlabel="Category",
                 ylabel="Value"):
        """
        Generic simple bar chart for comparing a set of values.

        :param categories: List of category labels (x-axis).
        :param values: List of values corresponding to each category (y-axis).
        :param title: Title of the plot.
        :param xlabel: Label for the x-axis.
        :param ylabel: Label for the y-axis.
        """
        plt.figure(figsize=(8, 5))

        colors = plt.cm.get_cmap('tab10', len(categories))  # Using a predefined colormap

        bars = plt.bar(categories, values, edgecolor='black', color=colors(range(len(categories))))

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(bars, categories, loc="best")
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.show()
