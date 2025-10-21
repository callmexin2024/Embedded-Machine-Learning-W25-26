
# Imports
import numpy as np
import matplotlib.pyplot as plt
import torch

# Convinience functions
def plot_model(model=None):
    # Visualize data
    plt.plot(torch.linspace(0, 1, 1000), ground_truth_function(torch.linspace(0, 1, 1000)), label='Ground truth')
    plt.plot(x_train, y_train, 'ob', label='Train data')
    plt.plot(x_test, y_test, 'xr', label='Test data')
    # Visualize model
    if model is not None:
        plt.plot(torch.linspace(0, 1, 1000), model(torch.linspace(0, 1, 1000)), label=f'Model of degree: {model.degree()}')

    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()

# Generate data
n_samples = 11
noise_amplitude = 0.15

def ground_truth_function(x):
    # Generate data of the form sin(2 * Pi * x)
    result = torch.sin(2 * np.pi * x)
    return result

torch.manual_seed(42)

x_test = torch.linspace(0, 1, n_samples)
y_test = ground_truth_function(x_test) + torch.normal(0., noise_amplitude, size=(n_samples,))
x_train = torch.linspace(0, 1, n_samples)
y_train = ground_truth_function(x_train) + torch.normal(0., noise_amplitude, size=(n_samples,))

# Test plotting
plot_model()
plt.savefig('Initial_data.png')
plt.show()
plt.clf()


# Model fitting

def error_function(model, x_data, y_data):
    y_pred = model(x_data)
    error = torch.sum((y_pred - y_data) ** 2)
    return error

model_degree = 3

model = np.polynomial.Polynomial.fit(x_train, y_train, deg=model_degree)
train_err = error_function(model, x_train, y_train)
test_err = error_function(model, x_test, y_test)

print(f"{train_err=}, {test_err=}")

# Result plotting
plot_model(model)
plt.savefig('Initial_fit.png')
plt.show()
plt.clf()

# ---- Continue with the exercises on the degree of the polynomial and the exploration of data size

overfit_degree = 11

overfit_model = np.polynomial.Polynomial.fit(x_train, y_train, deg=overfit_degree)

plot_model(overfit_model)
plt.title("Overfitted Polynomial of 11-th degree")
plt.savefig("Overfitted_fit.png")
plt.show()
plt.clf()

# Polynomial degree against the train and test error

def rms_error(model, x_data, y_data):
    y_pred = model(x_data)
    return torch.sqrt(torch.mean((y_pred - y_data) ** 2))

max_degree = 11
train_errors = []
test_errors = []

for degree in range(max_degree + 1):
    model = np.polynomial.Polynomial.fit(x_train, y_train, deg=degree)
    train_errors.append(rms_error(model, x_train, y_train))
    test_errors.append(rms_error(model, x_test, y_test))

plt.plot(range(max_degree + 1), train_errors, 'o-', label='Train RMS Error')
plt.plot(range(max_degree + 1), test_errors, 's-', label='Test RMS Error')
plt.xlabel("Polynomial Degree")
plt.ylabel("RMS Error")
plt.title("Polynomial Degree vs RMS Error")
plt.xticks(range(max_degree + 1))
plt.legend()
plt.grid(True)
plt.savefig("Degree_vs_RMS_Error.png")
plt.show()

# Sample size against RMS error

fixed_degree = 10
max_samples = 500

sample_sizes = []
train_errs = []
test_errs = []

for n in range(10, max_samples + 1, 20):
    x_train = torch.linspace(0, 1, n)
    y_train = ground_truth_function(x_train) + torch.normal(0., noise_amplitude, size=(n,))
    x_test = torch.linspace(0, 1, n)
    y_test = ground_truth_function(x_test) + torch.normal(0., noise_amplitude, size=(n,))
    
    model = np.polynomial.Polynomial.fit(x_train, y_train, deg=fixed_degree)
    
    train_err = rms_error(model, x_train, y_train).item()
    test_err = rms_error(model, x_test, y_test).item()
    
    sample_sizes.append(n)
    train_errs.append(train_err)
    test_errs.append(test_err)


plt.plot(sample_sizes, train_errs, 'o-', label="Train RMS Error")
plt.plot(sample_sizes, test_errs, 's-', label="Test RMS Error")
plt.xlabel("Sample Size Number")
plt.ylabel("RMS Error")
plt.title("Sample Size vs RMS Error")
plt.legend()
plt.grid(True)
plt.savefig("Sample_Size_vs_RMS_Error.png")
plt.show()
plt.clf()