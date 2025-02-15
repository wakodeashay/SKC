import matplotlib.pyplot as plt
import numpy as np
import csv

# Initialize lists for data
actual_sparsity = []
blocked_fraction = []
hilbert_len = []
bastar_len = []
unblocked_cell = []

actual_sparsity_ls = []
blocked_fraction_ls = []
hilbert_len_ls = []
bastar_len_ls = []
unblocked_cell_ls = []

# Read data from CSV
with open('experiment2.csv', mode='r') as file:
    csvFile = csv.reader(file)
    prev_iter = 2
    for line in csvFile:
        if int(float(line[7])) > 4 and float(line[4]) != 0.0 and float(line[4]) != 1.0:
            if int(float(line[0])) != prev_iter:
                prev_iter = int(float(line[0]))
                actual_sparsity.append(actual_sparsity_ls)
                blocked_fraction.append(blocked_fraction_ls)
                hilbert_len.append(hilbert_len_ls)
                bastar_len.append(bastar_len_ls)
                unblocked_cell.append(unblocked_cell_ls)

                actual_sparsity_ls = []
                blocked_fraction_ls = []
                hilbert_len_ls = []
                bastar_len_ls = []
                unblocked_cell_ls = []

            actual_sparsity_ls.append(float(line[4]))
            blocked_fraction_ls.append(float(line[3]))
            hilbert_len_ls.append(float(line[5]))
            bastar_len_ls.append(float(line[6]))
            unblocked_cell_ls.append(float(line[7]))

# Scatter plots
hilbert_metric = [i / j for i, j in zip(hilbert_len[1], unblocked_cell[1])]
bastar_metric = [i / j for i, j in zip(bastar_len[1], unblocked_cell[1])]

# hilbert_metric = hilbert_len[1]
# bastar_metric = bastar_len[1]

print(actual_sparsity[1])

plt.scatter(actual_sparsity[1], hilbert_metric, color='blue', label='Hilbert')
plt.scatter(actual_sparsity[1], bastar_metric, color='green', label='BA*Star')

# Fit a quadratic polynomial to Hilbert data
coeffs_hilbert = np.polyfit(actual_sparsity[1], hilbert_metric, 2)
poly_hilbert = np.poly1d(coeffs_hilbert)
x_vals = np.linspace(min(actual_sparsity[1]), max(actual_sparsity[1]), 100)
y_vals_hilbert = poly_hilbert(x_vals)

# Fit a quadratic polynomial to BA*Star data
coeffs_bastar = np.polyfit(actual_sparsity[1], bastar_metric, 2)
poly_bastar = np.poly1d(coeffs_bastar)
y_vals_bastar = poly_bastar(x_vals)

# Plot best-fit polynomials
plt.plot(x_vals, y_vals_hilbert, color='blue', linestyle='--', label='Hilbert Fit')
plt.plot(x_vals, y_vals_bastar, color='green', linestyle='--', label='BA*Star Fit')

# Add labels and legend
plt.title('Scatter Plot with Quadratic Fit')
plt.xlabel('Actual Sparsity')
plt.ylabel('Metric (Length / Unblocked Cells)')
plt.legend()

# Show plot
plt.show()
