import matplotlib.pyplot as plt
import numpy as np

# Data in hours for framework times
datasets = ["Breast Histopathology", "Brain Tumor MRI", "Soyleaf", "CIFAR10"]
framework_times = [12.6, 51, 81.5, 58.9]  # hours

# Exhaustive times in minutes and converted to hours
exhaustive_times_minutes = [(58**2) * 3.3, (58**4) * 4.9, (58**9) * 15.3, (58**10) * 5.6]
exhaustive_times_hours = [t / 60 for t in exhaustive_times_minutes]

# Log-transformed data for visual clarity
framework_times_log = [np.log10(time) for time in framework_times]
exhaustive_times_log = [np.log10(time) for time in exhaustive_times_hours]
exhaustive_times_labels = ["10^1.75", " 10^2.97", "10^9.70", "10^10.20"]  # readable labels

# Bar plot
font_size = 12
plt.figure(figsize=(10, 6))
font = {'size': font_size}
plt.rc('font', **font)
x = np.arange(len(datasets))
width = 0.35

plt.bar(x - width/2, framework_times_log, width, label="Framework (log hours)", color='skyblue')
plt.bar(x + width/2, exhaustive_times_log, width, label="Exhaustive (log hours)", color='salmon')

# Adding annotations for actual times in scientific notation
for i in range(len(datasets)):
    plt.text(x[i] - width/2, framework_times_log[i] + 0.1, f'{framework_times[i]} h', ha='center', color='blue')
    plt.text(x[i] + width/2, exhaustive_times_log[i] + 0.1, f'{exhaustive_times_labels[i]} h', ha='center', color='red')

plt.ylabel('Log(Time in Hours)')

plt.xticks(x, datasets)  # Set dataset names on x-axis
plt.legend()
plt.show()