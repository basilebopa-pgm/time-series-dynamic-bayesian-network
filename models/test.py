import pandas as pd


from models.ground_truth_model import create_ground_truth

ground_truth = create_ground_truth()


data = ground_truth.simulate(
    n_samples=1,
    n_time_slices=500,
    seed=41,
    return_format="long"
)


import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.plot(data['time'], data['Price'], linestyle='-', color='blue', label='Price')

# Formatting the plot
plt.title('Simulated Price Over Time')
plt.xlabel('Time Step')
plt.ylabel('Price Value')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()

# Show the plot
plt.show()