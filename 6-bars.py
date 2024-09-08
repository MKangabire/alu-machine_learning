#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

# Data for the plot
labels = ['Farrah', 'Fred', 'Felicia']
fruits = ['Apples', 'Bananas', 'Oranges', 'Peaches']
colors = {
    'Apples': 'red',
    'Bananas': 'yellow',
    'Oranges': '#ff8000',
    'Peaches': '#ffe5b4'
}

# Number of people and fruits
n_people = fruit.shape[1]
n_fruits = fruit.shape[0]

# Create a figure and axis
fig, ax = plt.subplots()

# Position of the bars on the x-axis
ind = np.arange(n_people)
width = 0.5  # Width of the bars

# Initialize the bottom of the bars
bottoms = np.zeros(n_people)

# Plot each type of fruit
for i, fruit_type in enumerate(fruits):
    ax.bar(ind, fruit[i, :], width, label=fruit_type, color=colors[fruit_type], bottom=bottoms)
    bottoms += fruit[i, :]  # Update the bottom for stacking

# Add labels, title, and legend
ax.set_xlabel('Person')
ax.set_ylabel('Quantity of Fruit')
ax.set_title('Number of Fruit per Person')
ax.set_xticks(ind)
ax.set_xticklabels(labels)
ax.set_ylim(0, 80)  # Set y-axis range
ax.set_yticks(np.arange(0, 81, 10))  # Set y-axis ticks
ax.legend()

# Show the plot
plt.show()
