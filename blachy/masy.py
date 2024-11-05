masy_8AL = [68, 68, 57, 56, 254,118,119]
masy_8ST = [233, 232, 497, 111, 111, 133, 133]
masy_1ST = [41,418,343,235,481,513,359,454,261,256,398,97,233,159,91,257,118,149,163,82,174,97,261,111]
masy_2ST = [304,204,258,184,54,149,144,232,118,69,155,641,143,102,119,31,137,61,162,162,163,649]
masy_3AL = [35,35,14,35,14,35,369,19,9,9,9,9,19,9,10,9,9,9,19,19,9,9,9,9,9]
masy_6ST = [762,1084,1070,337]
masy_7ST = [2028,165,139,552,530,768]
masy_5ST = [585,704,167,282,1459]
masy_4AL = [118,118,40,40,28,43,28,22,22,23,23,25,25,40,43,22,18,10,10,40,22,18]


import matplotlib.pyplot as plt

# Combine all masses into one list
all_masses = masy_8AL + masy_8ST + masy_1ST + masy_2ST + masy_3AL + masy_6ST + masy_7ST + masy_5ST + masy_4AL


# Calculate the number of elements in all_masses
num_elements = len(all_masses)
print(f"Liczba elementów w all_masses: {num_elements}")
# Display the name and number of elements in each vector
vectors = {
    "masy_8AL": masy_8AL,
    "masy_8ST": masy_8ST,
    "masy_1ST": masy_1ST,
    "masy_2ST": masy_2ST,
    "masy_3AL": masy_3AL,
    "masy_6ST": masy_6ST,
    "masy_7ST": masy_7ST,
    "masy_5ST": masy_5ST,
    "masy_4AL": masy_4AL
}

for name, vector in vectors.items():
    print(f"Liczba elementów w {name}: {len(vector)}")


# Define the bins
bins = list(range(0, max(all_masses) + 50, 50))

# Create the histogram
plt.hist(all_masses, bins=bins, edgecolor='black', color='gray')


plt.grid(axis='y', alpha=0.75)

# Add titles and labels
plt.xlabel('Masa [g]', font='times new roman', size=13)
plt.ylabel('Ilość detali [szt]', font='times new roman', size=13)
plt.xticks(fontname='Times New Roman', size=12, rotation=25, ticks=range(0, max(all_masses) + 200, 200))
plt.yticks(fontname='Times New Roman', size=12)

# Show the plot
plt.show()