masy_8AL = [68, 68, 57, 56, 254,118,119]
masy_8ST = [233, 232, 497, 111, 111, 133, 133]
masy_1ST = [40, 403, 228, 333, 469, 502, 352, 442, 155, 255, 389,251,95,228,89,251,116,145,116,79,159,170,253,94,107,253]
masy_2ST = [304,204,258,184,54,149,144,232,118,69,155,641,143,102,119,31,137,61,162,162,163,649]
masy_3AL = [35,35,14,35,14,35,369,19,9,9,9,9,19,9,10,9,9,9,19,19,9,9,9,9,9]
masy_6ST = [762,1084,1070,337]
masy_7ST = [2028,165,139,552,530,768]
masy_5ST = [585,704,167,282,1459]
masy_4AL = [118,118,40,40,28,43,28,22,22,23,23,25,25,40,43,22,18,10,10,40,22,18]


import matplotlib.pyplot as plt

# Combine all masses into one list
all_masses = masy_8AL + masy_8ST + masy_1ST + masy_2ST + masy_3AL + masy_6ST + masy_7ST + masy_5ST + masy_4AL

# Define the bins
bins = list(range(0, max(all_masses) + 50, 50))

# Create the histogram
plt.hist(all_masses, bins=bins, edgecolor='black', color='gray')

# Add titles and labels
plt.title('Distribution of Masses')
plt.xlabel('Mass Range')
plt.ylabel('Frequency')

# Show the plot
plt.show()