import matplotlib.pyplot as plt
import numpy as np

# Random test data
np.random.seed(19680801)
all_data = [-0.21279835062909633,-0.2094337732898827,-0.20844608211989057,
            -0.2067279437485256,-0.1866434361028442,-0.19910094905719808,-0.21742335498209245,
            -0.20029123395331666,-0.18762245388396814,-0.24127858075686276,-0.2027763344162455,
            -0.2002740262609603,-0.19835059990132634,-0.2190368046223404,-0.21054816398459747,
           -0.1781737026386157,-0.2236575390748718,-0.20420362374930243,
          -0.2007442057978, -0.2047966712400206,-0.2285345469502119]
labels = [' ']

fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))

# rectangular box plot
bplot1 = ax1.boxplot(all_data,
                     vert=True,  # vertical box alignment
                     patch_artist=True)  # will be used to label x-ticks
ax1.set_title('Correlation for All subjects')

# fill with colors
colors = ['pink', 'lightblue', 'lightgreen']
# for bplot in (bplot1):
#     for patch, color in zip(bplot['boxes'], colors):
#         patch.set_facecolor(color)

# adding horizontal grid lines
for ax in [ax1]:
    ax.yaxis.grid(True)
    ax.set_ylabel('Corr')

plt.show()