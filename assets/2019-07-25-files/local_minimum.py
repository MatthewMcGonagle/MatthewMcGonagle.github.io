'''
Test set consists of two squares that are vertically aligned but horizontally far apart. Start with
sub-optimal way of cycling throught the points using only vertical and horizontal moves.
'''

import numpy as np
import matplotlib.pyplot as plt
import my_src

# Make the local minimum cycle and the shorter cycle.
local_min, shorter = my_src.make_local_minimum(heights = [1, 2.5], separation = 6, return_shorter = True)
lengths = {'local_min' : my_src.find_length(local_min),
           'shorter' : my_src.find_length(shorter)}
print(lengths)

# Plot the local minimum cycle.
my_src.plot_cycle(local_min)
plt.title('Local Minimum, Length = ' + str(lengths['local_min']))
plt.tight_layout()
plt.savefig('local_min.svg')
plt.close()

# Plot the shorter cycle.
my_src.plot_cycle(shorter)
plt.title('Shorter Cycle, Length = ' +  '{:.2f}'.format(lengths['shorter']))
plt.tight_layout()
plt.savefig('shorter_cycle.svg')
plt.close()

# Get all possible local moves from the local minimum cycle and find their lengths.
# Note, that any flip is always equivalent to flipping the order for the segment
# of vertices contained in the array.

local_lengths = []
for i in range(0, len(local_min) - 1):
    for j in range(i + 1, len(local_min)):
        new_cycle = my_src.flip_segment_order(local_min, i, j)
        new_length = my_src.find_length(new_cycle)
        local_lengths.append(new_length)

        # Get a plot of what one of the moves looks like.
        if i == 0 and j == 4:
            my_src.plot_cycle(new_cycle)
            plt.title('Random Example Move for i = ' + str(i) + ', j = ' + str(j))
            plt.tight_layout()
            plt.savefig('local_move.svg')
            plt.close()

        # Make a plot if the length of the cycle is the same as the local min cycle.
        if new_length == lengths['local_min']:
            print('Move for i = ', i, 'j = ', j, 'has the same length as the local cycle.')
            my_src.plot_cycle(new_cycle, mark_begin_end = True)
            plt.title('Move for i = ' + str(i) + ' j = ' + str(j) + ', Length is Same as Local Min') 
            plt.tight_layout()
            plt.savefig('same_' + str(i) + '_' + str(j) + '.svg')
            plt.close()

        # If we see a cycle that is smaller, make a plot of the cycle and raise an exception. This
        # shouldn't ever happen. 
        if new_length < lengths['local_min']:
            my_src.plot_cycle(new_cycle)
            plt.title('Move for i = ' + str(i) + ' j = ' + str(j) + 'has Smaller Length = ' + 
                      '{:.2f}'.format(my_src.find_length(new_cycle)))
            plt.savefig('error.svg')
            raise Exception('Found a local move with SHORTER length! i = ' + str(i) + ', j = ' + str(j)) 

# Look at the differences in the lengths for the cycles local to the local minimum cycle.
diff_lengths = [l - lengths['local_min'] for l in local_lengths]
print('Minimal difference of lengths is ', min(diff_lengths))

# Plot the local lengths.
plt.scatter(local_lengths, np.zeros(len(local_lengths)))
plt.axvline(lengths['local_min'], color = 'red')
plt.title('Lengths of the Local Cycles')
plt.xlabel('Lengths')
plt.yticks([]) # Hide the y-axis.
plt.legend(['Local Min', 'Locals'])
plt.tight_layout()
plt.savefig('local_lengths.svg')
