import numpy as np
import matplotlib.pyplot as plt

def make_local_minimum(heights, separation, return_shorter = False):
    '''
    Local minimum cycle looks like the following:

                 ---------------------------------
    heights[1]   |                                |
                 |           separation           |
                 |     ----------------------     |
                 |     |                    |     |
                 |     |heights[0]          |     |
                 |_____|                    |_____|
                    1                          1

    A shorter cycle connects upper corners using diagonal segments and connects the bottom
    inside corners using a horizontal segment.

    Parameters
    ----------
    heights : Numpy array of heights of size 2
        The vertical heights as represented in the diagram.

    separation : Int or Float
        The horizontal separation as depicted in the diagram.

    return_shorter : Boolean
        Whether to return a shorter cycle too.

    Returns
    -------
    local_minimum : Numpy array of shape (n_vertices, 2)
        The vertices of the local minimum cycle in order.

    shorter_cycle : Numpy array of shape (n_vertices, 2)
        Returned ONLY when return_shorter is True. The vertices in the order of a shorter cycle.
    '''

    local_min = np.array([[0, heights[1]], [0, 0], [1, 0], [1, heights[0]],
                          [1 + separation, heights[0]], [1 + separation, 0],
                          [2 + separation, 0], [2 + separation, heights[1]]])
    if not return_shorter:
        return local_min 
    shorter = local_min[[0, 1, 2, 5, 6, 7, 4, 3]]
    return local_min, shorter 

def find_length(cycle):
    '''
    Find the length of the cycle. Make sure to include the distance between the first vertex and
    the last vertex.

    Parameters
    ----------
    cycle : Numpy array of shape (n_vertices, 2)
        The vertices of the cycle in the order of the cycle.

    Returns
    -------
    length : Float
        The length of the cycle.
    '''
    length = np.linalg.norm(cycle[1:] - cycle[:-1], axis = -1).sum()
    length += np.linalg.norm(cycle[0] - cycle[-1])
    return length

def plot_cycle(cycle, mark_begin_end = False):
    '''
    Plot the cycle.

    Parameters
    ----------
    cycle : Numpy array of shape (n_vertices, 2)
        The vertices of the cycle.

    mark_begin_end : Bool
        Whether to specially mark the segment connecting the first vertex to the last vertex.
    '''
    plt.plot(cycle.T[0], cycle.T[1], color = 'blue')
    if mark_begin_end:
        color = 'orange'
    else:
        color = 'blue'
    plt.plot(cycle.T[0][[0, -1]], cycle.T[1][[0, -1]], color = color)
    plt.scatter(cycle.T[0], cycle.T[1], color = 'red', zorder = 3)
    plt.xlabel('x')
    plt.ylabel('y')
    if mark_begin_end:
        plt.legend(['In Array', 'Begin to End of Array'])

def flip_segment_order(cycle, i, j):
    '''
    Flip the order of the cycle between two indices (inclusive).

    Parameters
    ----------
    cycle : Numpy array of shape (n_vertices, 2)
        The vertices of the cycle in the original order.

    i : Int
        Smaller end point. Should satisfy i < j.

    j : Int
        Larger end point. Should satisfy i < j.
    '''
    new_cycle = np.concatenate([cycle[:i],
                                np.flip(cycle[i:j+1], axis = 0),
                                cycle[j+1:]], axis = 0)
    return new_cycle

