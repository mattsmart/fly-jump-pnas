import numpy as np

"""
Currently un-used; consider removing. 

Could be integrated with plot_metrics_by_fly_id.py 
"""

# utility functions used within reduce_binary_timeseries(...)
def find_longest(l, target=0):
    """
    Given a list or 1d array, l, and a target element, target,
    Return the length of the longest sequence of consecutive element == target
    """
    counter = 0
    counters = []
    for elem in l:
        if elem != target:
            counters.append(counter)
            counter = 0
        else:
            counter += 1
    counters.append(counter)
    return max(counters)


def reduce_binary_timeseries(x, method='sum', mode='one_expt'):
    """
    x: 1-D arr of shape T
        A sequence of binary outcomes -- jump (1) or No jump (0) -- for one fly

    mode in ['one_expt', 'six_expt']
        'one_expt': x is assumed to be an array of length 200 or 50
        'six_expt': x is assumed to be an array of length 1050; a scalar is returned for each experimental window
               [0:200], [200:400], [400:600], [600:800], [800:1000] [1000:1050]
            How: this function is called recursively on each subset
                reduce_binary_timeseries(x_subset, method=method, mode='one_expt')

         mode='one_expt': One habituation or reactivity experiment
            a scalar is returned
         mode='six_expt': 5x habituation + 1x reactivity experiments, 6 in total
            an array of length 6 is returned

    methods
        'sum'      - return the sum of all jumps
        'ttc5'     - return the experimentalists' "time to criterion" of 5 consecutive non jumps
                      note: if criterion not met, or if the sequence is shorter than 5, returns the length of the seq
        'sum_diff' - return "jumps_A - jumps_B" where A and B are the first and second halves of the sequence
        'longest_1s'   - return the maximum number of consecutive jumps
        'longest_0s'   - return the maximum number of consecutive non-jumps

    """
    assert mode in ['one_expt', 'six_expt']
    assert method in ['sum', 'ttc5', 'sum_diff', 'longest_1s', 'longest_0s']  # TODO add more options...
    assert len(x.shape) == 1  # could generalize to 2D --  T x num_flies  -- later if needed

    x = x.astype(int)  # don't want float
    num_trials = x.shape[0]

    if mode == 'six_expt':
        assert num_trials == 1050
        return np.array([
            reduce_binary_timeseries(x_subset, method=method, mode='one_expt') for x_subset in [
                x[0:200],     # expt: habituation 1 of 5
                x[200:400],   # expt: habituation 2 of 5
                x[400:600],   # expt: habituation 3 of 5
                x[600:800],   # expt: habituation 4 of 5
                x[800:1000],  # expt: habituation 5 of 5
                x[1000:]      # expt: reactivity
            ]
        ])

    else:
        assert mode == 'one_expt'
        assert num_trials in [200, 50]

        if method == 'sum':
            val = x.sum()
        elif method == 'ttc5':
            x_as_str = str(x)[1:-1].replace(' ', '')  # e.g. converts [1, 0, 1] -> '101'
            num_non_jump = 5  # 'look for 5 consecutive non-jumps'
            target_non_jump_sequence = '0' * num_non_jump
            target_loc = str(x_as_str).find(target_non_jump_sequence)
            if target_loc == -1:
                val = num_trials
            else:
                val = target_loc

        elif method == 'sum_diff':
            half_trials = num_trials // 2
            val = x[0:half_trials].sum() - x[half_trials:].sum()

        elif method == 'longest_1s':
            val = find_longest(x, target=1)

        elif method == 'longest_0s':
            val = find_longest(x, target=0)

        else:
            val = None
            raise ValueError(f"Unknown method: {method}")

        return val


if __name__ == '__main__':
    x_example = np.ones(1050, dtype=int)
    x_example[10:15] = 0
    x_example[40:65] = 0
    print('First 20 elements of example jump timeseries:\n', x_example[0:20])

    for method in ['sum', 'ttc5', 'sum_diff', 'longest_1s', 'longest_0s']:
        val_one_expt = reduce_binary_timeseries(x_example[0:200], method=method, mode='one_expt')
        val_six_expt = reduce_binary_timeseries(x_example, method=method, mode='six_expt')
        print('\nmethod: %s' % method)
        print('\tval_one_expt (first 200 trials):', val_one_expt)
        print('\tval_six_expt:', val_six_expt)

