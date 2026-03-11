import numpy as np
import matplotlib.pyplot as plt

"""
Implements scalar functions f(x) which satisfy the following properties:

1) f(x) is non-increasing for x > 0
2) f(0) = 1
3) f(x -> inf) = 0
"""

def link_fn_hill(x, N=2):
    return 1 / (1 + x ** N)

def link_fn_exp(x, gamma=1):
    return np.exp(-gamma * x)


if __name__ == '__main__':
    x = np.linspace(0, 10, 100)

    plt.plot(x, link_fn_hill(x, N=2),
             '-', color='green',
             label='link_fn_hill(x, N=2)')
    plt.plot(x, link_fn_hill(x, N=0.5),
             '--', color='green',
             label='link_fn_hill(x, N=0.5)')

    plt.plot(x, link_fn_exp(x, gamma=1),
             '-', color='blue',
             label='link_fn_exp(x, gamma=1)')
    plt.plot(x, link_fn_exp(x, gamma=0.5),
             '--', color='blue',
             label='link_fn_exp(x, gamma=0.5)')

    plt.legend()
    plt.title('Examples of scalar link functions')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$f(x)$')
    plt.grid(alpha=0.5)
    plt.show()
