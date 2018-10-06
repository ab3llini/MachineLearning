from matplotlib import pyplot as plt


def plot(x, y, title='Graph', legend='y', color='blue', xlabel='x', ylabel='y'):
    plt.plot(x, y, 'o-', label=legend, color=color)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.grid(True, linestyle=':')
    plt.legend(loc="upper right")
    return plt