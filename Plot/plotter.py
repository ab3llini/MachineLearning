from matplotlib import pyplot as plt


def plot(x, y, title='Graph', legend='y', color='blue', xlabel='x', ylabel='y', fname='graph'):
    fig = plt.figure()
    plt.plot(x, y, 'o-', label=legend, color=color)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.grid(True, linestyle=':')
    plt.legend(loc="upper right")
    fig.savefig(fname + '.png', dpi=300)
    return plt