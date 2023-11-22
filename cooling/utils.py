def range_wend(a, b, step=1):
    return range(int(a), int(b+1), step)

def tento_label(expns):
    return [r'$10^{'+str(expn)+r'}$' for expn in expns]