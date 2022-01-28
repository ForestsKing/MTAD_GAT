from exp.exp import Exp
from utils.setseed import set_seed

if __name__ == '__main__':
    set_seed(0)
    group = '1-1'

    iters = 1
    epochs = 50
    batch_size = 32
    patience = 3
    lr = 0.001
    generate = False

    for it in range(iters):
        print("iter " + str(it) + ' is start...')
        exp = Exp(group, it, epochs, batch_size, patience, lr, generate)
        exp.fit()
        exp.predict(load=False)
        print("iter " + str(it) + ' is end!')
