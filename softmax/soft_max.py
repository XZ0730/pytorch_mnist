import numpy as np
def softmax(x):
    c = np.max(x)
    exp_a = np.exp(x - c)    # 溢出对策
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    
    return y

temp = np.array([[
    [1,2,3],[4,5,6]],[
    [1,2,3],[4,5,6] 
    ]])
res = softmax(temp)
print(res)