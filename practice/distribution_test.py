L = 1
N_1axis = 3

list_direct = []
for i in range(3):
    for j in range(3):
        for k in range(3):
            print(i,j,k)
            list_direct.append((i,j,k))

list_new = []
for n in range(N_1axis**3):
    i = n//(N_1axis*N_1axis)
    j = (n//N_1axis)%(N_1axis)
    k = n % (N_1axis)
    list_new.append((i,j,k))
for direct, attempt in zip(list_direct, list_new):
    print(direct, attempt, direct==attempt)
