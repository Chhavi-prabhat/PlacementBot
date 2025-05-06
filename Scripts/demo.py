def find_k_ele(nums, k):
    n = []
    numlist = list(map(int, nums)) 
    if len(numlist)==1:
        n=numlist
        # print(n)
    elif (len(numlist)>1):
        for i in range(0,len(numlist)):
            for j in range(i+1, len(numlist)):
                if numlist[i] == numlist[j]:
                    n.append(numlist[j])
        print(n)
    set_n=list(dict.fromkeys(n))
    return set_n

print(find_k_ele([1],2))