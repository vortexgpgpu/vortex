file = open("kernelCode.txt", 'r')
dictionary={}
for i in file:
    temp = list(i)
    inst=''
    ind=0
    for j in range(len(temp)-1):
        if temp[j] =='\t':
            ind=j+1
            while True:
                inst+=temp[ind]
                ind+=1
                if ind!=len(temp) and temp[ind] == '\t':
                    break
                elif ind==len(temp):
                    break
            if inst in dictionary:
                dictionary[inst]+=1
            else:
                dictionary[inst]=1
        if ind != len(temp) and temp[ind] == '\t':
            break
        elif ind == len(temp):
            break
#print(dictionary)
#del dictionary['<unknown>\n']
#del dictionary['\t...\n']
#del dictionary['<unknown>']
dictionary = dict(sorted(dictionary.items(), key=lambda item: item[1], reverse=True))

for i in dictionary:
    print(i, "\t", dictionary[i])
