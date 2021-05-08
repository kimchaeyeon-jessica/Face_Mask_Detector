a={'A':[4,5,6], 'B':[1,2,3], 'C':[7,8]}

print(a.items())

for key, value in a.items():
    print(key, value)
    print('===')
    for v in value:
        print(v)
    print('===')