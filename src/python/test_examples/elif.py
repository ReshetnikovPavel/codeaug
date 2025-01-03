if True:
    print('a')
elif True:
    print('b')
elif False:
    print('d')
else:
    print('c')


if not True:
    if True:
        print('b')
    else:
        print('c')
else:
    print('a')
