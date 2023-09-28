def convertTostr(list):
    st = ''
    for i in range(len(list)):
        if i < (len(list) - 1):
            st += list[i]
            st += ","
        else:
            st += "and "
            st += list[i]
    return st


print(convertTostr(['apples', 'bananas', 'tofu', 'cats']))
