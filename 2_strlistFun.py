def convertTostr(list):
    str = ''
    for i in range(len(list)):
        if i < (len(list) - 1):
            str += list[i]
            str +=","
        else:
            str += "and "
            str += list[i]
    return str

print(convertTostr(['apples', 'bananas', 'tofu', 'cats']))
