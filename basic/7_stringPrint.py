tableData = [['apples', 'oranges', 'cherries', 'banana'],
             ['Alice', 'Bob', 'Carol', 'David'],
             ['dogs', 'cats', 'moose', 'goose']]

def printTable(data):
    length = [0,0,0]
    for x in range(len(data)):
        for y in range(len(data[0])):
            if length[x] < len(data[x][y]):
                length[x] = len(data[x][y])

    for y in range(len(data[0])):
        for x in range(len(data)):
            print(data[x][y].rjust(length[x]), end=' ')
        print()

printTable(tableData)
