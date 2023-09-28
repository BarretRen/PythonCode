# write to file
file = open('./hello.txt', 'w')
file.write('hello world\n')
file.close()

# append to file
file = open('./hello.txt', 'a')
file.write('hello python\n')
file.close()

# read from file
file = open('./hello.txt', 'r')
text = file.read()
file.close()
print(text)

# use shelve module to store the variables
import shelve
shelf = shelve.open('data')
cats = ['zophie', 'pooka', 'simon']
shelf['cats'] = cats
shelf.close()

# use shelve module to read the data
shelf = shelve.open('data')
for cat in shelf['cats']:
    print(cat)




