stuff = {'rope': 1, 'torch': 6, 'gold coin': 42, 'dagger': 1, 'arrow': 12}


def printDict(inventory):
    print("inventory")
    total = 0
    for k, v in inventory.items():
        print(str(inventory[k]) + ' ' + str(k))
        total += inventory[k]
    print('Total number of items: ' + str(total))


def addToInventory(inventory, addedItems):
    for item in addedItems:
        if item in inventory.keys():
            inventory[item] += 1
        else:
            inventory[item] = 1

    return inventory


#  printDict(stuff)

inv = {'gold coin': 42, 'rope': 1}
dragonLoot = ['gold coin', 'dagger', 'gold coin', 'gold coin', 'ruby']

inv = addToInventory(inv, dragonLoot)
printDict(inv)
