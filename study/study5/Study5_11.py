def positive1(numberList):
    result = []
    for num in numberList:
        if num > 0:
            result.append(num)
    return result

def positive2(x):
    return x > 0

print(list(filter(positive2, [1,-3, 2, 0, -5, 6])))
print(positive1([1,-3, 2, 0, -5, 6]))