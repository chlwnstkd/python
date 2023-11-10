def two_times(numberList):
    result = []
    for number in numberList:
        result.append(number*2)
    return result
ex = [1,2,3,4]
result = two_times(ex)
print(result)
print(list(map(lambda a: a*2,ex)))
