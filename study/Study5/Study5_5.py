class FourCal:
    def setdata(self, first, second):
        self.first = first
        self.second = second

    def sum(self):
        result = self.first + self.second
        return result

    def mul(self):
        result = self.first * self.second
        return result

    def sub(self):
        result = self.first - self.second
        return result

    def div(self):
        result = self.first / self.second
        return result


a = FourCal()
b = FourCal()
print(type(a))
a.setdata(4, 2)
b.setdata(3, 7)

print("a")
print(a.first, a.second)
print(a.sum())
print(a.mul())
print(a.sub())
print(a.div())

print("")
print("b")
print(b.first, b.second)
print(b.sum())
print(b.mul())
print(b.sub())
print(b.div())