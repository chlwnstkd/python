import turtle


myT = None

myT = turtle.Turtle()
myT.shape('turtle')

for i in range(0,100) :
    myT.forward(200 + i * 5)
    myT.right(90)

turtle.done()