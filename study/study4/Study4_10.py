import turtle
import random

def screenLeftClick(x, y) :
     global r, g, b
     turtle.pencolor((r, g, b))
     turtle.pendown()
     turtle.goto(x, y)

def screenRightclick(x, y) :
     turtle.penup()
     turtle.goto(x, y)

def screenMidClick(x, y) :
     global r, g, b
     tSize = random.randrange(1, 10)
     turtle.shapesize(tSize)
     r = random.random()
     g = random.random()
     b = random.random()

pSize, tSize = 10, 0
r, g, b = 0.0, 0.0, 0.0

turtle.title('거북이 그림 그리기')
turtle.shape('turtle')
turtle.pensize(pSize)

turtle.onscreenclick(screenLeftClick, 1)
turtle.onscreenclick(screenMidClick, 2)
turtle.onscreenclick(screenRightclick, 3)

turtle.done()