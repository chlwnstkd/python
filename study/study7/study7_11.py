from tkinter import *

#전역 변수 선언 부분
btnList = [None] * 9
fnameList = ["froyo.gif", "gingerbread.gif", "honeycomb.gif", "icecream.gif",
             "jellybean.gif", "kitkat.gif", "lollipop.gif", "marshmallow.gif", "nougat.gif"]
photoList = [None] * 9
i, k = 0, 0
xPos,yPos = 0, 0
num = 0

#메인 코드 부분
window = Tk()
window.geometry("210x210")

for i in range(0, 9) :
    photoList[i] = PhotoImage