from tkinter import *
from tkinter import messagebox

#함수 선언 부분
def keyEvent(event) :
    messagebox.showinfo("키보드 이벤트", "눌린 키" + chr(event.keycode))
    # chr = 하나의 정수를 인자로 받고 해당 정수에 해당하는 유니코드 문자를 반환

# 메인 코드 부분
window = Tk()

window.bind("<Key>", keyEvent)

window.mainloop()