import random
myList = ["하나", "둘", "셋", "넷", "다섯", "여섯"]

while(True) :
    response = input("주사위 던지기를 걔속하시겠습니까(yes, no)");
    if response == "yes" :
        coin = random.choice(myList)
        print(coin)
    else :
        break
