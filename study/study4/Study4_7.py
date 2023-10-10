f = open("C:\python\새파일.txt", 'r')
lines = f.readlines()
for line in lines:
    print(line)
f.close()