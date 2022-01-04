#1.4.2 클래스 *개념중요! 개발자가 직접 정의하여 독자적 자료형 생성가능
class Man:
    def __init__(self, name):
        self.name=name
        print("Initialized!")

    def hello(self):
        print("Hello "+self.name + "!")

    def goodbye(self):
        print("Good-bye "+self.name + "!")
m =Man("David")
m.hello()
m.goodbye()
