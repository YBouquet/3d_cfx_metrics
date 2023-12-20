class Person:
    def __init__(self, name):
        self.name = name
        print(dir(self))
    def say_hi(self):
        print('Hello, my name is', self.name)


x = Person("Yann")
print(vars(x))