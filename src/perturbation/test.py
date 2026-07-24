import pickle

class Test:
    def __init__(self):
        self.X = 0
    

    def add(self, num):
        self.X += num
    
    def get_X(self):
        return self.X
    def save_class(self):
        filename = 'test.sav'
        with open(filename, 'wb') as file:
            pickle.dump(self, file)
    

#test = Test()
#test.add(10)
#test.add(100)
#print(test.get_X())
#test.save_class()
test_class = pickle.load(open('test.sav', 'rb'))
print(test_class.get_X())

