from cocobolo import Cocobolo

cocobolo = Cocobolo()

def test():
    # must be: 'index': 33, 'name': '17', 'label': '17-Pochota  fendleri [Pochote, Cedro pochote]'
    prediction = cocobolo.predict_test()
    print("prediction:", prediction)
    
test()