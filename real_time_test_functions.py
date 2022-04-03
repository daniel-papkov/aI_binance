import testing_functions,ai_testing_functions,numpy as np
import datetime as dt

def is_new_candle(closes1,closes2):
    if(closes1[-1]==closes2[-1]):
        return True
    else:
        return False

def get_new_closes(closes1,last_candle):
    closes1.append(last_candle)
    closes1.pop(0)
    return closes1

def cut_closes_for_Training(all_closes):
    closes=all_closes
    closes=np.array(closes)
    closes=closes[int(len(closes)*1/3):int(len(closes)*2/3)]

def get_training_data(all_closes,prediction_candles):
    test_data=all_closes[-prediction_candles]
    return test_data

def file_write(file,str):
    file.write(str)
    print(str)
    return

def log(last_price,new_price,prediction):
    now = dt.datetime.now()    
    print('logging!')
    f = open("buy_or_sell_log.txt", "a")
    file_write(f,f'\n{last_price} was the last price\n')
    file_write(f,f'{new_price} is the new price\n')
    file_write(f,f'{prediction} was the predicted next price\n')
    file_write(f,f'{now.month}/{now.day} - {now.hour}:{now.minute}:{now.second}\n')    
    f.close()

f = open("buy_or_sell_log.txt", "w")
f.close()
#print(log(5,6,7))