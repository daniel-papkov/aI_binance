import datetime as dt
import config
import binance
from binance.client import Client

def total_gain_possible(closes):
    start = closes[0]
    possible = 1.0
    for x in range(len(closes)-1):
        if(closes[x+1]>closes[x]):
            start=start*((float(closes[x+1]))/(float(closes[x])))*0.99875
            possible=possible*((float(closes[x+1]))/(float(closes[x])))*0.99875
    return possible

def total_loss_possible(closes):
    start = closes[0]
    possible = 1.0
    for x in range(len(closes)-1):
        if(closes[x+1]<closes[x]):
            start=start*((float(closes[x+1]))/(float(closes[x])))*0.99875
            possible=possible*((float(closes[x+1]))/(float(closes[x])))*0.99875
    return possible

def convert_to_angles(closes):
    angles=[]    
    for  x in range(len(closes)-1):        
        angles.append(round((closes[x+1])/closes[x],4))

    return angles

def convert_to_angles_longterm(closes):
    angles=[]
    start=closes[0]
    angles.append(0)
    for x in range(1,len(closes)):
        angles.append((closes[x]-start)/x)
    return angles

def convert_angles_to_price(startprice,angles):
    actual_price=[]
    actual_price.append(startprice)
    for x in range(0,len(angles)):        
        a=angles[x]
        b=actual_price[x]
        actual_price.append((b*a))
    return actual_price

def buy_or_sell_log(lastprice,prediction_for_next_candle,price_target,is_owned):
    now = dt.datetime.now()
    if(prediction_for_next_candle-lastprice>price_target and is_owned==False):
        f = open("buy_or_sell_log.txt", "w")
        f.write(f'\n{lastprice} was the last price\n')
        f.write(f'{prediction_for_next_candle} was the predicted next price\n')
        f.write(f'bought holdings at {now.month}/{now.day} - {now.minute}:{now.second}\n')
        f.write(f'for a profit of {prediction_for_next_candle-lastprice}\n')
        f.close()
        return prediction_for_next_candle-lastprice
    if(prediction_for_next_candle-lastprice<price_target and is_owned==True):
        f = open("buy_or_sell_log.txt", "w")
        f.write(f'\n{lastprice} was the last price\n')
        f.write(f'{prediction_for_next_candle} was the predicted next price\n')
        f.write(f'sold holdings at {now.month}/{now.day} - {now.minute}:{now.second}\n')
        f.write(f'for a profit of {prediction_for_next_candle-lastprice}\n')
        f.close()
        return prediction_for_next_candle-lastprice

def buy_or_Sell_action(last_price,enter_price,prediction,is_owned):
    if(prediction/last_price>1.0):
        if(is_owned>0):            
            return('#hold')
        if(is_owned==0):            
            return('#buy')
    if(prediction/last_price<1.0):
        if(is_owned>0):            
            return('#sell')           
        if(is_owned==0):
            return('#badbuy')
    if(prediction/enter_price<1.0):
        if(is_owned>0):
            return('#sell')
        if(is_owned==0):
            return('#badbuy')
    return('#noaction')

def buy_or_Sell_action_v2(last_price,enter_price,prediction,last_prediction,is_owned):
    if(prediction/last_prediction>1.0):
        if(is_owned>0):            
            return('#hold')
        if(is_owned==0):            
            return('#buy')
    if(prediction/last_prediction<1.0):
        if(is_owned>0):            
            return('#sell')           
        if(is_owned==0):
            return('#badbuy')
    if(last_price<enter_price):
        if(is_owned>0):
            return('#sell')
    if(prediction<enter_price):
        if(is_owned>0):
            return('#sell')
        if(is_owned==0):
            return('#badbuy')
    return('#noaction')

def buy_or_Sell_action_test(last_price,enter_price,prediction,last_prediction,is_owned,holdbuy_percent,sellignore_percent):
    if(prediction/last_prediction>holdbuy_percent):
        if(is_owned>0):            
            return('#hold')
        if(is_owned==0):            
            return('#buy')
    if(prediction/last_prediction<sellignore_percent):
        if(is_owned>0):            
            return('#sell')           
        if(is_owned==0):
            return('#badbuy')
    if(last_price<enter_price):
        if(is_owned>0):
            return('#sell')
    #if(prediction/enter_price<sellignore_percent):
    if(prediction<enter_price):
        if(is_owned>0):
            return('#sell')
        if(is_owned==0):
            return('#badbuy')
    return('#noaction')

def compare_ai_to_irl(action,cur_price):#tax is done in another function not here
    if(action=='#buy'):
        return 0-cur_price
    if(action=='#hold'):
        return 0.0        
    if(action=='#sell'):       
        return (cur_price)#to make nahman happy kekw
    if(action=='#badbuy'):
        return 0.0

def calc_fee(symbol):
    client = Client(config.API_KEY, config.SECRET_KEY)
    ticker = client.get_orderbook_ticker(symbol = symbol)
    fee_original = client.get_trade_fee(symbol = symbol)[0]
    fee_ask_bid = (float(ticker['bidPrice']) / float(ticker['askPrice'])) - float(1)
    fee_multiplier = float(1) - (float(fee_original['makerCommission']) - (fee_ask_bid / 2.0) )
    #print(fee_multiplier)
    return fee_multiplier

def check_profits(prediction_prices,prices,symbol):
    start_val=prices[0]
    target_val=start_val
    enter_price=start_val
    is_owned=0
    last_price=0
    last_prediction = start_val
    for x in range(0,len(prediction_prices)-1):
        prices=prices[:len(prediction_prices)+1]#might need to delete
        prediction = prediction_prices[x,0]#[x,0]
        last_price = prices[x]#one before prediction
        next_price = prices[x+1]#same time as prediction
        target_action=buy_or_Sell_action_v2(last_price,enter_price,prediction,last_prediction,is_owned)
        #target_action=buy_or_Sell_action(last_price,enter_price,prediction,is_owned)
        result_from_trade=compare_ai_to_irl(target_action,next_price)
        #print(target_action)
        print(f'{result_from_trade} {target_action}')
        last_prediction=prediction
        if(target_action=='#buy'):            
            print(f'buying {target_val/last_price} {symbol} for {last_price}')
            is_owned=target_val/last_price
            target_val+=result_from_trade
            enter_price=last_price
            continue

        if(target_action=='#sell'):
            print(f'selling {is_owned} {symbol} for {is_owned*last_price}')
            is_owned=0
            target_val+=result_from_trade
            enter_price=0
            continue

        if(target_action=='#hold'):
            print(f'holding {is_owned} {symbol} for total value of {is_owned*last_price}')
            continue

        if(target_action=='#badbuy'):
            #print('not buying cause of bad deal')
            continue  

        if(target_action=='#noaction'):
            continue
        else:
            print('nonetype here ?')
            #print('aint doing shit shouldnt have gotten here')

    if(is_owned>0):
        target_val+=is_owned*last_price
        is_owned=0
        print(target_val/start_val)

    return (target_val/start_val)

def check_profits_test(prediction_prices,prices,symbol,holdbuy_percent,sellignore_percent, trade_fee):
    start_val=prices[0]
    target_val=start_val
    enter_price=start_val
    is_owned=0
    last_price=0
    last_prediction = start_val
    for x in range(0,len(prediction_prices)-1):
        prices=prices[:len(prediction_prices)+1]#might need to delete
        prediction = prediction_prices[x,0]#[x,0]
        last_price = prices[x]#one before prediction
        next_price = prices[x+1]#same time as prediction
        target_action=buy_or_Sell_action_test(last_price,enter_price,prediction,last_prediction,is_owned,holdbuy_percent,sellignore_percent)
        #target_action=buy_or_Sell_action(last_price,enter_price,prediction,is_owned)
        result_from_trade=compare_ai_to_irl(target_action,next_price)
        print(target_action)
        print(f'{result_from_trade} {target_action}')
        last_prediction=prediction
        if(target_action=='#buy'):            
            print(f'buying {target_val/last_price} {symbol} for {last_price}')
            is_owned=(target_val/last_price)*trade_fee

            target_val+=result_from_trade
            enter_price=last_price
            continue

        if(target_action=='#sell'):
            print(f'selling {is_owned} {symbol} for {is_owned*last_price}, price {last_price}, profit in % {(last_price/enter_price)-(1-(trade_fee**2))}')
            is_owned=0
            target_val+=(result_from_trade*trade_fee)
            enter_price=0
            continue

        if(target_action=='#hold'):
            print(f'holding {is_owned} {symbol} for total value of {is_owned*last_price}, price {last_price}')
            continue

        if(target_action=='#badbuy'):
            #print('not buying cause of bad deal')
            continue  

        if(target_action=='#noaction'):
            continue        
            #print('nonetype here ?')
            #print('aint doing shit shouldnt have gotten here')

    if(is_owned>0):
        target_val+=is_owned*last_price
        is_owned=0
        print(target_val/start_val)

    return (target_val/start_val)

def check_profits_test_const_fee(prediction_prices,prices,symbol,holdbuy_percent,sellignore_percent,fee, to_log):
    start_val=prices[0]
    target_val=float(start_val)
    enter_price=float(start_val)
    is_owned=0
    last_price=0
    last_prediction = start_val
    trade_fee=fee
    for x in range(0,len(prediction_prices)-1):
        prices=prices[:len(prediction_prices)+1]#might need to delete
        prediction = prediction_prices[x,0]
        last_price = prices[x]#one before prediction
        next_price = prices[x+1]#same time as prediction
        target_action=buy_or_Sell_action_test(last_price,enter_price,prediction,last_prediction,is_owned,holdbuy_percent,sellignore_percent)
        #target_action=buy_or_Sell_action(last_price,enter_price,prediction,is_owned)
        result_from_trade=compare_ai_to_irl(target_action,next_price)
        if to_log: print(target_action)
        if to_log: print(f'{result_from_trade} {target_action}')
        last_prediction=prediction
        if(target_action=='#buy'):            
            if to_log: print(f'buying {target_val/last_price} {symbol} for {last_price}')
            is_owned=(target_val/last_price)*trade_fee
            target_val+=result_from_trade
            enter_price=last_price
            continue

        if(target_action=='#sell'):
            if to_log: print(f'selling {is_owned} {symbol} for {is_owned*last_price}')
            is_owned=0
            target_val+=(result_from_trade*trade_fee)
            enter_price=0
            continue

        if(target_action=='#hold'):
            if to_log: print(f'holding {is_owned} {symbol} for total value of {is_owned*last_price}')
            continue

        if(target_action=='#badbuy'):
            if to_log: print('not buying cause of bad deal')
            continue  

        if(target_action=='#noaction'):
            continue        
            #if to_log: print('nonetype here ?')
            #if to_log: print('aint doing shit shouldnt have gotten here')

    if(is_owned>0):
        target_val+=is_owned*last_price
        is_owned=0
        print(target_val/start_val)

    return (target_val/start_val)


# print(convert_angles_to_price(1,[20000,15000,13333.33333]))
# compare_ai_to_irl(buy_or_Sell_action(500,530,550,1),550)#hold example
# compare_ai_to_irl(buy_or_Sell_action(500,530,529,1),520)#sell example
# compare_ai_to_irl(buy_or_Sell_action(500,530,550,0),520)#buy example


# a=520
# a+=compare_ai_to_irl(buy_or_Sell_action(500,530,550,0),520)
# print(a)

#print(calc_fee('BTCUSDT'))

print()
print()