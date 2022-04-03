#coin_pair - USDT/BNB/BUSD etc.
import asyncio
from asyncio.tasks import Task
from asyncio.windows_events import NULL
from testing_functions import calc_fee
import threading
from time import time
from typing import Literal
import client as bin_client
from binance.client import Client
import binance
import time as time_lib
from multiprocessing import Pool

class CoinInfo:
    candles=[]
    all_closes=[]
    symbol=None
    client=None
    timestamp=None
    fee=1.0
    thread=None
    stop_execution=False
    def __init__(self,symbol:Literal):
        self.symbol = symbol
        self.client = bin_client.client
        self.timestamp = int(time_lib.time())
        self.candles = self.client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1MINUTE, "1 day ago UTC")
        for kline in self.candles:
            self.all_closes.append((float(kline[4])))
        self.thread = threading.Thread(target= self.WorkingThread)
        self.thread.start()
        self.fee=calc_fee(self.symbol)
        return
        
    async def StartCandlesUpdate(self):
        print('enter async')
        kekich = binance.BinanceSocketManager(self.client)
        socket_book = kekich.symbol_book_ticker_socket(self.symbol)
        async with socket_book as tick:
            while not self.stop_execution:
                res = await tick.recv()
                #print(f'sell:  {res["b"]}')
                #print(f'buy :  {res["a"]}')
                cur_tm = int(time_lib.time())
                if ( int(self.timestamp / 60) != int(cur_tm / 60) ):
                    print(f'appended new candle {res["a"]}')
                    self.all_closes.append(res["a"])
                    self.all_closes.pop(0)
                    #print(f'last time{ self.timestamp}-{cur_tm}')
                    self.timestamp = cur_tm
                else:
                    self.all_closes[-1] = res["a"]
        #await self.client.close_connection()
        return

    def WorkingThread(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.StartCandlesUpdate())
        loop.close()
        return

    def Stop(self):
        self.stop_execution = True
        return

def GetCoinsForTrading(coin_pair:Literal, start_balance:float):
    client = bin_client.client
    tickers = client.get_orderbook_tickers()

    #print (tickers)
    return tickers