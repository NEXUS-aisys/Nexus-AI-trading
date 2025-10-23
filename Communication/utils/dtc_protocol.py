"""
Sierra Chart DTC (Data and Trading Communications) Protocol Implementation
Production-grade binary protocol handler

DTC Protocol Documentation: https://dtcprotocol.org/
Version: 8 (Current)
"""

import struct
import logging
from enum import IntEnum
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)


# ============================================================================
# DTC MESSAGE TYPES (Protocol Version 8)
# ============================================================================

class DTCMessageType(IntEnum):
    """DTC Protocol Message Types"""
    # Logon/Heartbeat
    LOGON_REQUEST = 1
    LOGON_RESPONSE = 2
    HEARTBEAT = 3
    LOGOFF = 5
    ENCODING_REQUEST = 6
    ENCODING_RESPONSE = 7
    
    # Market Data
    MARKET_DATA_REQUEST = 101
    MARKET_DATA_REJECT = 103
    MARKET_DATA_SNAPSHOT = 104
    MARKET_DATA_UPDATE_TRADE = 107
    MARKET_DATA_UPDATE_BID_ASK = 108
    MARKET_DATA_UPDATE_SESSION_OPEN = 109
    MARKET_DATA_UPDATE_SESSION_HIGH = 110
    MARKET_DATA_UPDATE_SESSION_LOW = 111
    MARKET_DATA_UPDATE_SESSION_VOLUME = 112
    
    # Trading
    SUBMIT_NEW_SINGLE_ORDER = 208
    CANCEL_ORDER = 203
    OPEN_ORDERS_REQUEST = 300
    OPEN_ORDERS_REJECT = 302
    ORDER_UPDATE = 301
    HISTORICAL_ORDER_FILLS_REQUEST = 303
    HISTORICAL_ORDER_FILL_RESPONSE = 304
    CURRENT_POSITIONS_REQUEST = 305
    POSITION_UPDATE = 306
    
    # Account
    TRADE_ACCOUNT_REQUEST = 400
    TRADE_ACCOUNT_RESPONSE = 401
    EXCHANGE_LIST_REQUEST = 500
    EXCHANGE_LIST_RESPONSE = 501
    SYMBOLS_FOR_EXCHANGE_REQUEST = 502
    UNDERLYING_SYMBOLS_FOR_EXCHANGE_REQUEST = 503
    SYMBOL_SEARCH_REQUEST = 504
    SECURITY_DEFINITION_FOR_SYMBOL_REQUEST = 506
    SECURITY_DEFINITION_RESPONSE = 507
    
    # Historical Data
    HISTORICAL_PRICE_DATA_REQUEST = 700
    HISTORICAL_PRICE_DATA_RESPONSE_HEADER = 701
    HISTORICAL_PRICE_DATA_REJECT = 702
    HISTORICAL_PRICE_DATA_RECORD_RESPONSE = 703


class DTCOrderType(IntEnum):
    """DTC Order Types"""
    ORDER_TYPE_MARKET = 1
    ORDER_TYPE_LIMIT = 2
    ORDER_TYPE_STOP = 3
    ORDER_TYPE_STOP_LIMIT = 4
    ORDER_TYPE_MARKET_IF_TOUCHED = 5


class DTCOrderStatus(IntEnum):
    """DTC Order Status"""
    ORDER_STATUS_OPEN = 1
    ORDER_STATUS_FILLED = 2
    ORDER_STATUS_CANCELED = 3
    ORDER_STATUS_REJECTED = 4
    ORDER_STATUS_PARTIALLY_FILLED = 5


class DTCBuySell(IntEnum):
    """DTC Buy/Sell Indicator"""
    BUY = 1
    SELL = 2


class DTCTimeInForce(IntEnum):
    """DTC Time In Force"""
    TIF_DAY = 0
    TIF_GOOD_TILL_CANCELED = 1
    TIF_GOOD_TILL_DATE_TIME = 2
    TIF_IMMEDIATE_OR_CANCEL = 3
    TIF_FILL_OR_KILL = 4


# ============================================================================
# DTC MESSAGE STRUCTURES
# ============================================================================

@dataclass
class DTCLogonRequest:
    """DTC Logon Request Message"""
    ProtocolVersion: int = 8
    Username: str = ""
    Password: str = ""
    GeneralTextData: str = ""
    Integer_1: int = 0
    Integer_2: int = 0
    HeartbeatIntervalInSeconds: int = 10
    ClientName: str = "NEXUS_AI_Gateway"
    
    def encode(self) -> bytes:
        """Encode to binary format"""
        msg_type = DTCMessageType.LOGON_REQUEST
        size = 256  # Fixed size for logon
        
        return struct.pack(
            '<HH',  # Size, Type
            size, msg_type
        ) + struct.pack(
            '<I128s128s128sIIII128s',
            self.ProtocolVersion,
            self.Username.encode('utf-8')[:128],
            self.Password.encode('utf-8')[:128],
            self.GeneralTextData.encode('utf-8')[:128],
            self.Integer_1,
            self.Integer_2,
            self.HeartbeatIntervalInSeconds,
            0,  # Unused_1
            self.ClientName.encode('utf-8')[:128]
        )


@dataclass
class DTCLogonResponse:
    """DTC Logon Response Message"""
    ProtocolVersion: int
    Result: int
    ResultText: str
    ReconnectAddress: str
    Integer_1: int
    ServerName: str
    MarketDepthUpdatesBestBidAndAsk: int
    TradingIsSupported: int
    OCOOrdersSupported: int
    OrderCancelReplaceSupported: int
    SymbolExchangeDelimiter: str
    SecurityDefinitionsSupported: int
    HistoricalPriceDataSupported: int
    ResubscribeWhenMarketDataFeedAvailable: int
    MarketDepthIsSupported: int
    
    @staticmethod
    def decode(data: bytes) -> 'DTCLogonResponse':
        """Decode from binary format"""
        unpacked = struct.unpack(
            '<I I 128s 64s I 128s B B B B 8s B B B B',
            data
        )
        
        return DTCLogonResponse(
            ProtocolVersion=unpacked[0],
            Result=unpacked[1],
            ResultText=unpacked[2].decode('utf-8').rstrip('\x00'),
            ReconnectAddress=unpacked[3].decode('utf-8').rstrip('\x00'),
            Integer_1=unpacked[4],
            ServerName=unpacked[5].decode('utf-8').rstrip('\x00'),
            MarketDepthUpdatesBestBidAndAsk=unpacked[6],
            TradingIsSupported=unpacked[7],
            OCOOrdersSupported=unpacked[8],
            OrderCancelReplaceSupported=unpacked[9],
            SymbolExchangeDelimiter=unpacked[10].decode('utf-8').rstrip('\x00'),
            SecurityDefinitionsSupported=unpacked[11],
            HistoricalPriceDataSupported=unpacked[12],
            ResubscribeWhenMarketDataFeedAvailable=unpacked[13],
            MarketDepthIsSupported=unpacked[14]
        )


@dataclass
class DTCMarketDataRequest:
    """DTC Market Data Request"""
    RequestAction: int  # 1=Subscribe, 2=Unsubscribe
    SymbolID: int
    Symbol: str
    Exchange: str
    
    def encode(self) -> bytes:
        """Encode to binary format"""
        msg_type = DTCMessageType.MARKET_DATA_REQUEST
        size = 168  # Fixed size
        
        return struct.pack(
            '<HH I I 64s 16s',
            size, msg_type,
            self.RequestAction,
            self.SymbolID,
            self.Symbol.encode('utf-8')[:64],
            self.Exchange.encode('utf-8')[:16]
        )


@dataclass
class DTCMarketDataSnapshot:
    """DTC Market Data Snapshot"""
    SymbolID: int
    SessionSettlementPrice: float
    SessionOpenPrice: float
    SessionHighPrice: float
    SessionLowPrice: float
    SessionVolume: float
    SessionNumTrades: int
    OpenInterest: int
    BidPrice: float
    AskPrice: float
    BidQuantity: float
    AskQuantity: float
    LastTradePrice: float
    LastTradeVolume: float
    LastTradeDateTime: float
    BidAskDateTime: float
    
    @staticmethod
    def decode(data: bytes) -> 'DTCMarketDataSnapshot':
        """Decode from binary format"""
        unpacked = struct.unpack(
            '<I d d d d d I I d d d d d d d d',
            data
        )
        
        return DTCMarketDataSnapshot(
            SymbolID=unpacked[0],
            SessionSettlementPrice=unpacked[1],
            SessionOpenPrice=unpacked[2],
            SessionHighPrice=unpacked[3],
            SessionLowPrice=unpacked[4],
            SessionVolume=unpacked[5],
            SessionNumTrades=unpacked[6],
            OpenInterest=unpacked[7],
            BidPrice=unpacked[8],
            AskPrice=unpacked[9],
            BidQuantity=unpacked[10],
            AskQuantity=unpacked[11],
            LastTradePrice=unpacked[12],
            LastTradeVolume=unpacked[13],
            LastTradeDateTime=unpacked[14],
            BidAskDateTime=unpacked[15]
        )


@dataclass
class DTCMarketDataUpdateTrade:
    """DTC Market Data Update - Trade"""
    SymbolID: int
    AtBidOrAsk: int
    Price: float
    Volume: float
    DateTime: float
    
    @staticmethod
    def decode(data: bytes) -> 'DTCMarketDataUpdateTrade':
        """Decode from binary format"""
        unpacked = struct.unpack('<I I d d d', data)
        
        return DTCMarketDataUpdateTrade(
            SymbolID=unpacked[0],
            AtBidOrAsk=unpacked[1],
            Price=unpacked[2],
            Volume=unpacked[3],
            DateTime=unpacked[4]
        )


@dataclass
class DTCMarketDataUpdateBidAsk:
    """DTC Market Data Update - Bid/Ask"""
    SymbolID: int
    BidPrice: float
    BidQuantity: float
    AskPrice: float
    AskQuantity: float
    DateTime: float
    
    @staticmethod
    def decode(data: bytes) -> 'DTCMarketDataUpdateBidAsk':
        """Decode from binary format"""
        unpacked = struct.unpack('<I d d d d d', data)
        
        return DTCMarketDataUpdateBidAsk(
            SymbolID=unpacked[0],
            BidPrice=unpacked[1],
            BidQuantity=unpacked[2],
            AskPrice=unpacked[3],
            AskQuantity=unpacked[4],
            DateTime=unpacked[5]
        )


@dataclass
class DTCSubmitNewOrder:
    """DTC Submit New Order"""
    Symbol: str
    Exchange: str
    TradeAccount: str
    ClientOrderID: str
    OrderType: int  # DTCOrderType
    BuySell: int  # DTCBuySell
    Price1: float
    Price2: float
    Quantity: float
    TimeInForce: int  # DTCTimeInForce
    GoodTillDateTime: float
    IsAutomatedOrder: int
    
    def encode(self) -> bytes:
        """Encode to binary format"""
        msg_type = DTCMessageType.SUBMIT_NEW_SINGLE_ORDER
        size = 256  # Fixed size
        
        return struct.pack(
            '<HH',
            size, msg_type
        ) + struct.pack(
            '<64s 16s 32s 32s I I d d d I d B',
            self.Symbol.encode('utf-8')[:64],
            self.Exchange.encode('utf-8')[:16],
            self.TradeAccount.encode('utf-8')[:32],
            self.ClientOrderID.encode('utf-8')[:32],
            self.OrderType,
            self.BuySell,
            self.Price1,
            self.Price2,
            self.Quantity,
            self.TimeInForce,
            self.GoodTillDateTime,
            self.IsAutomatedOrder
        ) + b'\x00' * (256 - 4 - 64 - 16 - 32 - 32 - 4 - 4 - 8 - 8 - 8 - 4 - 8 - 1)


@dataclass
class DTCOrderUpdate:
    """DTC Order Update"""
    TotalNumMessages: int
    MessageNumber: int
    Symbol: str
    Exchange: str
    OrderStatus: int  # DTCOrderStatus
    OrderUpdateDateTime: float
    OrderQuantity: float
    FilledQuantity: float
    RemainingQuantity: float
    AverageFillPrice: float
    LastFillPrice: float
    LastFillDateTime: float
    LastFillQuantity: float
    UniqueFillExecutionID: str
    TradeAccount: str
    InfoText: str
    OrderType: int
    BuySell: int
    Price1: float
    Price2: float
    OrderID: str
    
    @staticmethod
    def decode(data: bytes) -> 'DTCOrderUpdate':
        """Decode from binary format"""
        unpacked = struct.unpack(
            '<I I 64s 16s I d d d d d d d d 64s 32s 96s I I d d 32s',
            data
        )
        
        return DTCOrderUpdate(
            TotalNumMessages=unpacked[0],
            MessageNumber=unpacked[1],
            Symbol=unpacked[2].decode('utf-8').rstrip('\x00'),
            Exchange=unpacked[3].decode('utf-8').rstrip('\x00'),
            OrderStatus=unpacked[4],
            OrderUpdateDateTime=unpacked[5],
            OrderQuantity=unpacked[6],
            FilledQuantity=unpacked[7],
            RemainingQuantity=unpacked[8],
            AverageFillPrice=unpacked[9],
            LastFillPrice=unpacked[10],
            LastFillDateTime=unpacked[11],
            LastFillQuantity=unpacked[12],
            UniqueFillExecutionID=unpacked[13].decode('utf-8').rstrip('\x00'),
            TradeAccount=unpacked[14].decode('utf-8').rstrip('\x00'),
            InfoText=unpacked[15].decode('utf-8').rstrip('\x00'),
            OrderType=unpacked[16],
            BuySell=unpacked[17],
            Price1=unpacked[18],
            Price2=unpacked[19],
            OrderID=unpacked[20].decode('utf-8').rstrip('\x00')
        )


@dataclass
class DTCHeartbeat:
    """DTC Heartbeat Message"""
    NumDroppedMessages: int
    CurrentDateTime: float
    
    def encode(self) -> bytes:
        """Encode to binary format"""
        msg_type = DTCMessageType.HEARTBEAT
        size = 16  # 4 (size+type) + 4 (NumDropped) + 8 (DateTime)
        
        return struct.pack(
            '<HH I d',
            size, msg_type,
            self.NumDroppedMessages,
            self.CurrentDateTime
        )
    
    @staticmethod
    def decode(data: bytes) -> 'DTCHeartbeat':
        """Decode from binary format"""
        unpacked = struct.unpack('<I d', data)
        return DTCHeartbeat(
            NumDroppedMessages=unpacked[0],
            CurrentDateTime=unpacked[1]
        )


# ============================================================================
# DTC PROTOCOL HANDLER
# ============================================================================

class DTCProtocolHandler:
    """
    Production DTC Protocol Handler
    Handles encoding/decoding of Sierra Chart DTC messages
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.DTCProtocolHandler")
        self.message_handlers = {
            DTCMessageType.LOGON_RESPONSE: self._handle_logon_response,
            DTCMessageType.HEARTBEAT: self._handle_heartbeat,
            DTCMessageType.MARKET_DATA_SNAPSHOT: self._handle_market_data_snapshot,
            DTCMessageType.MARKET_DATA_UPDATE_TRADE: self._handle_market_data_trade,
            DTCMessageType.MARKET_DATA_UPDATE_BID_ASK: self._handle_market_data_bid_ask,
            DTCMessageType.ORDER_UPDATE: self._handle_order_update,
        }
        
        self.callbacks = {
            'logon_response': None,
            'market_data': None,
            'order_update': None,
            'heartbeat': None,
        }
    
    def set_callback(self, event: str, callback):
        """Set callback function for events"""
        if event in self.callbacks:
            self.callbacks[event] = callback
        else:
            raise ValueError(f"Unknown event type: {event}")
    
    def parse_message(self, data: bytes) -> Optional[Any]:
        """
        Parse incoming DTC message
        
        Returns parsed message object or None
        """
        if len(data) < 4:
            self.logger.warning("Message too short (< 4 bytes)")
            return None
        
        # First 2 bytes: message size
        # Next 2 bytes: message type
        size, msg_type = struct.unpack('<HH', data[:4])
        
        if len(data) < size:
            self.logger.warning(f"Incomplete message: expected {size}, got {len(data)}")
            return None
        
        # Extract message body (skip size and type)
        message_body = data[4:size]
        
        # Handle message based on type
        try:
            msg_type_enum = DTCMessageType(msg_type)
            
            if msg_type_enum in self.message_handlers:
                return self.message_handlers[msg_type_enum](message_body)
            else:
                self.logger.debug(f"Unhandled message type: {msg_type_enum.name}")
                return None
        
        except ValueError:
            self.logger.warning(f"Unknown message type: {msg_type}")
            return None
        
        except Exception as e:
            self.logger.error(f"Error parsing message: {e}", exc_info=True)
            return None
    
    def _handle_logon_response(self, data: bytes):
        """Handle logon response"""
        response = DTCLogonResponse.decode(data)
        self.logger.info(f"Logon response: Result={response.Result}, Server={response.ServerName}")
        
        if self.callbacks['logon_response']:
            self.callbacks['logon_response'](response)
        
        return response
    
    def _handle_heartbeat(self, data: bytes):
        """Handle heartbeat"""
        heartbeat = DTCHeartbeat.decode(data)
        
        if self.callbacks['heartbeat']:
            self.callbacks['heartbeat'](heartbeat)
        
        return heartbeat
    
    def _handle_market_data_snapshot(self, data: bytes):
        """Handle market data snapshot"""
        snapshot = DTCMarketDataSnapshot.decode(data)
        
        if self.callbacks['market_data']:
            self.callbacks['market_data'](snapshot)
        
        return snapshot
    
    def _handle_market_data_trade(self, data: bytes):
        """Handle market data trade update"""
        trade = DTCMarketDataUpdateTrade.decode(data)
        
        if self.callbacks['market_data']:
            self.callbacks['market_data'](trade)
        
        return trade
    
    def _handle_market_data_bid_ask(self, data: bytes):
        """Handle market data bid/ask update"""
        bid_ask = DTCMarketDataUpdateBidAsk.decode(data)
        
        if self.callbacks['market_data']:
            self.callbacks['market_data'](bid_ask)
        
        return bid_ask
    
    def _handle_order_update(self, data: bytes):
        """Handle order update"""
        order_update = DTCOrderUpdate.decode(data)
        
        if self.callbacks['order_update']:
            self.callbacks['order_update'](order_update)
        
        return order_update
    
    # Message Creation Methods
    
    def create_logon_request(self, username: str = "", password: str = "") -> bytes:
        """Create logon request message"""
        request = DTCLogonRequest(
            Username=username,
            Password=password,
            ClientName="NEXUS_AI_Gateway"
        )
        return request.encode()
    
    def create_market_data_request(self, symbol: str, exchange: str = "", symbol_id: int = 1) -> bytes:
        """Create market data subscription request"""
        request = DTCMarketDataRequest(
            RequestAction=1,  # Subscribe
            SymbolID=symbol_id,
            Symbol=symbol,
            Exchange=exchange
        )
        return request.encode()
    
    def create_submit_order(self, symbol: str, side: str, quantity: float, 
                           order_type: str = "MARKET", price: float = 0.0,
                           client_order_id: str = "", exchange: str = "",
                           trade_account: str = "") -> bytes:
        """
        Create submit order message
        
        Args:
            symbol: Trading symbol
            side: "BUY" or "SELL"
            quantity: Order quantity
            order_type: "MARKET", "LIMIT", "STOP", etc.
            price: Limit/Stop price
            client_order_id: Unique client order ID
            exchange: Exchange name
            trade_account: Trading account
        """
        # Convert side
        buy_sell = DTCBuySell.BUY if side.upper() == "BUY" else DTCBuySell.SELL
        
        # Convert order type
        order_type_map = {
            "MARKET": DTCOrderType.ORDER_TYPE_MARKET,
            "LIMIT": DTCOrderType.ORDER_TYPE_LIMIT,
            "STOP": DTCOrderType.ORDER_TYPE_STOP,
            "STOP_LIMIT": DTCOrderType.ORDER_TYPE_STOP_LIMIT,
        }
        dtc_order_type = order_type_map.get(order_type.upper(), DTCOrderType.ORDER_TYPE_MARKET)
        
        order = DTCSubmitNewOrder(
            Symbol=symbol,
            Exchange=exchange,
            TradeAccount=trade_account,
            ClientOrderID=client_order_id,
            OrderType=dtc_order_type,
            BuySell=buy_sell,
            Price1=price,
            Price2=0.0,
            Quantity=quantity,
            TimeInForce=DTCTimeInForce.TIF_DAY,
            GoodTillDateTime=0.0,
            IsAutomatedOrder=1
        )
        
        return order.encode()
    
    def create_heartbeat(self, num_dropped: int = 0) -> bytes:
        """Create heartbeat message"""
        heartbeat = DTCHeartbeat(
            NumDroppedMessages=num_dropped,
            CurrentDateTime=datetime.now().timestamp()
        )
        return heartbeat.encode()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def dtc_datetime_to_unix(dtc_time: float) -> float:
    """Convert DTC DateTime to Unix timestamp"""
    # DTC uses Microsoft DateTime format (days since 12/30/1899)
    # Convert to Unix timestamp
    if dtc_time == 0:
        return 0.0
    
    # Days between 12/30/1899 and 1/1/1970
    epoch_offset = 25569.0
    seconds_per_day = 86400.0
    
    return (dtc_time - epoch_offset) * seconds_per_day


def unix_to_dtc_datetime(unix_time: float) -> float:
    """Convert Unix timestamp to DTC DateTime"""
    if unix_time == 0:
        return 0.0
    
    epoch_offset = 25569.0
    seconds_per_day = 86400.0
    
    return (unix_time / seconds_per_day) + epoch_offset


if __name__ == "__main__":
    # Test DTC protocol handler
    logging.basicConfig(level=logging.INFO)
    
    handler = DTCProtocolHandler()
    
    # Test logon request
    logon_msg = handler.create_logon_request(username="trader1", password="pass123")
    print(f"Logon request created: {len(logon_msg)} bytes")
    
    # Test market data request
    market_data_msg = handler.create_market_data_request(symbol="ESH25", exchange="CME")
    print(f"Market data request created: {len(market_data_msg)} bytes")
    
    # Test order submission
    order_msg = handler.create_submit_order(
        symbol="ESH25",
        side="BUY",
        quantity=1.0,
        order_type="MARKET",
        client_order_id="TEST001"
    )
    print(f"Order submission created: {len(order_msg)} bytes")
    
    print("\nDTC Protocol Handler initialized successfully ✓")

