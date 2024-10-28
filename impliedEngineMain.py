import math
import logging
from typing import List, Dict, Tuple, Callable, Optional, Set
from collections import defaultdict

# Configure logging at the module level
logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG for more detailed logs
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Logs will be output to the console
    ]
)

# Define Instrument class
class Instrument:
    def __init__(self, name: str, tick_size: float, contract_size: int = 1, settlement_price: float = 0.0, is_spread: bool = False):
        self.name = name
        self.tick_size = tick_size
        self.contract_size = contract_size  # Used to handle different contract sizes
        self.settlement_price = settlement_price
        self.is_spread = is_spread

    def __repr__(self):
        return f"Instrument({self.name})"

# Pricing Function for Inter-Commodity Spreads
def intercommodity_pricing_function(
    leg_info: Dict[str, Tuple[float, int]],
    spread: 'Spread'
) -> Tuple[Optional[float], Optional[float]]:
    """
    Pricing function for inter-commodity spreads based on the formula:
    Spread Price = Front Leg Price Change - Back Leg Price Change * Price Ratio
    """
    if len(spread.legs) != 2:
        logging.error("Intercommodity pricing function only supports spreads with two legs.")
        raise NotImplementedError("Intercommodity pricing function only supports spreads with two legs.")

    # Unpack legs
    (front_leg, front_ratio), (back_leg, back_ratio) = spread.legs

    # Ensure one leg is positive and one is negative
    if not ((front_ratio > 0 and back_ratio < 0) or (front_ratio < 0 and back_ratio > 0)):
        logging.error("Invalid spread ratios. One leg must have a positive ratio and the other a negative ratio.")
        raise ValueError("Invalid spread ratios. One leg must have a positive ratio and the other a negative ratio.")

    # Use provided price_ratio or calculate from ratios
    price_ratio = spread.price_ratio if spread.price_ratio is not None else abs(front_ratio / back_ratio)

    # Retrieve leg prices and settlement prices
    front_leg_bid = leg_info.get(f"{front_leg.name}_bid")
    front_leg_ask = leg_info.get(f"{front_leg.name}_ask")
    back_leg_bid = leg_info.get(f"{back_leg.name}_bid")
    back_leg_ask = leg_info.get(f"{back_leg.name}_ask")

    front_settlement = front_leg.settlement_price
    back_settlement = back_leg.settlement_price

    # Calculate price changes from settlement
    front_bid_change = None
    if front_leg_bid:
        front_bid_change = front_leg_bid[0] - front_settlement

    front_ask_change = None
    if front_leg_ask:
        front_ask_change = front_leg_ask[0] - front_settlement

    back_bid_change = None
    if back_leg_bid:
        back_bid_change = back_leg_bid[0] - back_settlement

    back_ask_change = None
    if back_leg_ask:
        back_ask_change = back_leg_ask[0] - back_settlement

    # Calculate spread bid price
    spread_bid_price = None
    if front_bid_change is not None and back_ask_change is not None:
        spread_bid_price = front_bid_change - back_ask_change * price_ratio
        logging.debug(f"Spread: {spread.name}, Spread Bid Price: {spread_bid_price}")

    # Calculate spread ask price
    spread_ask_price = None
    if front_ask_change is not None and back_bid_change is not None:
        spread_ask_price = front_ask_change - back_bid_change * price_ratio
        logging.debug(f"Spread: {spread.name}, Spread Ask Price: {spread_ask_price}")

    return (spread_bid_price, spread_ask_price)

# Define Spread class
class Spread:
    def __init__(
        self,
        name: str,
        legs: List[Tuple[Instrument, int]],  # List of tuples (Instrument, size_ratio)
        price_ratio: Optional[float] = None,  # Price ratio between legs
        pricing_function: Optional[Callable[[Dict[str, Tuple[float, int]], 'Spread'], Tuple[Optional[float], Optional[float]]]] = None,
    ):
        """
        :param name: Name of the spread
        :param legs: List of tuples (Instrument, size_ratio). Positive for long, negative for short
        :param price_ratio: Price ratio between legs (if different from size ratio)
        :param pricing_function: Custom pricing function
        """
        self.name = name
        self.legs = legs
        self.price_ratio = price_ratio
        self.pricing_function = pricing_function if pricing_function else intercommodity_pricing_function

    def calculate_spread_price(self, leg_info: Dict[str, Tuple[float, int]]) -> Tuple[Optional[float], Optional[float]]:
        """
        Calculates both spread bid and ask prices.
        """
        try:
            return self.pricing_function(leg_info, self)
        except Exception as e:
            logging.error(f"Pricing function error for spread {self.name}: {e}")
            return (None, None)

    def __repr__(self):
        return f"Spread({self.name})"

# Define OrderLayer class
class OrderLayer:
    def __init__(
        self,
        bid_price: float,
        bid_size: int,
        ask_price: float,
        ask_size: int,
        bid_is_implied: bool = False,
        ask_is_implied: bool = False,
        bid_source_instruments: Optional[Set[str]] = None,
        ask_source_instruments: Optional[Set[str]] = None,
    ):
        self.bid_price = bid_price
        self.bid_size = bid_size
        self.ask_price = ask_price
        self.ask_size = ask_size
        self.bid_is_implied = bid_is_implied
        self.ask_is_implied = ask_is_implied
        self.bid_source_instruments = bid_source_instruments or set()
        self.ask_source_instruments = ask_source_instruments or set()
        # Implied orders
        self.implied_bids: List['ImpliedOrder'] = []
        self.implied_asks: List['ImpliedOrder'] = []

    def add_implied_order(self, implied_order: 'ImpliedOrder'):
        if implied_order.is_bid:
            self.implied_bids.append(implied_order)
        else:
            self.implied_asks.append(implied_order)

    def __repr__(self):
        return (
            f"OrderLayer(Bid: {self.bid_price}x{self.bid_size}, "
            f"Ask: {self.ask_price}x{self.ask_size}, "
            f"Implied Bids: {self.implied_bids}, "
            f"Implied Asks: {self.implied_asks})"
        )

# Define OrderBook class
class OrderBook:
    def __init__(self, instrument: Instrument):
        self.instrument = instrument
        self.layers: List[OrderLayer] = []

    def add_layer(
        self,
        bid_price: float,
        bid_size: int,
        ask_price: float,
        ask_size: int,
        implied_orders: Optional[List['ImpliedOrder']] = None,
        bid_is_implied: bool = False,
        ask_is_implied: bool = False,
        bid_source_instruments: Optional[Set[str]] = None,
        ask_source_instruments: Optional[Set[str]] = None,
    ):
        layer = OrderLayer(
            bid_price, bid_size, ask_price, ask_size,
            bid_is_implied=bid_is_implied,
            ask_is_implied=ask_is_implied,
            bid_source_instruments=bid_source_instruments,
            ask_source_instruments=ask_source_instruments,
        )
        if implied_orders:
            for io in implied_orders:
                layer.add_implied_order(io)
        self.layers.append(layer)
        self.sort_layers()
        logging.debug(f"Added layer to {self.instrument.name}: {layer}")
        logging.debug(f"Current layers for {self.instrument.name}: {self.layers}")

    def sort_layers(self):
        # Sort layers based on bid_price (descending) and ask_price (ascending)
        self.layers.sort(
            key=lambda x: (
                -x.bid_price if x.bid_size > 0 else math.inf,
                x.ask_price if x.ask_size > 0 else math.inf
            )
        )

    def get_best_bid(self) -> Optional[Tuple[float, int, bool, Set[str]]]:
        for layer in self.layers:
            if layer.bid_size != 0:
                return (
                    layer.bid_price, layer.bid_size,
                    layer.bid_is_implied, layer.bid_source_instruments
                )
        return None

    def get_best_ask(self) -> Optional[Tuple[float, int, bool, Set[str]]]:
        for layer in self.layers:
            if layer.ask_size != 0:
                return (
                    layer.ask_price, layer.ask_size,
                    layer.ask_is_implied, layer.ask_source_instruments
                )
        return None

    def __repr__(self):
        return f"OrderBook({self.instrument.name}, Layers: {self.layers})"

# Define ImpliedOrder class to track implied orders
class ImpliedOrder:
    def __init__(
        self,
        instrument: Instrument,
        price: float,
        size: int,
        source: str,
        is_bid: bool,
        source_instruments: Optional[Set[str]] = None,
    ):
        self.instrument = instrument
        self.price = price
        self.size = size
        self.source = source  # Concise source information
        self.is_bid = is_bid  # True if bid, False if ask
        self.source_instruments = source_instruments or set()

    def __repr__(self):
        side = "Bid" if self.is_bid else "Ask"
        return (
            f"ImpliedOrder({side}, Price: {self.price}, Size: {self.size}, "
            f"Source: {self.source}, Source Instruments: {self.source_instruments})"
        )

# Define the main LiquidityEngine
class LiquidityEngine:
    def __init__(
        self,
        instruments: List[Instrument],
        spreads: List[Spread],
        method: str = 'iterative',  # 'iterative' or 'graph'
        max_iterations: int = 10,  # Maximum allowed iterations
        max_depth: int = 3,         # Maximum depth for implied liquidity calculation
        rounding: bool = False,
    ):
        self.instruments = {inst.name: inst for inst in instruments}
        self.spreads = spreads
        self.method = method
        self.max_iterations = max_iterations
        self.max_depth = max_depth
        self.rounding = rounding  # Whether to apply rounding to implied prices
        # Initialize order books for each instrument
        self.order_books: Dict[str, OrderBook] = {inst.name: OrderBook(inst) for inst in instruments}
        # Implied orders
        self.implied_orders: Dict[str, List[ImpliedOrder]] = defaultdict(list)
        # Construct adjacency list (graph representation)
        self.adjacency_list = self.build_adjacency_list()

        self.order_books.update({spread.name: OrderBook(spread) for spread in spreads})  # Create order books for spreads

    def build_adjacency_list(self) -> Dict[str, List[str]]:
        """
        Build an adjacency list representing the connections between spreads and instruments.
        """
        adjacency = defaultdict(list)
        for spread in self.spreads:
            for leg, _ in spread.legs:
                adjacency[spread.name].append(leg.name)
                adjacency[leg.name].append(spread.name)
        return adjacency

    def add_order(
        self,
        instrument_name: str,
        bid_price: float,
        bid_size: int,
        ask_price: float,
        ask_size: int,
    ):
        if instrument_name not in self.order_books:
            logging.error(f"Instrument {instrument_name} not found.")
            raise ValueError(f"Instrument {instrument_name} not found.")
        self.order_books[instrument_name].add_layer(
            bid_price, bid_size, ask_price, ask_size,
            bid_is_implied=False,
            ask_is_implied=False,
            bid_source_instruments={instrument_name},
            ask_source_instruments={instrument_name},
        )
        logging.info(f"Added order to {instrument_name}: Bid {bid_price}x{bid_size}, Ask {ask_price}x{ask_size}")

    def calculate_implied_liquidity(self):
        """
        Calculate implied liquidity using the iterative method.
        """
        new_implied = True
        iteration = 0
        while new_implied and iteration < self.max_iterations:
            new_implied = False
            iteration += 1
            logging.info(f"--- Iteration {iteration} ---")
            # Process implications from outrights to spreads
            for spread in self.spreads:
                logging.debug(f"Processing spread: {spread.name}")
                # Gather leg prices and sizes
                leg_info = self.get_leg_info(spread)
                # Calculate spread bid and ask prices
                spread_bid_price, spread_ask_price = spread.calculate_spread_price(leg_info)
                # Create implied orders
                new_implied |= self.create_implied_orders(spread, spread_bid_price, spread_ask_price, leg_info)
            # Process implications from spreads to outrights
            for spread in self.spreads:
                logging.debug(f"Processing reverse spread: {spread.name}")
                # Get spread's best bid and ask
                spread_book = self.order_books.get(spread.name)
                if not spread_book:
                    continue
                spread_best_bid = spread_book.get_best_bid()
                spread_best_ask = spread_book.get_best_ask()
                # Gather spread info
                spread_info = {}
                if spread_best_bid:
                    spread_info[f"{spread.name}_bid"] = spread_best_bid
                if spread_best_ask:
                    spread_info[f"{spread.name}_ask"] = spread_best_ask
                # Generate implied orders on legs
                new_implied |= self.create_implied_orders_on_legs(spread, spread_info)
            if not new_implied:
                logging.info("No new implied layers generated in this iteration.")

    def get_leg_info(self, spread: Spread) -> Dict[str, Tuple[float, int, bool, Set[str]]]:
        leg_info = {}
        for leg, _ in spread.legs:
            book = self.order_books.get(leg.name)
            if book and book.layers:
                best_bid = book.get_best_bid()
                best_ask = book.get_best_ask()
                if best_bid:
                    leg_info[f"{leg.name}_bid"] = best_bid
                    logging.debug(f"{leg.name} Best Bid: {best_bid[0]}, Size: {best_bid[1]}")
                if best_ask:
                    leg_info[f"{leg.name}_ask"] = best_ask
                    logging.debug(f"{leg.name} Best Ask: {best_ask[0]}, Size: {best_ask[1]}")
        return leg_info

    def create_implied_orders(
        self,
        spread: Spread,
        spread_bid_price: Optional[float],
        spread_ask_price: Optional[float],
        leg_info: Dict[str, Tuple[float, int, bool, Set[str]]]
    ) -> bool:
        new_implied = False
        # Initialize flags to track if bid and/or ask can be created
        can_create_bid = spread_bid_price is not None
        can_create_ask = spread_ask_price is not None

        if not (can_create_bid or can_create_ask):
            logging.info(f"Cannot calculate either spread bid or ask prices for {spread.name}. Skipping.")
            return False  # Cannot calculate either price

        if can_create_bid:
            logging.info(f"Spread: {spread.name}, Spread Bid Price: {spread_bid_price}")
        if can_create_ask:
            logging.info(f"Spread: {spread.name}, Spread Ask Price: {spread_ask_price}")

        # Apply optional rounding to valid tick sizes
        if self.rounding:
            if can_create_bid:
                spread_bid_price = self.round_price(
                    spread_bid_price,
                    is_bid=True,
                    tick_size=self.instruments[spread.name].tick_size
                )
                logging.debug(f"Rounded Spread Bid Price for {spread.name}: {spread_bid_price}")
            if can_create_ask:
                spread_ask_price = self.round_price(
                    spread_ask_price,
                    is_bid=False,
                    tick_size=self.instruments[spread.name].tick_size
                )
                logging.debug(f"Rounded Spread Ask Price for {spread.name}: {spread_ask_price}")

        # Determine implied sizes for bid and ask separately
        implied_size_bid, implied_size_ask = self.calculate_implied_sizes(spread, leg_info)

        # Create implied bid if possible
        if can_create_bid and implied_size_bid > 0:
            source_instruments = set()
            for leg, _ in spread.legs:
                # Collect source instruments from legs' bids and asks
                leg_bid = leg_info.get(f"{leg.name}_bid")
                leg_ask = leg_info.get(f"{leg.name}_ask")
                if leg_bid:
                    source_instruments.update(leg_bid[3])
                if leg_ask:
                    source_instruments.update(leg_ask[3])
            source_instruments.add(spread.name)
            new_implied |= self.add_implied_order(
                spread_name=spread.name,
                price=spread_bid_price,
                size=implied_size_bid,
                source=self.generate_source(spread, leg_info, is_bid=True),
                is_bid=True,
                source_instruments=source_instruments
            )

        # Create implied ask if possible
        if can_create_ask and implied_size_ask > 0:
            source_instruments = set()
            for leg, _ in spread.legs:
                leg_bid = leg_info.get(f"{leg.name}_bid")
                leg_ask = leg_info.get(f"{leg.name}_ask")
                if leg_bid:
                    source_instruments.update(leg_bid[3])
                if leg_ask:
                    source_instruments.update(leg_ask[3])
            source_instruments.add(spread.name)
            new_implied |= self.add_implied_order(
                spread_name=spread.name,
                price=spread_ask_price,
                size=implied_size_ask,
                source=self.generate_source(spread, leg_info, is_bid=False),
                is_bid=False,
                source_instruments=source_instruments
            )

        return new_implied

    def create_implied_orders_on_legs(
        self,
        spread: Spread,
        spread_info: Dict[str, Tuple[float, int, bool, Set[str]]]
    ) -> bool:
        new_implied = False
        # Unpack legs
        (front_leg, front_ratio), (back_leg, back_ratio) = spread.legs
        # Retrieve spread prices
        spread_best_bid = spread_info.get(f"{spread.name}_bid")
        spread_best_ask = spread_info.get(f"{spread.name}_ask")
        front_book = self.order_books.get(front_leg.name)
        back_book = self.order_books.get(back_leg.name)
        if not front_book or not back_book:
            return False

        # Use provided price_ratio or calculate from ratios
        price_ratio = spread.price_ratio if spread.price_ratio is not None else abs(front_ratio / back_ratio)
        size_ratio_front = abs(front_ratio)
        size_ratio_back = abs(back_ratio)

        # Generate implied orders on legs from spread ask (selling spread)
        if spread_best_ask:
            spread_ask_price, spread_ask_size, spread_ask_is_implied, spread_ask_source_instruments = spread_best_ask
            # Implied Ask on Front Leg
            back_best_bid = back_book.get_best_bid()
            if back_best_bid:
                back_bid_price, back_bid_size, back_bid_is_implied, back_bid_source_instruments = back_best_bid
                # Check for circular implication
                if spread_ask_is_implied and front_leg.name in spread_ask_source_instruments:
                    logging.info(f"Skipping implied ask on {front_leg.name} to prevent circular implication.")
                else:
                    front_implied_ask_price = spread_ask_price + back_bid_price
                    if self.rounding:
                        front_implied_ask_price = self.round_price(
                            front_implied_ask_price,
                            is_bid=False,
                            tick_size=front_leg.tick_size
                        )
                    # Calculate size
                    size = min(
                        spread_ask_size * size_ratio_front,
                        back_bid_size * size_ratio_front // size_ratio_back
                    )
                    if size > 0:
                        source_instruments = spread_ask_source_instruments.union(back_bid_source_instruments)
                        source_instruments.add(spread.name)
                        new_implied |= self.add_implied_order(
                            spread_name=front_leg.name,
                            price=front_implied_ask_price,
                            size=size,
                            source=f"{spread.name}_ask:{spread_ask_price} + {back_leg.name}_bid:{back_bid_price}",
                            is_bid=False,
                            source_instruments=source_instruments
                        )
            # Implied Bid on Back Leg
            front_best_ask = front_book.get_best_ask()
            if front_best_ask:
                front_ask_price, front_ask_size, front_ask_is_implied, front_ask_source_instruments = front_best_ask
                # Check for circular implication
                if spread_ask_is_implied and back_leg.name in spread_ask_source_instruments:
                    logging.info(f"Skipping implied bid on {back_leg.name} to prevent circular implication.")
                else:
                    back_implied_bid_price = front_ask_price - spread_ask_price
                    if self.rounding:
                        back_implied_bid_price = self.round_price(
                            back_implied_bid_price,
                            is_bid=True,
                            tick_size=back_leg.tick_size
                        )
                    # Calculate size
                    size = min(
                        spread_ask_size * size_ratio_back,
                        front_ask_size * size_ratio_back // size_ratio_front
                    )
                    if size > 0:
                        source_instruments = spread_ask_source_instruments.union(front_ask_source_instruments)
                        source_instruments.add(spread.name)
                        new_implied |= self.add_implied_order(
                            spread_name=back_leg.name,
                            price=back_implied_bid_price,
                            size=size,
                            source=f"{front_leg.name}_ask:{front_ask_price} - {spread.name}_ask:{spread_ask_price}",
                            is_bid=True,
                            source_instruments=source_instruments
                        )

        # Generate implied orders on legs from spread bid (buying spread)
        if spread_best_bid:
            spread_bid_price, spread_bid_size, spread_bid_is_implied, spread_bid_source_instruments = spread_best_bid
            # Implied Bid on Front Leg
            back_best_ask = back_book.get_best_ask()
            if back_best_ask:
                back_ask_price, back_ask_size, back_ask_is_implied, back_ask_source_instruments = back_best_ask
                # Check for circular implication
                if spread_bid_is_implied and front_leg.name in spread_bid_source_instruments:
                    logging.info(f"Skipping implied bid on {front_leg.name} to prevent circular implication.")
                else:
                    front_implied_bid_price = spread_bid_price + back_ask_price
                    if self.rounding:
                        front_implied_bid_price = self.round_price(
                            front_implied_bid_price,
                            is_bid=True,
                            tick_size=front_leg.tick_size
                        )
                    # Calculate size
                    size = min(
                        spread_bid_size * size_ratio_front,
                        back_ask_size * size_ratio_front // size_ratio_back
                    )
                    if size > 0:
                        source_instruments = spread_bid_source_instruments.union(back_ask_source_instruments)
                        source_instruments.add(spread.name)
                        new_implied |= self.add_implied_order(
                            spread_name=front_leg.name,
                            price=front_implied_bid_price,
                            size=size,
                            source=f"{spread.name}_bid:{spread_bid_price} + {back_leg.name}_ask:{back_ask_price}",
                            is_bid=True,
                            source_instruments=source_instruments
                        )
            # Implied Ask on Back Leg
            front_best_bid = front_book.get_best_bid()
            if front_best_bid:
                front_bid_price, front_bid_size, front_bid_is_implied, front_bid_source_instruments = front_best_bid
                # Check for circular implication
                if spread_bid_is_implied and back_leg.name in spread_bid_source_instruments:
                    logging.info(f"Skipping implied ask on {back_leg.name} to prevent circular implication.")
                else:
                    back_implied_ask_price = front_bid_price - spread_bid_price
                    if self.rounding:
                        back_implied_ask_price = self.round_price(
                            back_implied_ask_price,
                            is_bid=False,
                            tick_size=back_leg.tick_size
                        )
                    # Calculate size
                    size = min(
                        spread_bid_size * size_ratio_back,
                        front_bid_size * size_ratio_back // size_ratio_front
                    )
                    if size > 0:
                        source_instruments = spread_bid_source_instruments.union(front_bid_source_instruments)
                        source_instruments.add(spread.name)
                        new_implied |= self.add_implied_order(
                            spread_name=back_leg.name,
                            price=back_implied_ask_price,
                            size=size,
                            source=f"{front_leg.name}_bid:{front_bid_price} - {spread.name}_bid:{spread_bid_price}",
                            is_bid=False,
                            source_instruments=source_instruments
                        )

        return new_implied

    def calculate_implied_sizes(self, spread: Spread, leg_info: Dict[str, Tuple[float, int, bool, Set[str]]]) -> Tuple[int, int]:
        implied_size_bid = math.inf
        implied_size_ask = math.inf
        for leg, size_ratio in spread.legs:
            ratio_abs = abs(size_ratio)
            if size_ratio > 0:
                # For bid: use bid sizes
                available_bid = leg_info.get(f"{leg.name}_bid", (0, 0, False, set()))[1]
                possible_bid = available_bid // ratio_abs
                implied_size_bid = min(implied_size_bid, possible_bid)
                # For ask: use ask sizes
                available_ask = leg_info.get(f"{leg.name}_ask", (0, 0, False, set()))[1]
                possible_ask = available_ask // ratio_abs
                implied_size_ask = min(implied_size_ask, possible_ask)
            else:
                # For bid: use ask sizes
                available_bid = leg_info.get(f"{leg.name}_ask", (0, 0, False, set()))[1]
                possible_bid = available_bid // ratio_abs
                implied_size_bid = min(implied_size_bid, possible_bid)
                # For ask: use bid sizes
                available_ask = leg_info.get(f"{leg.name}_bid", (0, 0, False, set()))[1]
                possible_ask = available_ask // ratio_abs
                implied_size_ask = min(implied_size_ask, possible_ask)
        if implied_size_bid == math.inf:
            implied_size_bid = 0
        if implied_size_ask == math.inf:
            implied_size_ask = 0
        logging.info(f"Calculated Implied Sizes for {spread.name} - Bid: {implied_size_bid}, Ask: {implied_size_ask}")
        return implied_size_bid, implied_size_ask

    def generate_source(self, spread: Spread, leg_info: Dict[str, Tuple[float, int, bool, Set[str]]], is_bid: bool) -> str:
        (front_leg, front_ratio), (back_leg, back_ratio) = spread.legs
        if is_bid:
            source = f"{front_leg.name}_bid:{leg_info.get(f'{front_leg.name}_bid', ('N/A',))[0]}, {back_leg.name}_ask:{leg_info.get(f'{back_leg.name}_ask', ('N/A',))[0]}"
        else:
            source = f"{front_leg.name}_ask:{leg_info.get(f'{front_leg.name}_ask', ('N/A',))[0]}, {back_leg.name}_bid:{leg_info.get(f'{back_leg.name}_bid', ('N/A',))[0]}"
        return source

    def add_implied_order(
        self, spread_name: str, price: float, size: int,
        source: str, is_bid: bool, source_instruments: Set[str]
    ) -> bool:
        # Check for duplication
        existing_orders = self.implied_orders[spread_name]
        duplicate = any(
            order.price == price and order.size == size and order.is_bid == is_bid
            for order in existing_orders
        )
        if duplicate:
            return False
        # Create implied order
        implied_order = ImpliedOrder(
            instrument=self.instruments[spread_name],
            price=price,
            size=size,
            source=source,
            is_bid=is_bid,
            source_instruments=source_instruments,
        )
        # Add to implied orders
        self.implied_orders[spread_name].append(implied_order)
        logging.info(f"Created {'Bid' if is_bid else 'Ask'} Implied Order for {spread_name}: {implied_order}")

        # Merge into order book
        spread_book = self.order_books.get(spread_name)
        if not spread_book:
            # Initialize order book for spread if it doesn't exist
            spread_instrument = Instrument(spread_name, tick_size=self.instruments[spread_name].tick_size, is_spread=True)
            self.instruments[spread_name] = spread_instrument
            self.order_books[spread_name] = OrderBook(spread_instrument)
            spread_book = self.order_books[spread_name]
            logging.debug(f"Initialized Order Book for {spread_name}")
        # Add layer
        bid_price = price if is_bid else 0
        bid_size = size if is_bid else 0
        ask_price = price if not is_bid else 0
        ask_size = size if not is_bid else 0
        spread_book.add_layer(
            bid_price=bid_price,
            bid_size=bid_size,
            ask_price=ask_price,
            ask_size=ask_size,
            implied_orders=[implied_order],
            bid_is_implied=is_bid,
            ask_is_implied=not is_bid,
            bid_source_instruments=source_instruments if is_bid else None,
            ask_source_instruments=source_instruments if not is_bid else None,
        )
        logging.info(f"Added {'Bid' if is_bid else 'Ask'} Implied Order to Order Book for {spread_name}")
        return True

    def round_price(self, price: float, is_bid: bool, tick_size: float) -> float:
        """
        Round the price based on the tick size.
        Bids are rounded down, asks are rounded up.
        """
        multiplier = 1 / tick_size
        if is_bid:
            return math.floor(price * multiplier) / multiplier
        else:
            return math.ceil(price * multiplier) / multiplier

    def print_order_books(self):
        """
        Print all order books in a human-readable format.
        """
        for instrument_name, order_book in self.order_books.items():
            print(f"Order Book for {instrument_name}:")
            header = f"{'Implied Bids':>30} | {'Bid Price':>10} | {'Bid Size':>8} | {'Ask Price':>10} | {'Ask Size':>8} | {'Implied Asks':>30}"
            print(header)
            print("-" * len(header))
            if not order_book.layers:
                print(f"{'-':>30} | {'-':>10} | {'-':>8} | {'-':>10} | {'-':>8} | {'-':>30}")
                continue
            for layer in order_book.layers:
                # Format implied bids and asks
                implied_bids = ", ".join(
                    f"{io.price}@{io.size} ({io.source})" for io in layer.implied_bids
                ) or "-"
                implied_asks = ", ".join(
                    f"{io.price}@{io.size} ({io.source})" for io in layer.implied_asks
                ) or "-"
                # Format prices and sizes
                bid_price = f"{layer.bid_price:.4f}" if layer.bid_size != 0 else "-"
                bid_size = f"{layer.bid_size}" if layer.bid_size != 0 else "-"
                ask_price = f"{layer.ask_price:.4f}" if layer.ask_size != 0 else "-"
                ask_size = f"{layer.ask_size}" if layer.ask_size != 0 else "-"
                print(
                    f"{implied_bids:>30} | {bid_price:>10} | {bid_size:>8} | {ask_price:>10} | {ask_size:>8} | {implied_asks:>30}"
                )
            print("\n")

# Example usage demonstrating the LiquidityEngine
def main():
    # Define instruments
    zt = Instrument("ZT", 0.5, settlement_price=0.0)  # 2-Year Treasury Note Futures
    zf = Instrument("ZF", 0.5, settlement_price=0.0)  # 5-Year Treasury Note Futures
    # Assume settlement prices are zero for simplicity

    zt_zf_spread = Instrument("ZT_ZF", 0.5, is_spread=True)  # Spread Instrument

    instruments = [zt, zf, zt_zf_spread]

    # Define spreads
    # Spread: ZT - ZF (1:1 ratio)
    spread_zt_zf = Spread(
        name="ZT_ZF",
        legs=[(zt, 1), (zf, -1)],
        pricing_function=intercommodity_pricing_function
    )

    spreads = [spread_zt_zf]

    # Initialize LiquidityEngine with iterative method, rounding enabled
    engine = LiquidityEngine(
        instruments=instruments,
        spreads=spreads,
        method='iterative',
        max_iterations=5,
        max_depth=3,
        rounding=False
    )

    # Add sample orders to outrights
    # ZT Order Book
    engine.add_order("ZT", bid_price=111.0, bid_size=10, ask_price=111.5, ask_size=10)
    engine.add_order("ZT", bid_price=110.5, bid_size=10, ask_price=112.0, ask_size=10)
    engine.add_order("ZT", bid_price=110.0, bid_size=10, ask_price=112.5, ask_size=10)

    # ZF Order Book
    engine.add_order("ZF", bid_price=100.0, bid_size=10, ask_price=101.0, ask_size=10)
    engine.add_order("ZF", bid_price=99.0, bid_size=10, ask_price=102.0, ask_size=10)
    engine.add_order("ZF", bid_price=98.0, bid_size=10, ask_price=103.0, ask_size=10)

    # Add an order to a spread's order book to test bidirectional implied liquidity
    # Example: Adding an ask to ZT_ZF spread
    engine.add_order("ZT_ZF", bid_price=0.0, bid_size=0, ask_price=10.5, ask_size=10)

    # Run full simulation
    engine.calculate_implied_liquidity()

    # Print combined order books after simulation
    print("\nAfter Simulation:")
    engine.print_order_books()

if __name__ == "__main__":
    # Run the example usage
    main()
