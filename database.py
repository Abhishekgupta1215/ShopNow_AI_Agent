"""
database.py — ShopNow AI Voice Agent
Mock database simulating order history, product catalogue, customer records,
return eligibility, and payment details. Replaces a real DB layer for local dev.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional


# ---------------------------------------------------------------------------
# Domain models
# ---------------------------------------------------------------------------

@dataclass
class Customer:
    customer_id: str
    name: str
    phone: str
    email: str
    tier: str  # "standard" | "premium" | "vip"


@dataclass
class Order:
    order_id: str
    customer_id: str
    status: str          # "processing" | "shipped" | "delivered" | "cancelled" | "returned"
    items: list          # list[dict]
    order_date: str
    expected_delivery: str
    actual_delivery: Optional[str]
    total_amount: float
    payment_status: str  # "paid" | "pending" | "failed" | "refunded"
    tracking_number: Optional[str]
    shipping_address: str
    carrier: str


@dataclass
class Product:
    product_id: str
    name: str
    category: str
    price: float
    description: str
    specifications: dict
    in_stock: bool
    rating: float
    review_count: int


@dataclass
class ReturnRequest:
    return_id: str
    order_id: str
    customer_id: str
    reason: str
    status: str          # "requested" | "approved" | "rejected" | "processing" | "completed"
    refund_amount: Optional[float]
    created_date: str


# ---------------------------------------------------------------------------
# Mock database
# ---------------------------------------------------------------------------

class MockDatabase:
    """
    In-memory mock of ShopNow's backend.
    Provides helpers identical to what a real ORM / API layer would expose.
    """

    RETURN_WINDOW_DAYS = 30

    def __init__(self) -> None:
        self._customers = self._seed_customers()
        self._orders = self._seed_orders()
        self._products = self._seed_products()
        self._return_requests = self._seed_returns()

    # ------------------------------------------------------------------
    # Seed helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _seed_customers() -> dict[str, Customer]:
        return {
            "C001": Customer("C001", "Rahul Sharma",   "+91-9876543210", "rahul@example.com",  "premium"),
            "C002": Customer("C002", "Priya Patel",    "+91-8765432109", "priya@example.com",  "standard"),
            "C003": Customer("C003", "Arjun Nair",     "+91-7654321098", "arjun@example.com",  "vip"),
            "C004": Customer("C004", "Ananya Singh",   "+91-6543210987", "ananya@example.com", "standard"),
            "C005": Customer("C005", "Karthik Rajan",  "+91-9988776655", "karthik@example.com","premium"),
        }

    @staticmethod
    def _seed_orders() -> dict[str, Order]:
        today = datetime.now()
        d = lambda n: (today + timedelta(days=n)).strftime("%Y-%m-%d")

        return {
            "ORD001": Order(
                order_id="ORD001", customer_id="C001",
                status="shipped",
                items=[{"product_id": "P001", "name": "Wireless Headphones", "qty": 1, "price": 2999.0}],
                order_date=d(-3), expected_delivery=d(2), actual_delivery=None,
                total_amount=2999.0, payment_status="paid",
                tracking_number="TRK123456789",
                shipping_address="123 MG Road, Bangalore", carrier="BlueDart",
            ),
            "ORD002": Order(
                order_id="ORD002", customer_id="C002",
                status="delivered",
                items=[
                    {"product_id": "P002", "name": "Running Shoes",  "qty": 1, "price": 3499.0},
                    {"product_id": "P003", "name": "Sports Socks",   "qty": 2, "price": 299.0},
                ],
                order_date=d(-10), expected_delivery=d(-5), actual_delivery=d(-4),
                total_amount=4097.0, payment_status="paid",
                tracking_number="TRK987654321",
                shipping_address="456 FC Road, Pune", carrier="Delhivery",
            ),
            "ORD003": Order(
                order_id="ORD003", customer_id="C003",
                status="processing",
                items=[{"product_id": "P004", "name": "Smartphone", "qty": 1, "price": 24999.0}],
                order_date=d(-1), expected_delivery=d(4), actual_delivery=None,
                total_amount=24999.0, payment_status="paid",
                tracking_number=None,
                shipping_address="789 Anna Salai, Chennai", carrier="FedEx",
            ),
            "ORD004": Order(
                order_id="ORD004", customer_id="C001",
                status="delivered",
                items=[{"product_id": "P005", "name": "Coffee Maker", "qty": 1, "price": 4500.0}],
                order_date=d(-20), expected_delivery=d(-14), actual_delivery=d(-13),
                total_amount=4500.0, payment_status="paid",
                tracking_number="TRK555111222",
                shipping_address="123 MG Road, Bangalore", carrier="BlueDart",
            ),
            "ORD005": Order(
                order_id="ORD005", customer_id="C004",
                status="cancelled",
                items=[{"product_id": "P006", "name": "Yoga Mat", "qty": 1, "price": 1200.0}],
                order_date=d(-5), expected_delivery=d(1), actual_delivery=None,
                total_amount=1200.0, payment_status="failed",
                tracking_number=None,
                shipping_address="321 Park Street, Kolkata", carrier="DTDC",
            ),
            "ORD006": Order(
                order_id="ORD006", customer_id="C005",
                status="delivered",
                items=[{"product_id": "P007", "name": "Air Purifier", "qty": 1, "price": 8999.0}],
                order_date=d(-15), expected_delivery=d(-9), actual_delivery=d(-8),
                total_amount=8999.0, payment_status="paid",
                tracking_number="TRK778899001",
                shipping_address="12 Brigade Road, Bangalore", carrier="Shadowfax",
            ),
        }

    @staticmethod
    def _seed_products() -> dict[str, Product]:
        return {
            "P001": Product(
                "P001", "Wireless Headphones", "Electronics", 2999.0,
                "Premium noise-cancelling wireless headphones with 30 hr battery life.",
                {"battery": "30 hours", "connectivity": "Bluetooth 5.0", "weight": "250 g",
                 "driver_size": "40 mm", "frequency_response": "20 Hz – 20 kHz"},
                True, 4.5, 1240,
            ),
            "P002": Product(
                "P002", "Running Shoes", "Footwear", 3499.0,
                "Lightweight running shoes with advanced cushioning and breathable mesh upper.",
                {"sizes": "UK 6–12", "material": "Mesh upper, Rubber sole", "sole": "EVA foam",
                 "weight_per_shoe": "280 g"},
                True, 4.3, 890,
            ),
            "P003": Product(
                "P003", "Sports Socks", "Clothing", 299.0,
                "Anti-blister sports socks with moisture-wicking technology.",
                {"material": "80 % cotton, 20 % nylon", "sizes": "Free size",
                 "pack": "3 pairs"},
                True, 4.1, 2100,
            ),
            "P004": Product(
                "P004", "Smartphone", "Electronics", 24999.0,
                "Latest flagship smartphone with 50 MP camera and 5 G connectivity.",
                {"display": "6.7-inch AMOLED 120 Hz", "battery": "5000 mAh",
                 "storage": "256 GB", "ram": "8 GB", "os": "Android 14"},
                True, 4.6, 3450,
            ),
            "P005": Product(
                "P005", "Coffee Maker", "Kitchen", 4500.0,
                "Programmable 12-cup drip coffee maker with thermal carafe.",
                {"capacity": "12 cups / 1.5 L", "features": "Auto-brew, Keep-warm 2 hr",
                 "wattage": "1100 W", "dimensions": "30 × 20 × 35 cm"},
                True, 4.2, 567,
            ),
            "P006": Product(
                "P006", "Yoga Mat", "Sports", 1200.0,
                "Non-slip premium yoga mat with alignment lines and carrying strap.",
                {"thickness": "6 mm", "material": "TPE (eco-friendly)",
                 "dimensions": "183 × 61 cm", "weight": "1.2 kg"},
                True, 4.4, 1890,
            ),
            "P007": Product(
                "P007", "Air Purifier", "Home Appliances", 8999.0,
                "HEPA + activated-carbon air purifier covering up to 400 sq ft.",
                {"coverage": "400 sq ft", "filter": "True HEPA H13 + Activated Carbon",
                 "noise_level": "22–55 dB", "power": "45 W"},
                True, 4.3, 712,
            ),
        }

    @staticmethod
    def _seed_returns() -> dict[str, ReturnRequest]:
        today = datetime.now()
        return {
            "RET001": ReturnRequest(
                return_id="RET001", order_id="ORD002", customer_id="C002",
                reason="Wrong size",
                status="approved",
                refund_amount=3499.0,
                created_date=(today - timedelta(days=2)).strftime("%Y-%m-%d"),
            ),
        }

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def get_customer_by_id(self, customer_id: str) -> Optional[Customer]:
        return self._customers.get(customer_id)

    def get_orders_by_customer(self, identifier: str) -> list[Order]:
        """Accept customer_id or phone number."""
        customer_id: Optional[str] = None
        for cid, customer in self._customers.items():
            if cid == identifier or customer.phone == identifier:
                customer_id = cid
                break
        if not customer_id:
            return []
        return [o for o in self._orders.values() if o.customer_id == customer_id]

    def get_order_by_id(self, order_id: str) -> Optional[Order]:
        return self._orders.get(order_id.upper())

    def check_return_eligibility(self, order_id: str) -> dict:
        order = self.get_order_by_id(order_id)
        if not order:
            return {"eligible": False, "reason": "Order not found."}
        if order.status != "delivered":
            return {
                "eligible": False,
                "reason": f"Order status is '{order.status}'. Only delivered orders can be returned.",
            }
        if not order.actual_delivery:
            return {"eligible": False, "reason": "Delivery date not yet confirmed."}

        delivery_dt = datetime.strptime(order.actual_delivery, "%Y-%m-%d")
        days_elapsed = (datetime.now() - delivery_dt).days
        if days_elapsed > self.RETURN_WINDOW_DAYS:
            return {
                "eligible": False,
                "reason": (
                    f"Return window expired — {days_elapsed} days since delivery "
                    f"(limit: {self.RETURN_WINDOW_DAYS} days)."
                ),
            }
        return {
            "eligible": True,
            "days_remaining": self.RETURN_WINDOW_DAYS - days_elapsed,
            "refund_amount": order.total_amount,
        }

    def create_return_request(self, order_id: str, reason: str) -> dict:
        eligibility = self.check_return_eligibility(order_id)
        if not eligibility.get("eligible"):
            return {"success": False, "reason": eligibility.get("reason")}

        order = self.get_order_by_id(order_id)
        return_id = f"RET{len(self._return_requests) + 1:03d}"
        self._return_requests[return_id] = ReturnRequest(
            return_id=return_id,
            order_id=order_id,
            customer_id=order.customer_id,
            reason=reason,
            status="requested",
            refund_amount=order.total_amount,
            created_date=datetime.now().strftime("%Y-%m-%d"),
        )
        return {
            "success": True,
            "return_id": return_id,
            "refund_amount": order.total_amount,
            "estimated_refund_days": "5–7 business days",
        }

    def get_payment_details(self, order_id: str) -> Optional[dict]:
        order = self.get_order_by_id(order_id)
        if not order:
            return None
        return {
            "order_id": order.order_id,
            "payment_status": order.payment_status,
            "total_amount": order.total_amount,
            "payment_method": "Credit / Debit Card" if order.payment_status == "paid" else "Unknown",
        }

    def get_product_by_id_or_name(self, identifier: str) -> Optional[Product]:
        if identifier.upper() in self._products:
            return self._products[identifier.upper()]
        lower = identifier.lower()
        for p in self._products.values():
            if lower in p.name.lower() or lower in p.category.lower():
                return p
        return None

    def search_products(self, query: str) -> list[Product]:
        lower = query.lower()
        return [
            p for p in self._products.values()
            if lower in p.name.lower()
            or lower in p.category.lower()
            or lower in p.description.lower()
        ]


# Singleton for import
db = MockDatabase()
