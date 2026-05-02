"""
tools.py — ShopNow AI Voice Agent
Five LangChain tools, one per customer-support intent:
  1. check_order_status
  2. process_return_or_refund
  3. handle_payment_issue
  4. handle_delivery_complaint
  5. get_product_information

Each tool is decorated with @tool so LangChain can expose it as an
OpenAI function-call / tool-call to the underlying LLM.
"""

from __future__ import annotations

from langchain_core.tools import tool

from database import db


# ---------------------------------------------------------------------------
# 1. Order Status
# ---------------------------------------------------------------------------

@tool
def check_order_status(query: str) -> str:
    """
    Retrieve the status and details of a ShopNow order.

    Accepts an order ID (e.g. 'ORD001') OR a customer phone number
    (e.g. '+91-9876543210').  Returns current status, items, tracking
    information, and expected / actual delivery date.
    """
    query = query.strip()
    order_id_candidate = query.upper()

    # Try direct order-ID lookup
    order = db.get_order_by_id(order_id_candidate)
    if order:
        status_notes: dict[str, str] = {
            "processing": (
                "Your order is being prepared and will be dispatched soon."
            ),
            "shipped": (
                f"Your order has been shipped via {order.carrier}. "
                f"Tracking number: {order.tracking_number}. "
                f"Expected delivery: {order.expected_delivery}."
            ),
            "delivered": (
                f"Your order was successfully delivered on {order.actual_delivery}."
            ),
            "cancelled": (
                "Your order has been cancelled. If you were charged, a refund will "
                "be processed within 5–7 business days."
            ),
            "returned": "Your order has been marked as returned.",
        }
        items_str = ", ".join(
            f"{i['name']} × {i['qty']}" for i in order.items
        )
        return (
            f"Order ID     : {order.order_id}\n"
            f"Status       : {order.status.upper()}\n"
            f"Items        : {items_str}\n"
            f"Total        : ₹{order.total_amount:,.2f}\n"
            f"Payment      : {order.payment_status}\n"
            f"Shipped to   : {order.shipping_address}\n"
            f"Details      : {status_notes.get(order.status, 'Status unavailable.')}"
        )

    # Try phone-number / customer-ID lookup
    orders = db.get_orders_by_customer(query)
    if orders:
        recent = sorted(orders, key=lambda o: o.order_date, reverse=True)[:3]
        lines = [f"Found {len(orders)} order(s) for your account. Most recent:\n"]
        for o in recent:
            lines.append(
                f"  • {o.order_id}: {o.status.upper()} "
                f"| ₹{o.total_amount:,.2f} | Ordered {o.order_date}"
            )
        return "\n".join(lines)

    return (
        "No order found with that information. "
        "Please double-check your Order ID or the phone number linked to your account."
    )


# ---------------------------------------------------------------------------
# 2. Returns & Refunds
# ---------------------------------------------------------------------------

@tool
def process_return_or_refund(input_str: str) -> str:
    """
    Initiate a return or refund request for a delivered ShopNow order.

    Input format: 'ORDER_ID|REASON'  (pipe-separated)
    Example     : 'ORD002|defective product'

    Checks return-window eligibility (30 days from delivery) and, if
    eligible, registers the request and provides a Return ID plus
    estimated refund timeline.
    """
    parts  = input_str.split("|", 1)
    order_id = parts[0].strip().upper()
    reason   = parts[1].strip() if len(parts) > 1 else "Not specified"

    eligibility = db.check_return_eligibility(order_id)

    if not eligibility.get("eligible"):
        return (
            f"Return request for {order_id}: NOT ELIGIBLE\n"
            f"Reason: {eligibility.get('reason', 'Unknown')}"
        )

    result = db.create_return_request(order_id, reason)
    if result.get("success"):
        return (
            f"Return request APPROVED for order {order_id}\n"
            f"Return ID          : {result['return_id']}\n"
            f"Refund amount      : ₹{result['refund_amount']:,.2f}\n"
            f"Estimated refund   : {result['estimated_refund_days']}\n"
            f"Days remaining in window: {eligibility.get('days_remaining', 'N/A')}\n"
            "A pickup confirmation will be sent to your registered email address. "
            "Please keep the item ready for collection."
        )

    return (
        f"Could not process the return for {order_id}: "
        f"{result.get('reason', 'Unknown error. Please try again.')}"
    )


# ---------------------------------------------------------------------------
# 3. Payment Issues
# ---------------------------------------------------------------------------

@tool
def handle_payment_issue(order_id: str) -> str:
    """
    Look up payment details and provide resolution guidance for a ShopNow order.

    Input : order ID (e.g. 'ORD005')
    Covers: failed payments, pending authorisations, double-charges, and
    refund status queries.
    """
    order_id = order_id.strip().upper()
    details  = db.get_payment_details(order_id)

    if not details:
        return (
            f"No payment record found for order {order_id}. "
            "Please verify the order ID."
        )

    status = details["payment_status"]
    amount = details["total_amount"]
    method = details["payment_method"]

    responses: dict[str, str] = {
        "paid": (
            f"Payment for {order_id}: CONFIRMED ✓\n"
            f"Amount : ₹{amount:,.2f}   Method: {method}\n"
            "Your payment was successfully processed. "
            "If you notice a duplicate charge, please share your bank "
            "reference number with us and we will investigate within 24 hours."
        ),
        "failed": (
            f"Payment for {order_id}: FAILED ✗\n"
            f"Amount : ₹{amount:,.2f}\n"
            "Common reasons: insufficient balance, card declined by bank, "
            "or a temporary gateway issue.\n"
            "Your order is on hold. Please retry with the same or a different "
            "payment method. If the amount was debited, it will be auto-refunded "
            "within 5–7 business days — no action needed on your end."
        ),
        "pending": (
            f"Payment for {order_id}: PENDING ⏳\n"
            f"Amount : ₹{amount:,.2f}\n"
            "Payment verification is in progress (usually resolves in 24–48 h). "
            "If you were charged and the order remains on hold beyond 48 hours, "
            "please contact us with your bank transaction ID."
        ),
        "refunded": (
            f"Refund for {order_id}: PROCESSED ✓\n"
            f"Amount : ₹{amount:,.2f}\n"
            "The refund has been initiated. Please allow 5–7 business days for "
            "it to reflect in your account, depending on your bank."
        ),
    }

    return responses.get(
        status,
        f"Payment status for {order_id}: {status}. "
        "Please contact our billing team at billing@shopnow.in for further assistance.",
    )


# ---------------------------------------------------------------------------
# 4. Delivery Complaints
# ---------------------------------------------------------------------------

@tool
def handle_delivery_complaint(input_str: str) -> str:
    """
    Handle a delivery-related complaint for a ShopNow order.

    Input format : 'ORDER_ID|COMPLAINT_TYPE'  (pipe-separated)
    Complaint types:
      delayed      — shipment taking longer than expected
      wrong_item   — different product was received
      damaged      — item arrived damaged
      missing      — ordered item not found inside the package
      not_received — marked delivered but customer never received it

    Example: 'ORD001|delayed'
    """
    parts         = input_str.split("|", 1)
    order_id      = parts[0].strip().upper()
    complaint_type = parts[1].strip().lower() if len(parts) > 1 else "general"

    order = db.get_order_by_id(order_id)
    if not order:
        return (
            f"Order {order_id} not found. "
            "Please verify your order number and try again."
        )

    tracking = order.tracking_number or "Not yet assigned"
    items_str = ", ".join(i["name"] for i in order.items)

    resolutions: dict[str, str] = {
        "delayed": (
            f"Delivery Delay — Order {order_id}\n"
            f"Current status   : {order.status.upper()}\n"
            f"Expected delivery: {order.expected_delivery}\n"
            f"Tracking         : {tracking} via {order.carrier}\n"
            "We sincerely apologise for the delay. Our logistics team has been "
            "alerted. If the package is more than 3 days past the expected date, "
            "you are eligible for a ₹100 delivery-compensation voucher applied to "
            "your next order."
        ),
        "wrong_item": (
            f"Wrong Item Received — Order {order_id}\n"
            f"Expected: {items_str}\n"
            "We apologise for this error! A replacement will be dispatched "
            "within 24–48 hours after we lodge a pickup request for the wrong item. "
            "Please do NOT discard or return the wrong item yourself — our courier "
            "partner will collect it from your address. "
            "A pickup confirmation will be emailed to you shortly."
        ),
        "damaged": (
            f"Damaged Item — Order {order_id}\n"
            f"Items: {items_str}\n"
            "We are very sorry your item arrived in a damaged condition. "
            "Please take a few photos of the packaging and the damaged product "
            "for our records. A replacement has been automatically initiated and "
            "a pickup of the damaged item will be scheduled within 24 hours. "
            "The replacement will ship once the pickup is confirmed."
        ),
        "missing": (
            f"Missing Item — Order {order_id}\n"
            f"Ordered items: {items_str}\n"
            "A missing-item investigation has been raised with our warehouse team. "
            "We will verify the dispatch records and respond within 2 business days. "
            "If confirmed missing, a replacement will be shipped within 24 hours."
        ),
        "not_received": (
            f"Non-Delivery Report — Order {order_id}\n"
            f"Status   : {order.status.upper()}\n"
            f"Tracking : {tracking} via {order.carrier}\n"
            "If the system shows 'delivered' but you have not received the package, "
            "please:\n"
            "  1. Check with neighbours, building security, or your mailroom.\n"
            "  2. If still missing, contact your local police station and share "
            "      the complaint number with us.\n"
            "We will coordinate with the carrier and resolve this within 3 business days. "
            "If liability is confirmed, a full refund or replacement will be arranged."
        ),
    }

    return resolutions.get(
        complaint_type,
        (
            f"Delivery complaint registered for order {order_id}.\n"
            f"Tracking: {tracking} via {order.carrier}\n"
            "Our delivery team will contact you within 24 hours."
        ),
    )


# ---------------------------------------------------------------------------
# 5. Product Queries
# ---------------------------------------------------------------------------

@tool
def get_product_information(query: str) -> str:
    """
    Retrieve ShopNow product details, specifications, pricing, and availability.

    Input : product ID (e.g. 'P001'), product name (e.g. 'Wireless Headphones'),
            or a category keyword (e.g. 'Electronics', 'Kitchen').
    Returns full product details including specs, rating, and stock status.
    """
    query = query.strip()

    # Direct or fuzzy lookup
    product = db.get_product_by_id_or_name(query)
    if product:
        specs_lines = "\n".join(
            f"    {k.replace('_', ' ').title()}: {v}"
            for k, v in product.specifications.items()
        )
        return (
            f"Product    : {product.name}  (ID: {product.product_id})\n"
            f"Category   : {product.category}\n"
            f"Price      : ₹{product.price:,.2f}\n"
            f"Description: {product.description}\n"
            f"Specifications:\n{specs_lines}\n"
            f"In Stock   : {'Yes ✓' if product.in_stock else 'No — Out of Stock'}\n"
            f"Rating     : {product.rating}/5 ({product.review_count:,} reviews)"
        )

    # Broader search
    results = db.search_products(query)
    if results:
        header = f"Found {len(results)} product(s) matching '{query}':\n"
        lines  = [
            f"  • {p.name} [{p.category}] — ₹{p.price:,.2f} "
            f"({'In Stock' if p.in_stock else 'Out of Stock'})  "
            f"★ {p.rating}/5"
            for p in results[:5]
        ]
        return header + "\n".join(lines)

    return (
        f"No products found matching '{query}'. "
        "Try searching by product name, category (Electronics, Footwear, "
        "Kitchen, Sports, Clothing, Home Appliances), or product ID (e.g. P001)."
    )


# ---------------------------------------------------------------------------
# Exported collection  +  intent-mapping for analytics
# ---------------------------------------------------------------------------

SHOPNOW_TOOLS = [
    check_order_status,
    process_return_or_refund,
    handle_payment_issue,
    handle_delivery_complaint,
    get_product_information,
]

# Maps tool name → analytics intent key
TOOL_TO_INTENT: dict[str, str] = {
    "check_order_status":       "order_status",
    "process_return_or_refund": "returns_refunds",
    "handle_payment_issue":     "payment_issues",
    "handle_delivery_complaint":"delivery_complaints",
    "get_product_information":  "product_queries",
}
