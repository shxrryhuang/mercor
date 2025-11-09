You’re building an inventory analytics platform that needs a data validation and export feature before launch. Implement validate_and_export_inventory() to check data quality and export results in multiple formats.

Return a dictionary with four keys:

validation_passed → True/False (fails if any critical checks fail)

validation_issues → list of issue dicts with keys: check, severity, message, affected_rows

exports_generated → list of created filenames

export_data → the in-memory export structures for JSON, CSV, and TXT

Run all validation checks below. The critical checks fail validation:

Critical checks (implement helpers):

check_for_negative_counts() → negative on_hand, incoming, or sold_ytd

check_missing_required() → missing in sku, name, category, on_hand

check_category_validity() → category must be one of ['Electronics','Home','Beauty','Grocery']

check_duplicate_skus() → duplicate sku

Warning checks (do not fail validation):

check_high_return_rate() → return_rate_pct > 30

check_low_stock() → on_hand < 20 and incoming == 0

check_suspicious_pricing() → identical price_usd across all items in a category (flag affected rows)

Create three export files:

JSON inventory_summary.json

metadata: export_date (YYYY-MM-DD HH:MM:SS), total_skus, total_inventory_value_usd

category_snapshot: per category → item_count, sum_on_hand, avg_price_usd (2 decimals)

at_risk_items: top 5 items by highest return_rate_pct (list of SKUs)

CSV inventory_detailed.csv

All item fields plus a computed restock_tier using calculate_restock_tier(on_hand, incoming):

Immediate if on_hand < 10 and incoming == 0

Soon if on_hand < 25

Monitor if on_hand < 50

Healthy otherwise

Sorted by sku

TXT inventory_report.txt

Summary stats (totals, averages)

Category breakdown (counts, on-hand totals, avg price)

Top “at risk” items (by return_rate_pct)

Validation status and count of issues

Use helpers:

format_currency(value) → $123,456.78 (2 decimals)

calculate_restock_tier(on_hand, incoming) (as above)

Dates formatted as "YYYY-MM-DD HH:MM:SS". Round numeric outputs to 2 decimals. Run validation before exports. Store the final return value in a variable named result.