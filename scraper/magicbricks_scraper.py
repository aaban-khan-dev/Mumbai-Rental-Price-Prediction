"""
MagicBricks Rental Scraper 

   MULTI-URL: loops over many base URLs (property-type x city) to get more rows.
   SMARTER SCROLLING: incremental scroll steps + page-height detection to
    coax the site into lazy-loading more cards.

Still respectful (randomized delays), resilient (never crashes on a bad card),
and incremental (saves to CSV as it goes). Only scrapes permitted results pages.

"""

import asyncio
import random
import csv
import os
from datetime import datetime
from playwright.async_api import async_playwright

# URL GENERATION
# We build MANY URLs by combining locality x BHK x property-type, because each
# single URL caps around ~59 listings. Hundreds of URLs x ~59 = a large dataset.
#
# URL patterns confirmed from the live site (all WITHOUT `proptype=`):
#   flats-for-rent-in-{loc}-mumbai-pppfr
#   {n}-bhk-flats-for-rent-in-{loc}-mumbai-pppfr
#   independent-house-for-rent-in-{loc}-mumbai-pppfr
#   villa-for-rent-in-{loc}-mumbai-pppfr
# ---------------------------------------------------------------------------

CITY = "mumbai"

# Locality slugs (lowercase, hyphenated). Taken from the site's locality lists.

LOCALITIES = [
    "andheri-east", "andheri-west", "bandra-west", "bandra-east", "chembur",
    "mira-road", "powai", "goregaon-east", "goregaon-west", "worli",
    "malad-west", "borivali-west", "mulund-west", "chandivali", "kandivali-east",
    "thane-west", "vikhroli", "ghatkopar", "kanjurmarg", "dadar",
]

# BHK values to expand flats into (each is a separate URL slice).
BHK_VALUES = [1, 2, 3, 4, 5]

INCLUDE_FLATS = True         # flats-for-rent + {n}-bhk-flats-for-rent per locality
INCLUDE_HOUSES = True        # independent-house-for-rent per locality
INCLUDE_VILLAS = True        # villa-for-rent per locality


def build_targets():
    """Generate the full list of target URLs from localities x bhk x type."""
    base = "https://www.magicbricks.com"
    targets = []
    for loc in LOCALITIES:
        if INCLUDE_FLATS:
            # The general flats page for the locality (all BHK)
            targets.append({
                "url": f"{base}/flats-for-rent-in-{loc}-{CITY}-pppfr",
                "property_type": "flat", "city": CITY, "locality": loc,
            })
            # One page per BHK value for that locality
            for n in BHK_VALUES:
                targets.append({
                    "url": f"{base}/{n}-bhk-flats-for-rent-in-{loc}-{CITY}-pppfr",
                    "property_type": "flat", "city": CITY, "locality": loc,
                })
        if INCLUDE_HOUSES:
            targets.append({
                "url": f"{base}/independent-house-for-rent-in-{loc}-{CITY}-pppfr",
                "property_type": "house", "city": CITY, "locality": loc,
            })
        if INCLUDE_VILLAS:
            targets.append({
                "url": f"{base}/villa-for-rent-in-{loc}-{CITY}-pppfr",
                "property_type": "villa", "city": CITY, "locality": loc,
            })
    return targets


TARGETS = build_targets()

TARGET_ROWS_PER_URL = 500  
GLOBAL_ROW_TARGET = 6000       # stop the WHOLE run once we've collected this many NEW rows
OUTPUT_CSV = os.path.join("data","magicbricks_rentals.csv")

# Politeness 
MIN_DELAY = 2.5
MAX_DELAY = 5.0
HEADLESS = False               # keep False while watching it work

# --- Scrolling tuning (the upgrade) ---
SCROLL_STEP_PX = 1200          # scroll this many pixels at a time (incremental, not one big jump)
SCROLL_STEP_PAUSE = 1.2        # wait after each small scroll for lazy-load to fire
STEPS_PER_ROUND = 3            # how many small scrolls before re-harvesting
MAX_STALE_SCROLLS = 6          # give up only after THIS many no-new-card rounds (was 3 in v1)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


# FIELD EXTRACTION

async def safe_text(card, selector):
    try:
        el = await card.query_selector(selector)
        if el:
            txt = await el.inner_text()
            return txt.strip() if txt else None
    except Exception:
        pass
    return None


async def extract_summary_fields(card):
    result = {}
    try:
        items = await card.query_selector_all("div.mb-srp__card__summary__list--item")
        for item in items:
            try:
                label = await item.get_attribute("data-summary")
                value_el = await item.query_selector(".mb-srp__card__summary--value")
                value = (await value_el.inner_text()).strip() if value_el else None
                if label:
                    result[label] = value
            except Exception:
                continue
    except Exception:
        pass
    return result


async def extract_nearby(card):
    try:
        tags = await card.query_selector_all("span.mb-srp-m__card__nearby__tag--item")
        texts = []
        for t in tags:
            try:
                txt = (await t.inner_text()).strip()
                if txt:
                    texts.append(txt)
            except Exception:
                continue
        return " | ".join(texts) if texts else None
    except Exception:
        return None


async def extract_amenities(card):
    try:
        items = await card.query_selector_all(
            "ul.mb-srp__card__accomodation li, .mb-srp__card__accomodation__item"
        )
        return len(items) if items else None
    except Exception:
        return None


def parse_subtype(title):
    """
    Pull the finer property type from the title text.
    The URL-level `property_type` is flat/house/villa, but titles often say
    'Apartment', 'Studio', 'Penthouse', 'Builder Floor', etc. This captures that.
    Returns a lowercase subtype string, or None.
    """
    if not title:
        return None
    t = title.lower()
    for kw in ["penthouse", "studio", "builder floor", "independent house",
               "villa", "apartment", "flat"]:
        if kw in t:
            return kw
    return None


async def extract_card(card, property_type, city):
    title = await safe_text(card, "h2.mb-srp__card--title")
    price = await safe_text(card, "div.mb-srp__card__price--amount")
    price_sqft = await safe_text(card, "div.mb-srp__card__price--size")
    description = await safe_text(card, "div.mb-srp__card--desc-text")
    summary = await extract_summary_fields(card)
    nearby = await extract_nearby(card)
    amenity_count = await extract_amenities(card)

    return {
        "property_type": property_type,          # from the URL: flat / house / villa
        "property_subtype": parse_subtype(title), # from the title: apartment / studio / villa / ...
        "city": city,
        "title": title,
        "price_raw": price,
        "price_per_sqft_raw": price_sqft,
        "carpet_area_raw": summary.get("carpet-area"),
        "floor_raw": summary.get("floor"),
        "status_raw": summary.get("status"),
        "furnishing": summary.get("furnishing"),
        "facing": summary.get("facing"),
        "overlooking": summary.get("overlooking"),
        "amenity_count": amenity_count,
        "nearby_raw": nearby,
        "description_raw": description,
        "scraped_at": datetime.now().isoformat(timespec="seconds"),
    }



# CSV

FIELDNAMES = [
    "property_type", "property_subtype", "city", "locality", "title", "price_raw", "price_per_sqft_raw",
    "carpet_area_raw", "floor_raw", "status_raw", "furnishing", "facing",
    "overlooking", "amenity_count", "nearby_raw", "description_raw", "scraped_at",
]


def append_rows(rows):
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    file_exists = os.path.exists(OUTPUT_CSV) and os.path.getsize(OUTPUT_CSV) > 0
    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_existing_keys():
    """
    Read any rows already in the CSV and return their dedup keys.
    This lets you RUN THE SCRIPT MULTIPLE TIMES without deleting the file:
    listings already saved are skipped, only genuinely new ones get appended.
    So you can scrape in batches over time and never get duplicates.
    """
    keys = set()
    if not (os.path.exists(OUTPUT_CSV) and os.path.getsize(OUTPUT_CSV) > 0):
        return keys
    try:
        with open(OUTPUT_CSV, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                keys.add(dedup_key(row))
        print(f"Loaded {len(keys)} existing listings from {OUTPUT_CSV} (will skip these).")
    except Exception as e:
        print(f"Could not read existing CSV ({e}); starting fresh.")
    return keys



# SCRAPE ONE URL — improved incremental scrolling

def dedup_key(row):
    """
    A stronger dedup key than title alone. Two different flats can share a title
    like '2 BHK Flat for Rent in Andheri West', so we combine title + area + price.
    This collides far less often, so we avoid both re-saving duplicates AND
    wrongly dropping genuinely different listings.
    """
    return (
        (row.get("title") or "").strip().lower(),
        (row.get("carpet_area_raw") or "").strip().lower(),
        (row.get("price_raw") or "").strip().lower(),
    )


async def harvest_visible_cards(page, target, seen_keys):
    """Scrape all currently-rendered cards not seen yet. Returns list of new rows.

    `seen_keys` is shared GLOBALLY across all URLs in this run, so a property that
    somehow appears under two categories is saved only once per run.
    """
    cards = await page.query_selector_all("div.mb-srp__card")
    new_rows = []
    for card in cards:
        row = await extract_card(card, target["property_type"], target["city"])
        if not row.get("title"):
            continue                      # skip ads / placeholder cards with no title
        row["locality"] = target.get("locality")   # which locality-URL this came from
        key = dedup_key(row)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        new_rows.append(row)
    return new_rows


async def scrape_target(page, target, seen_keys):
    url = target["url"]
    print(f"\n=== Scraping {target['property_type']} in {target['city']} ===")
    print(f"URL: {url}")

    await page.goto(url, wait_until="domcontentloaded", timeout=60000)
    try:
        await page.wait_for_selector("div.mb-srp__card", timeout=30000)
    except Exception:
        print("  No cards appeared — skipping.")
        return 0

    collected = 0
    stale_scrolls = 0
    last_height = 0

    while collected < TARGET_ROWS_PER_URL:
        # 1. Harvest whatever is currently rendered.
        new_rows = await harvest_visible_cards(page, target, seen_keys)
        if new_rows:
            append_rows(new_rows)
            collected += len(new_rows)
            stale_scrolls = 0
            print(f"  +{len(new_rows)} new (total {collected})")
        else:
            stale_scrolls += 1

        if stale_scrolls >= MAX_STALE_SCROLLS:
            print("  No new cards after repeated scrolls — reached the end.")
            break

        # 2. INCREMENTAL scroll: several small steps rather than one big jump.
        for _ in range(STEPS_PER_ROUND):
            await page.evaluate(f"window.scrollBy(0, {SCROLL_STEP_PX})")
            await asyncio.sleep(SCROLL_STEP_PAUSE)

        # 3. Did the page actually grow (new content loaded)?
        current_height = await page.evaluate("document.body.scrollHeight")
        if current_height == last_height:
            stale_scrolls += 1   # no growth this round counts toward giving up
        last_height = current_height

        # 4. Polite randomized delay before the next round.
        await asyncio.sleep(random.uniform(MIN_DELAY, MAX_DELAY))

    print(f"  Done: {collected} listings for {target['property_type']} in {target['city']}")
    return collected


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=HEADLESS)
        context = await browser.new_context(
            user_agent=USER_AGENT,
            viewport={"width": 1366, "height": 768},
        )
        page = await context.new_page()

        grand_total = 0
        # GLOBAL dedup set, preloaded with anything already in the CSV.
        # -> prevents duplicates ACROSS urls AND across separate runs.
        # -> so you can run repeatedly / in batches WITHOUT deleting the file.
        seen_keys = load_existing_keys()
        start_count = len(seen_keys)
        print(f"Generated {len(TARGETS)} URLs to scrape.\n")
        for i, target in enumerate(TARGETS, 1):
            print(f"[URL {i}/{len(TARGETS)}]", end=" ")
            try:
                grand_total += await scrape_target(page, target, seen_keys)
            except Exception as e:
                print(f"  ERROR on {target['url']}: {e}")

            # Stop the whole run once we've gathered enough NEW rows this session.
            new_this_run = len(seen_keys) - start_count
            if new_this_run >= GLOBAL_ROW_TARGET:
                print(f"\nReached global target ({new_this_run} new rows). Stopping early.")
                break

            await asyncio.sleep(random.uniform(5, 10))  # human-like pause between URLs

        await browser.close()
        print(f"\n=== FINISHED. Total rows: {grand_total}. Saved to {OUTPUT_CSV} ===")


if __name__ == "__main__":
    asyncio.run(main())