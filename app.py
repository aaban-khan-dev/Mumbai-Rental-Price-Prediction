"""
Mumbai Rent Intelligence — Flask web app.
"""

import sys
import os
import json

from flask import Flask, render_template, request, jsonify

from src.constant import *
from src.exception import RentException
from src.logger import logging
from src.pipeline.prediction_pipeline import PredictionPipeline
from src.model.shap_explainer import ShapExplainer

app = Flask(__name__)

# One prediction pipeline for the whole app (model is cached inside it after first load)
pipeline = PredictionPipeline()

# The 19 localities and other dropdown options shown in the form.
LOCALITIES = [
    "andheri-east", "andheri-west", "bandra-east", "bandra-west", "borivali-west",
    "chandivali", "chembur", "dadar", "ghatkopar", "goregaon-east", "goregaon-west",
    "kandivali-east", "kanjurmarg", "malad-west", "mira-road", "mulund-west",
    "powai", "thane-west", "vikhroli", "worli",
]
FURNISHING = ["Unfurnished", "Semi-Furnished", "Furnished"]
FACING = ["East", "West", "North", "South", "North - East", "North - West",
          "South - East", "South -West", "Unknown"]
PROPERTY_TYPES = ["flat", "independent"]


def _build_user_input(form) -> dict:
    """Turn the submitted form into the dict the prediction pipeline expects."""
    def as_int(name, default=0):
        v = form.get(name)
        return int(v) if v not in (None, "") else default

    return {
        "property_type": form.get("property_type", "flat"),
        "locality": form.get("locality"),
        "furnishing": form.get("furnishing", "Semi-Furnished"),
        "facing": form.get("facing", "Unknown"),
        "bhk": as_int("bhk", 1),
        "carpet_area": float(form.get("carpet_area") or 0),
        "floor_num": as_int("floor_num", 0),
        "total_floors": as_int("total_floors", 1),
        # optional preference toggles (checkboxes send 'on' when ticked)
        "available_immediately": 1 if form.get("available_immediately") else 0,
        "overlooks_garden": 1 if form.get("overlooks_garden") else 0,
        "overlooks_pool": 1 if form.get("overlooks_pool") else 0,
        "overlooks_main_road": 1 if form.get("overlooks_main_road") else 0,
        "near_school": 1 if form.get("near_school") else 0,
        "near_hospital": 1 if form.get("near_hospital") else 0,
        "near_mall": 1 if form.get("near_mall") else 0,
        "near_bus": 1 if form.get("near_bus") else 0,
        "near_railway": 1 if form.get("near_railway") else 0,
        # metro: if the user ticks "near metro", send a small time; else leave blank
        "metro_mins": 5 if form.get("near_metro") else None,
    }


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template(
            "predict.html",
            localities=LOCALITIES, furnishing=FURNISHING,
            facing=FACING, property_types=PROPERTY_TYPES,
        )

    try:
        user_input = _build_user_input(request.form)

        # 1. predict the fair rent
        predicted_rent = pipeline.predict(user_input)

        # 2. SHAP breakdown ("why this price")
        model = pipeline.get_model()
        explainer = ShapExplainer(model)
        input_df = pipeline.prepare_input(user_input)
        breakdown = explainer.explain(input_df, top_n=8)

        # 3. fair-price meter (only if the user entered their asking rent)
        asking = request.form.get("asking_rent")
        fair_price = None
        if asking not in (None, ""):
            asking = float(asking)
            diff_pct = round((asking - predicted_rent) / predicted_rent * 100, 1)
            fair_price = {
                "asking": asking,
                "diff_pct": diff_pct,
                "verdict": (
                    "overpriced" if diff_pct > 5
                    else "underpriced" if diff_pct < -5
                    else "fair"
                ),
            }

        # a simple confidence range (±8%) shown around the point estimate
        low = round(predicted_rent * 0.92, -2)
        high = round(predicted_rent * 1.08, -2)

        return render_template(
            "result.html",
            rent=predicted_rent, low=low, high=high,
            breakdown=breakdown, fair_price=fair_price,
            user_input_obj=user_input,
        )

    except Exception as e:
        raise RentException(e, sys)


@app.route("/whatif", methods=["POST"])
def whatif():
    """Re-run the prediction on a tweaked input and return the new rent (for the sliders)."""
    try:
        data = request.get_json()
        base_input = data["base_input"]
        changes = data["changes"]        # e.g. {"bhk": 4, "furnishing": "Furnished"}
        tweaked = dict(base_input)
        tweaked.update(changes)
        new_rent = pipeline.predict(tweaked)
        return jsonify({"rent": new_rent})
    except Exception as e:
        raise RentException(e, sys)


@app.route("/insights")
def insights():
    """Market-insights view — locality price ranking built from the dataset."""
    try:
        # This data is precomputed from the EDA and stored as a small JSON file.
        insights_path = os.path.join("static", "data", "market_insights.json")
        if os.path.exists(insights_path):
            with open(insights_path) as f:
                market = json.load(f)
        else:
            market = {"localities": []}   # graceful empty state
        return render_template("insights.html", market=market)
    except Exception as e:
        raise RentException(e, sys)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
