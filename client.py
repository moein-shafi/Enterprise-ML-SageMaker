import argparse
import json
import os
import requests
import sys
from typing import Any, Dict, Optional



DEFAULT_URL = "http://localhost:8000/predict"
DEFAULT_TIMEOUT = 5.0


SAMPLE_DATA: Dict[str, Any] = {
    "MedInc": 4.8036,
    "HouseAge": 4.0,
    "AveRooms": 3.9246575342465753,
    "AveBedrms": 1.0359589041095891,
    "Population": 1050.0,
    "AveOccup": 1.797945205479452,
    "Latitude": 37.39,
    "Longitude": -122.08,
    "RoomsPerHousehold": 2.182857142857143,  # AveRooms / AveOccup
}


def predict(url: str, payload: Dict[str, Any], timeout: float = DEFAULT_TIMEOUT) -> Dict[str, Any]:
    """
    Send JSON payload to prediction API and return parsed JSON response.
    Raises requests.exceptions.RequestException on network problems.
    Raises ValueError on invalid response content.
    """
    with requests.Session() as session:
        resp = session.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        try:
            return resp.json()
        except json.JSONDecodeError as exc:
            raise ValueError("Response is not valid JSON") from exc


def format_prediction(result: Dict[str, Any]) -> str:
    """
    Extract prediction from API response and format as dollar amount.
    The model returns values in units of 100,000s.
    """
    if "prediction" not in result:
        raise KeyError("Missing 'prediction' in response")

    pred = result["prediction"]
    # Support either a single numeric value or a one-element list/tuple
    if isinstance(pred, (list, tuple)):
        if not pred:
            raise ValueError("Empty prediction list")
        pred_value = float(pred[0])
    else:
        pred_value = float(pred)

    dollars = pred_value * 100_000
    return f"Predicted house value: ${dollars:,.2f}"


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Request a house-value prediction from the API.")
    p.add_argument("--url", "-u", default=os.environ.get("API_URL", DEFAULT_URL), help="Prediction API URL")
    p.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="Request timeout in seconds")
    p.add_argument("--json", "-j", help="Path to JSON file with input data (overrides built-in sample)")
    return p.parse_args(argv)


def load_input(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return SAMPLE_DATA
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
        if not isinstance(data, dict):
            raise ValueError("Input JSON must be an object/dictionary")
        return data


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)
    try:
        payload = load_input(args.json)
    except Exception as e:
        print(f"Error loading input data: {e}", file=sys.stderr)
        return 2

    try:
        result = predict(args.url, payload, timeout=args.timeout)
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to API at {args.url}", file=sys.stderr)
        return 3
    except requests.exceptions.Timeout:
        print("Error: Request timed out", file=sys.stderr)
        return 4
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error: {e} - Response: {getattr(e, 'response', None)}", file=sys.stderr)
        return 5
    except Exception as e:
        print(f"Error during request: {e}", file=sys.stderr)
        return 6

    try:
        print(format_prediction(result))
        # Optionally show status if provided
        if "status" in result:
            print(f"Status: {result['status']}")
        return 0
    except Exception as e:
        print(f"Error processing response: {e}", file=sys.stderr)
        return 7


if __name__ == "__main__":
    raise SystemExit(main())
